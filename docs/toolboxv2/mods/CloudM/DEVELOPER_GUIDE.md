# LiveSync — Entwickler-Dokumentation

**Architektur, Datenfluss, Protokoll und Erweiterungspunkte.**

---

## Architektur-Übersicht

```
┌──────────────┐     HTTP/API      ┌───────────────────┐     S3 API     ┌──────────┐
│  ToolBoxV2   │ ←────────────────→ │   SyncServer      │ ←────────────→ │  MinIO   │
│  (Supervisor)│  start/stop/status │   (eigener Prozess)│  Store/Retrieve│  (Store) │
│  __init__.py │                    │   server.py        │                │          │
└──────────────┘                    └─────────┬──────────┘                └──────────┘
                                              │
                                     WebSocket (Metadaten ONLY)
                                              │
                                    ┌─────────┴──────────┐
                               ┌────┴─────┐        ┌────┴─────┐
                               │ Client A │        │ Client B │
                               │ client.py│        │ client.py│
                               └──────────┘        └──────────┘
```

**Kernprinzip:** WebSocket transportiert **NIE** Dateiinhalte. Nur Metadaten und MinIO-Keys. Clients laden Dateien direkt von MinIO.

---

## Modul-Übersicht

```
LiveSync/
├── __init__.py        387 LOC   Supervisor: subprocess, share CRUD, @export
├── protocol.py        347 LOC   Pydantic Message-Models, MsgType enum, ignore rules
├── config.py          126 LOC   SyncConfig, ShareToken, ENV loading
├── crypto.py          191 LOC   AES-256-GCM + zlib, wraps toolboxv2 cryp
├── index.py           221 LOC   aiosqlite LocalIndex (files + sync_log)
├── minio_helper.py    301 LOC   MinIO I/O, key helpers, CredentialBroker
├── conflict.py        153 LOC   Konflikt-Erkennung, Merge-Marker, .sync-trash
├── server.py          662 LOC   WS-Server, Auth, Broadcast, Watchdog→asyncio
└── client.py          791 LOC   WS-Client, Debounce, Upload/Download Engine
```

---

## Datenfluss: Eine Datei synchronisieren

```
1. Client A: Watchdog erkennt Änderung
2. Client A: Event → asyncio Queue (thread-safe, KEIN create_task aus sync!)
3. Client A: DebounceBatch sammelt 2s, dedupliziert
4. Client A: Checksum prüfen — skip wenn Index-Match
5. Client A: zlib compress → AES-256-GCM encrypt
6. Client A: Upload zu MinIO (direkt, nicht durch Server)
7. Client A: WS → Server: {type: "file_changed", path, checksum, minio_key}
8. Server:   Index updaten, Konflikt prüfen
9. Server:   Broadcast an Client B: {type: "file_changed", ..., source_client: "A"}
10. Client B: Download von MinIO → decrypt → decompress
11. Client B: Checksum verifizieren
12. Client B: Atomic write (.sync-tmp → rename)
13. Client B: Index updaten
```

---

## WebSocket-Protokoll

Alle Messages folgen dem Envelope-Format:

```json
{
  "type": "file_changed",
  "payload": { ... },
  "timestamp": 1713379200.0,
  "msg_id": "a1b2c3d4e5f6"
}
```

### Message-Types

| Type | Richtung | Payload-Felder | Zweck |
|------|----------|---------------|-------|
| `auth` | C→S | client_id, device_type, share_id | Verbindungsaufbau |
| `auth_success` | S→C | client_id, minio_credentials, checksums | Auth OK + Init-State |
| `file_changed` | C→S, S→C | path, checksum, minio_key, file_type, source_client | Datei geändert |
| `file_deleted` | C→S, S→C | path, source_client | Datei gelöscht |
| `file_renamed` | C→S, S→C | old_path, new_path, checksum, minio_key | Datei umbenannt |
| `request_full` | C→S | — | Full-State anfordern |
| `full_state_ready` | S→C | minio_key, file_count | Index-DB in MinIO bereit |
| `conflict` | S→C | path, local_checksum, remote_checksum, resolution | Konflikt erkannt |
| `ack` | S→C | path, checksum | Änderung bestätigt |
| `ping` / `pong` | bidirektional | — | Keepalive (30s) |
| `error` | S→C | message, path | Fehler |

Alle Payload-Models sind in `protocol.py` als Pydantic-Klassen definiert. Factory-Methoden: `SyncMessage.file_changed(...)`, `SyncMessage.auth(...)`, etc.

---

## MinIO-Layout

```
livesync/                          ← Bucket
└── {share_id}/
    ├── notes.md.enc               ← verschlüsselt + komprimiert
    ├── sub/deep/file.txt.enc
    └── .meta/
        ├── notes.md.json          ← {checksum, mtime, source_client}
        └── index.db.gz            ← Full-State Export (Szenario S5)
```

Key-Helpers in `minio_helper.py`:

```python
make_object_key("share1", "sub/notes.md")  → "share1/sub/notes.md.enc"
make_meta_key("share1", "sub/notes.md")    → "share1/.meta/sub/notes.md.json"
rel_path_from_object_key("share1", "share1/sub/notes.md.enc") → "sub/notes.md"
```

---

## Credential-Flow

```
1. Host erstellt Share → share_id + AES-Key generiert
2. Token enthält: share_id, minio_endpoint, enc_key, ws_endpoint (KEINE Creds!)
3. Client verbindet per WS → sendet auth mit share_id
4. Server ruft CredentialBroker.vend_share_credentials(share_id) auf
5. MinIO Service Account wird erstellt (scoped auf livesync/{share_id}/*)
6. Credentials kommen im auth_success zurück
7. Client initialisiert eigenen Minio-Client mit diesen Credentials
8. Alle Transfers gehen direkt Client↔MinIO
```

`CredentialBroker` kommt aus `toolboxv2.mods.CloudM.auth.minio_policy`. Fallback: Admin-Credentials mit Warnung im Log.

---

## SQLite-Index

Jeder Client und der Server halten eine lokale SQLite-DB (`aiosqlite`):

```sql
-- Datei-Tracking
CREATE TABLE files (
    rel_path    TEXT PRIMARY KEY,
    mtime       REAL,
    size        INTEGER,
    checksum    TEXT,           -- SHA-256, erste 16 Hex-Zeichen
    sync_state  TEXT,           -- 'synced', 'pending_upload', 'pending_download', 'conflict'
    remote_key  TEXT,
    updated_at  REAL
);

-- Audit-Log
CREATE TABLE sync_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rel_path    TEXT,
    action      TEXT,           -- 'upload', 'download', 'conflict', 'delete'
    checksum    TEXT,
    timestamp   REAL,
    client_id   TEXT
);
```

Für Full-State Sync (Szenario S5): `index.export_gzipped()` → hochladen nach MinIO → Client lädt herunter → `index.import_gzipped()`.

---

## Watchdog → asyncio Bridge

**Problem:** Watchdog-Callbacks laufen in einem separaten Thread. `asyncio.create_task()` aus einem Nicht-asyncio-Thread crasht.

**Lösung:** `loop.call_soon_threadsafe(queue.put_nowait, event)` in `AsyncWatchdogHandler` / `ClientWatchdogHandler`.

```python
class AsyncWatchdogHandler(FileSystemEventHandler):
    def __init__(self, loop, queue, vault_path):
        self.loop = loop
        self.queue = queue
        ...

    def on_modified(self, event):
        # Thread-safe! Kein create_task!
        self.loop.call_soon_threadsafe(
            self.queue.put_nowait, ("modified", rel_path)
        )
```

Hauptschleife draint die Queue → DebounceBatch (2s sammeln, dedup) → Batch-Prozess.

---

## Konflikterkennung

Ablauf im Server bei eingehender `file_changed`:

```
1. Server hat Index-Eintrag für path? → Nein: kein Konflikt, speichern
2. Ja: Server-Checksum == Client-Checksum? → Ja: kein Konflikt
3. Nein: KONFLIKT erkannt
4. FileType prüfen:
   - .md → resolution = "merge_markers"
   - Alles andere → resolution = "latest_wins"
5. Conflict-Notification an ALLE Clients
6. sync_log Eintrag
7. Index trotzdem updaten (latest writer wins am Index)
```

Merge-Marker werden client-seitig eingefügt (`conflict.resolve_md_conflict()`). Der Client der den Merge macht, lädt die zusammengeführte Version wieder hoch.

---

## Sicherheitsregeln (Safety Invariants)

1. **Kein stiller Datenverlust.** Jeder Fehler wird geloggt und gemeldet.
2. **Backup vor Überschreiben.** `create_backup()` vor jedem Download/Overwrite.
3. **Löschen → .sync-trash.** Remote-Deletes werden nie sofort ausgeführt.
4. **Checksum-Verifikation.** Nach jedem Download wird die SHA-256 geprüft. Bei Mismatch: Retry (3x), dann `pending_download` + Error-Log.
5. **Atomic Writes.** Immer `.sync-tmp` → `os.rename()`. Nie direkt in die Zieldatei schreiben.
6. **Konflikte sind sichtbar.** Console-Log + sync_log + WS-Notification.

---

## Szenarien-Mapping

| Szenario | Beschreibung | Implementiert in |
|----------|-------------|------------------|
| S1 | Client offline, kommt wieder | `client._catchup_sync()` |
| S2 | Beide online, Live-Sync | `client._ws_recv_loop()` + `_watch_loop()` |
| S3 | Gleichzeitige Änderung (Konflikt) | `server._check_conflict()` + `conflict.py` |
| S4 | Netzwerk-Abbruch bei Download | `client._download_file()` mit Retry + Checksum |
| S5 | Neuer Client, Full State | `server._send_full_state()` + `index.export_gzipped()` |
| S6 | Datei gelöscht | `client._handle_remote_delete()` → `.sync-trash/` |

---

## Erweiterungspunkte

**Neuen Dateityp zur Konfliktstrategie hinzufügen:**
→ `protocol.py`: `_EXT_MAP` erweitern, ggf. neuen `FileType` Enum-Wert
→ `conflict.py`: Handler für neuen Typ in Server/Client

**Anderer Encryption-Algorithmus:**
→ `crypto.py`: `_encrypt_raw()` / `_decrypt_raw()` anpassen. Interface bleibt gleich.

**Web-UI / Dashboard:**
→ `__init__.py` exportiert alle Status-Funktionen. HTTP-Endpoint um `get_sync_status()` + `get_sync_log()` wrappen.

**Delta-Sync (nur Diffs statt Full-File):**
→ Aktuell: Full-File Replace. Für Delta: `crypto.py` um Diff-Logik erweitern, `minio_helper.py` um Patch-Upload, `protocol.py` um `FILE_PATCHED` Message.

**Multi-User / ACLs:**
→ `CredentialBroker` unterstützt bereits per-User Policies. `server.py` um User-Management erweitern, Auth per JWT statt client_id.

---

## Tests ausführen

```bash
# Alle 95 Tests
python -m unittest discover tests/ -v

# Einzelnes Modul
python -m unittest tests/test_crypto.py -v

# Nur Server-Tests
python -m unittest tests/test_server.py -v
```

Alle Tests nutzen `unittest` (kein pytest). MinIO-Calls sind gemockt — kein laufender MinIO nötig für Tests.

---

## Dependencies

| Package | Version | Zweck |
|---------|---------|-------|
| `websockets` | ≥12.0 | WS-Server/Client |
| `watchdog` | ≥3.0 | Filesystem Events |
| `minio` | ≥7.2 | S3/MinIO Client |
| `cryptography` | ≥41.0 | AES-256-GCM |
| `aiosqlite` | ≥0.19 | Async SQLite |
| `pydantic` | ≥2.0 | Message Models |
| `zlib` | stdlib | Kompression |

---

## Log-Format

```
[LiveSync] 2026-04-17T15:30:00 INFO  Sync started for share abc123
[LiveSync] 2026-04-17T15:30:01 INFO  File synced: notes.md from desktop-markin
[LiveSync] 2026-04-17T15:30:02 WARN  Conflict detected: notes.md (server=abc, client=def)
[LiveSync] 2026-04-17T15:30:03 ERROR MinIO upload failed: connection refused — retrying in 30s
[LiveSync] 2026-04-17T15:30:04 INFO  Reconnected — caught up 5 files
```

Logger-Name: `"LiveSync"`. Konfigurierbar über Standard-Python-Logging.
