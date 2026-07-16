# LiveSync — Benutzerhandbuch

**Obsidian-Vault live zwischen Geräten synchronisieren.**

Datei auf dem Desktop ändern → in Sekunden auf dem Handy.

---

## Was ist LiveSync?

LiveSync hält einen Ordner (z.B. deinen Obsidian-Vault) automatisch zwischen mehreren Geräten synchron. Alle Dateien werden vor dem Upload verschlüsselt — der Server sieht nur Ciphertext.

Unterstützte Geräte: Linux Desktop, Laptop, Termux (Android), weitere Linux-Rechner.

---

## Voraussetzungen

**Auf dem Server (RYZEN):**

- MinIO läuft und ist erreichbar
- Python 3.10+
- ToolBoxV2 installiert

**Auf jedem Client-Gerät:**

```bash
pip install websockets watchdog minio cryptography aiosqlite pydantic
```

---

## Schnellstart

### 1. Share erstellen (auf dem Server)

```bash
# Über ToolBoxV2
tb CloudM.LiveSync create_share --vault_path /srv/obsidian/vaults/main --port 8765
```

Ausgabe:

```
{
  "ok": true,
  "share_id": "a1b2c3d4",
  "token": "eyJ2IjogMSwg...",
  "info": "Share created. Distribute token to join."
}
```

Den **Token** kopieren — das ist dein Zugangsschlüssel.

### 2. Client verbinden (Desktop/Laptop)

```bash
python -m toolboxv2.mods.CloudM.LiveSync.client \
  --token "eyJ2IjogMSwg..." \
  --vault /home/markin/obsidian-vault
```

LiveSync erstellt den Ordner falls nötig, lädt alle Dateien herunter und hält ab dann alles synchron.

### 3. Client verbinden (Termux/Android)

```bash
# In Termux
pkg install python
pip install websockets watchdog minio cryptography aiosqlite pydantic

python -m toolboxv2.mods.CloudM.LiveSync.client \
  --token "eyJ2IjogMSwg..." \
  --vault /data/data/com.termux/files/home/obsidian
```

Danach die Obsidian-App auf den gleichen Ordner zeigen lassen.

---

## Alltag

### Status prüfen

```bash
tb CloudM.LiveSync sync_status
```

### Aktive Shares anzeigen

```bash
tb CloudM.LiveSync list_shares
```

### Share stoppen

```bash
tb CloudM.LiveSync stop_share --share_id a1b2c3d4
```

### Server neustarten

```bash
tb CloudM.LiveSync restart_sync --vault_path /srv/obsidian/vaults/main
```

### Sync-Log anschauen

```bash
tb CloudM.LiveSync get_sync_log --share_id a1b2c3d4 --limit 20
```

### Abhängigkeiten prüfen

```bash
tb CloudM.LiveSync selftest
```

---

## Was wird synchronisiert?

| Dateityp | Beispiele | Synchronisiert? |
|----------|-----------|-----------------|
| Markdown | .md, .txt | Ja |
| Daten | .json, .csv | Ja |
| Bilder | .png, .jpg, .gif, .webp | Ja |
| PDFs | .pdf | Ja |
| Obsidian-Config | .obsidian/* | Nein (ignoriert) |
| Git | .git/* | Nein (ignoriert) |
| Temp-Dateien | .tmp, .sync-tmp | Nein (ignoriert) |

Maximale Dateigröße: **50 MB** pro Datei.

---

## Konflikte

Wenn zwei Geräte gleichzeitig die gleiche Datei ändern, passiert Folgendes:

**Markdown-Dateien (.md):** Beide Versionen werden mit Merge-Markern zusammengeführt. Du siehst dann:

```markdown
<<<<<<< LOCAL (desktop-markin @ 18:30:01)
Deine lokale Version...
=======
Die andere Version...
>>>>>>> REMOTE (handy @ 18:30:00)
```

Lösche einfach die Version die du nicht brauchst und die Marker-Zeilen.

**Bilder und PDFs:** Die neuere Version gewinnt. Die ältere wird als Backup gespeichert.

**Wichtig:** Konflikte werden NIE stillschweigend gelöst. Du siehst immer eine Meldung im Log.

---

## Gelöschte Dateien

Wenn eine Datei auf einem Gerät gelöscht wird, wird sie auf den anderen Geräten **nicht sofort gelöscht**, sondern nach `.sync-trash/` verschoben. Du kannst sie dort jederzeit wiederherstellen.

---

## Offline-Nutzung

LiveSync funktioniert auch offline. Änderungen werden lokal gesammelt und beim nächsten Verbinden automatisch synchronisiert. Nichts geht verloren.

Beim Reconnect siehst du im Log:

```
[LiveSync] Reconnected — caught up 5 files
```

---

## Troubleshooting

**"MinIO unreachable"**
→ Prüfe ob MinIO läuft: `systemctl status minio`
→ Prüfe die ENV-Variablen: `MINIO_ENDPOINT`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`

**"websockets not installed"**
→ `pip install websockets`

**Client verbindet sich nicht**
→ Ist Port 8765 offen? `ss -tlnp | grep 8765`
→ Firewall-Regel prüfen

**"Checksum mismatch after 3 retries"**
→ Die Datei auf dem Server ist korrupt. Manuell prüfen und ggf. neu hochladen.

**Dateien werden nicht erkannt**
→ Liegt die Datei in `.obsidian/` oder `.git/`? Diese werden absichtlich ignoriert.
→ Ist die Datei größer als 50 MB?

**Token ungültig**
→ Token muss vollständig kopiert werden (keine Zeilenumbrüche). Am besten per QR-Code oder Clipboard.

---

## Umgebungsvariablen

| Variable | Default | Beschreibung |
|----------|---------|-------------|
| `MINIO_ENDPOINT` | `127.0.0.1:9000` | MinIO Server |
| `MINIO_ROOT_USER` | `admin` | MinIO Admin User |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | MinIO Admin Passwort |
| `MINIO_SECURE` | `false` | HTTPS verwenden |
| `LIVESYNC_WS_HOST` | `0.0.0.0` | WebSocket Bind-Adresse |
| `LIVESYNC_WS_PORT` | `8765` | WebSocket Port |
| `LIVESYNC_BUCKET` | `livesync` | MinIO Bucket Name |

---

## Sicherheit

- Dateien werden **vor dem Upload** mit AES-256-GCM verschlüsselt
- Der Server sieht nur verschlüsselte Daten
- Jeder Share hat einen eigenen Verschlüsselungsschlüssel
- MinIO-Zugangsdaten werden pro Client individuell erstellt und sind auf den jeweiligen Share beschränkt
- Der Share-Token enthält **keine** MinIO-Credentials — die kommen erst nach WebSocket-Authentifizierung
