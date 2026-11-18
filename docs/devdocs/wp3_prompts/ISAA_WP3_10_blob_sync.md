# ISAA-Auftrag WP3-10 – Blob-Sync über die Data-API

> Basis: `master` **nach** WP3-08 (gepusht). Welle 3. Owner-Entscheidung 2026-11-18: Blob-Sync gehört wieder zu Ziel 7; Löschungen werden übertragen, auch für Blobs.
> Hintergrund: `d672504` hat `CloudM/sync/blob_sync.py` gelöscht (`git show d672504^:toolboxv2/mods/CloudM/sync/blob_sync.py`). Der war eine zweite Sync-Maschine neben LiveSync. **Nicht wiederherstellen.**

## 0. Rahmen (gleich in allen WP3-Startprompts)

**Du bist eine ISAA-Instanz und arbeitest diesen Auftrag selbstständig ab.** Verbindlich und vor dem ersten Edit vollständig zu lesen (liegen auf `master`):

1. `docs/devdocs/agent_arbeitsprotokoll.md` – Rahmen, die 11 Pflichtschritte je Änderung, isolierte Umgebung, Bericht.
2. `docs/devdocs/testing_guide.md` – Anti-Muster und Regeln. Besonders: Doubles nur an der Prozessgrenze (Regel 2), Eingang wie im Betrieb (Regel 3), **kein Quelltext als Testobjekt** (kein `inspect.getsource`, keine Wortsuche), Gegenprobe (Regel 7), Wheel-Test bei Paket-/Importänderungen (Regel 8).
3. `docs/devdocs/WP3_ordnung_und_betriebstests_2026-11.md` – Entscheidungen (Abschnitt 2), Befunde (Abschnitte 4, 9).

Kurzfassung der harten Regeln (die Dateien gehen vor):

- **Nur deine Dateien** (Abschnitt 1 unten). Alles andere, was dir auffällt: Bericht, „Gefunden, nicht geändert“.
- **Keine Dopplung:** Vor jeder neuen Funktion `git grep -n` nach dem **Zweck**; Fund in den Bericht. Kein Wrapper, kein Fallback-Zweig, keine zweite Prüfung neben einer vorhandenen.
- **Hypothese vor Edit** (Protokoll Schritte 3–4), **isolierter Lauf mit vollen Abhängigkeiten nach jedem Edit** (Schritt 7, Protokoll Abschnitt 3), **Gegenprobe** mit `git stash` (Schritt 8).
- **Kein Grün durch Wegsehen:** kein `skip`/`xfail`/`importorskip`, keine abgeschwächte Assertion, kein Test gelöscht, außer der Auftrag nennt ihn ausdrücklich zum Ersetzen.
- **Owner-Umgebung schützen:** Ports 8467/8468 nicht belegen, wenn die Owner-Instanz läuft (dann fragen, nicht ausweichen). Daten nur unter `$T` (Protokoll 3.1). Keine Geheimnisse in Dateien, Logs, Commits, Bericht.
- **Stoppen und melden** bei: Hypothese widerlegt und keine neue belegbar; Datei außerhalb der Liste nötig; Code sieht anders aus als hier beschrieben; dritter Reparaturversuch am selben Fehler.

### 0.1 Arbeitskopie (einmal)

```powershell
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2
git fetch origin
git worktree add -B isaa/10 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa10 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa10
uv venv -p 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install pytest pytest-asyncio pytest-timeout build
uv pip install -e toolboxv2/mods/isaa
uv pip install -e ".[web]"
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
python -c "import toolboxv2; print(toolboxv2.__file__)"   # muss in ...Toolboxv2-isaa10 liegen
```

Nichts von Hand in die venv installieren, was nicht in `pyproject.toml` steht. Sonst misst die Gegenprobe deine Umgebung statt des Codes (Testleitfaden, Fall 12).

### 0.2 Gate

Skript `C:\Users\Markin\Workspace\ISAABOX\tb_gate.py` (liegt aus WP2 bereits dort; Inhalt sonst: `git show origin/claude/keen-goldberg-o59mt9:docs/devdocs/ci_analysis/prompts/ISAA_06_session_per_user.md`, Abschnitt 0.4). Gate-Bereich: siehe Abschnitt 1 unten.

```powershell
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py baseline C:\Users\Markin\Workspace\ISAABOX\isaa10_baseline.json @GATE   # vor dem ersten Edit
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py check    C:\Users\Markin\Workspace\ISAABOX\isaa10_baseline.json @GATE   # nach jedem Edit, muss "NEU rot: 0"
```

### 0.3 Commit und Push (Owner-Entscheidung: direkt auf `master`)

Erst wenn alle Pflichtfelder des Protokolls belegt sind und der Bericht geschrieben ist. Ein Commit je Teilaufgabe; Dateien einzeln hinzufügen (`git add <pfad>`), nie `git add -A`/`.`/`-a`. Nachricht: erste Zeile `WP3 A10.<n>: <kurz>`, darunter Hypothese (ein Satz) und die Schlusszeile des Abnahmelaufs.

```powershell
git fetch origin
git rebase origin/master
# nach dem Rebase: Gate-Check erneut
git push origin HEAD:master
```

Abgelehnt (`non-fast-forward`): höchstens dreimal wiederholen, dann melden. Kein `--force`, kein Merge-Commit.

### 0.4 Bericht

`C:\Users\Markin\Workspace\ISAABOX\wp3_A10_bericht.md` (außerhalb des Repos), Format nach Protokoll Abschnitt 4: je Teilaufgabe die Tabelle mit den 11 Pflichtfeldern, Zustand, „Gefunden, nicht geändert“, Patch-Inventar der Tests. Denselben Text als Antwort ausgeben.


## 1. Deine Dateien

- **neu, freigegeben:** `toolboxv2/mods/CloudM/LiveSync/blob_bridge.py` (eine Datei)
- `toolboxv2/mods/CloudM/LiveSync/data_api_sync.py` (nur `sync_all`: einen Schritt „Blobs“ einfügen)
- **neu:** `toolboxv2/tests/test_core/test_wp3_blob_sync.py`

Gate-Bereich:
```powershell
$GATE = @('toolboxv2/tests/test_core/test_a4_betriebsbefunde.py','toolboxv2/tests/test_core/test_data_api_sync.py','toolboxv2/tests/test_core/test_account_bridge.py','toolboxv2/tests/test_core/test_wp3_pairing.py','toolboxv2/tests/test_mini/test_extras/test_blobs.py','toolboxv2/tests/test_mini/test_blob_db.py')
```

**Nicht anfassen:** `utils/extras/blobs.py` (nur benutzen), `UserDataAPI.py`, `pairing.py`, `sync_folder`, `pyproject.toml`, `docs/`. Kein MinIO vom Gerät: `BlobStorage.sync()`, `_cloud_minio`, `get_secure_storage_credentials` werden nicht aufgerufen (WP1-Entscheidung 3).

## 2. Verfahren (festgelegt): Blobs über den vorhandenen Spiegel-Mechanismus

Es gibt **eine** Sync-Maschine: `sync_folder` mit `DataApiTransport`, `LocalIndex`, Verschlüsselung (WP3-08), Konflikten und Löschungen (WP3-09). Der Konto-Sync benutzt sie über einen Spiegelordner (`account_bridge.export_account` → `sync_folder` → `import_account`). Blobs gehen **genauso**:

- `blob_bridge.export_blobs(storage, mirror) -> int` und `blob_bridge.import_blobs(storage, mirror) -> int`, nach dem Vorbild von `account_bridge.py` (gleiche Form, gleiche Behandlung von Konfliktkopien wie dort nach WP3-09).
- Spiegel: `app.data_dir/account_sync/blobs_mirror/<blob_id>/<datei>`; Präfix in der Data-API `livesync/blobs/` (Konstante neben `PREFIX_GLOBAL`).
- `sync_all`: nach dem Konto-Spiegel `export_blobs` → `sync_folder(..., PREFIX_BLOBS, eigener Index)` → `import_blobs`.
- Löschungen: Blob lokal gelöscht → Spiegeldatei weg → `sync_folder` überträgt die Löschung (WP3-09). Remote gelöscht → Spiegeldatei weg → `import_blobs` löscht den lokalen Blob. Der Massenlösch-Schutz aus WP3-09 gilt.
- Größe: `MAX_BYTES` aus `data_api_sync` gilt; größere Blobs stehen in `skipped`.

## 3. Teilaufgaben

### 3.1 Lesen (Schritt 1–4 mit Beleg)

1. Welche `BlobStorage`-Instanz hält auf dem Gerät die Nutzerblobs? Kandidat `app.root_blob_storage` (benutzt in `DB/ui.py:47`, `PasswordManager.py:155`). Belegen: Wo wird sie erzeugt, mit welchem `mode`, welchem Verzeichnis, und versucht sie dabei MinIO zu erreichen (Befund: Die App greift beim Start auf `localhost:9000`/`dl.min.io` zu, auch offline)? Gibt es **mehrere** Blob-Speicher mit Nutzerdaten (`FileWidget.py:214`, `ScopedBlobStorage`)? Welche gehören zu Ziel 7? Mit Datei:Zeile in den Bericht.
2. Blob-API: `create_blob`/`read_blob`/`update_blob`/`list_blobs`/`delete_blob` (`utils/extras/blobs.py`) – Signaturen, ob Inhalte lokal verschlüsselt liegen (`encrypt`-Parameter), Form der `list_blobs`-Einträge.
3. **Stopp-Punkt:** Ist die Antwort auf 1 nicht eindeutig (mehr als ein Speicher mit Nutzerdaten, oder der Speicher lässt sich ohne MinIO-Zugriff nicht öffnen): Bericht mit Befund schreiben und stoppen. Der Owner entscheidet, welche Speicher synchronisiert werden.

### 3.2 Tests zuerst (rot)

`test_wp3_blob_sync.py`, zwei Geräte (zwei Datenordner, zwei echte `BlobStorage` im Offline-Modus in Temp-Ordnern, gemeinsamer Data-API-Fake aus WP3-09/08):
1. Blob (Binärdaten mit allen 256 Bytewerten) auf A → nach `sync_all` A, B inhaltsgleich auf B.
2. Zweiter Lauf ohne Änderung → 0 Übertragungen.
3. Von 3 Blobs wird nur der geänderte übertragen.
4. Blob auf A gelöscht → nach `sync_all` A, B weg auf B.
5. Ohne Gerätekonto → `skipped`.
6. Kein Aufruf ins Netz außer über den Fake: Prüfung, dass `Minio`-Clients nicht erzeugt werden (Grenze: der `minio`-Import bzw. `Minio.__init__` – das ist eine Prozessgrenze, Patch erlaubt, mit Begründung im Test).
7. Server-Fake sieht nur Chiffrat (kein Blob-Inhalt im Klartext in den gespeicherten Werten).

### 3.3 Umsetzung, Abnahme

Kleinster Eingriff nach Abschnitt 2; nach jedem Edit isolierter Lauf (Protokoll 3.2) mit echter App und echtem `root_blob_storage`. Abnahme im echten Ablauf wie WP3-08, Abschnitt 2.5, mit einem Blob statt einer Datei.

## 4. Fertig, wenn

`test_wp3_blob_sync.py` grün, Gate „NEU rot: 0“, Gegenprobe (Schritt `sync_all`-Blobs mit `git stash` entfernt → Tests 1–4 rot), Bericht mit Abschnitt 3.1.
