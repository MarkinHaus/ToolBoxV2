# ToolBoxV2 Logging System

> `toolboxv2/utils/system/tb_logger.py`

Strukturiertes JSON-Logging mit lokaler Verschlüsselung, Offline-First Persistenz und explizitem Cloud-Sync.

---

## Architektur

```
App Code                    RAM Queue              MobileDB (SQLite)           MinIO (Cloud)
  │                            │                        │                          │
  │  logger.info("…")         │                        │                          │
  ├──────────────────────►  Queue  ──── 5s/100 ────►  logs/node/date/            │
  │                        (non-blocking)             system_*.jsonl              │
  │                                                   audit_*.jsonl              │
  │  audit.log_action(…)                              (encrypted=True)           │
  │                                                        │                     │
  │                                                        │  sync_all()         │
  │                                                        ├────────────────►  bucket/
  │                                                        │  sync_time_range()  app_id/logs/…
  │                                                        │  start_auto_sync()  │
```

**Formate:** Alle Log-Einträge sind JSONL (eine JSON-Zeile pro Event). Kompatibel mit Grafana Loki, OpenSearch, ELK, Fluentd.

**Keine externen Dependencies:** Der `JsonLogFormatter` nutzt nur `json` + `datetime` aus der stdlib.

---

## Quick Start

### 1. MobileDB registrieren (einmalig, vor `set_logger`)

```python
from toolboxv2.utils.system.tb_logger import set_log_db
from toolboxv2.utils.extras.db.mobile_db import create_mobile_db

log_db = create_mobile_db(path="data/system_logs.db", max_size_mb=200)
set_log_db(log_db, node_id="desktop-01")
```

### 2. Logger initialisieren (unveränderte API)

```python
# Bestehender Code funktioniert weiterhin:
logger, filename = setup_logging(
    logging.DEBUG,
    name="toolbox-live",
    interminal=True,
    file_level=logging.WARNING,
    app_name="myapp",
)

# Oder über set_logger in toolbox.py (keine Änderung nötig):
logger_info_str, self.logger, self.logging_filename = self.set_logger(args.debug, self.logger_prefix)
```

### 3. Loggen (identische API)

```python
from toolboxv2.utils.system.tb_logger import get_logger

logger = get_logger()

# Standard-Logging
logger.info("Server started", extra={"port": 8080})
logger.warning("Rate limit reached", extra={"user": "u_42", "endpoint": "/api"})
logger.error("Connection failed", exc_info=True)
```

**Output (JSONL):**
```json
{"timestamp":"2026-02-23T14:30:00+00:00","level":"INFO","name":"toolbox-live","filename":"server.py","lineno":42,"funcName":"start","message":"Server started","app_id":"myapp","node_id":"desktop-01","port":8080}
```

---

## Audit Logger

Erzwingt ein striktes Schema für Compliance- und Business-Events.

### Events loggen

```python
from toolboxv2.utils.system.tb_logger import AuditLogger

audit = AuditLogger(logger, db=log_db, node_id="desktop-01")

audit.log_action(
    user_id="user_42",
    action="LOGIN",
    resource="/auth/session",
    status="SUCCESS",
)

audit.log_action(
    user_id="user_42",
    action="DELETE",
    resource="/api/project/123",
    status="FAILURE",
    details={"reason": "permission_denied"},
)
```

### Audit-Events abfragen

```python
# Letzte 20 Aktionen eines Users
entries = audit.get_user_actions("user_42", last_n=20)

# Alle LOGIN-Events dieser Woche
entries = audit.query(
    action="LOGIN",
    date_from="2026-02-17",
    date_to="2026-02-23",
)

# Alle fehlgeschlagenen Events
entries = audit.query(status="FAILURE", limit=100)

# Zusammenfassung: {user_id: {action: count}}
summary = audit.get_actions_summary(
    date_from="2026-02-01",
    date_to="2026-02-23",
)
# → {"user_42": {"LOGIN": 15, "DELETE": 3}, "user_99": {"LOGIN": 8}}
```

---

## Log Sync (Lokal → Remote MinIO)

### Setup

```python
from minio import Minio
from toolboxv2.utils.system.tb_logger import LogSyncManager

minio_client = Minio("ryzen.local:9000", access_key="…", secret_key="…", secure=False)

sync = LogSyncManager(
    db=log_db,
    minio_client=minio_client,
    bucket="system-audit-logs",
    app_id="toolbox-instance-01",
    node_id="desktop-01",
)
sync.ensure_bucket()
```

### Manueller Sync

```python
# Alles was DIRTY ist
stats = sync.sync_all()

# Nur bestimmte Zeitspanne
stats = sync.sync_time_range(date_from="2026-02-20", date_to="2026-02-23")

# Nur Audit-Logs
stats = sync.sync_audit_only(date_from="2026-02-23")

# Nur System-Logs der letzten Woche
stats = sync.sync_system_only(date_from="2026-02-17")

print(stats)
# → {"uploaded": 12, "skipped": 0, "errors": [], "bytes_transferred": 48320, "status": "complete"}
```

### Automatischer Sync

```python
# Alle 5 Minuten im Hintergrund
sync.start_auto_sync(interval_seconds=300)

# Status prüfen
print(sync.is_auto_syncing)  # True

# Stoppen
sync.stop_auto_sync()
```

### Pending Stats

```python
stats = sync.get_pending_stats()
# → {"total_dirty": 5, "by_date": {"2026-02-22": 2, "2026-02-23": 3}, "audit": 1, "system": 4}
```

---

## Integration in `toolbox.py`

```python
# In App.__init__:

from .system.tb_logger import set_log_db, AuditLogger, LogSyncManager
from ..extras.db.mobile_db import create_mobile_db

# 1. Log-DB erstellen + registrieren
self.log_db = create_mobile_db(path=f"{self.data_dir}/system_logs.db", max_size_mb=200)
set_log_db(self.log_db, node_id=node())

# 2. Logger (unverändert)
logger_info_str, self.logger, self.logging_filename = self.set_logger(args.debug, self.logger_prefix)

# 3. Audit Logger
self.audit_logger = AuditLogger(self.logger, db=self.log_db, node_id=node())

# 4. Optional: Sync zum RYZEN Server
def setup_log_sync(self, minio_client, auto_interval=None):
    self.log_sync = LogSyncManager(
        db=self.log_db, minio_client=minio_client,
        bucket="system-audit-logs", app_id=self.id, node_id=node(),
    )
    self.log_sync.ensure_bucket()
    if auto_interval:
        self.log_sync.start_auto_sync(interval_seconds=auto_interval)
```

---

## Pfad-Schema

Logs werden in der MobileDB unter folgendem Pfad gespeichert:

```
logs/{node_id}/{YYYY-MM-DD}/{type}_{timestamp_ms}.jsonl
```

Beispiele:
```
logs/desktop-01/2026-02-23/system_1740300000000.jsonl
logs/desktop-01/2026-02-23/audit_1740300005000.jsonl
```

Im MinIO-Bucket:
```
{app_id}/logs/{node_id}/{YYYY-MM-DD}/{type}_{timestamp_ms}.jsonl
```

---

## API-Referenz

| Funktion / Klasse | Beschreibung |
|---|---|
| `setup_logging(level, name, …)` | Logger-Setup (alte Signatur, intern jetzt JSONL + MobileDB) |
| `setup_production_logging(level, app_id, node_id, local_db)` | Neue API ohne File-Rotation |
| `get_logger()` | Aktiven Logger holen |
| `set_log_db(db, node_id)` | MobileDB global registrieren |
| `get_log_db()` | Registrierte MobileDB holen |
| `AuditLogger(logger, db, node_id)` | Audit-Logger mit Query-API |
| `AuditLogger.log_action(user_id, action, resource, status, details)` | Event loggen |
| `AuditLogger.query(user_id, action, date_from, date_to, status, limit)` | Events filtern |
| `AuditLogger.get_user_actions(user_id, last_n)` | Letzte N Events eines Users |
| `AuditLogger.get_actions_summary(date_from, date_to)` | Aggregation pro User |
| `LogSyncManager(db, minio_client, bucket, app_id, node_id)` | Sync-Manager |
| `LogSyncManager.sync_all()` | Alle dirty Chunks pushen |
| `LogSyncManager.sync_time_range(date_from, date_to)` | Zeitspanne pushen |
| `LogSyncManager.sync_audit_only(date_from, date_to)` | Nur Audit-Chunks |
| `LogSyncManager.sync_system_only(date_from, date_to)` | Nur System-Chunks |
| `LogSyncManager.start_auto_sync(interval_seconds)` | Auto-Sync starten |
| `LogSyncManager.stop_auto_sync()` | Auto-Sync stoppen |
| `LogSyncManager.get_pending_stats()` | Offene Chunks anzeigen |

---

## Design-Entscheidungen

- **Kein `python-json-logger`** – stdlib-only Formatter, null Dependencies
- **Queue + Batch-Worker** – Main-Thread wird beim Loggen nie blockiert
- **5s / 100 Einträge** – Flush-Schwelle für Balance zwischen Latenz und I/O
- **`encrypted=True`** – Logs liegen lokal und in MinIO verschlüsselt (via MobileDB)
- **Expliziter Sync** – Kein automatischer Upload ohne bewusste Entscheidung
- **DIRTY-Tracking** – MobileDB markiert neue Chunks automatisch als sync-pflichtig
- **Pfad-basiertes Date-Filtering** – String-Vergleich auf `YYYY-MM-DD` Ordnernamen, kein SQL
**Usage in `toolbox.py`:**

```python
from .system.minio_helper import create_minio_client

# Lokal (defaults)
client = create_minio_client()

# RYZEN Server (explizit)
client = create_minio_client(endpoint="ryzen.local:9000", access_key="…", secret_key="…")

# Über Env-Vars (z.B. in Docker / Produktion)
# MINIO_ENDPOINT=ryzen.local:9000 MINIO_ACCESS_KEY=… MINIO_SECRET_KEY=…
client = create_minio_client()

# Direkt mit LogSyncManager
self.setup_log_sync(client, auto_interval=300)
```
// ─────────────────────────────────────────────────────────────────────
// FRONTEND INTEGRATION — in your app init (e.g. tbjs/index.js or main.js)
// ─────────────────────────────────────────────────────────────────────

import Logger, { setLogConsent, AuditLogger } from './core/logger.js';

// 1. Init Logger (optional: set remote endpoint for Tauri/cross-origin)
Logger.init({
    logLevel: 'debug',
    // remoteEndpoint: 'http://localhost:5000/api/client-logs',  // nur nötig wenn cross-origin
});

// 2. Consent UI — binde an deinen Settings-Dialog / Cookie-Banner
//    setLogConsent('all');       // User sagt ja zu allem
//    setLogConsent('essential'); // Nur WARN + ERROR + AUDIT
//    setLogConsent('errors');    // Nur ERROR + AUDIT
//    setLogConsent('none');      // Nix (default)

// 3. Audit Events im Frontend
AuditLogger.log('PAGE_VIEW', '/dashboard');
AuditLogger.log('FORM_SUBMIT', 'settings-form', 'SUCCESS', { changed: ['email'] });
AuditLogger.log('LOGIN', '/auth', 'SUCCESS', { provider: 'google' });
AuditLogger.log('API_CALL', '/api/projects', 'FAILURE', { status: 403 });

// 4. Cleanup bei SPA unmount
// Logger.destroy();


# ToolBoxV2 Observability — Setup Guide

> `toolboxv2/utils/system/` — `observability_adapter.py` · `openobserve_setup.py` · `tb_logger.py`

End-to-End Anleitung für Datenanalyse und Visualisierung mit OpenObserve.
Drei Szenarien, ein Adapter — jederzeit austauschbar.

---

## Architektur-Überblick

```
┌─────────────────────────────────────────────────────────────┐
│  Dein Code                                                   │
│  logger.info("…") / audit.log_action(…)                     │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├──────────────────────────────────────────────────┐
           │                                                   │
           ▼                                                   ▼
┌──────────────────────┐                    ┌──────────────────────────────┐
│  MobileDBLogHandler  │                    │  ObservabilityLogHandler     │
│  5s batch → SQLite   │                    │  5s batch → HTTP POST        │
│  (Offline-First)     │                    │  (Live Dashboard)            │
└──────────┬───────────┘                    └──────────┬───────────────────┘
           │                                           │
           ▼                                           │
┌──────────────────────┐                               │
│  LogSyncManager      │──► ObservabilityAdapter ──┐   │
│  (periodisch)        │   (nach MinIO-Upload)     │   │
└──────────┬───────────┘                           │   │
           │                                       │   │
           ▼                                       ▼   ▼
     ┌─────────┐                           ┌───────────────┐
     │  MinIO  │                           │  OpenObserve  │
     │  (S3)   │                           │  Dashboard    │
     │ Backup  │                           │  Alerts, SQL  │
     └─────────┘                           └───────────────┘

  Zwei Wege zum Dashboard:
    LIVE:   ObservabilityLogHandler → direkt (< 5s Latenz)
    SYNC:   LogSyncManager → Adapter (nach MinIO-Upload, periodisch)
```

**Datei-Übersicht:**

| Datei | Aufgabe |
|---|---|
| `observability_adapter.py` | Abstraktes Interface + OpenObserve-Implementierung |
| `openobserve_setup.py` | Docker-Lifecycle + historischer MinIO-Import |
| `observability_helper.py` | End-to-End Helper für alle 3 Szenarien |
| `tb_logger.py` | Logger mit Adapter-Hook (3 Zeilen geändert) |

---

## Szenario 1 — Lokales Debugging

**Ziel:** Logs in einem Dashboard durchsuchen, filtern und visualisieren — direkt auf deinem Entwicklungsrechner.

### Variante A: Mit Docker (empfohlen)

```bash
# Einzeiler — startet OpenObserve + verbindet Logger
python -m toolboxv2.utils.system.observability_helper local-debug

# Oder mit eigenen Credentials
OPENOBSERVE_PASSWORD=mein-passwort \
python -m toolboxv2.utils.system.observability_helper local-debug
```

**Was passiert:**
1. OpenObserve Docker-Container startet auf `http://localhost:5080`
2. Logger-Adapter wird konfiguriert
3. Test-Logs (System + Audit) werden gesendet
4. Browser öffnet das Dashboard

**Manuell Schritt für Schritt:**

```python
from toolboxv2.utils.system.openobserve_setup import OpenObserveManager
from toolboxv2.utils.system.observability_adapter import OpenObserveAdapter

# 1. OpenObserve starten
import os
os.environ["OPENOBSERVE_PASSWORD"] = "dev-password-123"
mgr = OpenObserveManager()
mgr.deploy()

# 2. Adapter an bestehenden LogSyncManager hängen
adapter = OpenObserveAdapter(
    endpoint="http://localhost:5080",
    credentials=("root@example.com", "dev-password-123"),
)
sync_manager.set_observability_adapter(adapter)

# 3. Logs landen jetzt automatisch in OpenObserve nach jedem sync
sync_manager.sync_all()

# Dashboard: http://localhost:5080
# Stream "system_logs" → System-Logs
# Stream "audit_logs"  → Audit-Events
```

### Variante B: Ohne Docker (Standalone Binary)

Für Rechner ohne Docker — OpenObserve läuft als einzelne Binary.

```bash
# 1. Binary herunterladen (einmalig)
# Linux:
curl -L https://github.com/openobserve/openobserve/releases/latest/download/openobserve-linux-amd64 -o openobserve
chmod +x openobserve

# macOS:
curl -L https://github.com/openobserve/openobserve/releases/latest/download/openobserve-darwin-arm64 -o openobserve
chmod +x openobserve

# Windows (PowerShell):
Invoke-WebRequest -Uri "https://github.com/openobserve/openobserve/releases/latest/download/openobserve-windows-amd64.exe" -OutFile openobserve.exe

# 2. Starten
ZO_ROOT_USER_EMAIL="root@example.com" \
ZO_ROOT_USER_PASSWORD="dev-password-123" \
./openobserve

# 3. Helper verbindet sich
python -m toolboxv2.utils.system.observability_helper local-debug --no-docker
```

**Programmatisch ohne Docker:**

```python
from toolboxv2.utils.system.observability_adapter import OpenObserveAdapter

# OpenObserve läuft schon (manuell gestartet)
adapter = OpenObserveAdapter(
    endpoint="http://localhost:5080",
    credentials=("root@example.com", "dev-password-123"),
)

# Health-Check
assert adapter.health_check(), "OpenObserve nicht erreichbar!"

# An Sync-Manager hängen
sync_manager.set_observability_adapter(adapter)
```

### Dashboard nutzen

Nach dem Start: **http://localhost:5080**

### Live-Logs von jeder laufenden Instanz

Damit Logs innerhalb von ~5 Sekunden im Dashboard erscheinen, muss jede ToolBox-Instanz den Live-Handler aktivieren. **Einmalig** in `toolbox.py`:

```python
# In App.__init__ oder App.run_app — NACH set_logger():

from .system.observability_adapter import OpenObserveAdapter
from .system.tb_logger import enable_live_observability

def enable_observability(self):
    """Aktiviert Live-Log-Streaming zu OpenObserve (wenn erreichbar)."""
    adapter = OpenObserveAdapter(
        # ENV-Vars: OPENOBSERVE_ENDPOINT, OPENOBSERVE_USER, OPENOBSERVE_PASSWORD
        # oder explizit:
        # endpoint="http://localhost:5080",
        # credentials=("root@example.com", "dev-password-123"),
    )

    if adapter.health_check():
        enable_live_observability(adapter)
        self.logger.info("Live observability enabled")
    else:
        self.logger.debug("OpenObserve not reachable, live observability skipped")
```

**Das ist alles.** Ab diesem Aufruf landet jeder `logger.info()`, `logger.error()`, `audit.log_action()` — aus jedem Modul das `get_logger()` nutzt — innerhalb von 5 Sekunden im Dashboard.

**Wie es funktioniert:**

```
logger.info("Server started")
    │
    ├── Console (sofort, wie bisher)
    ├── File Handler (sofort, wie bisher)
    ├── MobileDBLogHandler (5s batch → SQLite, wie bisher)
    └── ObservabilityLogHandler (5s batch → HTTP POST → OpenObserve) ← NEU
```

Der `ObservabilityLogHandler` ist ein normaler Python `logging.Handler` der parallel zu allen bestehenden Handlern läuft. Er blockiert nicht, nutzt die gleiche Queue+Worker Architektur wie der MobileDB-Handler.

**Mehrere Instanzen:**

Jede Instanz die `enable_live_observability()` aufruft, erscheint automatisch im Dashboard — unterscheidbar über `node_id` und `app_id` im JSON:

```sql
-- Logs von allen Instanzen
SELECT * FROM system_logs ORDER BY _timestamp DESC

-- Nur von einer bestimmten Instanz
SELECT * FROM system_logs WHERE node_id = 'desktop-01' ORDER BY _timestamp DESC

-- Fehler über alle Instanzen
SELECT node_id, COUNT(*) as errors
FROM system_logs
WHERE level = 'ERROR'
GROUP BY node_id
```

**Deaktivieren:**

```python
from toolboxv2.utils.system.tb_logger import disable_live_observability
disable_live_observability()
```

**Prüfen:**

```python
from toolboxv2.utils.system.tb_logger import is_live_observability_enabled
print(is_live_observability_enabled())  # True/False
```

**Erste Schritte im UI:**
1. Login mit den konfigurierten Credentials
2. Links → **Logs** → Stream `system_logs` auswählen
3. Zeitfilter auf "Last 1 Hour" setzen
4. SQL-Query für Fehler: `SELECT * FROM system_logs WHERE level = 'ERROR' ORDER BY _timestamp DESC`
5. Audit-Events: Stream `audit_logs` → Filter auf `audit_action = 'LOGIN'`

**Nützliche Queries:**

```sql
-- Fehler der letzten Stunde
SELECT * FROM system_logs
WHERE level = 'ERROR'
ORDER BY _timestamp DESC
LIMIT 100

-- Audit: Alle fehlgeschlagenen Aktionen
SELECT * FROM audit_logs
WHERE status = 'FAILURE'
ORDER BY _timestamp DESC

-- Audit: Login-Übersicht pro User
SELECT user_id, COUNT(*) as login_count
FROM audit_logs
WHERE audit_action = 'LOGIN'
GROUP BY user_id
ORDER BY login_count DESC

-- Log-Volumen pro Level (letzte 24h)
SELECT level, COUNT(*) as count
FROM system_logs
GROUP BY level
```

---

## Szenario 2 — Server Setup (Produktion)

**Ziel:** OpenObserve auf dem RYZEN-Server deployen mit MinIO als Storage-Backend, historische Logs importieren, Auto-Sync aktivieren.

### Quick Start

```bash
# Auf dem RYZEN Server:
python -m toolboxv2.utils.system.observability_helper server-setup \
    --minio-endpoint http://localhost:9000 \
    --minio-access-key DEIN_KEY \
    --minio-secret-key DEIN_SECRET \
    --import-bucket system-audit-logs \
    --import-from 2026-01-01
```

### Schritt für Schritt

#### 1. ENV-Datei vorbereiten

```bash
# /srv/toolbox/openobserve.env
OPENOBSERVE_PASSWORD=sicheres-produktions-passwort
OPENOBSERVE_USER=admin@toolbox.local
OPENOBSERVE_PORT=5080
OPENOBSERVE_DATA_DIR=/srv/openobserve/data

# MinIO als Storage-Backend (Parquet-Daten in MinIO statt lokal)
OPENOBSERVE_STORAGE=s3
OPENOBSERVE_S3_ENDPOINT=http://localhost:9000
OPENOBSERVE_S3_ACCESS_KEY=<minio-key>
OPENOBSERVE_S3_SECRET_KEY=<minio-secret>
OPENOBSERVE_S3_BUCKET=openobserve-data

# Historischer Import
OPENOBSERVE_IMPORT_ENABLED=true
OPENOBSERVE_IMPORT_BUCKET=system-audit-logs
OPENOBSERVE_IMPORT_PREFIX=toolbox-instance-01/
OPENOBSERVE_IMPORT_FROM=2026-01-01

# MinIO-Verbindung für Import
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=<minio-key>
MINIO_SECRET_KEY=<minio-secret>
```

#### 2. Deploy + Import

```python
from toolboxv2.utils.system.openobserve_setup import OpenObserveManager

# Lädt Config aus ENV
mgr = OpenObserveManager()

# Docker starten (pulled Image, wartet auf healthz)
mgr.deploy()

# Historische Logs aus MinIO importieren
stats = mgr.import_historical_logs(
    bucket="system-audit-logs",
    prefix="toolbox-instance-01/",
    date_from="2026-01-01",
)
print(stats)
# → {"system_sent": 12450, "audit_sent": 890, "files_processed": 245, ...}
```

#### 3. Auto-Sync mit Adapter aktivieren

```python
# In App.__init__ (toolbox.py) — nach bestehendem Sync-Setup:
from .system.observability_adapter import OpenObserveAdapter

def setup_observability(self):
    """Verbindet LogSyncManager mit OpenObserve (additiv zu MinIO)."""
    adapter = OpenObserveAdapter(
        endpoint="http://localhost:5080",
        # Credentials aus ENV oder explizit
    )

    if adapter.health_check():
        self.log_sync.set_observability_adapter(adapter)
        self.logger.info("Observability adapter connected to OpenObserve")
    else:
        self.logger.warning("OpenObserve not reachable, adapter not connected")
```

#### 4. Nginx Reverse Proxy (optional)

```nginx
# /etc/nginx/sites-available/openobserve
server {
    listen 443 ssl;
    server_name obs.toolbox.local;

    ssl_certificate     /etc/ssl/certs/toolbox.pem;
    ssl_certificate_key /etc/ssl/private/toolbox.key;

    location / {
        proxy_pass http://127.0.0.1:5080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket für Live-Tail
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### 5. MinIO Bucket vorbereiten

```bash
# OpenObserve-Data Bucket (für OZ internes Storage)
mc mb myminio/openobserve-data

# Prüfen ob Import-Bucket existiert
mc ls myminio/system-audit-logs/
```

---

## Szenario 3 — Remote Live-Logs senden

**Ziel:** Logs von einer lokalen Instanz, einem CI-Runner oder GitHub Action live an den RYZEN-Server senden.

### Von einer lokalen Instanz

```python
from toolboxv2.utils.system.observability_adapter import OpenObserveAdapter

# Adapter zeigt auf den RYZEN-Server
adapter = OpenObserveAdapter(
    endpoint="https://obs.toolbox.local",   # oder http://ryzen.local:5080
    credentials=("admin@toolbox.local", "prod-password"),
)

# An lokalen LogSyncManager hängen
sync_manager.set_observability_adapter(adapter)

# Ab jetzt: jeder sync_all() pusht auch nach OpenObserve
sync_manager.start_auto_sync(interval_seconds=60)
```

### Ohne LogSyncManager (Standalone Push)

```python
from toolboxv2.utils.system.observability_adapter import OpenObserveAdapter

adapter = OpenObserveAdapter(
    endpoint="https://obs.toolbox.local",
    credentials=("ci-user@toolbox.local", "ci-password"),
)

# Direkt Logs senden
adapter.send_batch([
    {"timestamp": "2026-02-23T14:30:00Z", "level": "INFO",
     "message": "Build started", "app_id": "ci", "node_id": "github-runner"},
    {"timestamp": "2026-02-23T14:31:00Z", "level": "ERROR",
     "message": "Test failed", "app_id": "ci", "test": "test_auth"},
], stream="ci_logs")

# Audit-Events
adapter.send_audit_batch([
    {"timestamp": "2026-02-23T14:30:00Z", "audit_action": "DEPLOY",
     "user_id": "github-actions", "resource": "toolbox-v2",
     "status": "SUCCESS", "details": {"commit": "abc1234", "branch": "main"}},
])
```

### GitHub Action

```yaml
# .github/workflows/deploy.yml
name: Deploy with Observability

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e .

      - name: Run tests + send logs
        env:
          OPENOBSERVE_ENDPOINT: ${{ secrets.OPENOBSERVE_ENDPOINT }}
          OPENOBSERVE_USER: ${{ secrets.OPENOBSERVE_USER }}
          OPENOBSERVE_PASSWORD: ${{ secrets.OPENOBSERVE_PASSWORD }}
          OPENOBSERVE_ORG: default
        run: |
          python -m toolboxv2.utils.system.observability_helper ci-push \
            --stream ci_logs \
            --message "Deploy started" \
            --level INFO \
            --meta '{"commit":"${{ github.sha }}","branch":"${{ github.ref_name }}","run":"${{ github.run_id }}"}'

      - name: Run tests
        run: python -m unittest discover -s tests

      - name: Report result
        if: always()
        env:
          OPENOBSERVE_ENDPOINT: ${{ secrets.OPENOBSERVE_ENDPOINT }}
          OPENOBSERVE_USER: ${{ secrets.OPENOBSERVE_USER }}
          OPENOBSERVE_PASSWORD: ${{ secrets.OPENOBSERVE_PASSWORD }}
        run: |
          STATUS="${{ job.status }}"
          python -m toolboxv2.utils.system.observability_helper ci-push \
            --stream ci_logs \
            --message "Deploy $STATUS" \
            --level $([ "$STATUS" = "success" ] && echo "INFO" || echo "ERROR") \
            --audit-action DEPLOY \
            --audit-status $(echo "$STATUS" | tr '[:lower:]' '[:upper:]') \
            --meta '{"commit":"${{ github.sha }}","branch":"${{ github.ref_name }}"}'
```

**GitHub Secrets einrichten:**

| Secret | Wert |
|---|---|
| `OPENOBSERVE_ENDPOINT` | `https://obs.toolbox.local` |
| `OPENOBSERVE_USER` | `ci-user@toolbox.local` |
| `OPENOBSERVE_PASSWORD` | CI-spezifisches Passwort |

### Frontend (logger.js) → Backend → OpenObserve

Der Frontend-Logger (`logger.js`) sendet Batches an `/api/client-logs`. Im Python-Worker diesen Endpoint an OpenObserve weiterleiten:

```python
# In deinem API-Worker (z.B. Flask/FastAPI):
from toolboxv2.utils.system.observability_adapter import OpenObserveAdapter

adapter = OpenObserveAdapter(endpoint="http://localhost:5080", ...)

@app.post("/api/client-logs")
async def receive_client_logs(request):
    body = await request.json()
    logs = body.get("logs", [])

    system = [e for e in logs if e.get("type") != "audit"]
    audit  = [e for e in logs if e.get("type") == "audit"]

    if system:
        adapter.send_batch(system, stream="frontend_logs")
    if audit:
        adapter.send_audit_batch(audit)

    return {"status": "ok"}
```

---

## ENV-Referenz

| Variable | Default | Beschreibung |
|---|---|---|
| `OPENOBSERVE_ENDPOINT` | `http://localhost:5080` | API-Endpoint |
| `OPENOBSERVE_USER` | `root@example.com` | Root-User E-Mail |
| `OPENOBSERVE_PASSWORD` | *(pflicht)* | Root-User Passwort |
| `OPENOBSERVE_ORG` | `default` | Organisation |
| `OPENOBSERVE_PORT` | `5080` | Exposed Port |
| `OPENOBSERVE_DATA_DIR` | `./openobserve-data` | Host Volume |
| `OPENOBSERVE_DOCKER_IMAGE` | `public.ecr.aws/zinclabs/openobserve:latest` | Docker Image |
| `OPENOBSERVE_CONTAINER` | `toolbox-openobserve` | Container-Name |
| `OPENOBSERVE_STORAGE` | `disk` | `disk` oder `s3` |
| `OPENOBSERVE_S3_ENDPOINT` | | MinIO URL für OZ-Storage |
| `OPENOBSERVE_S3_ACCESS_KEY` | | S3 Access Key |
| `OPENOBSERVE_S3_SECRET_KEY` | | S3 Secret Key |
| `OPENOBSERVE_S3_BUCKET` | `openobserve-data` | OZ-Data Bucket |
| `OPENOBSERVE_IMPORT_ENABLED` | `false` | Auto-Import bei Deploy |
| `OPENOBSERVE_IMPORT_BUCKET` | `system-audit-logs` | Quell-Bucket |
| `OPENOBSERVE_IMPORT_PREFIX` | | Objekt-Prefix Filter |
| `OPENOBSERVE_IMPORT_FROM` | | Start-Datum `YYYY-MM-DD` |
| `OPENOBSERVE_IMPORT_TO` | | End-Datum `YYYY-MM-DD` |
| `OPENOBSERVE_STREAM` | `system_logs` | Default System-Stream |
| `OPENOBSERVE_AUDIT_STREAM` | `audit_logs` | Default Audit-Stream |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO für Import |
| `MINIO_ACCESS_KEY` | | MinIO Key |
| `MINIO_SECRET_KEY` | | MinIO Secret |

---

## Streams

| Stream | Inhalt | Quelle |
|---|---|---|
| `system_logs` | Python-Logger (INFO, WARN, ERROR, DEBUG) | `tb_logger.py` |
| `audit_logs` | Audit-Events (LOGIN, DELETE, etc.) | `AuditLogger` |
| `frontend_logs` | Browser-Logs aus `logger.js` | `/api/client-logs` Worker |
| `ci_logs` | CI/CD Pipeline Events | GitHub Actions / Helper |

---

## Eigenen Adapter schreiben

```python
from toolboxv2.utils.system.observability_adapter import ObservabilityAdapter

class MeinBackendAdapter(ObservabilityAdapter):
    def __init__(self, url, token):
        self.url = url
        self.token = token

    def send_batch(self, entries, stream="default"):
        # Deine Logik: HTTP POST, gRPC, Datei, etc.
        ...
        return {"sent": len(entries), "failed": 0, "errors": []}

    def health_check(self):
        # Connectivity prüfen
        return True

# Einhängen — fertig
sync_manager.set_observability_adapter(MeinBackendAdapter(...))
```
