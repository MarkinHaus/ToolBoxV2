# How-To: ToolBoxV2 als Server betreiben

## TL;DR
Server-Profil zeigt ASCII-Statusübersicht. Konfiguration über tb-manifest.yaml.

<!-- verified: __main__.py::_run_server_overview -->
<!-- verified: __main__.py::ProfileType -->

## Profil: Server
Das Server-Profil ist für IT-Infrastruktur und verteilte Systeme.

<!-- verified: utils/clis/first_run.py::PROFILES -->

## Server-Übersicht (ASCII)

Beim Ausführen von `tb` im Server-Profil:

```
════════════════════════════════════════════════════
  ToolBoxV2 Server Overview  •  2024-03-19 12:00:00
════════════════════════════════════════════════════
  Services : 3/5 running
  ✅ http_main        ▶ running   pid=1234
  ✅ ws_main          ▶ running   pid=1235
  ✅ worker_manager   ▶ running   pid=1236
  🔴 minio            ■ stopped
  🔴 registry         ■ stopped
════════════════════════════════════════════════════
```

<!-- verified: __main__.py::_run_server_overview -->

## Konfiguration (tb-manifest.yaml)

### Grundstruktur

<!-- verified: utils/manifest/schema.py::DatabaseMode -->

```yaml
manifest_version: "1.0.0"

app:
  name: ToolBoxV2
  profile: server        # ← Server-Profil setzen
  environment: production
  debug: false
  log_level: INFO

database:
  mode: LC              # LC | LR | RR | CB
```

### Datenbank-Modi

| Modus | Bedeutung | Konfiguration |
|-------|-----------|---------------|
| `LC` | Local Dict | JSON-Datei (lokal) |
| `LR` | Local Redis | Redis auf localhost |
| `RR` | Remote Redis | Externer Redis-Server |
| `CB` | Cluster Blob | MinIO Storage |

<!-- verified: utils/manifest/schema.py::DatabaseMode -->

### Services konfigurieren

```yaml
services:
  enabled:
    - workers
    - db

  zmq:
    pub_endpoint: "tcp://127.0.0.1:5555"
    sub_endpoint: "tcp://127.0.0.1:5556"
    req_endpoint: "tcp://127.0.0.1:5557"

  manager:
    web_ui_enabled: true
    web_ui_port: 9000
```

### Workers konfigurieren

```yaml
workers:
  http:
    - name: http_main
      host: "0.0.0.0"
      port: 8000
      workers: 4
      max_concurrent: 100

  websocket:
    - name: ws_main
      host: "0.0.0.0"
      port: 8100
      max_connections: 10000
```

## Manifest-Commands

```bash
# Manifest anzeigen
tb manifest show

# Validieren
tb manifest validate

# Sub-Configs generieren
tb manifest apply

# Services synchronisieren
tb manifest sync

# Service-Status
tb manifest status
```

<!-- verified: manifest_cli.py::create_parser -->

## Config-Generation

Der `ConfigConverter` generiert automatisch:

<!-- verified: utils/manifest/converter.py::ConfigConverter -->

| Ausgabe | Beschreibung |
|---------|---------------|
| `.config.yaml` | Python-Worker Config |
| `bin/config.toml` | Rust-Server Config |
| `services.json` | Auto-Start Services |

```bash
# Dry-Run (was würde generiert)
tb manifest apply --dry-run

# Mit Force (überschreiben)
tb manifest apply --force
```

## Server starten

```bash
# Server starten
tb

# Oder direkt mit Services
tb workers start
tb registry start

# Status prüfen
tb status
```

## Nginx-Konfiguration

```yaml
nginx:
  enabled: true
  server_name: "example.com"
  listen_port: 80
  listen_ssl_port: 443
  ssl_enabled: false
  static_enabled: true
  rate_limit_enabled: true
```
