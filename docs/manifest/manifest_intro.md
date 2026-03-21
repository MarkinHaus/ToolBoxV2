# tb-manifest.yaml — Einführung

**Single Source of Truth** für deine ToolBoxV2 Installation.

<!-- verified: example-manifest.yaml::header -->

## Was ist das Manifest?

Die `tb-manifest.yaml` ist die zentrale Konfigurationsdatei, die alle Einstellungen an einem Ort bündelt. Änderungen werden über `tb manifest apply` auf alle Unterkonfigurationen verteilt.

<!-- verified: schema.py::TBManifest -->

## Kernbereiche

| Bereich | Beschreibung |
|---------|-------------|
| `app` | Anwendungsidentität (Name, Version, Environment, Debug-Modus) |
| `database` | Datenspeicher-Modus: LC (lokal), LR/RR (Redis), CB (MinIO) |
| `workers` | HTTP- und WebSocket-Worker-Instanzen |
| `nginx` | Reverse Proxy mit SSL und Rate-Limiting |
| `auth` | Session- und Authentifizierungs-Einstellungen |
| `paths` | Verzeichnisstruktur (data, logs, dist, mods) |
| `observability` | Log-Sync, Dashboard, automatische Bereinigung |

<!-- verified: schema.py::DatabaseMode -->

## Datenbank-Modi

```yaml
database:
  mode: LC  # LOCAL_DICT — JSON-Datei, keine额外 Konfiguration
  # mode: LR  # LOCAL_REDIS — Lokaler Redis-Server
  # mode: RR  # REMOTE_REDIS — Remote Redis
  # mode: CB  # CLUSTER_BLOB — MinIO Blob-Speicher
```

## Environment-Overwrites

Das Manifest unterstützt umgebungsspezifische Überschreibungen:

```yaml
environments:
  development:
    app.debug: true
    database.mode: LC
  production:
    app.debug: false
    database.mode: CB
    nginx.ssl_enabled: true
```

<!-- verified: schema.py::TBManifest.environments -->

## Generierte Dateien

`tb manifest apply` erzeugt:

1. **`.config.yaml`** — Python-Worker Konfiguration
2. **`bin/config.toml`** — Rust-Server Konfiguration
3. **`services.json`** — Auto-Start Services

<!-- verified: converter.py::ConfigConverter.apply_all -->

## CLI-Befehle

```bash
tb manifest show      # Manifest anzeigen
tb manifest validate  # Schema prüfen
tb manifest apply     # Unterkonfigs generieren
tb manifest init      # Standard-Manifest erstellen
```

<!-- verified: example-manifest.yaml::CLI-Commands-Sektion -->