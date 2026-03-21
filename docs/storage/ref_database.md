# ToolBoxV2 Database Modes Reference

Dokumentation der unterstützten Datenbank-Modi für ToolBoxV2.

<!-- verified: types.py::DatabaseModes -->

---

## DatabaseModes Enum

Definiert die verfügbaren Datenbank-Modi.

<!-- verified: types.py::DatabaseModes -->

| Modus | Wert | Beschreibung |
|-------|------|-------------|
| **LC** | LOCAL_DICT | Lokale JSON-Datei (Standard) |
| **LR** | LOCAL_REDIS | Lokaler Redis-Server |
| **RR** | REMOTE_REDIS | Remote Redis-Server |
| **CB** | CLUSTER_BLOB | MinIO/S3 Blob Storage |

### Methoden

#### `DatabaseModes.crate(mode: str) -> DatabaseModes`
Erstellt eine DatabaseModes-Instanz aus einem String.

```python
from toolboxv2.mods.DB.types import DatabaseModes

mode = DatabaseModes.crate("LC")  # -> DatabaseModes.LC
```

<!-- verified: types.py::DatabaseModes.crate -->

---

## AuthenticationTypes Enum

Definiert die Authentifizierungstypen für Datenbankverbindungen.

<!-- verified: types.py::AuthenticationTypes -->

| Typ | Wert | Verwendung |
|-----|------|------------|
| UserNamePassword | password | Klassische Benutzer/Passwort-Auth |
| Uri | url | Connection-String in URI-Form |
| PassKey | passkey | Auth via Passkey |
| location | location | Dateibasierte Auth (Standard für LC) |
| none | none | Keine Auth erforderlich |

<!-- verified: types.py::AuthenticationTypes -->

---

## Konfiguration

### tb-manifest.yaml Beispiel

```yaml
database:
  mode: LC  # oder LR, RR, CB

  local:
    path: ".data/MiniDictDB.json"

  redis:
    url: "${DB_CONNECTION_URI:redis://localhost:6379}"
    username: "${DB_USERNAME:}"
    password: "${DB_PASSWORD:}"

  minio:
    endpoint: "${MINIO_ENDPOINT:localhost:9000}"
    access_key: "${MINIO_ACCESS_KEY:minioadmin}"
    secret_key: "${MINIO_SECRET_KEY:minioadmin}"
    bucket: user-data-enc
```

<!-- verified: schema.py::DatabaseMode -->
