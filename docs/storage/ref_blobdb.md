# BlobDB Reference

Server-Blob-Datenbank basierend auf MinIO für skalierbaren Storage.

---

## BlobDB Overview

BlobDB ist eine Key-Value-Datenbank für Server-Daten mit MinIO-Backend.

<!-- verified: blob_instance.py::BlobDB -->

### Kern-Features

| Feature | Beschreibung |
|---------|-------------|
| **MinIO Backend** | Lokaler MinIO + optionaler Cloud Sync |
| **Offline-Fallback** | SQLite bei keiner MinIO-Verbindung |
| **Cache mit TTL** | Konfigurierbares Caching |
| **Manifest-Tracking** | Schnelle Key-Suche ohne Scan |

<!-- verified: blob_instance.py::Config -->

---

## Offline Fallback

### Aktivierung

```bash
export IS_OFFLINE_DB=true
```

### Funktionsweise

```
┌─────────────────────────────────────────────────────┐
│                    BlobDB                           │
├─────────────────────────────────────────────────────┤
│  IS_OFFLINE_DB=true?                                │
│  ├── JA → SQLite Fallback (lokal)                  │
│  └── NEIN → MinIO (lokal) → Cloud MinIO → SQLite   │
└─────────────────────────────────────────────────────┘
```

### SQLite Schema

```sql
CREATE TABLE blobs (
    path TEXT PRIMARY KEY,
    data BLOB NOT NULL,
    checksum TEXT,
    updated_at REAL,
    sync_status TEXT DEFAULT 'dirty'
);

CREATE TABLE manifest (
    key TEXT PRIMARY KEY,
    created_at REAL
);
```

<!-- verified: blob_instance.py::SQLiteCache -->

### Sync-Status

| Status | Bedeutung |
|--------|----------|
| `dirty` | Geändert, nicht gesynct |
| `synced` | Mit Cloud synchronisiert |

<!-- verified: blob_instance.py::SQLiteCache.mark_synced -->

---

## When to use BlobDB vs LC

### Local Dictionary (LC) - Wann?

| Kriterium | Empfehlung |
|-----------|-----------|
| Datenvolumen | < 10 MB |
| Zugriffe | < 100/s |
| Verteilung | Single-Instance |
| Komplexität | Einfache Key-Value |

**Typische Use-Cases:**
- User Settings
- Mod-Konfiguration
- Lokale Caches

<!-- verified: schema.py::LocalDBConfig -->

### BlobDB (MinIO) - Wann?

| Kriterium | Empfehlung |
|-----------|-----------|
| Datenvolumen | > 10 MB bis TBs |
| Zugriffe | > 100/s |
| Verteilung | Multi-Instance / Cluster |
| Komplexität | Enterprise-Grade |

**Typische Use-Cases:**
- Server-Sessions
- Binäre Assets
- Audit Logs
- Backup-Storage

<!-- verified: blob_instance.py::BlobDB -->

---

## Environment Variables

### Lokaler MinIO

| Variable | Default | Pflicht |
|----------|---------|--------|
| `MINIO_ENDPOINT` | `127.0.0.1:9000` | Nein |
| `MINIO_ACCESS_KEY` | `admin` | Nein |
| `MINIO_SECRET_KEY` | - | **Ja** |
| `MINIO_SECURE` | `false` | Nein |

### Cloud MinIO (optional)

| Variable | Pflicht |
|----------|---------|
| `CLOUD_ENDPOINT` | Nein |
| `CLOUD_ACCESS_KEY` | Nein |
| `CLOUD_SECRET_KEY` | Nein |
| `CLOUD_SECURE` | Nein |

### Betrieb

| Variable | Default | Beschreibung |
|----------|---------|-------------|
| `IS_OFFLINE_DB` | `false` | Nur SQLite, kein MinIO |
| `SERVER_ID` | `hostname` | Server-Identifier |
| `DB_CACHE_TTL` | `60` | Cache-Lebensdauer (Sekunden) |

<!-- verified: blob_instance.py::Config -->

---

## Usage Example

```python
from toolboxv2.mods.DB.blob_instance import create_db

# Initialisierung
db = create_db("my_server")

# CRUD
db.set("users/123", {"name": "Alice"})
result = db.get("users/123")
print(result.get())  # {"name": "Alice"}

# Pattern Matching
db.set("config/theme", "dark")
results = db.get("config/*")

# Cloud Sync
db.sync_to_cloud()

# Cleanup
db.exit()
```

<!-- verified: blob_instance.py::create_db -->

---

## Bucket & Path Structure

```
Bucket: tb-servers
│
└── {SERVER_ID}/
    ├── _manifest.json      # Key-Index
    ├── users/
    │   └── 123.json
    ├── config/
    │   └── theme.json
    └── ...
```

<!-- verified: blob_instance.py::BlobDB._key_to_path -->
