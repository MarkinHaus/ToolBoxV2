# Developers Guide - Registry Entwicklung

**Version**: 1.1
**Stand**: 2026-04-28

---

## Inhaltsverzeichnis

1. [Architektur](#architektur)
2. [Setup & Installation](#setup--installation)
3. [Admin Management](#admin-management)
4. [API-Referenz](#api-referenz)
5. [Datenbank-Schema](#datenbank-schema)
6. [Authentifizierung](#authentifizierung)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

---

## Architektur

### System-Übersicht

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Layer                          │
├─────────────────────────────────────────────────────────────┤
│  TB CLI (tb registry)  │  Web UI  │  HTTP API  │  CloudM   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  FastAPI  │  Routes  │  Services  │  Auth (CloudM.Auth)     │
│  Router prefix: /api/v1                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                      Data Layer                             │
├─────────────────────────────────────────────────────────────┤
│  SQLite  │  Repositories  │  MinIO (Storage)                │
└─────────────────────────────────────────────────────────────┘
```

### Verzeichnisstruktur

```
tb-registry/
├── admin_cli.py                  # Server-side Admin Tool
├── registry/
│   ├── __init__.py
│   ├── app.py                    # FastAPI Application
│   ├── config.py                 # Configuration (Pydantic)
│   │
│   ├── api/                      # API Layer
│   │   ├── __init__.py
│   │   ├── deps.py               # Dependencies (Auth, DB, etc.)
│   │   ├── router.py             # Main router configuration
│   │   └── routes/               # API Routes
│   │       ├── auth.py           # Authentication endpoints
│   │       ├── packages.py       # Package CRUD + Admin endpoints
│   │       ├── artifacts.py      # Artifact CRUD + Build upload
│   │       ├── publishers.py     # Publisher management + Verification
│   │       ├── search.py         # Search functionality
│   │       ├── versions.py       # Batch version query (update-ping)
│   │       ├── diff.py           # Incremental update diffs
│   │       ├── resolve.py        # Dependency resolution
│   │       └── health.py         # Health checks
│   │
│   ├── auth/                     # Authentication
│   │   └── cloudm_client.py      # CloudM.Auth JWT client
│   │
│   ├── db/                       # Database Layer
│   │   ├── database.py           # SQLite connection & schema
│   │   └── repositories/
│   │       ├── user_repo.py
│   │       ├── package_repo.py
│   │       └── artifact_repo.py
│   │
│   ├── models/                   # Data Models
│   │   ├── user.py               # User, Publisher, VerificationStatus
│   │   ├── package.py            # Package, Version, PackageType, Visibility
│   │   └── artifact.py           # Artifact, ArtifactType, ArtifactBuild
│   │
│   ├── services/                 # Business Logic
│   │   ├── package_service.py    # Package operations
│   │   ├── artifact_service.py   # Artifact operations
│   │   └── verification.py       # Publisher verification
│   │
│   ├── storage/                  # Storage Layer
│   │   └── manager.py            # MinIO/S3 wrapper
│   │
│   ├── resolver/                 # Dependency Resolution
│   │   └── dependency.py         # SemVer resolver
│   │
│   ├── diff.py                   # Diff generator for incremental updates
│   └── exceptions.py             # Custom exceptions
│
├── tests/
├── migrations/
└── pyproject.toml
```

### Route Mounting

Alle Routes werden unter `/api/v1` gemounted (`app.include_router(router, prefix="/api/v1")`):

| Route File | Prefix | Resultierende URL |
|---|---|---|
| `health.py` | (none) | `/api/v1/health`, `/api/v1/ready` |
| `auth.py` | `/auth` | `/api/v1/auth/me`, `/api/v1/auth/publisher` |
| `packages.py` | `/packages` | `/api/v1/packages`, `/api/v1/packages/{name}` |
| `artifacts.py` | `/artifacts` | `/api/v1/artifacts`, `/api/v1/artifacts/{name}` |
| `publishers.py` | `/publishers` | `/api/v1/publishers`, `/api/v1/publishers/verify` |
| `search.py` | `/search` | `/api/v1/search`, `/api/v1/search/suggest` |
| `resolve.py` | `/resolve` | `/api/v1/resolve`, `/api/v1/resolve/check` |
| `versions.py` | `/versions` | `/api/v1/versions` |
| `diff.py` | `/api/v1` | `/api/v1/api/v1/packages/{name}/diff/...` |

**Hinweis**: `diff.py` hat prefix `/api/v1` im Router, was zu doppeltem Prefix führt. Dies sollte zu `/diff` oder leerem Prefix korrigiert werden.

---

## Setup & Installation

### Lokale Entwicklung

```bash
# Repository klonen
git clone https://github.com/MarkinHaus/ToolBoxV2.git
cd ToolBoxV2/tb-registry

# Virtuelle Umgebung
python -m venv .venv
source .venv/bin/activate

# Dependencies installieren
pip install -e ".[dev]"

# Umgebung konfigurieren
cp .env.example .env
# CLOUDM_JWT_SECRET=your_secret_here

# Datenbank initialisieren
python -c "from registry.db.database import Database; import asyncio; asyncio.run(Database('sqlite:///./data/registry.db').initialize())"

# Server starten
uvicorn registry.app:app --reload --host 127.0.0.1 --port 4025
```

---

## Admin Management

### Erster Admin (Bootstrap)

Der erste Admin wird **direkt auf dem Server** über das Admin-CLI erstellt. Dies ist der einzige Weg einen Admin zu erstellen — es gibt keinen HTTP-Endpoint dafür.

```bash
# Auf dem Registry-Server ausführen
python admin_cli.py --db ./data/registry.db
```

**Voraussetzung**: Der User muss sich mindestens einmal eingeloggt haben (wird automatisch bei erstem JWT-Login erstellt via `get_current_user` in `deps.py`).

### Admin-CLI Befehle

Das Admin-CLI ist ein interaktives Menü:

| # | Befehl | Beschreibung |
|---|--------|-------------|
| 1 | List users | Alle User anzeigen |
| 2 | List publishers | Alle Publisher anzeigen |
| 3 | Make publisher | User zum Publisher machen + optional sofort verifizieren |
| 4 | Remove publisher | Publisher-Status entfernen |
| 5 | Edit publisher | Publisher-Felder bearbeiten |
| 6 | Set publisher status | Verification-Status ändern (unverified/pending/verified/rejected/suspended) |
| 7 | Toggle admin | Admin-Rechte an/aus |
| 8 | Raw SQL | Direkte SQL-Queries (read-only, `!`-Prefix für write) |

### Admin über API (nach Bootstrap)

Sobald ein Admin existiert, kann dieser über die Registry-CLI Publisher verwalten:

```bash
# Pending Publisher auflisten
tb registry admin publisher list --status pending

# Oder nur offene Requests
tb registry admin publisher open

# Publisher verifizieren
tb registry admin publisher verify --target <publisher-id>

# Publisher ablehnen
tb registry admin publisher reject --target <publisher-id> --notes "Reason"

# Verification widerrufen
tb registry admin publisher revoke --target <publisher-id>
```

### Admin API Endpoints

| Endpoint | Beschreibung |
|---|---|
| `GET /api/v1/packages/admin/pending` | Pending Publisher auflisten |
| `POST /api/v1/packages/admin/{publisher_id}/verify` | Publisher verifizieren |
| `POST /api/v1/packages/admin/{publisher_id}/reject` | Publisher ablehnen |
| `POST /api/v1/packages/admin/{publisher_id}/revoke` | Verification widerrufen |

Alle Admin-Endpoints erfordern `is_admin = true` auf dem User.

---

## API-Referenz

Siehe [API Reference](API_REFERENCE.md) für vollständige Endpoint-Dokumentation.

### Authentifizierung

#### JWT Token erstellen (für Tests)

```python
import jwt
import time

payload = {
    "user_id": "usr_test123",
    "username": "testuser",
    "email": "test@example.com",
    "level": 1,
    "provider": "magic_link",
    "exp": int(time.time()) + 3600,
    "iat": int(time.time()),
    "jti": f"jti_{int(time.time())}",
}
token = jwt.encode(payload, "your_jwt_secret", algorithm="HS256")
```

### Debug-Mode

Bei `DEBUG=True` und ohne konfiguriertem JWT-Secret gibt `verify_cloudm_token` einen Mock-User zurück:

```python
TokenPayload(
    user_id="user_debug",
    username="debug_user",
    email="debug@example.com",
    level=1,
    provider="debug",
)
```

---

## Datenbank-Schema

### Users Table

```sql
CREATE TABLE users (
    cloudm_user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    username TEXT NOT NULL,
    publisher_id TEXT REFERENCES publishers(id),
    is_admin BOOLEAN DEFAULT 0,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Publishers Table

```sql
CREATE TABLE publishers (
    id TEXT PRIMARY KEY,
    cloudm_user_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    email TEXT NOT NULL,
    website TEXT,
    github TEXT,
    status TEXT DEFAULT 'unverified',
    verified_at TIMESTAMP,
    verified_by TEXT,
    verification_notes TEXT,
    can_publish_public BOOLEAN DEFAULT 0,
    can_publish_artifacts BOOLEAN DEFAULT 0,
    max_package_size_mb INTEGER DEFAULT 100,
    packages_count INTEGER DEFAULT 0,
    total_downloads INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Packages Table

```sql
CREATE TABLE packages (
    name TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    package_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    publisher_id TEXT NOT NULL REFERENCES publishers(id),
    visibility TEXT DEFAULT 'public',
    description TEXT DEFAULT '',
    readme TEXT DEFAULT '',
    homepage TEXT,
    repository TEXT,
    license TEXT,
    keywords TEXT DEFAULT '[]',
    latest_version TEXT,
    total_downloads INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Package Versions Table

```sql
CREATE TABLE package_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    package_name TEXT NOT NULL REFERENCES packages(name) ON DELETE CASCADE,
    version TEXT NOT NULL,
    released_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    changelog TEXT DEFAULT '',
    dependencies TEXT DEFAULT '[]',
    toolbox_version TEXT,
    yanked BOOLEAN DEFAULT 0,
    yank_reason TEXT,
    UNIQUE(package_name, version)
);
```

### Enums im Code

**PackageType**: `mod`, `artifact`, `library`, `theme`, `plugin`

**ArtifactType**: `tauri_app`, `cli_executable`, `browser_extension`, `mobile_app`, `library`

**Visibility**: `public`, `private`, `unlisted`

**VerificationStatus**: `unverified`, `pending`, `verified`, `rejected`, `suspended`

**Platform**: `all`, `windows`, `linux`, `macos`, `android`, `ios`

**Architecture**: `all`, `x64`, `x86`, `arm64`, `arm32`

---

## Authentifizierung

### CloudM.Auth Integration

1. **Client** bekommt Token von CloudM.Auth (ToolBox Hauptanwendung)
2. **Client** sendet Token an Registry im `Authorization: Bearer` Header
3. **Registry** validiert Token lokal mit `CLOUDM_JWT_SECRET`
4. **User-Erststellung**: Bei erstem Login wird automatisch ein User in der DB erstellt (`deps.py: get_current_user`)

### JWT Token Struktur

```json
{
  "user_id": "usr_abc123",
  "username": "johndoe",
  "email": "john@example.com",
  "level": 1,
  "provider": "discord",
  "exp": 1740451200,
  "iat": 1740447600,
  "jti": "jti_usr_abc123_1740447600"
}
```

---

## Testing

### Unit Tests

```bash
# Alle Tests (unittest, nicht pytest!)
python -m unittest discover tests/ -v

# Spezifisches Modul
python -m unittest tests.test_packages -v
```

### Integration Tests

```bash
python -m unittest discover tests/integration/ -v
```

---

## Deployment

### Production Setup

1. **Environment Variables**
```bash
CLOUDM_JWT_SECRET=<strong-random-secret>
DATABASE_URL=sqlite:///./data/registry.db
MINIO_PRIMARY_ENDPOINT=minio.simplecore.app
MINIO_PRIMARY_ACCESS_KEY=access_key
MINIO_PRIMARY_SECRET_KEY=secret_key
DEBUG=False
```

2. **Start Application**
```bash
uvicorn registry.app:app --host 0.0.0.0 --port 4025 --workers 4
```

3. **Admin Bootstrap**
```bash
# Erster Admin muss auf dem Server erstellt werden
python admin_cli.py --db ./data/registry.db
# → Option 7: Toggle admin für deinen User
```

---

## Troubleshooting

### "Authentication service not configured"

```bash
# CLOUDM_JWT_SECRET nicht gesetzt
export CLOUDM_JWT_SECRET=your_secret
```

### "Admin access required" (403)

```bash
# User ist kein Admin
# Auf dem Server: python admin_cli.py → Option 7
```

### Diff-Route doppelter Prefix

Die `diff.py` Route ist mit prefix `/api/v1` im Router gemounted, was zu `/api/v1/api/v1/packages/...` führt. Fix: Prefix in `router.py` zu leerem String ändern oder die Pfade in `diff.py` anpassen.

---

**Letzte Aktualisierung**: 2026-04-28
