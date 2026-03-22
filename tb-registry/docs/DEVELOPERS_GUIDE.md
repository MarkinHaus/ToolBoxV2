# Developers Guide - Registry Entwicklung

**Version**: 1.0
**Stand**: 2026-02-25

---

## Inhaltsverzeichnis

1. [Architektur](#architektur)
2. [Setup & Installation](#setup--installation)
3. [API-Referenz](#api-referenz)
4. [Datenbank-Schema](#datenbank-schema)
5. [Authentifizierung](#authentifizierung)
6. [Testing](#testing)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## Architektur

### System-Übersicht

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Layer                          │
├─────────────────────────────────────────────────────────────┤
│  TB CLI │  Web UI │  HTTP API │  Webhooks                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  FastAPI │  Routes │  Services │  Auth (CloudM.Auth)        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                      Data Layer                             │
├─────────────────────────────────────────────────────────────┤
│  SQLite │  Repositories │  MinIO (Storage)                  │
└─────────────────────────────────────────────────────────────┘
```

### Verzeichnisstruktur

```
tb-registry/
├── registry/
│   ├── __init__.py
│   ├── app.py                    # FastAPI Application
│   ├── config.py                 # Configuration (Pydantic)
│   │
│   ├── api/                      # API Layer
│   │   ├── deps.py               # Dependencies (Auth, DB, etc.)
│   │   └── routes/               # API Routes
│   │       ├── auth.py           # Authentication endpoints
│   │       ├── packages.py       # Package CRUD
│   │       ├── artifacts.py      # Artifacts
│   │       ├── publishers.py     # Publisher management
│   │       ├── search.py         # Search functionality
│   │       ├── health.py         # Health checks
│   │       └── resolve.py        # Dependency resolution
│   │
│   ├── auth/                     # Authentication
│   │   └── cloudm_client.py      # CloudM.Auth JWT client
│   │
│   ├── db/                       # Database Layer
│   │   ├── database.py           # SQLite connection & schema
│   │   └── repositories/         # Data access layer
│   │       ├── user_repo.py
│   │       ├── package_repo.py
│   │       └── artifact_repo.py
│   │
│   ├── models/                   # Data Models
│   │   ├── user.py               # User, Publisher
│   │   └── package.py            # Package, Version, etc.
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
│   └── exceptions.py             # Custom exceptions
│
├── tests/                        # Tests
│   ├── test_auth.py              # Auth tests
│   ├── test_packages.py          # Package tests
│   ├── test_zip_security.py      # ZIP security tests
│   └── integration/              # E2E tests
│
├── migrations/                   # DB migrations
├── docs/                         # Documentation
└── pyproject.toml                # Python package config
```

---

## Setup & Installation

### Lokale Entwicklung

```bash
# Repository klonen
git clone https://github.com/toolboxv2/tb-registry.git
cd tb-registry

# Virtuelle Umgebung erstellen
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Dependencies installieren
pip install -e ".[dev]"

# Umgebung konfigurieren
cp .env.example .env
# .env bearbeiten:
# CLOUDM_JWT_SECRET=your_secret_here
# DATABASE_URL=sqlite:///./data/registry.db

# Datenbank initialisieren
python -c "from registry.db.database import Database; import asyncio; asyncio.run(Database('sqlite:///./data/registry.db').initialize())"

# Server starten
uvicorn registry.app:app --reload --host 127.0.0.1 --port 4025
```

### Docker Setup

```bash
# Docker Compose
docker-compose up -d

# Oder manuell
docker build -t tb-registry .
docker run -p 4025:4025 \
  -e CLOUDM_JWT_SECRET=secret \
  -e DATABASE_URL=sqlite:///./data/registry.db \
  tb-registry
```

### Entwicklung-Tools

```bash
# Code-Formatierung
ruff format registry/
ruff check registry/

# Typ-Prüfung
mypy registry/

# Tests
pytest tests/ -v
pytest tests/integration/ -v

# Coverage
pytest --cov=registry --cov-report=html
```

---

## API-Referenz

### Authentication

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

#### API mit Token aufrufen

```bash
curl -H "Authorization: Bearer $TOKEN" \
  https://registry.tb2.app/api/v1/auth/me
```

### Packages Endpoints

#### Package auflisten

```http
GET /api/v1/packages?page=1&per_page=20
```

**Response:**
```json
{
  "packages": [
    {
      "name": "CloudM",
      "display_name": "CloudM Module",
      "version": "2.0.0",
      "description": "Cloud management module",
      "author": "ToolBoxV2",
      "downloads": 1234,
      "visibility": "public"
    }
  ],
  "total": 1,
  "page": 1,
  "per_page": 20
}
```

#### Package erstellen

```http
POST /api/v1/packages
Authorization: Bearer $TOKEN
Content-Type: application/json

{
  "name": "my-mod",
  "display_name": "My Mod",
  "description": "My awesome mod",
  "package_type": "mod",
  "visibility": "public"
}
```

#### Package upload

```http
POST /api/v1/packages/{name}/upload
Authorization: Bearer $TOKEN
Content-Type: multipart/form-data

file: <binary>
version: "1.0.0"
changelog: "Initial release"
```

#### Package Details

```http
GET /api/v1/packages/{name}
```

#### Package downloaden

```http
GET /api/v1/packages/{name}/versions/{version}/download
```

### Search Endpoints

```http
GET /api/v1/search?q=discord&page=1
```

### Health Endpoints

```http
GET /api/v1/health
GET /api/v1/ready
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
    can_publish_public BOOLEAN DEFAULT 0,
    packages_count INTEGER DEFAULT 0,
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

---

## Authentifizierung

### CloudM.Auth Integration

Die Registry verwendet **CloudM.Auth** für die JWT-Validierung:

1. **Client** bekommt Token von CloudM.Auth (ToolBox Hauptanwendung)
2. **Client** sendet Token an Registry im `Authorization: Bearer` Header
3. **Registry** validiert Token lokal mit `CLOUDM_JWT_SECRET`

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

### Configuration

```python
# registry/config.py
class Settings(BaseSettings):
    cloudm_jwt_secret: str = Field(
        default="",
        description="Shared secret for CloudM.Auth JWT validation"
    )
    cloudm_auth_url: Optional[str] = Field(
        default=None,
        description="URL to CloudM.Auth service (fallback validation)"
    )
```

### Dependency Usage

```python
from fastapi import Depends
from registry.api.deps import get_current_user

@router.get("/api/v1/auth/me")
async def get_my_info(user: User = Depends(get_current_user)):
    return {"user_id": user.cloudm_user_id, "email": user.email}
```

---

## Testing

### Unit Tests

```bash
# Alle Tests
pytest tests/ -v

# Spezifisches Modul
pytest tests/test_packages.py -v

# Mit Coverage
pytest --cov=registry --cov-report=html
```

### Integration Tests

```bash
# E2E Tests
pytest tests/integration/ -v

# Spezifischer Test
pytest tests/integration/test_auth_migration.py::test_full_auth_flow -v
```

### Test-Beispiel

```python
import pytest
from httpx import AsyncClient
from registry.app import create_app

@pytest.mark.asyncio
async def test_get_package():
    app = create_app()
    async with AsyncClient(app=app) as client:
        response = await client.get("/api/v1/packages/CloudM")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "CloudM"
```

---

## Deployment

### Production Setup

1. **Environment Variables**
```bash
CLOUDM_JWT_SECRET=<strong-random-secret>
DATABASE_URL=postgresql://user:pass@host/db
MINIO_PRIMARY_ENDPOINT=minio.example.com
MINIO_PRIMARY_ACCESS_KEY=access_key
MINIO_PRIMARY_SECRET_KEY=secret_key
DEBUG=False
```

2. **Database Migration**
```bash
python migrations/002_add_cloudm_user_id.py
```

3. **Start Application**
```bash
uvicorn registry.app:app --host 0.0.0.0 --port 4025 --workers 4
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install -e .

COPY registry/ registry/

EXPOSE 4025
CMD ["uvicorn", "registry.app:app", "--host", "0.0.0.0", "--port", "4025"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tb-registry
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tb-registry
  template:
    metadata:
      labels:
        app: tb-registry
    spec:
      containers:
      - name: registry
        image: tb-registry:latest
        ports:
        - containerPort: 4025
        env:
        - name: CLOUDM_JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: registry-secrets
              key: jwt-secret
```

---

## Troubleshooting

### Häufige Fehler

#### "Authentication service not configured"

```bash
# Lösung: CLOUDM_JWT_SECRET setzen
export CLOUDM_JWT_SECRET=your_secret
```

#### "No such column: cloudm_user_id"

```bash
# Lösung: Migration ausführen
python migrations/002_add_cloudm_user_id.py
```

#### "Package not found"

```bash
# Prüfen: Package existiert?
curl https://registry.tb2.app/api/v1/packages/{name}
```

### Debug-Mode

```python
# config.py
debug: bool = Field(default=False)

# deps.py
if settings.debug:
    # Returns mock user for testing
    return TokenPayload(user_id="debug_user", ...)
```

---

## Weiterführende Links

- [User Guide](USER_GUIDE.md) - Für End-Nutzer
- [Contributors Guide](CONTRIBUTORS_GUIDE.md) - Für Mod-Entwickler
- [Security Audit](../migrations/003_security_audit.md) - Security-Checklist
- [Deployment Checklist](../DEPLOYMENT_CHECKLIST.md) - Production-Deployment

---

**Letzte Aktualisierung**: 2026-02-25
