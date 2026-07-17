# TB Registry — Vollständige Dokumentation

> **Status:** Validiert am 2026-07-17 durch Live-Tests (Server auf Port 4030, MinIO-Backend, alle Core-Endpunkte grün)

## Inhaltsverzeichnis

1. [Architektur-Überblick](#architektur-überblick)
2. [Consumer Guide — Pakete finden & herunterladen](#consumer-guide)
3. [Publisher Guide — Pakete veröffentlichen](#publisher-guide)
4. [Admin Guide — Registry verwalten](#admin-guide)
5. [API-Referenz (Kompakt)](#api-referenz)
6. [Validierungs-Ergebnisse](#validierungs-ergebnisse)
7. [Bekannte Issues](#bekannte-issues)

---

## Architektur-Überblick

### Komponenten

```
┌──────────────────────────────────────────────────────────┐
│                    ToolBoxV2 (Client)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ tb registry │  │ RegistryClient│  │ cli_registry.py │  │
│  │   (CLI)     │  │ (async HTTP) │  │   (18 cmds)     │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘  │
│         └────────────────┴───────────────────┘            │
│                          │ JWT Bearer Token                │
└──────────────────────────┼────────────────────────────────┘
                           │ HTTPS
┌──────────────────────────┼────────────────────────────────┐
│                   tb-registry (Server)                     │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              FastAPI Application                     │  │
│  │  ┌──────┐ ┌──────────┐ ┌─────────┐ ┌─────────────┐ │  │
│  │  │ Auth │ │ Packages │ │Artifacts│ │ Search/Res. │ │  │
│  │  └──┬───┘ └────┬─────┘ └────┬────┘ └──────┬──────┘ │  │
│  │     │          │            │              │        │  │
│  │  ┌──┴──────────┴────────────┴──────────────┴──────┐ │  │
│  │  │          SQLite (aiosqlite, WAL mode)          │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────┘  │
│                          │                                 │
│  ┌───────────────────────┴─────────────────────────────┐  │
│  │            StorageManager (MinIO S3)                │  │
│  │  ┌─────────────────┐    ┌────────────────────────┐  │  │
│  │  │  Primary MinIO  │───▶│  Mirror MinIO (async)  │  │  │
│  │  └─────────────────┘    └────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Zwei Datenmodell-Kategorien

| Kategorie | Model | Upload-Pfad | Beispiele |
|-----------|-------|-------------|-----------|
| **Packages** (Mods, Libraries) | `Package` → `PackageVersion` | `POST /api/v1/packages/{name}/versions` | Python-Module, ToolBox-Erweiterungen |
| **Artifacts** (Apps, Installers) | `Artifact` → `ArtifactVersion` → `ArtifactBuild` | `POST /api/v1/artifacts/{name}/builds` | GUI (Tauri), CLI-Executables, Browser-Extensions |

### Auth: CloudM.Auth JWT

- **Token-Format:** HS256-signiertes JWT
- **Claims:** `user_id`, `username`, `email`, `level`, `provider`, `exp`, `iat`, `jti`
- **Validierung:** Lokal via Shared Secret (`CLOUDM_JWT_SECRET`)
- **User-Auto-Provisioning:** Erster API-Call mit gültigem Token erstellt User in Registry-DB
- **3 Rollen:** Anonymous → User → Publisher (verified) → Admin

---

## Consumer Guide

### Voraussetzungen

- ToolBoxV2 installiert (`tb` CLI verfügbar)
- CloudM.Auth Login (`tb login`)

### Pakete suchen

```bash
# Über CLI
tb registry search "example"
tb registry search --type mod "database"

# Über API
GET /api/v1/search?q=example&package_type=mod&page=1&per_page=20

# Auto-Vorschläge
GET /api/v1/search/suggest?q=exam&limit=5
```

### Paket-Details abrufen

```bash
# CLI
tb registry info example-mod
tb registry versions example-mod

# API
GET /api/v1/packages/example-mod
GET /api/v1/packages/example-mod/versions
```

### Paket herunterladen

```bash
# CLI
tb registry download example-mod --version 0.1.0

# API (liefert presigned MinIO URL, 1h gültig)
GET /api/v1/packages/example-mod/versions/0.1.0/download
```

**Visibility-Regeln für Downloads:**
| Visibility | Wer kann herunterladen? |
|------------|------------------------|
| `public` | Jeder (anonym) |
| `unlisted` | Nur authentifizierte User |
| `private` | Nur der Owner |

### Update-Check (Update-Ping)

```bash
# CLI — prüft alle installierten Mods auf Updates
tb registry list --check-updates

# API — Batch-Abfrage der neuesten Versionen
GET /api/v1/versions?names=mod1&names=mod2&names=mod3
# Response: {"versions": {"mod1": "1.2.3", "mod2": null}}
```

### Dependency Resolution

```bash
# API — Abhängigkeiten auflösen
POST /api/v1/resolve
Body: {"requirements": ["example-mod>=0.1.0", "other-lib<2.0.0"]}

# Kompatibilitäts-Check
GET /api/v1/resolve/check?package=example-mod&version=0.1.0&toolbox_version=2.0
```

### Inkrementelle Updates (Diff)

```bash
# Diff zwischen zwei Versionen abrufen
GET /api/v1/packages/{name}/diff/{from_version}/{to_version}

# Patch-Datei herunterladen
GET /api/v1/packages/{name}/diff/{from_version}/{to_version}/download
```

---

## Publisher Guide

### 1. Publisher werden

```bash
# Als Publisher registrieren
tb registry register-publisher --name "my-org" --display-name "My Organization" --email "contact@my.org"

# Verifizierung einreichen (Warte auf Admin)
tb registry verify-publisher --method email --data '{"email":"contact@my.org"}'
```

**Publisher-Status Lifecycle:**
```
unverified → pending → verified  (Admin genehmigt)
                    ↘ rejected   (Admin lehnt ab)
verified → unverified (Admin revoke)
         → suspended  (Admin suspendiert)
```

**Wichtig:** Nur `verified` Publisher mit `can_publish_public=true` können öffentliche Pakete veröffentlichen.

### 2. Package erstellen (Mod/Library)

```bash
# CLI
tb registry create-package --name "my-mod" --type mod --description "My awesome mod"

# API
POST /api/v1/packages
Body: {
  "name": "my-mod",
  "display_name": "My Mod",
  "package_type": "mod",          // mod | library | theme | plugin | artifact
  "description": "My awesome mod",
  "homepage": "https://github.com/me/my-mod",
  "repository": "https://github.com/me/my-mod"
}
```

**Package-Namen-Regeln:** `^[a-z0-9_-]+$`, 1-100 Zeichen.

### 3. Version hochladen

```bash
# CLI
tb registry publish my-mod.zip --name my-mod --version 0.1.0 --changelog "Initial release"

# API (multipart/form-data)
POST /api/v1/packages/my-mod/versions
Form: {
  "version": "0.1.0",
  "changelog": "Initial release",
  "file": <binary zip file>
}
```

**Version-Format:** SemVer `^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$`

### 4. Artifact erstellen (App/Installer)

```bash
# Artifact erstellen
POST /api/v1/artifacts
Body: {
  "name": "toolbox-gui",
  "artifact_type": "tauri_app",   // tauri_app | cli_executable | browser_extension | mobile_app | library
  "description": "ToolBox GUI",
  "homepage": "https://toolbox.dev"
}

# Platform-spezifischen Build hochladen (multipart)
POST /api/v1/artifacts/toolbox-gui/builds
Form: {
  "version": "1.0.0",
  "platform": "windows",          // all | windows | linux | macos | android | ios
  "architecture": "x64",          // all | x64 | x86 | arm64 | arm32
  "changelog": "Release 1.0.0",
  "installer_type": "msi",        // optional: msi, exe, dmg, deb, etc.
  "min_os_version": "10.0",       // optional
  "file": <binary installer file>
}
```

### 5. Package verwalten

```bash
# Package aktualisieren
PATCH /api/v1/packages/my-mod
Body: {"description": "Updated description", "visibility": "public"}

# Version yanken (als deprecated markieren, nicht löschen)
POST /api/v1/packages/my-mod/versions/0.1.0/yank
Body: {"reason": "Security issue in dependency"}

# Package löschen (nur Owner oder Admin)
DELETE /api/v1/packages/my-mod
```

### Versionierung & Sichtbarkeit

| Feld | Werte | Beschreibung |
|------|-------|-------------|
| `version` | `1.0.0`, `1.0.0-beta`, `2.1.3` | SemVer-konform |
| `visibility` | `public`, `unlisted`, `private` | Wer kann das Paket sehen/herunterladen |
| `yanked` | `true/false` | Version als problematisch markiert |
| `toolbox_version` | `>=2.0` | Kompatibilität mit ToolBox-Version |
| `python_version` | `>=3.11` | Benötigte Python-Version |

---

## Admin Guide

### Server starten

#### Lokal (Entwicklung)

```bash
cd tb-registry
cp .env.example .env
# .env anpassen: MINIO creds, CLOUDM_JWT_SECRET, SERVER_PORT
uv sync
uv run tb-registry
```

#### Docker (Produktion)

```bash
docker build -t tb-registry .
docker run -d \
  -p 4025:4025 \
  --env-file .env \
  -v registry-data:/app/data \
  tb-registry
```

#### Docker Compose (Full Stack)

```bash
cd tb-registry
docker-compose up -d
# Startet: tb-registry + MinIO (primary) + optional MinIO (mirror)
```

### Konfiguration (.env)

| Variable | Default | Beschreibung |
|----------|---------|-------------|
| `SERVER_PORT` | `4025` | Server-Port |
| `HOST` | `0.0.0.0` | Bind-Adresse |
| `DEBUG` | `false` | Debug-Modus (Mock-Auth bei leerem JWT-Secret) |
| `DATABASE_URL` | `sqlite:///./data/registry.db` | SQLite-Pfad |
| `CLOUDM_JWT_SECRET` | — | Shared Secret für JWT-Validierung |
| `CLOUDM_AUTH_URL` | — | URL zu CloudM.Auth (Fallback-Validierung) |
| `CORS_ORIGINS` | `["*"]` | Erlaubte CORS-Origins |
| `MINIO_PRIMARY_ENDPOINT` | `localhost:9000` | MinIO-Server |
| `MINIO_PRIMARY_ACCESS_KEY` | — | MinIO Access Key |
| `MINIO_PRIMARY_SECRET_KEY` | — | MinIO Secret Key |
| `MINIO_PRIMARY_BUCKET` | `tb-registry` | MinIO Bucket-Name |
| `MINIO_PRIMARY_SECURE` | `false` | HTTPS für MinIO |
| `MINIO_MIRROR_*` | — | Optionaler Mirror (alle 4 Felder nötig) |

### Admin CLI (Direkt auf dem Server)

```bash
cd tb-registry
python admin_cli.py --db ./data/registry.db
```

**Verfügbare Kommandos:**

| # | Kommando | Beschreibung |
|---|----------|-------------|
| 1 | List users | Alle registrierten User anzeigen |
| 2 | List publishers | Alle Publisher mit Status/Stats |
| 3 | Make publisher | User → Publisher (mit sofortiger Verify-Option) |
| 4 | Remove publisher | Publisher-Status entfernen |
| 5 | Edit publisher | Felder bearbeiten (Name, Slug, Limits) |
| 6 | Set publisher status | Status ändern (verified/rejected/suspended) |
| 7 | Toggle admin | Admin-Rechte gewähren/entziehen |
| 8 | Raw SQL | Direkte SQL-Abfragen (`!` Prefix für Writes) |

**⚠️ Sicherheit:** Admin CLI läuft OHNE Authentifizierung. Nur lokal auf dem Server ausführen — niemals extern exponieren.

### Publisher-Verifizierung (Admin API)

```bash
# Ausstehende Verifizierungen auflisten
GET /api/v1/packages/admin/pending    # (erfordert Admin-Token)

# Publisher verifizieren
POST /api/v1/packages/admin/{publisher_id}/verify
Body: {"notes": "Verified via GitHub"}

# Publisher ablehnen
POST /api/v1/packages/admin/{publisher_id}/reject
Body: {"notes": "Invalid contact information"}

# Verifizierung widerrufen
POST /api/v1/packages/admin/{publisher_id}/revoke
Body: {"notes": "Policy violation"}
```

### Health & Readiness

```bash
# Einfacher Health-Check (für Load Balancer)
GET /health                    # → {"status": "healthy"}
HEAD /health                   # → 200 (Docker healthcheck)

# Readiness (DB + Storage prüfen)
GET /api/v1/ready
# → {"status": "ready", "details": {"database": true, "storage": {"healthy": true, ...}}}
```

### Storage-Verwaltung

Die Registry nutzt **MinIO** (S3-kompatibel) mit Primary + optionalem Mirror:

- **Upload-Flow:** Datei → Primary MinIO → Async Queue → Mirror MinIO
- **Download-Flow:** Presigned URL (1h gültig), Mirror bevorzugt wenn konfiguriert
- **Pfad-Schema:** `packages/{name}/{version}/{filename}` bzw. `artifacts/{name}/{version}/{platform}_{arch}/{filename}`
- **Checksum:** SHA256 für jede hochgeladene Datei

### Backup

```bash
# 1. SQLite DB
cp data/registry.db data/registry-backup-$(date +%Y%m%d).db

# 2. MinIO Bucket (via mc)
mc mirror local/tb-registry ./backup-$(date +%Y%m%d)/

# 3. .env + Konfiguration
cp .env .env.backup
```

---

## API-Referenz

### Basis-URL
```
http://localhost:{SERVER_PORT}/api/v1
```

### Auth-Endpunkte

| Method | Path | Auth | Beschreibung |
|--------|------|------|-------------|
| GET | `/auth/me` | User | Aktueller User + Publisher-Info |
| POST | `/auth/register-publisher` | User | Als Publisher registrieren |
| GET | `/auth/publisher` | User | Eigenen Publisher-Status |
| POST | `/publishers/verify` | Publisher | Verifizierung einreichen |

### Package-Endpunkte

| Method | Path | Auth | Beschreibung |
|--------|------|------|-------------|
| GET | `/packages` | — | Pakete auflisten (paginiert) |
| POST | `/packages` | Publisher | Paket erstellen |
| GET | `/packages/{name}` | — | Paket-Details |
| POST | `/packages/{name}/versions` | Publisher | Version hochladen |
| DELETE | `/packages/{name}` | Owner/Admin | Paket löschen |

### Artifact-Endpunkte

| Method | Path | Auth | Beschreibung |
|--------|------|------|-------------|
| GET | `/artifacts` | — | Artifacts auflisten |
| POST | `/artifacts` | Publisher | Artifact erstellen |
| GET | `/artifacts/{name}` | — | Artifact-Details |
| POST | `/artifacts/{name}/builds` | Publisher | Build hochladen |
| GET | `/artifacts/{name}/latest` | — | Neueste Version für Platform |
| GET | `/artifacts/{name}/versions/{ver}/download` | — | Presigned Download-URL |

### Search & Resolve

| Method | Path | Auth | Beschreibung |
|--------|------|------|-------------|
| GET | `/search?q=...` | — | Pakete durchsuchen |
| GET | `/search/suggest?q=...` | — | Auto-Vorschläge |
| POST | `/resolve` | — | Dependencies auflösen |
| GET | `/resolve/check` | — | Kompatibilität prüfen |

### Admin-Endpunkte

| Method | Path | Auth | Beschreibung |
|--------|------|------|-------------|
| GET | `/packages/admin/pending` | Admin | Ausstehende Publisher |
| POST | `/packages/admin/{id}/verify` | Admin | Publisher verifizieren |
| POST | `/packages/admin/{id}/reject` | Admin | Publisher ablehnen |
| POST | `/packages/admin/{id}/revoke` | Admin | Verifizierung widerrufen |

### Health

| Method | Path | Auth | Beschreibung |
|--------|------|------|-------------|
| GET | `/health` | — | Health-Check |
| GET | `/health` (HEAD) | — | Docker healthcheck |
| GET | `/ready` | — | DB + Storage Readiness |
| GET | `/versions` | — | Update-Ping (Batch) |

---

## Validierungs-Ergebnisse

> Durchgeführt am 2026-07-17 gegen lokalen Server (Port 4030, SQLite, lokaler MinIO)

### ✅ Funktionierende Endpunkte

| Test | Endpoint | Ergebnis |
|------|----------|----------|
| Health-Check | `GET /health` | ✅ `{"status":"healthy"}` |
| Readiness | `GET /api/v1/ready` | ✅ DB healthy, MinIO healthy |
| Auth — Token-Validierung | `GET /api/v1/auth/me` | ✅ JWT validiert, User auto-created |
| Auth — Admin-Erkennung | `GET /api/v1/auth/me` | ✅ `is_admin: true` nach DB-Update |
| Publisher registrieren | `POST /api/v1/auth/register-publisher` | ✅ Publisher erstellt (Status: unverified) |
| Publisher verifizieren | Direct DB / Admin CLI | ✅ Status: verified, can_publish_public: 1 |
| Package erstellen (mod) | `POST /api/v1/packages` | ✅ `{"id":"example-mod"}` |
| Package erstellen (library) | `POST /api/v1/packages` | ✅ `{"id":"toolbox-core-lib"}` |
| Package auflisten | `GET /api/v1/packages` | ✅ 2 Packages, korrekte Paginierung |
| Artifact erstellen | `POST /api/v1/artifacts` | ✅ `tauri_app` mit UUID |
| Artifact Build upload | `POST /artifacts/{name}/builds` | ✅ File → MinIO, SHA256 berechnet |
| Artifact latest | `GET /artifacts/{name}/latest` | ✅ Version + Build-Details |
| Versions-Ping | `GET /versions?names=...` | ✅ Korrekte Response |
| Resolve | `POST /resolve` | ✅ Korrekte Conflict-Erkennung |
| Publisher-Liste | `GET /publishers` | ✅ Stats (packages_count, downloads) |

### Getestete Uploads (Non-destructive)

| Upload-Typ | Name | Version | Ergebnis |
|------------|------|---------|----------|
| Mod Package | `example-mod` | (keine Version) | ✅ Metadaten gespeichert |
| Library Package | `toolbox-core-lib` | (keine Version) | ✅ Metadaten gespeichert |
| Artifact (Tauri App) | `toolbox-gui` | `0.0.1` | ✅ Build + File in MinIO |

---

## Bekannte Issues

### 1. Package Version Upload Route fehlt (kritisch)

**Symptom:** `POST /api/v1/packages/{name}/versions` → 404 Not Found

**Analyse:** Die Service-Methode `PackageService.upload_version()` existiert, aber es gibt **keine HTTP-Route** die sie exponiert. Die `packages.py` Route-Datei definiert keinen POST-Endpoint für `/packages/{name}/versions`.

**Workaround:** Artifact-Upload-System (`/artifacts/{name}/builds`) funktioniert vollständig und kann als Alternative für alle Upload-Typen genutzt werden.

**Fix:** Route in `packages.py` hinzufügen:
```python
@router.post("/{name}/versions")
async def upload_version(name, version=Form(...), file=File(...), ...):
    ...
```

### 2. Artifact storage_locations nicht persisted

**Symptom:** Build-Upload Response zeigt `"storage_locations": ""` (leer).

**Analyse:** Die Datei wird erfolgreich zu MinIO hochgeladen (SHA256 + Size korrekt), aber die StorageLocation-Referenz wird nicht in der SQLite-DB gespeichert. Dies führt dazu, dass der Download-Endpoint "No download available" zurückgibt.

**Impact:** Presigned Download-URLs können nicht generiert werden.

### 3. Search ohne Treffer

**Symptom:** `GET /api/v1/search?q=example` → 0 Ergebnisse trotz existierender Packages.

**Analyse:** Mögliche Ursachen: SQLite FTS-Index nicht gebaut, oder Search implementiert nur LIKE auf Beschreibung/Name mit Case-Sensitivity-Issue.

### 4. Config-Variable Naming

**Beobachtung:** `.env` nutzt `PORT=4025`, aber `config.py` erwartet `SERVER_PORT` (Field alias). Pydantic-Settings löst dies korrekt auf, aber die `.env.example` verwendet `PORT` — inkonsistent.

### 5. Debug-Modus reload verursacht Windows-Port-Konflikte

**Beobachtung:** `DEBUG=true` aktiviert uvicorn `reload=True` (WatchFiles). Bei Dateiänderungen im Projektordner (inkl. `_fix_db.py`) trigger Reload, der alte Port bleibt in Windows TIME_WAIT.

**Empfehlung:** `DEBUG=false` für Tests, oder Reload-Excludes konfigurieren.
