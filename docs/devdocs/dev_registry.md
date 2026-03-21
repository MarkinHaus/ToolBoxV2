# ToolBoxV2 Registry CLI Reference

## SHARED_CONTEXT
Du arbeitest im ToolBoxV2 Doc-Writer System. Deine einzige Wahrheitsquelle ist der Code. Nie etwas erfinden oder aus alten Docs übernehmen ohne Code-Verifikation.

**CODE-REFERENZ REGEL**: Jeden Claim mit belegen. Format: `<!-- verified: <dateiname>::<klasse_oder_funktion> -->`

---

# Registry CLI Dokumentation

**WARNUNG**: Diese Dokumentation basiert ausschließlich auf dem implementierten Code in `cli_registry.py`.
Keine Funktion die dort nicht existiert wird dokumentiert.

<!-- verified: cli_registry.py -->

---

## Übersicht

Die Registry CLI ermöglicht:
- **Server Management**: Start/Stop/Status
- **Package Operations**: Search, List, Info, Download, Versions
- **Publishing**: Create, Upload, Delete, Yank
- **Authentication**: Login, Logout, WhoAmI
- **Admin**: Publisher Management

---

## Server Commands

### Start Registry Server

<!-- verified: cli_registry.py::registry_start -->
```bash
tb registry server start [OPTIONS]

OPTIONS:
  --host TEXT         Host to bind (default: 127.0.0.1)
  --port INTEGER      Port to bind (default: 4025)
  -b, --background    Run in background
  --reload            Enable auto-reload
```

**Beispiel**:
```bash
tb registry server start --host 0.0.0.0 --port 4025 --background
```

---

### Stop Registry Server

<!-- verified: cli_registry.py::registry_stop -->
```bash
tb registry server stop
```

---

### Registry Server Status

<!-- verified: cli_registry.py::registry_status -->
```bash
tb registry server status
```

Zeigt:
- Status (running/stopped)
- PID
- Host & Port

---

## Package Commands

### Search Packages

<!-- verified: cli_registry.py::registry_search -->
```bash
tb registry search QUERY [OPTIONS]

OPTIONS:
  --limit INTEGER    Max results (default: 50)
  -r, --registry-url TEXT   Registry URL
```

**Beispiel**:
```bash
tb registry search discord --limit 10
```

Ausgabe:
```
┌─────────────────────────────────────────────────────────┐
│  Name                  Version    Publisher    Downloads│
├─────────────────────────────────────────────────────────┤
│  toolboxv2-discord    1.2.0      simplecore    1,234    │
│  ...                                                    │
└─────────────────────────────────────────────────────────┘
```

---

### List Packages

<!-- verified: cli_registry.py::registry_list -->
```bash
tb registry list [OPTIONS]

OPTIONS:
  --type TEXT         Filter: mod, library, artifact
  --sort TEXT         Sort by: name, downloads, recent (default: name)
  --limit INTEGER     Max results (default: 50)
  -r, --registry-url TEXT   Registry URL
```

**Beispiel**:
```bash
# Alle mods nach Downloads sortiert
tb registry list --type mod --sort downloads
```

---

### Package Info

<!-- verified: cli_registry.py::registry_info -->
```bash
tb registry info PACKAGE [OPTIONS]

OPTIONS:
  --versions          Show version history
  -r, --registry-url TEXT   Registry URL
```

**Beispiel**:
```bash
tb registry info CloudM --versions
```

Zeigt:
- Name, Version, Visibility
- Publisher, License
- Homepage, Repository
- Description, Keywords
- Version History (mit --versions)

---

### Download Package

<!-- verified: cli_registry.py::registry_download -->
```bash
tb registry download PACKAGE [OPTIONS]

OPTIONS:
  --version TEXT      Specific version (default: latest)
  -o, --output DIR    Output directory (default: .)
  -r, --registry-url TEXT   Registry URL
```

**Beispiel**:
```bash
# Download latest
tb registry download CloudM

# Specific version
tb registry download CloudM --version 2.0.0 --output ./mods/
```

---

### Package Versions

<!-- verified: cli_registry.py::registry_versions -->
```bash
tb registry versions PACKAGE [OPTIONS]

OPTIONS:
  -r, --registry-url TEXT   Registry URL
```

**Beispiel**:
```bash
tb registry versions CloudM
```

Ausgabe:
```
┌────────────────────┬───────────────────────┬────────────┬──────────┐
│ Version            │ Published             │ Downloads  │ Status   │
├────────────────────┼───────────────────────┼────────────┼──────────┤
│ 2.0.0              │ 2024-01-15            │ 500        │ Active   │
│ 1.5.0 (YANKED)     │ 2023-12-01            │ 200        │ YANKED   │
└────────────────────┴───────────────────────┴────────────┴──────────┘
```

---

## Publishing Commands

### Publish Package

<!-- verified: cli_registry.py::registry_publish -->
```bash
tb registry publish PACKAGE [OPTIONS]

OPTIONS:
  --create            Create new package
  --upload            Upload new version
  --visibility TEXT   Set: public, private, unlisted
  -m, --metadata FILE Path to metadata JSON
  -r, --registry-url TEXT   Registry URL
```

**Metadaten JSON**:
```json
{
  "name": "mein-mod",
  "display_name": "Mein Mod",
  "package_type": "mod",
  "visibility": "unlisted",
  "description": "Ein tolles Mod",
  "readme": "README.md",
  "homepage": "https://github.com/user/repo",
  "repository": "https://github.com/user/repo",
  "license": "MIT",
  "keywords": ["toolbox", "mod"]
}
```

**Beispiel**:
```bash
# Neues Paket erstellen
tb registry publish ./mein-mod --create --metadata meta.json

# Version hochladen
tb registry publish ./mein-mod --upload --metadata meta.json

# Sichtbarkeit ändern
tb registry publish mein-mod --visibility public
```

---

### Upload with Diff Support

<!-- verified: cli_registry.py::registry_upload -->
```bash
tb registry upload PACKAGE -m METADATA [OPTIONS]

OPTIONS:
  -m, --metadata FILE       Path to metadata JSON (REQUIRED)
  --diff-threshold INTEGER Max diff ratio % (default: 50)
  --force-full             Force full upload (no diff)
  -r, --registry-url TEXT   Registry URL
```

**Beispiel**:
```bash
tb registry upload ./mein-mod.zip \\
  --metadata meta.json \\
  --diff-threshold 30
```

Ausgabe:
```
✅ Upload successful!
Type: DIFF
Uploaded: 45 KB / 200 KB
Saved: 155 KB (77.5%)
Diffed from: 1.0.0
```

---

### Delete Package

<!-- verified: cli_registry.py::registry_delete -->
```bash
tb registry delete PACKAGE [--force]
```

**Beispiel**:
```bash
# Mit Bestätigung
tb registry delete mein-mod

# Ohne Bestätigung
tb registry delete mein-mod --force
```

---

### Yank Version

<!-- verified: cli_registry.py::registry_yank -->
```bash
tb registry yank PACKAGE VERSION [OPTIONS]

OPTIONS:
  --reason TEXT       Reason for yanking
  --undo             Unyank (restore) the version
```

**Beispiel**:
```bash
# Version yanken
tb registry yank mein-mod 1.0.0 --reason "Security vulnerability"

# Version wiederherstellen
tb registry yank mein-mod 1.0.0 --undo
```

---

## Authentication Commands

### Login

<!-- verified: cli_registry.py::registry_login -->
```bash
tb registry login [-r REGISTRY_URL]
```

Nutzt **CloudM.Auth** für Authentifizierung:
1. Token wird von CloudM.Auth bezogen
2. Token wird mit Registry verifiziert
3. Token wird lokal gespeichert (`~/.tb-registry/auth_token.txt`)

**Beispiel**:
```bash
tb registry login
```

---

### Logout

<!-- verified: cli_registry.py::registry_logout -->
```bash
tb registry logout
```

**Beispiel**:
```bash
tb registry logout
✅ Logged out successfully
```

---

### WhoAmI

<!-- verified: cli_registry.py::registry_whoami -->
```bash
tb registry whoami [-r REGISTRY_URL]
```

Zeigt aktuellen Benutzer:
- User ID
- Username
- Email
- Verified Status
- Admin Status
- Publisher ID

---

## Admin Commands

### Publisher Management

<!-- verified: cli_registry.py::registry_admin_publisher -->
```bash
tb registry admin publisher ACTION [OPTIONS]

ACTIONS:
  list                List all publishers
  open                List pending publishers
  verify TARGET       Verify a publisher
  reject TARGET       Reject a publisher
  revoke TARGET       Revoke verification

OPTIONS:
  --target TEXT       Publisher ID
  --status TEXT       Filter: unverified, pending, verified, rejected
  --notes TEXT        Action notes / reason
```

**Beispiel**:
```bash
# Liste alle Publisher
tb registry admin publisher list

# Zeige ausstehende Anträge
tb registry admin publisher open

# Publisher verifizieren
tb registry admin publisher verify publisher_abc123

# Publisher ablehnen
tb registry admin publisher reject publisher_xyz789 --notes "Invalid repository"

# Verifizierung zurückziehen
tb registry admin publisher revoke publisher_abc123
```

---

## Utility Commands

### Health Check

<!-- verified: cli_registry.py::registry_health -->
```bash
tb registry health [-r REGISTRY_URL]
```

**Beispiel**:
```bash
tb registry health
✅ Registry is healthy: https://registry.simplecore.app
```

---

## Global Options

Alle Commands akzeptieren:

```bash
-r, --registry-url TEXT   Registry URL (default: https://registry.simplecore.app)
-v, --verbose             Verbose output
```

---

## Error Codes

| Code | Bedeutung |
|------|----------|
| 0 | Erfolgreich |
| 1 | Fehler |
| 130 | Interrupted (Ctrl+C) |

---

## Troubleshooting

### Authentication failed
```bash
# Prüfe ob eingeloggt
tb registry whoami

# Erneut einloggen
tb registry login
```

### Package not found
```bash
# Prüfe Schreibweise
tb registry search mein-paket

# Prüfe Registry URL
tb registry list -r https://registry.simplecore.app
```

### Upload fehlgeschlagen
```bash
# Mit verbose
tb registry upload ./mein-mod.zip --metadata meta.json -v

# Full Upload erzwingen
tb registry upload ./mein-mod.zip --metadata meta.json --force-full
```
