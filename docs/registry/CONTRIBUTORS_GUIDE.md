# Contributors Guide - Mods veröffentlichen

**Version**: 1.1
**Stand**: 2026-04-28

---

## Inhaltsverzeichnis

1. [Erste Schritte](#erste-schritte)
2. [Mod-Struktur](#mod-struktur)
3. [Publishing-Workflow](#publishing-workflow)
4. [Sichtbarkeit & Zugriff](#sichtbarkeit--zugriff)
5. [Best Practices](#best-practices)
6. [CI/CD Integration](#cicd-integration)

---

## Erste Schritte

### Account erstellen

Accounts werden automatisch beim ersten Login erstellt — kein separater Registrierungsschritt nötig.

```bash
# ToolBoxV2 CLI installiert?
tb --version

# Einloggen (mit CloudM.Auth)
tb registry login
# → Dein Account wird automatisch erstellt

# Prüfen ob Login erfolgreich
tb registry whoami
```

### Publisher erstellen

Jeder Mod braucht einen Publisher (Entwickler/Organisation):

```bash
# Option A: Über den interaktiven Manager
tb -c CloudM mods manager
# → REGISTRY → Register as Publisher

# Option B: Via HTTP API
curl -X POST https://registry.simplecore.app/api/v1/auth/register-publisher \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-publisher",
    "display_name": "My Publisher",
    "email": "contact@example.com",
    "homepage": "https://example.com"
  }'
```

### Publisher Status prüfen

```bash
# Eigenen Publisher-Status anzeigen
tb registry whoami
# → Zeigt Publisher-ID und ob Admin

# Via API
curl -H "Authorization: Bearer $TOKEN" \
  https://registry.simplecore.app/api/v1/auth/publisher
```

---

## Mod-Struktur

### Minimale Struktur

```
my-mod/
├── __init__.py          # Mod-Initialisierung
├── my_mod.py           # Hauptdatei
└── my_mod.yaml         # Metadaten (Pflicht!)
```

### Empfohlene Struktur

```
my-mod/
├── __init__.py
├── my_mod.py
├── my_mod.yaml         # Metadaten
├── requirements.txt     # Python-Abhängigkeiten
├── README.md            # Dokumentation
├── CHANGELOG.md         # Änderungshistorie
├── LICENSE              # Lizenz
└── assets/              # Optionale Assets
    ├── icon.png         # 64x64 PNG
    └── banner.png       # Optionales Banner
```

### my_mod.yaml Referenz

```yaml
# Pflichtfelder
name: my_mod                    # Interner Name (keine Leerzeichen!)
display_name: My Awesome Mod     # Angezeigter Name
version: 1.0.0                   # Semantische Versionierung
description: Eine Beschreibung   # Was macht der Mod?
author: awesome-mods              # Publisher-Slug
license: MIT                      # SPDX Lizenz-Identifier

# Optionale Felder
homepage: https://github.com/user/my-mod
repository: https://github.com/user/my-mod.git
keywords: ["utility", "discord", "automation"]

# Plattform-Support
platforms:
  server:                         # Server-seitige Komponenten
    files: ["*.py", "requirements.txt"]
    required: true
  client:                         # Client-seitige Komponenten (optional)
    files: ["assets/**"]
    required: false

# Abhängigkeiten
dependencies:
  - CloudM >= 2.0.0               # Mindestversion
  - isaa_core >= 1.5.0            # Exakte Version
  - some-mod >= 1.0.0 < 2.0.0    # Version range

# Toolbox-Kompatibilität
toolbox_version: ">=0.1.20"

# Sichtbarkeit (wenn nicht überschrieben)
visibility: public                # public | unlisted | private
```

---

## Publishing-Workflow

### 1. Mod entwickeln

```bash
# Mod lokal entwickeln
cd my-mod

# tbConfig generieren
tb -c CloudM mods gen-config my-mod
```

### 2. Metadata-Datei erstellen

Erstelle eine `metadata.json` für das Publishing:

```json
{
  "name": "my-mod",
  "display_name": "My Awesome Mod",
  "package_type": "mod",
  "version": "1.0.0",
  "description": "This mod does awesome things",
  "visibility": "unlisted",
  "homepage": "https://github.com/user/my-mod",
  "repository": "https://github.com/user/my-mod.git",
  "license": "MIT",
  "keywords": ["utility", "automation"]
}
```

### 3. Package erstellen (erster Upload)

```bash
# Package in der Registry registrieren
tb registry publish ./my-mod/ --create --metadata metadata.json

# Output:
# ✓ Package 'my-mod' created successfully
```

### 4. Version hochladen

```bash
# Version hochladen
tb registry publish ./my-mod/ --upload --metadata metadata.json

# Oder mit Diff-Support (spart Bandbreite bei Updates)
tb registry upload ./my-mod.zip --metadata metadata.json
```

### 5. Update veröffentlichen

```bash
# Version in metadata.json erhöhen: 1.0.0 -> 1.1.0
# changelog hinzufügen

# Update hochladen
tb registry publish ./my-mod/ --upload --metadata metadata.json

# Mit Diff-Optimierung (nur Änderungen hochladen)
tb registry upload ./my-mod.zip \
  --metadata metadata.json \
  --diff-threshold 50
```

### 6. Sichtbarkeit ändern

```bash
# Auf Public (erfordert verifizierten Publisher)
tb registry publish my-mod --visibility public

# Auf Unlisted (für Beta/Testing)
tb registry publish my-mod --visibility unlisted

# Auf Private (nur für dich)
tb registry publish my-mod --visibility private
```

---

## Sichtbarkeit & Zugriff

### Public Mods

**Für jeden sichtbar und downloadbar.**

**Voraussetzungen:**
- Publisher ist **verifiziert**
- Mod hat vollständige Metadaten

**Verification beantragen:**

```bash
# Via HTTP API
curl -X POST https://registry.simplecore.app/api/v1/publishers/verify \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"method": "github", "data": {"username": "dein-github"}}'
```

**Verification-Status:**
- `unverified` - Noch nicht beantragt
- `pending` - Warten auf Admin-Review
- `verified` - Kann Public Mods veröffentlichen
- `rejected` - Grund prüfen und erneut beantragen
- `suspended` - Temporär gesperrt

### Unlisted Mods

**Nicht in Suche, aber mit Link/Namen downloadbar.**

```bash
# Unlisted Mod
tb registry publish my-mod --visibility unlisted

# Andere können downloaden wenn sie den Namen kennen:
tb registry download my-mod
```

### Private Mods

**Nur für den Owner downloadbar.**

```bash
# Private Mod
tb registry publish my-mod --visibility private

# Nur du kannst downloaden:
tb registry download my-mod
# Andere erhalten: 403 Forbidden
```

---

## Best Practices

### Versionierung

Verwende **Semantic Versioning** (SemVer):

```
MAJOR.MINOR.PATCH

1.0.0  -> Erstes Release
1.1.0  -> Neues Feature (Backward Compatible)
1.1.1  -> Bug Fix (Backward Compatible)
2.0.0  -> Breaking Changes
```

### Changelog

Trage den Changelog in die `metadata.json` ein:

```json
{
  "name": "my-mod",
  "version": "1.1.0",
  "changelog": "Added: Feature X for doing Y\nFixed: Bug when running on Windows\nChanged: Improved performance by 50%"
}
```

### Dokumentation

Dein README.md sollte enthalten:

1. Kurze Beschreibung
2. Installationsanleitung
3. Konfiguration
4. Beispiele
5. Bekannte Issues

---

## CI/CD Integration

### GitHub Actions Beispiel

```yaml
name: Publish to Registry

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install TB CLI
        run: pip install toolboxv2

      - name: Create metadata
        run: |
          VERSION=${GITHUB_REF#refs/tags/}
          cat > metadata.json << EOF
          {
            "name": "${{ github.event.repository.name }}",
            "version": "$VERSION",
            "package_type": "mod",
            "changelog": "${{ github.event.release.body }}"
          }
          EOF

      - name: Publish to Registry
        env:
          TB_TOKEN: ${{ secrets.TB_REGISTRY_TOKEN }}
        run: |
          tb registry login
          tb registry publish ./ --upload --metadata metadata.json
```

---

## Troubleshooting

### Upload fehlschlägt

```bash
# Error: Package already exists (409 CONFLICT)
# Lösung: Version erhöhen in metadata.json

# Error: Must be a registered publisher (403)
# Lösung: Publisher registrieren (siehe "Erste Schritte")

# Error: Authentication required (401)
# Lösung: Einloggen
tb registry login
```

### Verification abgelehnt

```bash
# Status prüfen
tb registry whoami

# Feedback über API abrufen
curl -H "Authorization: Bearer $TOKEN" \
  https://registry.simplecore.app/api/v1/auth/publisher

# Erneut beantragen (nach Korrekturen)
curl -X POST https://registry.simplecore.app/api/v1/publishers/verify \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"method": "github", "data": {"username": "dein-github"}}'
```

---

## CLI-Befehle Übersicht

| Befehl | Beschreibung |
|--------|-------------|
| `tb registry login` | Einloggen |
| `tb registry logout` | Ausloggen |
| `tb registry whoami` | Eigene Info anzeigen |
| `tb registry search <query>` | Mods suchen |
| `tb registry list` | Mods auflisten |
| `tb registry info <name>` | Mod-Details |
| `tb registry versions <name>` | Versionen auflisten |
| `tb registry download <name>` | Mod herunterladen |
| `tb registry publish <path>` | Mod erstellen/updaten/visibility |
| `tb registry upload <file>` | Upload mit Diff-Support |
| `tb registry delete <name>` | Mod löschen |
| `tb registry yank <name> <ver>` | Version zurückziehen |
| `tb registry health` | Registry-Status prüfen |
| `tb registry admin publisher` | Publisher-Verwaltung (Admin) |

---

**Weiterführende Links:**
- [User Guide](USER_GUIDE.md) - Für Mod-Nutzer
- [Developers Guide](DEVELOPERS_GUIDE.md) - Für Registry-Entwickler
- [API Reference](API_REFERENCE.md) - HTTP-API Endpunkte

---

**Letzte Aktualisierung**: 2026-04-28
