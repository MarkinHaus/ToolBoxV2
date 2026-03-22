# Contributors Guide - Mods veröffentlichen

**Version**: 1.0
**Stand**: 2026-02-25

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

```bash
# ToolBoxV2 CLI installiert?
tb --version

# Einloggen (mit CloudM.Auth)
tb login
```

### Publisher erstellen

Jeder Mod braucht einen Publisher (Entwickler/Organisation):

```bash
#TODO fill
```

### Publisher Status prüfen

```bash
#TODO fill
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

# Mod testen
tb test my-mod

# Lokale Installation testen
tb install .
```

### 2. Mod bauen/packen

```bash
# Als .tbx packen
tb pack my-mod

# Erzeugt: my-mod-1.0.0.tbx
```

### 3. Mod hochladen

```bash
# Erster Upload (neues Mod)
tb registry upload ./my-mod/

# Output:
# ✓ Package registered: my-mod v1.0.0
# ✓ Public URL: https://registry.tb2.app/packages/my-mod
```

### 4. Update veröffentlichen

```bash
# Version erhöhen (in my_mod.yaml)
# version: 1.0.0 -> 1.1.0

# Update hochladen
tb registry upload ./my-mod/ --new-version 1.1.0

# Mit Changelog
tb registry upload ./my-mod/ \
  --new-version 1.1.0 \
  --changelog "Added feature X, fixed bug Y"
```

### 5. Sichtbarkeit ändern

```bash
# Auf Public (nach Verification!)
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
- Mod ist getestet

**Verification beantragen:**

```bash
#TODO fill
```

**Verification-Status:**
- `pending` - Warten auf Review
- `verified` - Kann Public Mods veröffentlichen
- `rejected` - Grund prüfen und erneut beantragen

### Unlisted Mods

**Nicht in Suche, aber mit Link/Namen downloadbar.**

**Anwendungsfälle:**
- Beta-Tests
- Private Verteilung anselected Users
- Mods in Entwicklung

```bash
# Unlisted Mod
tb registry publish my-mod --visibility unlisted

# Link teilen:
# https://registry.tb2.app/packages/my-mod

# Oder CLI-Download:
tb registry download my-mod
```

### Private Mods

**Nur für den Owner downloadbar.**

**Anwendungsfälle:**
- Persönliche Tools
- Mods in früher Entwicklung
- Mods mit sensiblen Daten

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

```bash
# Gute Changelogs:
tb registry upload ./my-mod/ \
  --new-version 1.1.0 \
  --changelog "
  Added:
  - Feature X for doing Y
  - Integration with Service Z

  Fixed:
  - Bug when running on Windows
  - Memory leak in long-running processes

  Changed:
  - Improved performance by 50%
  "
```

### Testing

```bash
# Mod vor Upload testen
tb test my-mod --full

# Auf verschiedenen Plattformen testen
tb test my-mod --platform windows
tb test my-mod --platform linux
tb test my-mod --platform macos
```

### Dokumentation

```markdown
# README.md sollte enthalten:

1. Kurze Beschreibung
2. Installationsanleitung
3. Konfiguration
4. Beispiele
5. Bekannte Issues
6. Roadmap (optional)
```

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
        run: |
          curl -sSL https://install.tb2.app | sh
          echo "$HOME/.tb/bin" >> $GITHUB_PATH

      - name: Pack Mod
        run: tb pack .

      - name: Publish to Registry
        env:
          TB_TOKEN: ${{ secrets.TB_REGISTRY_TOKEN }}
        run: |
          VERSION=${GITHUB_REF#refs/tags/}
          tb registry upload ./ \
            --new-version $VERSION \
            --token $TB_TOKEN
```

### GitLab CI Beispiel

```yaml
publish:
  stage: deploy
  only:
    - tags
  script:
    - pip install toolboxv2
    - tb pack .
    - tb registry upload ./ --new-version $CI_COMMIT_TAG
```

---

## Troubleshooting

### Upload fehlschlägt

```bash
# Error: Package already exists
# Lösung: Neue Version verwenden
tb registry upload ./my-mod/ --new-version 1.1.0

# Error: Publisher not verified
# Lösung: Verification beantragen oder Unlisted verwenden
tb registry publish my-mod --visibility unlisted

# Error: Invalid manifest
# Lösung: my_mod.yaml prüfen
tb validate ./my-mod/my_mod.yaml
```

### Verification abgelehnt

```bash
# Status prüfen
tb publisher status

# Feedback anzeigen
tb publisher status --verbose

# Erneut beantragen (nach Korrekturen)
tb publisher verify --reason "Fixed all issues"
```

---

## API-Referenz

### CLI-Befehle

| Befehl | Beschreibung |
|--------|-------------|
| `tb register` | Account erstellen |
| `tb login` | Einloggen |
| `tb publisher create` | Publisher erstellen |
| `tb publisher verify` | Verification beantragen |
| `tb registry upload` | Mod hochladen |
| `tb registry publish` | Sichtbarkeit ändern |
| `tb registry delete` | Mod löschen |

### HTTP API

```bash
# Mod hochladen (HTTP API)
curl -X POST https://registry.tb2.app/api/v1/packages \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@my-mod-1.0.0.tbx"

# Mod infos
curl https://registry.tb2.app/api/v1/packages/my-mod

# Versionen auflisten
curl https://registry.tb2.app/api/v1/packages/my-mod/versions
```

---

**Weiterführende Links:**
- [User Guide](USER_GUIDE.md) - Für Mod-Nutzer
- [Developers Guide](DEVELOPERS_GUIDE.md) - Für Registry-Entwickler
- [API Documentation](https://registry.tb2.app/api/docs)

---

**Letzte Aktualisierung**: 2026-02-25
