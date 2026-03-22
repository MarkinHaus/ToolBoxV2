# ToolBoxV2 Registry - Benutzerhandbuch

**Version**: 1.0
**Stand**: 2026-02-25

---

## Inhaltsverzeichnis

1. [Über das Registry](#über-das-registry)
2. [Schnellstart](#schnellstart)
3. [Mods herunterladen](#mods-herunterladen)
4. [Mods installieren](#mods-installieren)
5. [Mods aktualisieren](#mods-aktualisieren)
6. [Sichtbarkeit von Mods](#sichtbarkeit-von-mods)

---

## Über das Registry

Die ToolBoxV2 Registry ist die zentrale Verwaltungsplattform für alle ToolBoxV2 Module (Mods). Hier findest du:

- **Mods**: Erweiterungen für die ToolBoxV2
- **Libraries**: Nützliche Code-Bibliotheken
- **Artifacts**: Kompilierte Binaries und Assets

---

## Schnellstart

### Voraussetzungen

```bash
# ToolBoxV2 muss installiert sein
tb --version

# Registry-Client ist integriert
tb registry --help
```

### Registry konfigurieren

```bash
# Registry-URL setzen (standard: https://registry.tb2.app)
tb config set registry.url https://registry.tb2.app

# Authentifizierung mit CloudM.Auth
tb login
```

---

## Mods herunterladen

### Suche nach Mods

```bash
# Alle Mods auflisten
tb registry list

# Suche nach bestimmten Mods
tb registry search discord

# Infos zu einem Mod
tb registry info CloudM
```

### Mod herunterladen

```bash
# Mod herunterladen
tb registry download CloudM

# Spezifische Version
tb registry download CloudM --version 2.0.0

# In bestimmten Ordner herunterladen
tb registry download CloudM --output ./mods/
```

---

## Mods installieren

### Installation aus Registry

```bash
# Mod direkt installieren
tb install CloudM

# Mit Abhängigkeiten
tb install CloudM --with-deps

# Update erzwingen
tb install CloudM --force
```

### Lokale Installation

```bash
# Aus .tbx Datei installieren
tb install ./my_mod.tbx

# Aus Ordner installieren
tb install ./my_mod/
```

---

## Mods aktualisieren

### Alle Mods aktualisieren

```bash
# Check for updates
tb registry check-updates

# Update alle Mods
tb update --all
```

### Spezifisches Mod aktualisieren

```bash
# Ein bestimmtes Mod aktualisieren
tb update CloudM

# Auf bestimmte Version
tb update CloudM --version 2.1.0
```

### Rollback bei Fehlern

```bash
# Wenn ein Update fehlschlägt, wird automatisch ein Rollback durchgeführt
# Manuell auf vorherige Version:
tb rollback CloudM --to-version 2.0.0
```

---

## Sichtbarkeit von Mods

### Sichtbarkeitsstufen

Mods können drei Sichtbarkeitsstufen haben:

| Stufe | Beschreibung | Zugriff |
|-------|-------------|---------|
| **Public** | Öffentlich sichtbar | Jeder kann downloaden |
| **Unlisted** | Nicht gelistet | Nur mit direktem Link/Namen |
| **Private** | Privat | Nur der Owner |

### Public Mods

```bash
# Öffentliche Mods sind in der Suche sichtbar
tb registry search utility

# Jeder kann sie herunterladen
tb registry download MyUtilityMod
```

### Unlisted Mods

```bash
# Nicht in der Suche gelistet
# Aber herunterladen mit Name funktioniert:
tb registry download MyUnlistedMod

# Nützlich für Beta-Tests oder Invite-Only Mods
```

### Private Mods

```bash
# Private Mods benötigen Authentifizierung
tb login

# Download nur für den Owner
tb registry download MyPrivateMod

# Andere bekommen: 403 Forbidden
```

---

## Contributors Guide

### Registrierung als Contributor

1. **Account erstellen**
   ```bash
   #TODO fill
   ```

2. **Publisher erstellen**
   ```bash
   #TODO fill
   ```

3. **Verification beantragen** (optional, für Public Mods)
   ```bash
   #TODO fill
   ```

### Mod hochladen

#### Mod vorbereiten

```bash
# Mod-Verzeichnis strukturieren
my-mod/
├── mod.py              # Hauptdatei
├── mod.yaml            # Metadaten
├── requirements.txt    # Abhängigkeiten
├── README.md           # Dokumentation
└── assets/             # Bilder, Icons, etc.
```

#### mod.yaml Beispiel

```yaml
name: my_mod
display_name: My Awesome Mod
version: 1.0.0
description: Eine tolle Mod für ToolBoxV2
author: MyPublisher
license: MIT
homepage: https://github.com/user/my-mod

# Plattformen
platforms:
  server:
    files: ["*.py"]
    required: true
  client:
    files: ["assets/**"]
    required: false

# Abhängigkeiten
dependencies:
  - CloudM >= 2.0.0
  - isaa_core >= 1.5.0
```

#### Mod hochladen

```bash
# Mod registrieren (erster Upload)
tb registry upload ./my-mod/

# Update hochladen
tb registry upload ./my-mod/ --new-version 1.1.0

# Mit Changelog
tb registry upload ./my-mod/ --new-version 1.1.0 --changelog "Fixed bugs"
```

### Sichtbarkeit festlegen

#### Public Mod

```bash
# Public Mods sind für jeden sichtbar
tb registry publish my-mod --visibility public
```

**Voraussetzungen:**
- Publisher muss verifiziert sein
- Mod muss Beschreibung und README haben
- Mod muss getestet sein

#### Unlisted Mod

```bash
# Unlisted Mods sind nicht in der Suche
tb registry publish my-mod --visibility unlisted

# Kann mit direktem Link geteilt werden:
# https://registry.tb2.app/packages/my-mod
```

#### Private Mod

```bash
# Private Mods nur für dich
tb registry publish my-mod --visibility private

# Nur du (der Owner) kann herunterladen
tb registry download my-mod  # Funktioniert für dich
```

### Mod verwalten

#### Infos anzeigen

```bash
# Mod-Details
tb registry info my-mod

# Versionen auflisten
tb registry versions my-mod
```

#### Mod löschen

```bash
# Mod vollständig löschen (Vorsicht!)
tb registry delete my-mod --force

# Bestimmte Version löschen
tb registry delete my-mod --version 1.0.0
```

#### Yank (Zurückziehen)

```bash
# Version als "yanked" markieren (bleibt gelistet, aber nicht downloadbar)
tb registry yank my-mod --version 1.0.0 --reason "Critical bug"

# Yank aufheben
tb registry yank my-mod --version 1.0.0 --undo
```

---

## Troubleshooting

### Authentifizierungsprobleme

```bash
# Nicht eingeloggt?
Error: Authentication required

# Lösung:
tb login

# Token abgelaufen?
Error: Token expired

# Lösung:
tb login --refresh
```

### Download-Probleme

```bash
# Mod nicht gefunden?
Error: Package not found

# Lösung: Name prüfen (case-sensitive!)
tb registry search mod-name

# Version nicht gefunden?
Error: Version not found

# Lösung: Verfügbare Versionen prüfen
tb registry versions mod-name
```

### Installationsprobleme

```bash
# Abhängigkeiten fehlen?
Error: Missing dependencies

# Lösung: Mit --with-deps installieren
tb install mod-name --with-deps

# Datei beschädigt?
Error: Checksum mismatch

# Lösung: Cache leeren und neu downloaden
tb cache clear
tb registry download mod-name --force
```

---

## Häufige Fragen (FAQ)

### Wie werde ich Publisher?

```bash
#TODO fill
```

### Was ist der Unterschied zwischen Public und Unlisted?

| Public | Unlisted |
|--------|----------|
| In Suche sichtbar | Nicht in Suche |
| Jeder kann downloaden | Jeder mit Link kann downloaden |
| Verification benötigt | Keine Verification nötig |

### Kann ich meinen Mod später von Private auf Public ändern?

```bash
# Ja, mit re-publish:
tb registry publish my-mod --visibility public
```

### Wie teile ich einen Unlisted Mod?

Teile den direkten Link:
```
https://registry.tb2.app/packages/my-mod
```

Oder den Namen für CLI-Download:
```bash
tb registry download my-mod
```

---

## Weiterführende Links

- [Contributors Guide](CONTRIBUTORS_GUIDE.md) - Ausführliche Anleitung für Contributors
- [Developers Guide](DEVELOPERS_GUIDE.md) - Für System-Entwickler
- [API Documentation](https://registry.tb2.app/api/docs) - Interaktive API-Doku
- [GitHub Repository](https://github.com/toolboxv2/registry) - Source Code

---

**Letzte Aktualisierung**: 2026-02-25
**Kontakt**: support@toolboxv2.app
