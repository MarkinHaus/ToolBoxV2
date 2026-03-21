# How-To: ToolBoxV2 als Developer nutzen

## TL;DR
Developer-Profil für Mod-Entwicklung. Nutze `tb manifest pack` um Features zu verteilen.

<!-- verified: __main__.py::ProfileType -->
<!-- verified: manifest_cli.py::create_parser -->

## Profil: Developer
Das Developer-Profil ist für Mod-Entwicklung und Core-Arbeit.

<!-- verified: utils/clis/first_run.py::PROFILES -->

## Mod-Entwicklung starten

```bash
# Developer-Profil setzen
# In tb-manifest.yaml:
app:
  profile: developer
  environment: development
  debug: true

mods:
  watch_modules:
    - MyMod
```

## Feature packen

### CLI Command

```bash
# Einzelnes Feature packen
tb manifest pack <feature>

# Beispiel
tb manifest pack cli
tb manifest pack web
tb manifest pack mymod
```

<!-- verified: manifest_cli.py::create_parser -->

### Pack-Vorgang

Der `pack` Command:

1. Liest Feature aus `features/{name}/`
2. Erstellt ZIP-Archiv in `features_packed/`
3. Inkludiert alle relevanten Dateien

```bash
# Feature-Struktur wird gescannt
features/
  cli/
    feature.yaml      ← Metadaten
    bin/              ← Ausführbare Dateien
    lib/              ← Libraries
    docs/             ← Dokumentation

# Output
features_packed/
  cli.zip             ← Verteilbares Archiv
```

## Feature entpacken

```bash
# Feature aus ZIP entpacken
tb manifest unpack <path-to-zip>

# Beispiel
tb manifest unpack /path/to/myfeature.zip
```

<!-- verified: manifest_cli.py::create_parser -->

## Mod erstellen

### Struktur

```
mods_dev/
  MyMod/
    __init__.py
    main.py           ← Mod-Logik
    feature.yaml      ← Mod-Metadaten
    requirements.txt
```

### feature.yaml Beispiel

```yaml
name: MyMod
version: 0.1.0
description: Mein erstes Modul
author: Developer

dependencies:
  - requests

entry_point: MyMod.main:MyTool
```

## Mod laden

```bash
# Entwicklungsmodus mit Hot-Reload
tb dev

# Mods im Watch-Modus
tb manifest apply
# → mods.watch_modules wird überwacht
```

## Mod registrieren

```python
# mods_dev/MyMod/__init__.py
from toolboxv2 import MainTool

class MyTool(MainTool):
    name = "mymod"
    version = "0.1.0"

    async def run(self, args):
        print("MyMod ausgeführt!")
```

## Mod testen

```bash
# Alle Mods testen
tb --test
```

## Mods installieren

```bash
# Mod aus Registry installieren
tb registry install mymod

# Mod aus lokaler Datei
tb registry install --from ./mymod.zip

# Mod aus Git
tb registry install --git https://github.com/user/mymod
```

## Manifest für Development

```yaml
mods:
  init_modules:
    - CloudM
    - MyMod

  watch_modules:
    - MyMod         # Auto-Reload bei Änderungen

  open_modules:
    - CloudM.AuthManager
```

## CLI-Tools für Developer

| Command | Beschreibung |
|---------|---------------|
| `tb manifest list` | Alle Features anzeigen |
| `tb manifest pack <f>` | Feature packen |
| `tb manifest unpack <p>` | Feature entpacken |
| `tb manifest enable <f>` | Feature aktivieren |
| `tb manifest disable <f>` | Feature deaktivieren |
| `tb manifest files <f>` | Feature-Dateien anzeigen |

<!-- verified: manifest_cli.py::create_parser -->

## Tipps

1. **Hot-Reload**: Setze `watch_modules` für Auto-Reload
2. **Debug**: `tb dev --debug` für verbose Output
3. **Testing**: Nutze `tb test --mod MyMod`
4. **Dokumentation**: Erstelle `docs/` im Mod-Verzeichnis
