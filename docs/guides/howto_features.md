# How-To: ToolBoxV2 Features verstehen und nutzen

## TL;DR
Features werden automatisch beim Import entpackt basierend auf installierten Extras.

<!-- verified: feature_loader.py::EXTRA_TO_FEATURES -->
<!-- verified: feature_loader.py::FEATURE_DETECTION -->

## Architektur

ToolBoxV2 verwendet ein modulares Feature-System:

```
toolboxv2/features_packed/   ← ZIP-Archive
toolboxv2/features/          ← Entpackte Features
```

<!-- verified: feature_loader.py::get_features_packed_dir -->
<!-- verified: feature_loader.py::get_features_dir -->

## Feature-Extras

| Extra | Features | Abhängigkeiten |
|-------|----------|----------------|
| `cli` | cli | prompt_toolkit, rich, readchar |
| `web` | web | starlette, uvicorn, httpx |
| `desktop` | desktop | PyQt6 |
| `exotic` | exotic | scipy, matplotlib, pandas |
| `isaa` | isaa | litellm, langchain_core, groq |
| `production` | core, cli, web | - |
| `dev` | Alle | - |
| `all` | Alle | - |

<!-- verified: feature_loader.py::EXTRA_TO_FEATURES -->
<!-- verified: feature_loader.py::FEATURE_DETECTION -->

## Installation mit Features

### Nur Core (Standard)
```bash
pip install toolboxv2
```
→ Nur Core wird entpackt

### Mit extras
```bash
pip install toolboxv2[web]     # Core + Web
pip install toolboxv2[cli]    # Core + CLI
pip install toolboxv2[all]    # Alles
pip install toolboxv2[isaa]  # AI-Agent
```

<!-- verified: feature_loader.py::__init__.py:ensure_features_loaded -->

## Automatisches Entpacken

Beim `import toolboxv2` wird automatisch:

1. `ensure_features_loaded()` aufgerufen
2. Prüfen welche Extras installiert sind
3. Passende ZIPs entpacken
4. Feature-Status setzen

```python
from toolboxv2 import ensure_features_loaded, get_feature_status

# Alle installierten Features
status = get_feature_status()
print(status)
```

<!-- verified: feature_loader.py::is_feature_installed -->

## Feature prüfen

Ein Feature gilt als installiert wenn:
1. Verzeichnis existiert: `features/{name}/`
2. `feature.yaml` existiert
3. `.installed` Marker existiert

```python
from toolboxv2.feature_loader import is_feature_installed

if is_feature_installed("web"):
    print("Web-Feature ist aktiv")
```

## Verfügbare Features

| Feature | Beschreibung |
|---------|---------------|
| `core` | Basis-System |
| `cli` | Kommandozeilen-Tools |
| `web` | Web-Interface |
| `desktop` | Desktop-GUI |
| `exotic` | Wissenschaftliche Tools |
| `isaa` | AI-Agent |

<!-- verified: feature_loader.py::CORE_FEATURE -->

## CLI-Commands für Features

```bash
# Features anzeigen
tb manifest list

# Feature aktivieren
tb manifest enable <feature>

# Feature deaktivieren
tb manifest disable <feature>

# Packed Features anzeigen
tb manifest packed

# Feature packen
tb manifest pack <feature>

# Feature entpacken
tb manifest unpack <path>
```

<!-- verified: manifest_cli.py::create_parser -->
