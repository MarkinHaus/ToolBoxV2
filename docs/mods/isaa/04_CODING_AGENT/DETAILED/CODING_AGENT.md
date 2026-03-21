# ISAA Coding Agent

## Übersicht

Der Coding Agent (`isaa_mod/CodingAgent/`) ist ein spezialisierter Agent für Code-Generierung und Projekt-Entwicklung.

## Komponenten

| Komponente | Datei | Beschreibung |
|------------|-------|---------------|
| `Coder` | `coder.py` | Haupt-Coder Engine |
| `CoderToolset` | `coder_toolset.py` | Coding Tools |
| `Live` | `live.py` | Live-Coding Mode |
| `Manager` | `manager.py` | Projekt-Management |

## Usage

```python
from isaa_mod.CodingAgent import Coder

# Coder erstellen
coder = Coder()

# Code generieren
code = await coder.generate(
    prompt=\"Erstelle eine REST API mit FastAPI\",
    language=\"python\"
)

# Projekt erstellen
project = await coder.create_project(
    name=\"my_api\",
    template=\"fastapi\"
)
```

## Live Mode

```python
# Live Coding Session
async with coder.live() as session:
    await session.write(\"main.py\", \"print('Hello')\")
    await session.run(\"python main.py\")
```

## Project Development UI

```python
from isaa_mod.CodingAgent.project_dev_ui import ProjectDevUI

ui = ProjectDevUI()

# Projekt öffnen
await ui.open_project(\"./my_project\")

# Parallel Connector für Multi-File
from project_dev_ui.parallel_connector import ParallelConnector
connector = ParallelConnector()
```
