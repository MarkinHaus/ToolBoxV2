# ProjectDeveloperEngine V3

> Multi-File Code Generation System für das ToolBoxV2 Ökosystem

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Übersicht

Die **ProjectDeveloperEngine V3** ist eine produktionsreife Code-Generierungs-Engine, die den AtomicCoder V2 vollständig refaktoriert und erweitert. Sie integriert nativ die bestehenden ToolBoxV2-Module:

- **DocsSystem** (mkdocs.py) - Projekt-Indizierung, Context Graph, semantische Suche
- **Executors** (executors.py) - Sichere Code-Ausführung via Docker/RestrictedPython
- **FlowAgent** (flow_agent.py) - LLM-Orchestrierung mit Chain-Patterns

### Kernfunktionen

| Feature | Beschreibung |
|---------|-------------|
| 🗂️ **Multi-File Support** | Generiert und validiert mehrere zusammenhängende Dateien |
| 🔍 **Context Graph** | Nutzt Upstream/Downstream-Abhängigkeiten aus DocsSystem |
| 🔬 **Research Phase** | Automatische API-Dokumentations-Recherche für unbekannte Imports |
| 🐳 **Sichere Ausführung** | Docker → RestrictedPython → Subprocess Fallback-Chain |
| 🔧 **Auto-Fix Loop** | Automatische Fehlerkorrektur mit LSP + Runtime-Feedback |
| 📊 **Token-Optimierung** | ContextBundle statt vollständiger Dateien im Prompt |

## 📦 Installation

```bash
# Basis-Installation
pip install pydantic pyyaml

# Für Docker-Executor (empfohlen)
pip install docker

# Für RestrictedPython-Executor
pip install restrictedpython

# Für LSP-Integration
pip install python-lsp-server pyflakes
```

## 🚀 Quickstart

```python
import asyncio
from toolboxv2 import get_app
from project_developer import create_project_developer

async def main():
    # Setup
    app = get_app()
    isaa = app.get_mod("isaa")
    await isaa.init_isaa()
    agent = await isaa.get_agent("coder")

    # Engine erstellen
    developer = create_project_developer(
        agent=agent,
        workspace_path="./my_project",
        prefer_docker=True,
        verbose=True
    )

    try:
        # Multi-File Task ausführen
        success, files = await developer.execute(
            task="Erstelle eine REST API mit FastAPI, Pydantic Models und SQLite",
            target_files=[
                "app/main.py",
                "app/models.py",
                "app/database.py"
            ],
            max_retries=3,
            auto_research=True
        )

        if success:
            print(f"✅ {len(files)} Dateien generiert")
        else:
            print("❌ Generierung fehlgeschlagen")

    finally:
        await developer.close()

asyncio.run(main())
```

## 🔄 State Machine

Die Engine arbeitet als 6-Phasen State Machine:

```
┌─────────┐
│  IDLE   │
└────┬────┘
     ▼
┌─────────────┐     DocsSystem.get_task_context()
│  ANALYSIS   │───► Context Graph laden
└─────┬───────┘     Unbekannte APIs identifizieren
      │
      ▼ (wenn unknown_apis)
┌─────────────┐     MCP/Web Search
│  RESEARCH   │───► API-Dokumentation holen
└─────┬───────┘     ResearchResults sammeln
      │
      ▼
┌─────────────┐     ProjectSpec erstellen
│ MULTI_SPEC  │───► FileActions planen
└─────┬───────┘     Dependency-Order festlegen
      │
      ▼
┌─────────────┐     Iterativ pro FileAction
│ GENERATION  │───► ContextBundle nutzen
└─────┬───────┘     Code generieren
      │
      ▼
┌─────────────┐     LSP Diagnostics
│ VALIDATION  │───► Runtime Tests (Docker)
└─────┬───────┘     Auto-Fix Loop (max 3x)
      │
      ▼
┌─────────────┐
│    SYNC     │───► Dateien schreiben
└─────┬───────┘     DocsSystem Index updaten
      │
      ▼
┌─────────────┐
│ COMPLETED   │
└─────────────┘
```

## 📐 Architektur

### Pydantic Models

```python
# Projekt-Spezifikation
class ProjectSpec(BaseModel):
    task_id: str
    intent: str                    # Aufgabenbeschreibung
    summary: str                   # Änderungszusammenfassung
    actions: List[FileAction]      # Geordnete Datei-Operationen
    upstream_deps: List[Dict]      # Abhängigkeiten
    downstream_usage: List[Dict]   # Verwendungsstellen
    research_results: List[ResearchResult]

# Einzelne Datei-Operation
class FileAction(BaseModel):
    action: FileActionType         # CREATE | MODIFY | DELETE
    file_path: str
    language: LanguageType         # PYTHON | JAVASCRIPT | ...
    description: str
    dependencies: List[str]        # Abhängige Dateien
    target_symbols: List[str]      # Zu erstellende Symbole
    priority: int                  # Ausführungsreihenfolge
    generated_code: Optional[str]
    validation_passed: bool

# Research-Ergebnis
class ResearchResult(BaseModel):
    source: str                    # docs | web | mcp
    topic: str
    content: str
    url: Optional[str]
    relevance: float
```

### Komponenten

```
ProjectDeveloperEngine
├── FlowAgent              # LLM-Interaktion
├── LSPManager             # Statische Analyse (Python, JS, TS)
├── SafeExecutor           # Code-Ausführung
│   ├── DockerCodeExecutor     (bevorzugt)
│   ├── RestrictedPythonExecutor
│   └── SubprocessFallback
└── DocsSystem             # Context & Graph
    ├── ContextEngine
    ├── IndexManager
    └── CodeAnalyzer
```

## 🔧 Konfiguration

### Factory-Parameter

```python
developer = create_project_developer(
    agent=agent,                    # FlowAgent Instanz (required)
    workspace_path="./project",     # Arbeitsverzeichnis (required)
    docs_system=None,               # Vorinitialisiertes DocsSystem
    auto_lsp=True,                  # LSP Server auto-starten
    prefer_docker=True,             # Docker bevorzugen
    verbose=True                    # Logging aktivieren
)
```

### Executor-Auswahl

Die Engine wählt automatisch den sichersten verfügbaren Executor:

| Priorität | Executor | Voraussetzung |
|-----------|----------|---------------|
| 1 | DockerCodeExecutor | Docker installiert + läuft |
| 2 | RestrictedPythonExecutor | `restrictedpython` installiert |
| 3 | SubprocessFallback | Immer verfügbar |

```python
# Executor-Typ prüfen
print(developer.executor.executor_type)  # "docker" | "restricted" | "subprocess"
```

## 📊 API Reference

### Hauptmethoden

#### `execute(task, target_files, max_retries=3, auto_research=True)`

Führt einen Multi-File-Entwicklungsauftrag aus.

```python
success, generated_files = await developer.execute(
    task="Implementiere Feature X",
    target_files=["src/feature.py", "src/utils.py"],
    max_retries=3,
    auto_research=True
)
```

**Returns:** `Tuple[bool, Dict[str, str]]` - (Erfolg, {Pfad: Code})

#### `get_state(execution_id)`

Gibt den aktuellen Ausführungszustand zurück.

```python
state = developer.get_state("abc123")
print(state.phase)           # DeveloperPhase.GENERATION
print(state.generated_files) # {"src/main.py": "..."}
print(state.errors)          # ["Error 1", "Error 2"]
```

#### `list_executions()`

Listet alle Ausführungen.

```python
executions = developer.list_executions()
# [{"id": "abc123", "task": "...", "phase": "completed", "success": True}]
```

#### `close()`

Räumt Ressourcen auf (LSP Server, etc.).

```python
await developer.close()
```

## 🧪 Testing

Tests mit `unittest` ausführen:

```bash
python -m unittest test_project_developer.py -v
```

## 🔍 Debugging

### Verbose Mode

```python
developer = create_project_developer(
    ...,
    verbose=True  # Aktiviert detailliertes Logging
)
```

### Phase History

```python
state = developer.get_state(execution_id)
for phase, timestamp in state.phase_history:
    print(f"{phase.value}: {timestamp}")
```

### Error Tracking

```python
state = developer.get_state(execution_id)
for error in state.errors:
    print(f"Error: {error}")
```

## 🔗 Integration mit ToolBoxV2

### Mit bestehendem DocsSystem

```python
from toolboxv2.mods.isaa.base.CodingAgent.mkdocs import create_docs_system

# DocsSystem separat initialisieren
docs = create_docs_system(
    project_root="./my_project",
    docs_root="./my_project/docs"
)
await docs.initialize()

# An Engine übergeben
developer = create_project_developer(
    agent=agent,
    workspace_path="./my_project",
    docs_system=docs  # Wiederverwendung
)
```

### Mit FlowAgent Chain Pattern

```python
# Chain für komplexe Workflows
from toolboxv2.mods.isaa.base.Agent.chain import Chain

analysis_agent = await isaa.get_agent("analyzer")
coder_agent = await isaa.get_agent("coder")

# Analyse → Coding Pipeline
pipeline = analysis_agent >> coder_agent
result = await pipeline.run("Analysiere und implementiere Feature X")
```

## 📝 Changelog

### V3.0.0 (Current)

- ✨ Multi-File Support mit ProjectSpec
- 🔗 Native DocsSystem Integration (ContextBundle)
- 🐳 Docker/RestrictedPython Executor Chain
- 🔍 Research Phase für unbekannte APIs
- 🔄 6-Phasen State Machine
- 📊 Token-optimierte Prompts

### V2.0.0 (AtomicCoder)

- Single-File Fokus
- Eigene CodeAnalyzer Implementation
- Unsicherer SandboxExecutor (exec())
- Keine Context Graph Integration

## 📄 Lizenz

MIT License - siehe [LICENSE](LICENSE)

## 🤝 Contributing

1. Fork das Repository
2. Feature Branch erstellen (`git checkout -b feature/amazing`)
3. Änderungen committen (`git commit -m 'Add amazing feature'`)
4. Branch pushen (`git push origin feature/amazing`)
5. Pull Request öffnen

---

**Entwickelt für das ToolBoxV2 Ökosystem** 🛠️
