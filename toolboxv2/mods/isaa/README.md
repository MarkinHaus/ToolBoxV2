# ISAA Mod - Intelligent Semantic Agent Architecture

> **Version:** 0.3.0 | **Status:** Production Ready | **Python:** 3.10+

---

## 🎯 Übersicht

Die **ISAA Mod** ist eine produktionsreife, modulare Agent-Architektur für das ToolBoxV2 Ökosystem. Sie bietet:

- 🤖 **Multi-Agent System** - Erstelle und verwalte mehrere spezialisierte Agenten
- 🧠 **Advanced Memory** - Hybrides Vektorspeicher-System mit BM25 + FAISS + Graph
- 🔧 **CodingAgent** - Automatische Code-Generierung und Validierung
- 🌐 **Kernel System** - Multi-Plattform Integration (Discord, Telegram, WhatsApp)
- 🔗 **Chain Support** - Agent-Ketten für komplexe Workflows
- 💾 **Agent Export/Import** - Agenten als `.tar.gz` Archive speichern und teilen

---

## 📦 Installation

```bash
# Aus dem ToolBoxV2 Repository
cd toolboxv2/mods/isaa

# Dependencies installieren
pip install -r requirements.txt

# Optional: Entwicklungstools
pip install pytest black ruff mypy
```

### Requirements

```
beautifulsoup4>=4.12.0
langchain-core>=0.1.20
litellm~=1.74.14
networkx>=3.1
pydantic>=2.5.0
groq>=0.31.0
requests>=2.31.0
tiktoken>=0.5.0
tqdm>=4.66.0
pypdf2>=3.0.1
nest-asyncio>=1.5.0
schedule>=1.2.0
pyvis>=0.3.0
redis>=5.0.0
python-a2a[all]>=0.5.1
mcp>=1.6.0
google-cloud-aiplatform>=1.38.0
pyyaml>=6.0
docker>=7.0.0
hnswlib>=0.8.0
faiss-cpu>=1.12.0
langchain-community>=0.3.29
```

---

## 🚀 Quickstart

### Basis-Initialisierung

```python
import asyncio
from toolboxv2 import get_app

async def main():
    # App initialisieren
    app = get_app()
    isaa = app.get_mod("isaa")
    
    # ISAA starten
    await isaa.init_isaa()
    
    # Agent erstellen
    agent = await isaa.get_agent("my_agent")
    
    # Task ausführen
    response = await agent.a_run("Erkläre mir das ISAA System")
    print(response)

asyncio.run(main())
```

### Agent mit Custom Tools

```python
from toolboxv2.mods.isaa.base.Agent.builder import FlowAgentBuilder

# Builder konfigurieren
builder = isaa.get_agent_builder(
    name="specialist",
    add_base_tools=True,
    with_dangerous_shell=False
)

# Custom Tools hinzufügen
def my_tool(input: str) -> str:
    return f"Processed: {input}"

builder.add_tool(
    my_tool,
    name="my_tool",
    description="Mein spezialisiertes Tool",
    category=["custom"]
)

# Agent registrieren und nutzen
await isaa.register_agent(builder)
agent = await isaa.get_agent("specialist")
```

---

## 📐 Architektur

### Verzeichnisstruktur

```
isaa/
├── 📁 base/                    # Core Agent Systeme
│   ├── 📁 Agent/              # FlowAgent, Builder, Chain
│   ├── 📁 IntelligentRateLimiter/  # Rate Limiting
│   ├── 📁 VectorStores/       # FAISS, Redis, Taichi
│   ├── 📁 audio_io/          # STT/TTS
│   ├── 📁 bench/             # Benchmarking
│   ├── 📁 chain/             # Chain Tools
│   ├── 📁 rl/                # Reinforcement Learning
│   ├── 📁 tbpocketflow/      # PocketFlow Integration
│   ├── AgentUtils.py         # Utils & Controller
│   ├── KnowledgeBase.py      # V1 Knowledge Base
│   ├── ai_semantic_memory.py # V2 Memory Wrapper
│   ├── hybrid_memory.py      # Hybrid Memory Store
│   └── MemoryKnowledgeActor.py  # Memory Actor
│
├── 📁 CodingAgent/            # Code-Generierung
│   ├── coder.py              # Core Coder
│   ├── coder_toolset.py      # Coding Tools
│   ├── live.py               # Live Coding
│   ├── project_developer.py  # Project Developer V3
│   └── 📁 project_dev_ui/    # Web UI
│
├── 📁 kernel/                 # Kernel System
│   ├── instace.py           # Kernel Core
│   ├── models.py            # Learning, Memory, Scheduler
│   ├── types.py             # Signal, States
│   └── 📁 kernelin/         # Interface Implementations
│       ├── kernelin_discord.py
│       ├── kernelin_telegram.py
│       ├── kernelin_whatsapp.py
│       └── 📁 st/           # Streamlit UI
│
├── 📁 extras/                 # Extras & Integrations
│   ├── discord_interface/   # Discord Bot
│   ├── web_helper/          # Web Search (SearXNG)
│   ├── toolkit/             # Google Calendar/Gmail
│   └── 📁 sub/              # Sub-Agent Adapter
│
├── 📁 docs/                   # Dokumentation
├── 📁 examples/               # Beispiele
├── module.py                # Main ISAA Module
├── chainUi.py               # Chain UI
├── hud.py                   # HUD Display
└── requirements.txt         # Dependencies
```

---

## 🧠 Memory System V2

### Überblick

Das **Memory V2** System ist ein produktionsreifes hybrides Speichersystem mit:

- 🎯 **SQLite + FAISS + FTS5** - Hybrid Storage für maximale Performance
- 🔍 **RRF Fusion** - Vector + BM25 + Relation Search kombiniert
- 🕸️ **Entity Graph** - Beziehungen zwischen Konzepten
- 📊 **Thread-Safe** - RLock + thread-local connections
- 💾 **Multi-Space** - Verschiedene Wissensbereiche
- 🔄 **Auto-Migration** - V1 → V2 automatisch

### Usage

```python
from toolboxv2.mods.isaa.base.ai_semantic_memory import AISemanticMemory

# Singleton Instanz
memory = AISemanticMemory(base_path="./data/memory")

# Space nutzen/erstellen
kb = memory.get("my_project")

# Daten hinzufügen
await kb.add_data(
    text="ISAA ist ein modulares Agent-System",
    concepts=["isaa", "agent", "modular"],
    metadata={"source": "docs"}
)

# Suchen
results = await kb.query(
    query_text="Was ist ISAA?",
    search_mode="auto"  # "vector", "bm25", "relation", "auto"
)

# Speichern
memory.save_memory("my_project", "./backup/")
```

### MemoryKnowledgeActor

Der **MemoryKnowledgeActor** ist ein Mini-Agent für Memory-Operationen:

```python
from toolboxv2.mods.isaa.base.MemoryKnowledgeActor import MemoryKnowledgeActor

mka = MemoryKnowledgeActor(memory=memory, space_name="my_project")

# Konzept-basierte Suche
results = await mka.search_by_concept("agent", k=5)

# Entity-Relationen
related = await mka.get_related_entities("ISAA", depth=2)

# Autonome Analyse
history = await mka.start_analysis_loop(
    user_task="Analysiere die ISAA Architektur",
    max_iterations=5
)
```

---

## 🔧 CodingAgent

### ProjectDeveloper V3

Der **ProjectDeveloper** ist eine Multi-File Code-Generierungs-Engine:

```python
from toolboxv2.mods.isaa.CodingAgent.project_developer import create_project_developer

developer = create_project_developer(
    agent=agent,
    workspace_path="./my_project",
    prefer_docker=True,
    verbose=True
)

try:
    success, files = await developer.execute(
        task="Erstelle eine REST API mit FastAPI",
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
finally:
    await developer.close()
```

### Features

| Feature | Beschreibung |
|---------|-------------|
| 🗂️ Multi-File | Generiert mehrere zusammenhängende Dateien |
| 🔍 Context Graph | Nutzt Abhängigkeiten aus DocsSystem |
| 🔬 Research Phase | API-Dokumentations-Recherche |
| 🐳 Safe Execution | Docker → RestrictedPython → Subprocess |
| 🔧 Auto-Fix | LSP + Runtime Feedback Loop |
| 📊 Token-Optimized | ContextBundle statt voller Dateien |

---

## 🔗 Chain System

### Chain erstellen

```python
# Einfache Chain
chain = isaa.create_chain(agent1, agent2, agent3)

# Mit Formatting
from toolboxv2.mods.isaa.base.Agent.chain import CF

class MyModel(BaseModel):
    summary: str
    keywords: list[str]

chain = isaa.create_chain(
    agent1,
    CF(MyModel),
    agent2
)

# Chain ausführen
result = await isaa.run_chain(
    chain=chain,
    query="Analysiere diesen Text",
    session_id="my_session"
)
```

### Chain aus Agent-Namen

```python
# Lazy Chain - resolved Agents on first run
chain = isaa.chain_from_agents("analyzer", "summarizer", "formatter")

result = await chain.run("Zu analysierender Text")
```

---

## 🌐 Kernel System

### Kernel Architecture

Das **Kernel System** orchestriert Agent-Interaktionen über verschiedene Plattformen:

```
┌─────────────────────────────────────────────┐
│                  Kernel                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────────┐  │
│  │SignalBus│ │Percept. │ │DecisionEng. │  │
│  └─────────┘ └─────────┘ └─────────────┘  │
└─────────────────┬───────────────────────────┘
                  │
          ┌───────▼───────┐
          │  IOManager    │
          └───────┬───────┘
     ┌────────────┼────────────┐
     ▼            ▼            ▼
  Discord     Telegram    WhatsApp
```

### Transports

```python
from toolboxv2.mods.isaa.kernel.kernelin.run_unified_kernels import main

# Startet alle Kernel-Interfaces
await main()
```

**Verfügbare Interfaces:**
- **Discord** - `kernelin_discord.py` - Full Discord Bot Support
- **Telegram** - `kernelin_telegram.py` - Telegram Bot
- **WhatsApp** - `kernelin_whatsapp.py` - WhatsApp Integration
- **Streamlit** - `st/kernel_streamlit_ui.py` - Web UI

---

## 💾 Agent Export/Import

### Agent speichern

```python
success, manifest = await isaa.save_agent(
    agent_name="my_agent",
    path="./my_agent.tar.gz",
    include_checkpoint=True,
    include_tools=True,
    notes="Mein spezieller Agent"
)

if success:
    print(f"✅ {manifest.tool_count} Tools exportiert")
```

**Archive Struktur:**
```
my_agent.tar.gz/
├── manifest.json          # Export Metadata
├── config.json            # AgentConfig
├── checkpoint.json        # Agent State (optional)
├── tools.dill            # Serialized Tools (optional)
└── tools_manifest.json   # Tool Info
```

### Agent laden

```python
agent, manifest, warnings = await isaa.load_agent(
    path="./my_agent.tar.gz",
    override_name="loaded_agent",
    load_tools=True,
    register=True
)

if agent:
    print(f"✅ Agent '{agent.amd.name}' geladen")
    
for warning in warnings:
    print(f"⚠️  {warning}")
```

### Agent Network

```python
# Mehrere Agenten exportieren
success, msg = await isaa.export_agent_network(
    agent_names=["agent1", "agent2", "agent3"],
    path="./network.tar.gz",
    entry_agent="agent1",
    include_checkpoints=True
)

# Network importieren
agents, warnings = await isaa.import_agent_network(
    path="./network.tar.gz",
    name_prefix="prod_",
    restore_bindings=True
)
```

---

## 🛠️ ISAA Tools API

### Hauptmethoden

| Methode | Beschreibung |
|---------|-------------|
| `init_isaa()` | ISAA initialisieren |
| `get_agent(name)` | Agent instanziieren |
| `get_agent_builder(name)` | Builder erstellen |
| `register_agent(builder)` | Agent registrieren |
| `run_agent(name, text)` | Agent ausführen |
| `mini_task_completion()` | Kleine Task absolvieren |
| `format_class()` | Strukturierte Ausgabe |

### Chain Methoden

| Methode | Beschreibung |
|---------|-------------|
| `create_chain(*agents)` | Chain erstellen |
| `run_chain(chain, query)` | Chain ausführen |
| `chain_from_agents(*names)` | Lazy Chain |

### Export/Import

| Methode | Beschreibung |
|---------|-------------|
| `save_agent(name, path)` | Agent exportieren |
| `load_agent(path, name)` | Agent importieren |
| `export_agent_network()` | Network export |
| `import_agent_network()` | Network import |

---

## 🔬 Extras

### Discord Interface

```python
from toolboxv2.mods.isaa.extras.discord_interface import DiscordInterface

discord = DiscordInterface(token="YOUR_TOKEN")
await discord.start()
```

### Web Helper

```python
from toolboxv2.mods.isaa.extras.web_helper.web_search import web_search

results = await web_search("ISAA Agent System")
```

### Google Toolkit

```python
from toolboxv2.mods.isaa.extras.toolkit.google_calendar_toolkit import (
    GoogleCalendarToolkit
)

toolkit = GoogleCalendarToolkit()
tools = toolkit.get_tools()
```

---

## 📊 Benchmarks

### Memory V2 Performance

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Add Latency (p95) | 2.3ms | <10ms | ✅ PASS |
| Query Latency (p95) | 15ms | <50ms | ✅ PASS |
| Save/Load (10k) | 1.2s | <5s | ✅ PASS |
| Concept Lookup | 0.1ms | <1ms | ✅ PASS |
| Dedup Check | O(1) | O(1) | ✅ PASS |
| Concurrency | 100 req/s | >50 req/s | ✅ PASS |

---

## 🧪 Testing

```bash
# Memory Tests
python -m pytest test_hybrid_memory.py -v

# CodingAgent Tests
python -m unittest test_project_developer.py

# Alle Tests
pytest tests/ -v --cov=isaa
```

---

## 📚 Weitere Dokumentation

- **[Memory V2 Production Readiness](base/MEMORY_V2_PRODUCTION_READINESS.md)**
- **[ProjectDeveloper V3 Guide](CodingAgent/readme_project_developer.md)**
- **[Kernel Refactoring Plan](kernel/Plan.md)**
- **[Feature Implementation Guide](FEATURE_IMPLEMENTATION_GUIDE.md)**

---

## 🤝 Contributing

1. Fork das Repository
2. Feature Branch erstellen (`git checkout -b feature/amazing`)
3. Änderungen committen (`git commit -m 'Add amazing feature'`)
4. Branch pushen (`git push origin feature/amazing`)
5. Pull Request öffnen

---

## 📄 Lizenz

MIT License - siehe [LICENSE](LICENSE)

---

**Entwickelt für das ToolBoxV2 Ökosystem** 🛠️