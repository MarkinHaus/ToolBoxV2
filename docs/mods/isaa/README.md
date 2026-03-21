# ISAA Modul Dokumentation

## Übersicht

Das ISAA-Modul (`toolboxv2.mods.isaa`) ist das Kern-System für Agent-Management in ToolBoxV2.

**Hauptdatei:** `toolboxv2/mods/isaa/module.py`

---

## Modul-Zugriff

```python
from toolboxv2 import Application

app = Application()
isaa = app.get_mod(\"isaa\")
```

**Wichtig:** Bei Tippfehler (`\"isaaa\"` statt `\"isaa\"`) wird KEIN Fehler geworfen!

---

## Kern-Komponenten

| Komponente | Datei | Beschreibung |
|------------|-------|---------------|
| `ISAA` | `module.py` | Hauptklasse - Agent Registry, Chains, Import/Export |
| `FlowAgent` | `base/Agent/flow_agent.py` | Runtime-Instanz eines Agenten |
| `FlowAgentBuilder` | `base/Agent/builder.py` | Builder-Pattern für Agent-Konfiguration |

---

## 📚 Komplette Dokumentation

### 🚀 Erste Schritte
- **[Quickstart](./01_GETTING_STARTED/QUICKSTART.md)** - 5 Minuten zum ersten Agent
- **[Beispiele](./08_EXAMPLES/EXAMPLES.md)** - Code-Beispiele

### 🏗️ Architektur
- **[Übersicht](./02_ARCHITECTURE/OVERVIEW.md)** - Gesamtarchitektur
- **[Agent Architektur](./02_ARCHITECTURE/DETAILED/AGENT_ARCHITECTURE.md)** - FlowAgent Details
- **[Memory Architektur](./02_ARCHITECTURE/DETAILED/MEMORY_ARCHITECTURE.md)** - Memory System
- **[Chain Architektur](./02_ARCHITECTURE/DETAILED/CHAIN_ARCHITECTURE.md)** - Chain Pipeline

### 🔧 API Referenz
- **[Module API](./03_API_REFERENCE/MODULE_API.md)** - Komplette API-Dokumentation
- **[Agent Management](./AGENT_MANAGEMENT.md)** - get_agent, register_agent, etc.
- **[Chain System](./CHAIN_SYSTEM.md)** - Pipeline-APIs
- **[Import/Export](./IMPORT_EXPORT.md)** - save_agent, load_agent
- **[format_class](./MINITASK_FORMATCLASS.md)** - Strukturierte Ausgabe

### 💻 Coding Agent
- **[Coding Agent](./04_CODING_AGENT/DETAILED/CODING_AGENT.md)** - Code-Generierung

### ⚙️ Kernel & Interfaces
- **[Kernel System](./05_KERNEL/KERNEL_SYSTEM.md)** - Discord/Telegram Integration

### 🔌 Extras
- **[Discord Interface](./06_EXTRAS/DISCORD_INTERFACE.md)** - Discord Bot
- **[Web Helper](./06_EXTRAS/WEB_HELPER.md)** - Web-Suche
- **[Zen System](./06_EXTRAS/ZEN_SYSTEM.md)** - Meta-Learning

### ⏰ Jobs & Scheduling
- **[Jobs System](./07_JOBS/JOBS_SYSTEM.md)** - Geplante Agent-Ausführungen

### 🧠 Deep Analysis
- **[RL System](./09_DEEP_ANALYSIS/RL_SYSTEM.md)** - Reinforcement Learning

### 📖 Zusätzliche Docs
- **[ISAAMODUL_INIT.md](./ISAAMODUL_INIT.md)** - Modul-Lifecycle
- **[PHASE1_VORANALYSE.md](./PHASE1_VORANALYSE.md)** - Planung

---

## ⚠️ Entfernte Features (V1 → V2)

| Feature | Status | Alternative |
|---------|--------|-------------|
| `minitask()` | ❌ Entfernt | `format_class()` |
| `ToolsInterface` | ❌ Entfernt | Native ToolManager |

---

## Schnellstart

```python
from toolboxv2 import Application

app = Application()
isaa = app.get_mod(\"isaa\")

# 1. Agent holen (erstellt automatisch wenn nicht vorhanden)
agent = await isaa.get_agent(\"my_agent\")

# 2. Agent ausführen
result = await agent.a_run(\"Erkläre Python\")

# 3. Chain erstellen
chain = isaa.chain_from_agents(\"analyzer\", \"writer\")
result = await chain.a_run(\"Analysiere das...\")

# 4. Structurierte Ausgabe
from pydantic import BaseModel

class MyOutput(BaseModel):
    summary: str
    tags: list[str]

result = await isaa.format_class(
    format_schema=MyOutput,
    task=\"Analysiere: ...\",
    agent_name=\"TaskCompletion\"
)
```

---

## Version

Aktuelle Version: **0.4.0**

Änderungen in V2:
- Native Chain-Unterstützung
- Agent Export/Import (.tar.gz mit dill)
- Kernel-System für Discord/Telegram
- Zen Meta-Learning
- Jobs Scheduling
- RL Training System
