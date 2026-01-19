Überabrite die docs für den agent neune docs zum executoer # ExecutionEngine V3 - Dokumentation

## Überblick

ExecutionEngine V3 ist das Herzstück der FlowAgent-Orchestrierung. Es wurde speziell für **kleine/günstige LLMs** optimiert und bietet:

- **Dynamic Tool Loading** mit max. 5 gleichzeitigen Tools
- **Working/Permanent History Separation** für Token-Effizienz
- **Skills System** für gelerntes Verhalten
- **Sub-Agent System** für parallele Aufgaben
- **Loop Detection** für autonome Sicherheit

---

## Architektur

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION ENGINE V3                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  SkillsManager  │    │   ToolManager   │    │  SubAgentManager │         │
│  │  (skills.py)    │    │   (existing)    │    │  (sub_agent.py)  │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         ExecutionContext                              │  │
│  │  ├─ run_id                    ├─ working_history                     │  │
│  │  ├─ matched_skills            ├─ dynamic_tools (max 5)               │  │
│  │  ├─ tool_relevance_cache      ├─ auto_focus: AutoFocusTracker        │  │
│  │  └─ current_iteration         └─ loop_detector: LoopDetector         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         EXECUTION LOOP                                │  │
│  │                                                                       │  │
│  │  Query → Skill Match → Tool Relevance → Preload → LLM Loop →         │  │
│  │  final_answer → Compression → Commit → Learn                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dateien

| Datei | Zeilen | Beschreibung |
|-------|--------|--------------|
| `execution_engine.py` | ~1360 | Main Engine, Context, Compression, Tool Management |
| `skills.py` | ~1080 | Skill, SkillsManager, ToolGroup, Predefined Skills |
| `sub_agent.py` | ~715 | SubAgentManager, RestrictedVFSWrapper, Tools |
| `test_execution_engine_v3.py` | ~990 | 39 Unit Tests |

---

## Installation

```python
# In deinem FlowAgent Projekt:
from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine
from toolboxv2.mods.isaa.base.Agent.skills import SkillsManager, Skill
from toolboxv2.mods.isaa.base.Agent.sub_agent import SubAgentManager
```

---

## Schnellstart

```python
from execution_engine import ExecutionEngine

# Erstelle Engine
agent = YourFlowAgent()  # Muss tool_manager, session_manager haben
engine = ExecutionEngine(agent)

# Führe Query aus
result = await engine.execute(
    query="Recherchiere X, Y, Z parallel und vergleiche dann",
    session_id="session_123",
    max_iterations=15
)

print(result)
```

---

## Komponenten

### 1. ExecutionEngine

**Initialisierung:**
```python
engine = ExecutionEngine(
    agent,                          # FlowAgent Instanz
    human_online=False,             # Human überwacht?
    callback=None,                  # Progress Callback
    is_sub_agent=False,             # Ist dies ein Sub-Agent?
    sub_agent_output_dir=None,      # VFS Write Restriction
    sub_agent_budget=5000           # Token Budget für Sub-Agents
)
```

**Hauptmethode:**
```python
result = await engine.execute(
    query="User Query",
    session_id="session_id",
    max_iterations=15
)
```

---

### 2. SkillsManager

**Predefined Skills (12):**

| Skill | Triggers | Beschreibung |
|-------|----------|--------------|
| `user_preference_save` | merke, speicher, remember | Präferenzen speichern |
| `user_preference_recall` | was mag ich, erinnerst du | Präferenzen abrufen |
| `user_context_update` | ich bin jetzt, update | Kontext aktualisieren |
| `habits_tracking` | gewohnheit, habit, täglich | Habits tracken |
| `habits_analysis` | analyse, statistik, streak | Habits analysieren |
| `habits_setup` | neue gewohnheit, erstellen | Habits einrichten |
| `multi_step_task` | mehrere schritte, projekt | Komplexe Aufgaben planen |
| `clarification_needed` | unklar, was meinst du | Klarstellung anfordern |
| `error_recovery` | fehler, funktioniert nicht | Fehlerbehandlung |
| `vfs_info_persist` | merke dir, wichtig, notiz | Info im VFS speichern |
| `vfs_task_planning` | plane, projekt, workflow | Pläne im VFS |
| `vfs_knowledge_base` | wissen, docs, sammlung | Knowledge Base in /info/ |
| `parallel_subtasks` | parallel, gleichzeitig | Sub-Agent Parallelisierung |

**Skill Matching (Hybrid):**
```python
# Keyword first (fast)
matches = manager.match_skills("Merke dir meine Präferenz")
# → user_preference_save

# Embedding fallback (wenn Memory vorhanden)
matches = await manager.match_skills_async("Speichere das für später")
```

**Skill Learning:**
```python
# Automatisch nach erfolgreichem Run
await manager.learn_from_run(
    query="Erstelle Flask API",
    tools_called=["vfs_write", "vfs_read", "http_request"],
    final_answer="Ich habe eine Flask API erstellt...",
    success=True,
    llm_completion_func=agent.a_run_llm_completion
)
# → Neuer Skill mit confidence=0.3
# → Wird aktiv wenn confidence >= 0.6 (nach ~3 erfolgreichen Verwendungen)
```

---

### 3. Sub-Agent System

**Konzept:**
- Main Agent kann bis zu N Sub-Agents parallel spawnen
- Sub-Agents können **NICHT** weitere Sub-Agents spawnen (Max Depth = 1)
- Sub-Agents können **NUR** in ihren `output_dir` schreiben
- Sub-Agents können das **gesamte VFS lesen**

**Tools:**

```python
# spawn_sub_agent - Starte Sub-Agent
spawn_sub_agent(
    task="Recherchiere Thema X, schreibe Zusammenfassung",
    output_dir="research_x",    # → /sub/research_x/
    wait=False,                 # True = blockierend, False = async
    budget=5000                 # Token Budget
)
# → Returns: "sub_abc123" (ID)

# wait_for - Warte auf Sub-Agents
wait_for(
    sub_agent_ids=["sub_abc123", "sub_def456"],
    timeout=300
)
# → Returns: {id: SubAgentResult, ...}
```

**Beispiel-Flow:**
```
User: "Vergleiche A, B und C"

Main Agent:
1. spawn_sub_agent(task="Recherchiere A", output_dir="research_a", wait=False) → sub_1
2. spawn_sub_agent(task="Recherchiere B", output_dir="research_b", wait=False) → sub_2
3. spawn_sub_agent(task="Recherchiere C", output_dir="research_c", wait=False) → sub_3
4. wait_for([sub_1, sub_2, sub_3])

   [Sub-Agents laufen parallel]
   - sub_1 schreibt /sub/research_a/result.md
   - sub_2 schreibt /sub/research_b/result.md
   - sub_3 schreibt /sub/research_c/result.md

5. vfs_read("/sub/research_a/result.md")
6. vfs_read("/sub/research_b/result.md")
7. vfs_read("/sub/research_c/result.md")
8. Vergleiche und final_answer()
```

---

### 4. Tool Management

**Static Tools (immer verfügbar, zählen nicht zum Limit):**
- `think` - Scratchpad für Reasoning
- `final_answer` - Aufgabe abschließen
- `list_tools` - Verfügbare Tools anzeigen
- `load_tools` - Tools laden
- `vfs_read`, `vfs_write`, `vfs_list`, `vfs_navigate`, `vfs_control`
- `spawn_sub_agent`, `wait_for` (nur Main Agent)

**Dynamic Tools (max 5 gleichzeitig):**
- Werden bei Bedarf geladen: `load_tools(["discord_send", "http_request"])`
- Auto-Unload: Niedrigste Relevanz wird entfernt wenn Limit erreicht
- Relevanz wird einmalig bei Query-Start berechnet (Keyword Overlap)

---

### 5. History Compression

**Zwei Trigger:**

1. **TRIGGER 1: final_answer**
   - Komprimiert gesamte `working_history` zu Summary
   - Summary + User + Assistant → `permanent_history`
   - Summary wird auch in RAG gespeichert

2. **TRIGGER 2: load_tools + Kategorie-Wechsel + len > 3**
   - Partielle Kompression der ältesten Einträge
   - Behält letzte 3 Messages
   - Verhindert Context Overflow während langer Runs

**Summary Format:**
```
ABGESCHLOSSENE AKTIONEN:
• Erstellt: 2 Datei(en)
• Gelesen: 1 Datei(en)
• Tools genutzt: vfs_write, vfs_read, http_request
• Gesamt Tool-Calls: 5
```

---

### 6. AutoFocusTracker

Verhindert "Ich habe vergessen was ich getan habe" bei kleinen Modellen.

```python
# Nach jedem Tool Call:
tracker.record("vfs_write", {"path": "/app.py"}, "Created file")

# Wird vor User Query injiziert:
"LETZTE AKTIONEN (zur Erinnerung):
- ✏️ Wrote /app.py (45 lines)
- 📖 Read /requirements.txt (50 chars)
- 🔍 Searched, found 3 results"
```

---

### 7. LoopDetector

Erkennt wenn Agent stecken bleibt:

**Erkannte Patterns:**
1. **Exact Repeat:** `tool1(args) → tool1(args) → tool1(args)` (3x gleich)
2. **Ping-Pong:** `A → B → A → B`

**Intervention:**
```
⚠️ LOOP ERKANNT: Du hast 'vfs_write' mehrfach mit gleichen Argumenten aufgerufen.

OPTIONEN:
1. Falls du blockiert bist → Nutze final_answer um das Problem zu erklären
2. Falls du andere Daten brauchst → Ändere deinen Ansatz
3. Falls du auf User-Input wartest → Sage das ehrlich in final_answer
```

---

## Konfiguration

### ExecutionContext Limits

```python
@dataclass
class ExecutionContext:
    max_dynamic_tools: int = 5      # Max gleichzeitig geladene Tools
    # ...

class AutoFocusTracker:
    max_actions: int = 5            # Max getrackte Aktionen
    max_chars: int = 500            # Max Zeichen im Focus Message

class LoopDetector:
    max_repeats: int = 3            # Ab wann Loop erkannt wird
```

### Sub-Agent Limits

```python
class SubAgentConfig:
    max_tokens: int = 5000          # Token Budget
    max_iterations: int = 10        # Max Iterationen
    timeout_seconds: int = 300      # Timeout
```

---

## Checkpoints

**Speichern:**
```python
# Skills
checkpoint = engine.skills_manager.to_checkpoint()
# → {"skills": {...}, "tool_groups": {...}}

# Kann in AgentSessionV2.to_checkpoint() integriert werden
```

**Laden:**
```python
engine.skills_manager.from_checkpoint(checkpoint)
```

---

## Fehlerbehandlung

### Max Iterations Reached

Agent generiert ehrliche Antwort:
```
Ich konnte die Aufgabe leider nicht vollständig abschließen.

ABGESCHLOSSENE AKTIONEN:
• Erstellt: 1 Datei(en)
• Fehler: 2

**Warum?**
Die Aufgabe war möglicherweise zu komplex oder ich bin in einer Schleife gelandet.

**Mögliche nächste Schritte:**
1. Die Aufgabe in kleinere Teile aufteilen
2. Mir mehr Kontext oder Details geben
3. Eine spezifischere Frage stellen
```

### Sub-Agent Timeout

```python
SubAgentResult(
    success=False,
    status=SubAgentStatus.TIMEOUT,
    error="Timeout after 300 seconds"
)
```

---

## Tests

```bash
# Alle 39 Tests ausführen
python test_execution_engine_v3.py -v

# Einzelne Test-Klasse
python -m unittest test_execution_engine_v3.TestSkillsManager -v
```

**Test Coverage:**
- `TestSkill` - Skill Dataclass
- `TestSkillsManager` - Matching, Learning, Checkpoints
- `TestToolGroup` - Tool Grouping
- `TestAutoFocusTracker` - Focus Tracking
- `TestLoopDetector` - Loop Detection
- `TestHistoryCompressor` - Compression
- `TestExecutionContext` - Context Management
- `TestRestrictedVFSWrapper` - VFS Restriction
- `TestSubAgentManager` - Sub-Agent Spawning
- `TestExecutionEngineIntegration` - Full Integration
- `TestAsyncOperations` - Async Methods

---

## Migration von RuleSet

`RuleSet` → `SkillsManager`:

| Alt (RuleSet) | Neu (Skills) |
|---------------|--------------|
| `SituationRule` | `Skill` |
| `RuleResult` | Skill Matching Score |
| `LearnedPattern` | Learned Skill (confidence) |
| `ToolGroup` | ✅ Behalten |
| `auto_group_tools_by_name_pattern` | ✅ Behalten |

---

## Best Practices

1. **Klare Task-Beschreibungen für Sub-Agents**
   - Sub-Agents können NICHT nachfragen
   - Alle Infos müssen im `task` Parameter sein

2. **VFS für Persistenz nutzen**
   - `/info/` für Knowledge Base
   - `/plan.md` für Task Planning
   - `/sub/*/result.md` für Sub-Agent Results

3. **Skill Learning beobachten**
   - Neue Skills starten mit confidence=0.3
   - Erst aktiv bei confidence >= 0.6
   - Bei Fehlern sinkt confidence

4. **Loop Detection beachten**
   - Agent wird bei Loop gewarnt
   - Danach sollte er Ansatz ändern oder aufgeben

---

## Changelog

### V3.0 (2025-01-19)
- Initial Release
- Dynamic Tool Loading mit Keyword Relevance
- Skills System mit Hybrid Matching
- Sub-Agent System für Parallelisierung
- History Compression (Rule-based)
- AutoFocus und LoopDetector
- 12 Predefined Skills
- 39 Unit Tests
