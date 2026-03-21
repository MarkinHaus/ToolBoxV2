# Chain System API

## Datei
`toolboxv2/mods/isaa/module.py:460`

---

## Schnellstart

```python
# Chain erstellen
chain = isaa.chain_from_agents(\"analyzer\", \"summarizer\", \"formatter\")

# Chain ausführen - Agents werden lazy geladen
result = await chain.a_run(\"Analysiere diese Daten...\")
```

---

## `chain_from_agents(*agent_names)`

**Signatur:** `module.py:460`

```python
def chain_from_agents(self, *agent_names: str) -> Chain
```

### Parameter

| Parameter | Typ | Beschreibung |
|-----------|-----|---------------|
| `agent_names` | `str...` | Namen der Agents in Ausführungsreihenfolge |

### Return
`LazyAgentChain` - Ketten-Klasse

### Logik (Code-Referenz)

```python
# module.py:474
async def get_agents():
    agents = []
    for name in agent_names:
        agent = await self.get_agent(name)  # Lazy Loading!
        agents.append(agent)
    return agents

# Lazy Wrapper (module.py:483)
class LazyAgentChain(Chain):
    async def a_run(self, query, **kwargs):
        if not self._resolved:
            for name in self._agent_names:
                agent = await self._isaa_ref.get_agent(name)
                self.tasks.append(agent)  # Erst hier: Agents laden
            self._resolved = True
        return await super().a_run(query, **kwargs)
```

---

## Lazy Loading Mechanismus

### Warum Lazy?

**Problem:** Wenn du 10 Agents in einer Chain definierst, aber nur 2 brauchst:

```python
# Ohne Lazy: 10 Agent-Instanzen werden gebaut
chain = isaa.chain_from_agents(\"a1\", \"a2\", \"a3\", ..., \"a10\")
# → 10x FlowAgentBuilder.build() → 10x LLM-Init

# Mit Lazy: Erst wenn a_run() aufgerufen wird
chain = isaa.chain_from_agents(\"a1\", \"a2\")
result = await chain.a_run(query)
# → Nur die 2 Agents werden initialisiert
```

### Vorteile

1. **Schneller Import** - Keine Agent-Initialisierung beim Import
2. **Bedingte Nutzung** - Nur Agents laden die wirklich gebraucht werden
3. **Spezialisierte Agents** - Verschiedene Chains für verschiedene Tasks

---

## Chain-Klasse

**Datei:** `toolboxv2/mods/isaa/base/Agent/chain.py`

### Architektur

```
Chain
├── tasks: list[FlowAgent]  # Agent-Instanzen
├── results: list[str]      # Zwischenergebnisse
│
├── a_run(query)           # Pipeline starten
├── add(agent)             # Agent hinzufügen
├── clear()                # Reset
└── get_results()          # Zwischenergebnisse holen
```

### a_run() Logik

```python
async def a_run(self, query, **kwargs):
    current_input = query
    
    for task in self.tasks:
        # Output des vorherigen Agents → Input des nächsten
        result = await task.a_run(current_input, **kwargs)
        self.results.append(result)
        current_input = result  # Chaining!
    
    return current_input  # Finale Ausgabe
```

---

## Praxis-Beispiele

### 1. Einfache Pipeline

```python
chain = isaa.chain_from_agents(
    \"researcher\",    # Sucht Informationen
    \"analyzer\",      # Analysiert Ergebnisse  
    \"writer\"         # Schreibt finale Antwort
)

result = await chain.a_run(\"Was sind die neuesten AI-Entwicklungen?\")
```

**Datenfluss:**
```
\"Was sind...\" 
    → researcher.a_run() 
    → \"Research Ergebnis...\" 
    → analyzer.a_run() 
    → \"Analyse:...\" 
    → writer.a_run() 
    → Finale Antwort
```

### 2. Parallele Verarbeitung

```python
# Nicht Teil des Chain-Systems, aber verwandt
async def parallel_run(query):
    # 3 Agents parallel starten
    results = await asyncio.gather(
        isaa.get_agent(\"variant_a\").a_run(query),
        isaa.get_agent(\"variant_b\").a_run(query),
        isaa.get_agent(\"variant_c\").a_run(query),
    )
    return results
```

### 3. Conditional Chain

```python
class ConditionalChain(Chain):
    async def a_run(self, query, **kwargs):
        classifier = await self._isaa_ref.get_agent(\"classifier\")
        classification = await classifier.a_run(query)
        
        if \"technical\" in classification:
            return await self.tasks[0].a_run(query)  # Tech-Agent
        else:
            return await self.tasks[1].a_run(query)  # Creative-Agent
```

---

## Fehlerbehandlung

### Agent nicht gefunden

```python
try:
    chain = isaa.chain_from_agents(\"unknown_agent\")
    await chain.a_run(\"test\")
except Exception as e:
    print(f\"Agent fehlt: {e}\")
    # Lösung: Agent erstellen
    builder = isaa.get_agent_builder(\"unknown_agent\")
    await isaa.register_agent(builder)
```

### Agent Crashed

```python
async def robust_run(chain, query):
    for i, agent in enumerate(chain.tasks):
        try:
            result = await agent.a_run(input)
            input = result
        except Exception as e:
            print(f\"Agent {i} failed: {e}\")
            # Fallback oder Error-Handling
            return f\"Pipeline failed at step {i}\"
    return input
```

---

## Vor- und Nachteile

### ✅ Vorteile

| Aspekt | Begründung |
|--------|------------|
| **Separation of Concerns** | Jeder Agent macht genau eine Sache |
| **Wiederverwendbarkeit** | Agent in verschiedenen Chains |
| **Debugging** | Step-by-Step Output sichtbar |
| **Skalierung** | Einzelne Agents können ersetzt werden |

### ⚠️ Nachteile

| Aspekt | Risiko |
|--------|--------|
| **Latenz** | Jeder Agent = zusätzlicher LLM-Call |
| **Context-Verlust** | Zwischenergebnisse können Kontext verlieren |
| **Komplexität** | Viele kleine Agents = komplexes System |
| **Fehler-Kaskaden** | Ein Agent-Fehler = Pipeline-Stop |

### 💡 Wann Chain nutzen?

```
CHAIN JA:
├── Komplexe Tasks mit klaren Phasen
├── Verschiedene Expertisen nötig
├── Debugging wichtig
└── Wiederverwendbare Bausteine

CHAIN NEIN:
├── Einfache One-Shot Tasks
├── Schnelle Antwort nötig
└── Single-Purpose Agent reicht
```

---

## Erweiterungen (Zukunft)

| Feature | Status | Beschreibung |
|---------|--------|---------------|
| `parallel` Parameter | TODO | Agents parallel statt sequentiell |
| `timeout` pro Agent | TODO | Timeout pro Chain-Step |
| `retry` Konfiguration | TODO | Retry-Strategie pro Agent |
| `branch` / `if` | TODO | Conditional Routing |

---

## Nächste Schritte

👉 **[Import/Export](./IMPORT_EXPORT.md)** - Agents speichern und laden