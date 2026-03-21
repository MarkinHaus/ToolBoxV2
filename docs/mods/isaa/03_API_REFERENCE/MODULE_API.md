# ISAA Module API - Tools Klasse

## Referenz: `toolboxv2.mods.isaa.module.Tools`

Die `Tools` Klasse ist der Haupteinstiegspunkt für das ISAA Modul.

---

## Initialisierung

```python
from toolboxv2 import get_app

app = get_app()
isaa = app.get_mod(\"isaa\")
```

**Parameter:**
- Keine direkten Parameter
- Liest Konfiguration aus `file_handler`
- Setzt Environment-Variablen für Modelle

---

## Agent Management

### `get_agent(name: str) -> FlowAgent`

Ruft einen Agenten ab oder erstellt ihn bei Bedarf.

```python
agent = await isaa.get_agent(\"my_agent\")
```

| Parameter | Typ | Beschreibung |
|-----------|-----|-------------|
| `name` | `str` | Name des Agenten |

**Returns:** `FlowAgent` Instanz

---

### `register_agent(builder: FlowAgentBuilder) -> FlowAgent`

Registriert einen neuen Agenten aus einem Builder.

```python
builder = isaa.get_agent_builder(name=\"specialist\")
builder.add_tool(my_tool, name=\"custom_tool\")
agent = await isaa.register_agent(builder)
```

| Parameter | Typ | Beschreibung |
|-----------|-----|-------------|
| `builder` | `FlowAgentBuilder` | Konfigurierter Builder |

**Returns:** `FlowAgent` Instanz

---

### `delete_agent(name: str)`

Löscht einen Agenten.

```python
await isaa.delete_agent(\"old_agent\")
```

---

### `get_all_agents() -> list[str]`

Liste aller registrierten Agenten.

```python
agents = await isaa.get_all_agents()
print(agents)  # [\"agent1\", \"agent2\", ...]
```

---

### `get_agent_builder(name: str, **kwargs) -> FlowAgentBuilder`

Erstellt einen neuen Builder für einen Agenten.

```python
builder = isaa.get_agent_builder(
    name=\"my_agent\",
    add_base_tools=True,
    with_dangerous_shell=False
)
```

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|-------------|
| `name` | `str` | - | Agent Name |
| `add_base_tools` | `bool` | `True` | Basis-Tools hinzufügen |
| `with_dangerous_shell` | `bool` | `False` | Shell-Tool erlauben |

---

## Chain Management

### `create_chain(*agents_or_components) -> Chain`

Erstellt eine Chain aus Agenten und/oder Komponenten.

```python
# Einfache Chain
chain = isaa.create_chain(agent1, agent2, agent3)

# Mit Format-Klasse
chain = isaa.create_chain(agent1, CF(OutputModel), agent2)

# Mit Bedingung
chain = isaa.create_chain(
    agent1,
    IS(\"status\", \"success\"),  # Wenn status == \"success\"
    agent2
)

# Mit Funktion
chain = isaa.create_chain(agent1, lambda x: x.upper(), agent2)
```

**Unterstützte Komponenten:**
- `FlowAgent` - Agent
- `CF(FormatClass)` - Output Formatierung
- `IS(key, value)` - Bedingung
- `Function(func)` - Python Funktion
- `Chain` - Verschachtelte Chain

---

### `run_chain(chain: Chain, query: str, session_id: str = \"default\", **kwargs) -> Any`

Führt eine Chain aus.

```python
result = await isaa.run_chain(
    chain,
    \"Analysiere diese Daten\",
    session_id=\"analysis_001\"
)
```

| Parameter | Typ | Beschreibung |
|-----------|-----|-------------|
| `chain` | `Chain` | Auszuführende Chain |
| `query` | `str` | Eingabe-Query |
| `session_id` | `str` | Session ID (default: \"default\") |

---

### `chain_from_agents(*agent_names: str) -> Chain`

Erstellt eine Chain aus Agent-Namen (lazy resolution).

```python
chain = isaa.chain_from_agents(\"researcher\", \"summarizer\", \"formatter\")
# Agenten werden erst bei Ausführung aufgelöst
```

---

## Export / Import

### `save_agent(name: str, path: str)`

Speichert einen Agenten als `.tar.gz` Archiv.

```python
await isaa.save_agent(\"my_agent\", \"./backup/my_agent.tar.gz\")
```

**Inhalt des Archives:**
- `manifest.json` - Metadaten
- `config.json` - Agent-Konfiguration
- `tools/` - Serialisierte Tools (via dill/cloudpickle)
- `memory/` - Agent Memory (optional)

---

### `load_agent(name: str, path: str) -> FlowAgent`

Lädt einen Agenten aus einem Archiv.

```python
agent = await isaa.load_agent(\"my_agent\", \"./backup/my_agent.tar.gz\")
```

**Hinweis:** Bei Deserialisierungsfehlern werden Hinweise zur manuellen Wiederherstellung gegeben.

---

### `export_agent_network(path: str)`

Exportiert alle Agenten als Netzwerk.

```python
await isaa.export_agent_network(\"./network_export.tar.gz\")
```

---

### `import_agent_network(path: str)`

Importiert ein Agenten-Netzwerk.

```python
await isaa.import_agent_network(\"./network_export.tar.gz\")
```

---

## Memory System

### `get_memory(space_name: str) -> AISemanticMemory`

Ruft den Memory-Space eines Agenten ab.

```python
memory = await isaa.get_memory(\"my_agent\")

# Daten hinzufügen
await memory.add_data(
    text=\"Wichtige Information\",
    concepts=[\"info\", \" wichtig\"],
    metadata={\"source\": \"user\"}
)

# Suchen
results = await memory.query(\"Wichtige?\")
```

---

## Jobs

### `job_add(name: str, query: str, trigger: TriggerConfig, agent_name: str)`

Erstellt einen geplanten Job.

```python
from toolboxv2.mods.isaa.extras.jobs import TriggerConfig

trigger = TriggerConfig(
    type=\"cron\",
    cron=\"0 9 * * *\"  # Täglich um 9 Uhr
)

await isaa.job_add(
    name=\"morning_report\",
    query=\"Erstelle Bericht\",
    trigger=trigger,
    agent_name=\"reporter\"
)
```

---

### `job_list() -> list[dict]`

Liste aller Jobs mit Status.

```python
jobs = await isaa.job_list()
for job in jobs:
    print(f\"{job['name']}: {job['status']}\")
```

---

### `job_remove(name: str)`

Löscht einen Job.

```python
await isaa.job_remove(\"morning_report\")
```

---

### `job_pause(name: str)` / `job_resume(name: str)`

Pausiert/Setzt einen Job fort.

```python
await isaa.job_pause(\"morning_report\")
# ...
await isaa.job_resume(\"morning_report\")
```

---

## Web Search

### `web_search(query: str, **dork_kwargs) -> str`

Web-Suche mit Google Dorks Support.

```python
# Einfache Suche
results = await isaa.web_search(\"Python async\")

# Mit Dorks
results = await isaa.web_search(
    \"filetype:pdf machine learning\",
    site=\"arxiv.org\"
)
```

**Dork Parameter:**
- `site` - Auf bestimmte Domain beschränken
- `filetype` - Dateityp filtern (pdf, doc, etc.)
- `inurl` - URL enthält String
- `intitle` - Titel enthält String
- `exclude` - Ausschluss

---

## Konfiguration

### `config` Property

```python
# Aktuelle Konfiguration
print(isaa.config)

# Modelle
isaa.config[\"FASTMODEL\"] = \"openrouter/anthropic/claude-3-haiku\"
```

**Standard-Felder:**
| Key | Default | Beschreibung |
|-----|---------|-------------|
| `FASTMODEL` | `ollama/llama3.1` | Schnelles Modell |
| `AUDIOMODEL` | `groq/whisper-large-v3-turbo` | Audio Modell |
| `BLITZMODEL` | `ollama/llama3.1` | Blitz Modell |
| `COMPLEXMODEL` | `ollama/llama3.1` | Komplexes Modell |
| `SUMMARYMODEL` | `ollama/llama3.1` | Zusammenfassungsmodell |
| `IMAGEMODEL` | `ollama/llama3.1` | Bildmodell |
| `DEFAULTMODELEMBEDDING` | `gemini/text-embedding-004` | Embedding Modell |
