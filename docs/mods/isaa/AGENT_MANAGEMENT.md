# Agent Management API

## Datei
`toolboxv2/mods/isaa/module.py:1679`

---

## Schnellstart

```python
# ISAA Modul holen
isaa = app.get_mod(\"isaa\")

# Agent holen (erstellt automatisch wenn nicht vorhanden)
agent = await isaa.get_agent(\"my_agent\")

# Agent ausführen
result = await agent.a_run(\"Erkläre Python\")
```

---

## `get_agent(agent_name, model_override=None)`

**Signatur:** `module.py:1679`

```python
async def get_agent(
    self, 
    agent_name=\"Normal\", 
    model_override: str | None = None
) -> FlowAgent
```

### Parameter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|---------------|
| `agent_name` | `str` | `\"Normal\"` | Name des Agents |
| `model_override` | `str` | `None` | Erzwingt anderes LLM-Modell |

### Return
`FlowAgent` Instance (Thread-safe, gecacht)

### Logik (Code-Referenz)

```python
# 1. Cache-Check (module.py:1686)
instance_key = f\"agent-instance-{agent_name}\"
if instance_key in self.config:
    return self.config[instance_key]  # Sofort zurück

# 2. Builder aus Registry holen (module.py:1697)
if agent_name in row_agent_builder_sto:
    builder = row_agent_builder_sto[agent_name]

# 3. Oder Config laden (module.py:1704)
elif agent_name in self.agent_data:
    config = AgentConfig(**self.agent_data[agent_name])
    builder = FlowAgentBuilder(config=config)

# 4. Oder Default-Builder (module.py:1714)
else:
    builder = self.get_agent_builder(agent_name)

# 5. Bauen + Cachen (module.py:1736)
agent_instance = await builder.build()
self.config[instance_key] = agent_instance
```

### Cache-Invalidation

```python
# Model-Override erzwingt Rebuild
if model_override and agent.amd.fast_llm_model != model_override:
    self.config.pop(instance_key, None)  # Cache löschen
```

### Beispiel: Verschiedene Modelle pro Request

```python
# Agent mit Default-Modell
agent_gpt4 = await isaa.get_agent(\"assistant\")

# Gleicher Agent, anderes Modell
agent_claude = await isaa.get_agent(\"assistant\", model_override=\"claude-3-5-sonnet\")
```

---

## `get_agent_builder(name, extra_tools, add_base_tools, with_dangerous_shell)`

**Signatur:** `module.py:1183`

```python
def get_agent_builder(
    self,
    name=\"self\",
    extra_tools=None,
    add_base_tools=True,
    with_dangerous_shell=False,
) -> FlowAgentBuilder
```

### Parameter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|---------------|
| `name` | `str` | `\"self\"` | Agent-Name für Config-Keys |
| `extra_tools` | `list` | `None` | Zusätzliche Tools |
| `add_base_tools` | `bool` | `True` | Basis-Tools hinzufügen |
| `with_dangerous_shell` | `bool` | `False` | Shell-Tool erlauben |

### Logik (Code-Referenz)

```python
# 1. Modelle aus Config holen (module.py:1197)
config = AgentConfig(
    name=name,
    fast_llm_model=self.config.get(
        f\"{name.upper()}MODEL\",  # Z.B. \"SELFMODEL\" für name=\"self\"
        self.config[\"FASTMODEL\"]  # Fallback
    ),
    complex_llm_model=self.config.get(
        f\"{name.upper()}MODEL\",
        self.config[\"COMPLEXMODEL\"]
    ),
    # System-Message aus Manifest oder Default
    system_message=... 
)
```

### Model-Resolution

| Config Key | Agent Name | Liest |
|-----------|------------|-------|
| `SELFMODEL` | `\"self\"` | `FASTMODEL` Fallback |
| `ASSISTANTMODEL` | `\"assistant\"` | `FASTMODEL` Fallback |
| `ANALYZERMODEL` | `\"analyzer\"` | `FASTMODEL` Fallback |

### Beispiel: Custom Agent mit Extra-Tools

```python
builder = isaa.get_agent_builder(
    name=\"coder\",
    extra_tools=[\"filesystem\", \"docker\"],
    add_base_tools=True,
    with_dangerous_shell=True
)
```

---

## `register_agent(builder)`

**Signatur:** `module.py:1668`

```python
async def register_agent(self, agent_builder: FlowAgentBuilder) -> None
```

### Logik

```python
# Builder in Registry speichern (module.py:1676)
row_agent_builder_sto[agent_builder.config.name] = agent_builder
print(f\"FlowAgent '{agent_name}' registered.\")
```

**Wichtig:** Registry ist APP-Lebensdauer, nicht Session!

### Beispiel

```python
builder = FlowAgentBuilder(config=my_config)
builder.with_tools([filesystem_tool])

await isaa.register_agent(builder)
# Ab jetzt: await isaa.get_agent(my_config.name) funktioniert
```

---

## `delete_agent(agent_name)`

**Signatur:** `module.py:1755`

```python
async def delete_agent(self, agent_name: str) -> bool
```

### Was wird gelöscht?

1. **RAM:** Instanz aus `config[\"agent-instance-{name}\"]`
2. **Registry:** Builder aus `row_agent_builder_sto`
3. **Disk:** `data/Agents/{name}/`
4. **Memory:** `data/Memory/{name}/`

### Logik (Code-Referenz)

```python
# 1. Instanz stoppen (module.py:1783)
if instance_key in self.config:
    agent = self.config.pop(instance_key)
    await agent.close()  # Graceful shutdown

# 2. Registry leeren (module.py:1794)
row_agent_builder_sto.pop(agent_name, None)

# 3. Config-Datei löschen (module.py:1810)
agent_config_path = Path(data_dir) / \"Agents\" / agent_name
if agent_config_path.exists():
    shutil.rmtree(agent_config_path)

# 4. Memory-Ordner löschen (module.py:1832)
memory_path = Path(data_dir) / \"Memory\" / agent_name
if memory_path.exists():
    shutil.rmtree(memory_path)
```

### Beispiel

```python
success = await isaa.delete_agent(\"temp_agent\")
if success:
    print(\"Agent vollständig entfernt\")
```

---

## `get_agents()`

**Signatur:** `module.py:482`

```python
async def get_agents() -> list[FlowAgent]
```

Gibt alle registrierten Agent-Instanzen zurück.

```python
all_agents = await isaa.get_agents()
for agent in all_agents:
    print(f\"- {agent.amd.name}\")
```

---

## FlowAgentBuilder Deep-Dive

**Datei:** `toolboxv2/mods/isaa/base/Agent/builder.py:171`

### Konfiguration (AgentConfig)

```python
class AgentConfig(BaseModel):
    name: str
    fast_llm_model: str
    complex_llm_model: str
    system_message: str
    
    # Optional
    enable_lsp: bool = False
    enable_docker: bool = False
    vfs_max_window_lines: int = 500
    handler_path_or_dict: str | dict = None
```

### Builder-Methoden

```python
builder = FlowAgentBuilder(config=AgentConfig(...))

# Chain Style
agent = (
    builder
    .with_name(\"my_agent\")
    .with_models(fast=\"gpt-4o-mini\", complex=\"gpt-4o\")
    .with_tools([filesystem_tool])
    .with_persona(custom_persona)
    .build()
)
```

---

## Praxis-Beispiele

### 1. Singleton-Pattern (Empfohlen)

```python
# Einmal holen, mehrfach nutzen
isaa = app.get_mod(\"isaa\")

async def handle_request(user_id: str):
    agent = await isaa.get_agent(f\"user_{user_id}\")
    return await agent.a_run(query)
```

### 2. Lazy Initialization

```python
# Agent wird erst beim ersten use gebaut
agent = await isaa.get_agent(\"heavy_agent\")  # Kein LLM-Aufruf hier
result = await agent.a_run(query)  # Erst hier
```

### 3. Model-A/B Testing

```python
# Version A
agent_a = await isaa.get_agent(\"task\", model_override=\"gpt-4o\")

# Version B  
agent_b = await isaa.get_agent(\"task\", model_override=\"claude-3-5-sonnet\")

# Vergleich
result_a = await agent_a.a_run(query)
result_b = await agent_b.a_run(query)
```

---

## Nächste Schritte

👉 **[Chain System](./CHAIN_SYSTEM.md)** - Mehrere Agents verketten