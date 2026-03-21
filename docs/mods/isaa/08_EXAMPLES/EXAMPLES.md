# ISAA Code Examples

## Agent Erstellung

```python
from toolboxv2 import Application

app = Application()
isaa = app.get_mod(\"isaa\")

# Einfacher Agent
agent = await isaa.get_agent(\"helper\")
result = await agent.a_run(\"Hilf mir bei Python\")

# Custom Agent
builder = isaa.get_agent_builder(
    name=\"coder\",
    add_base_tools=True,
    with_dangerous_shell=True
)
builder.add_system_prompt(\"Du bist ein Python Experte\")
agent = await isaa.register_agent(builder)
```

## Chain Examples

```python
# Recherche → Analyse → Ausgabe
chain = isaa.chain_from_agents(
    \"researcher\",
    \"analyzer\",
    \"formatter\"
)
result = await chain.a_run(\"Neueste AI News\")

# Mit Bedingung
from isaa_mod.base.Agent.chain import IS

chain = isaa.create_chain(
    agent1,
    IS(\"confidence\", \"high\"),
    agent2
)
```

## Memory Usage

```python
memory = await isaa.get_memory(\"my_agent\")

# Speichern
await memory.add_data(
    text=\"Wichtige Info\",
    concepts=[\"info\", \"wichtig\"],
    metadata={\"source\": \"user\"}
)

# Abrufen
results = await memory.query(\"Wichtige?\")
```

## Jobs Scheduling

```python
from isaa_mod.extras.jobs import TriggerConfig

trigger = TriggerConfig(
    type=\"cron\",
    cron=\"0 */6 * * *\"  # Alle 6 Stunden
)

await isaa.job_add(
    name=\"status_check\",
    query=\"System Status prüfen\",
    trigger=trigger,
    agent_name=\"monitor\"
)
```
