# ISAA Quickstart Guide

## Installation & Setup

```bash
pip install toolboxv2
```

## Erster Agent

```python
from toolboxv2 import Application

app = Application()
isaa = app.get_mod(\"isaa\")

# Agent erstellen
agent = await isaa.get_agent(\"my_first_agent\")
```

## Agent ausführen

```python
# Synchrone Ausführung
result = agent.run(\"Erkläre Python\")

# Asynchrone Ausführung
result = await agent.a_run(\"Erkläre Python\")
```

## Chain erstellen

```python
chain = isaa.chain_from_agents(\"researcher\", \"summarizer\")
result = await chain.a_run(\"Dein Query hier\")
```

## Nächste Schritte

- [Agent Management](../AGENT_MANAGEMENT.md) - Agenten konfigurieren
- [Chain System](../CHAIN_SYSTEM.md) - Pipelines bauen
- [API Referenz](../03_API_REFERENCE/MODULE_API.md) - Alle Methoden
