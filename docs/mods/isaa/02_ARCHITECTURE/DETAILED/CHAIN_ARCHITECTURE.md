# ISAA Chain Architektur

## Übersicht

Chains ermöglichen die Verkettung von Agenten und Komponenten.

## Chain Typen

### Sequentielle Chain
```python
chain = isaa.create_chain(agent1, agent2, agent3)
# A → B → C → D
```

### Parallele Chain
```python
chain = isaa.create_chain(
    agent1,
    CF(OutputModel),  # Parallel
    [agent2, agent3]  # Beide parallel
)
```

### Bedingte Chain
```python
chain = isaa.create_chain(
    agent1,
    IS(\"status\", \"success\"),  # Wenn Bedingung erfüllt
    agent2,
    ELSE(agent3)                 # Sonst
)
```

## Chain Components

| Component | Beschreibung |
|-----------|-------------|
| `FlowAgent` | Agent-Ausführung |
| `CF(cls)` | Output Formatierung |
| `IS(key, val)` | Bedingung |
| `Function(func)` | Python Funktion |
| `Chain` | Verschachtelte Chain |

## Lazy Loading

```python
# Namen werden erst bei Ausführung aufgelöst
chain = isaa.chain_from_agents(
    \"researcher\",
    \"analyzer\",
    \"formatter\"
)
```
