# ISAA Agent Architektur

## FlowAgent

Der FlowAgent (`base/Agent/flow_agent.py`) ist die Runtime-Instanz eines Agenten.

### Kern-Architektur

```
User Input
    │
    ▼
┌─────────────┐
│   Session   │ ← Session Manager
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Execution  │ ← Execution Engine
│   Engine    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Tools    │ ← Tool Manager
└─────────────┘
       │
       ▼
    Output
```

### Execution Engine

Die Execution Engine (`execution_engine.py`) steuert die Agent-Ausführung.

### Builder Pattern

```python
builder = FlowAgentBuilder(name=\"agent\")
builder.add_tool(search_tool)
builder.add_tool(write_tool)
builder.set_model(\"gpt-4\")
agent = builder.build()
```

## Session Management

- **Session Manager** - `session_manager.py`
- **Agent Session** - `agent_session_v2.py`
- **Live State** - `agent_live_state.py`

## Memory Integration

```
Agent ◄──► AISemanticMemory
         │
         └──► VectorStores
         │
         └──► HybridMemory
```
