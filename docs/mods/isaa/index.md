# ISAA — Agent Framework

> **ISAA** = Intelligent System for Autonomous Agents
> **File:** `toolboxv2/mods/isaa/`

Core subsystem for creating, managing, and executing AI agents with tool-calling, session persistence, checkpointing, and hybrid memory.

## Architecture

```
┌──────────────────────────────────────────────┐
│                    ISAA Mod                    │
│  on_start → init_isaa → register_agents      │
└──────┬───────┬────────┬──────────┬───────────┘
       │       │        │          │
       ▼       ▼        ▼          ▼
┌─────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐
│ Agent   │ │ Tool   │ │Session │ │ Checkpoint   │
│ Builder │ │Manager │ │Manager │ │ Manager      │
│(Fluent) │ │(Unified│ │(VFS/LSP│ │(Pickle+Meta) │
│         │ │ Registry│ │/Docker)│ │ Auto-Recovery│
└─────────┘ └────────┘ └────────┘ └──────────────┘
       │                      │
       ▼                      ▼
┌──────────────┐   ┌──────────────────┐
│ HybridMemory │   │ ExecutionEngine  │
│ (SQLite+     │   │ a_run (silent)   │
│  FAISS+FTS5) │   │ a_stream (dict)  │
│              │   │ a_stream_verbose │
└──────────────┘   └──────────────────┘
```

## Core Components

### AgentBuilder (`builder.py`)

Fluent builder for creating agents. Build pipeline:

```python
builder = app.get_agent_builder()
agent = (
    builder("my_agent")
    .fast_model("gpt-4o-mini")
    .complex_model("gpt-4o")
    .max_iterations(25)
    .history_length(20)
    .add_tool("tool_name")
    .build()
)
```

| Method | Description |
|--------|-------------|
| `__call__(name)` | Start building agent with given name |
| `.fast_model(model)` | Set model for simple/quick tasks |
| `.complex_model(model)` | Set model for complex reasoning |
| `.max_iterations(n)` | Max execution steps |
| `.history_length(n)` | Chat history window |
| `.add_tool(name)` | Register tool for agent |
| `.add_tools([names])` | Register multiple tools |
| `.build()` | Finalize and register agent |

### AgentManager (`module.py`)

Manages agent lifecycle through the ISAA mod:

| Method | Description |
|--------|-------------|
| `init_isaa(app)` | Initialize ISAA, create base agents |
| `get_agent(name) → Agent` | Get registered agent instance |
| `get_agent_builder() → AgentBuilder` | Get builder for creating new agents |
| `register_agent(agent)` | Register an agent |
| `list_agents() → list` | List all registered agents |

### ToolManager (`tool_manager.py`)

Unified tool registry supporting local, MCP, CLI, and A2A tools.

| Method | Description |
|--------|-------------|
| `register(func, name, description, category, flags, ...)` | Register a tool |
| `register_cli_tool(name, executable, ...)` | Register CLI command as tool |
| `register_mcp_tools(server_name, tools)` | Register MCP server tools |
| `get(name) → ToolEntry` | Get tool by name |
| `execute(name, **kwargs)` | Execute tool |
| `get_all_litellm(...) → list` | Export in LiteLLM/OpenAI format |
| `health_check_all() → dict` | Health check all tools |
| `unregister(name)` | Remove tool |

Features:
- Auto-wraps sync functions as async (`asyncio.to_thread`)
- `no_thread` flag for GUI/Win32 calls (runs on event loop thread)
- `result_contract` validation (type, non-null, empty string checks)
- Checkpoint serialization (function references NOT serialized)
- `register_cli_tool` auto-discovers --help for documentation

### SessionManager (`session_manager.py`)

Manages agent sessions with persistence:

| Method | Description |
|--------|-------------|
| `create_session(agent_name) → session_id` | Start new session |
| `get_session(session_id) → ChatSession` | Load session |
| `save_session(session)` | Persist session state |
| `delete_session(session_id)` | Remove session |
| `list_sessions(agent_name) → list` | List sessions for agent |

Sessions persist to VFS, with optional LSP, Docker, and Web container support.

### Execution Modes

| Mode | Method | Output |
|------|--------|--------|
| **Silent** | `a_run(prompt, ...)` | Final result only, auto-resume on failure |
| **Stream (dict)** | `a_stream(prompt)` | Yields dict chunks (token, tool_call, progress) |
| **Stream (verbose)** | `a_stream_verbose(prompt)` | Yields ANSI-formatted terminal output for live UX |

### Hybrid Memory (`AISemanticMemory` + `HybridMemoryStore`)

| Component | Backend | Purpose |
|-----------|---------|---------|
| `AISemanticMemory` | Singleton | FAISS vector search, embeddings |
| `HybridMemoryStore` | SQLite + FAISS + FTS5 | Triple-mode retrieval: vector, keyword, metadata |
| Agent Memory Tools | Via ToolManager | `memory_recall`, `memory_save`, `memory_analyse` |

### CheckpointManager (`checkpoint_manager.py`)

| Method | Description |
|--------|-------------|
| `save_checkpoint(state, label)` | Save state to pickle + JSON meta |
| `load_checkpoint(path) → state` | Load and verify checkpoint |
| `list_checkpoints(agent_name) → list` | List available checkpoints |
| `rotate(max_checkpoints)` | Auto-rotation (oldest removed) |

## Tools Available to Agents

Base tools registered by ISAA:

| Tool | Description |
|------|-------------|
| `memory_recall` | Query long-term memory (vector + BM25) |
| `memory_save` | Save important facts permanently |
| `memory_analyse` | Deep multi-step memory analysis |
| `shell` | Execute shell commands |
| `write_code` | Write code files (auto static analysis) |
| `patch_code` | Patch files via unique str-replace |
| `analyze_code` | Static analysis (lint, security, complexity) |
| `run_tests` | Execute tests (optional runtime analysis) |
| `docs_read` | Read/search documentation |
| `docs_lookup` | Find code elements |
| `docs_sync` | Sync docs index |
| `manifest_show/get/set` | Read/write configuration |
| `tb` | Execute CLI commands |
| `toolbox_execute` | Run any mod function |
| `cloudm_action` | CloudM user/folder operations |

## Configuration (Manifest)

```yaml
isaa:
  self_agent:
    fast_model: "gpt-4o-mini"
    complex_model: "gpt-4o"
    max_iterations: 25
    history_length: 20
  agent_store: "~/.local/share/ToolBoxV2/agents/"
  checkpoint_dir: "~/.local/share/ToolBoxV2/checkpoints/"
```

## Related

- [AgentBuilder Reference](builder.md) (detailed API)
- [ToolManager Reference](tool_manager.md) (detailed API)
- [Session Management](session.md)
- [CloudM Auth](../CloudM/auth.md) — session validation for agents
- [Flows](../../flows/index.md) — Chains, MiniCLI using ISAA agents
