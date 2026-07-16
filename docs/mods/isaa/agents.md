# FlowAgent

###### CORE CLASS · VERIFIED AGAINST `31a117e`

`FlowAgent` (`base/Agent/flow_agent.py`) is a production-ready autonomous agent with session isolation. One instance serves many sessions; each session has its own history and virtual file system.

## Running

| Method | Use when |
|---|---|
| `a_run(query, session_id="default", execution_id=None, human_online=False, max_iterations=…)` | You want the final answer as a string |
| `a_stream(...)` | You want live progress — yields dicts during execution |
| `a_stream_verbose(...)` | Human-readable terminal stream ("Zen" output) |
| `chat(query, is_new=False, with_tools=True, stream=False)` | Lightweight conversational turn |
| `a_run_llm_completion(messages, model_preference="fast", ...)` | Raw LLM call through the agent's router, no agent loop |

## Execution lifecycle — pause, resume, cancel

Long-running executions are first-class objects with IDs:

```python
execs   = agent.list_executions()                 # active + paused
state   = agent.get_execution_state(exec_id)
await agent.pause_execution(exec_id)
await agent.resume_execution(exec_id, content="continue with option B")
await agent.cancel_execution(exec_id)

# After a restart: hot (in-memory) + cold (on-disk) resumables
agent.get_all_resumable()
await agent.resume_last_execution()
```

An execution that hits `max_iterations` is not lost — it lands in the resumable set.

## VFS coding

The agent writes code into its session VFS through a dedicated coding LLM:

```python
await agent.write_file("src/parser.py",  task="CSV parser with type inference", session_id="dev")
await agent.write_patch("src/parser.py", task="Handle BOM and empty lines",     session_id="dev")
```

## Meta-learning ("Dreamer")

```python
report = await agent.a_dream()          # blocking meta-learning cycle
async for ev in agent.a_dream_stream(): ...   # same, streaming
```

## Audio

`setup_audio(tts_config=None, player="null", ...)` attaches an `AudioStreamPlayer`; `set_audio_player_device(-1)` selects output.

<!-- verified: toolboxv2/mods/isaa/base/Agent/flow_agent.py::FlowAgent @ 31a117e -->

---

# FlowAgentBuilder

Fluent configuration; `build()` produces the agent. Persistable via `save_config(path)` / `FlowAgentBuilder.from_config_file(path)`.

###### OPTION SURFACE

| Area | Methods |
|---|---|
| Identity | `with_name` · `with_system_message` · `with_temperature` · `verbose` · `with_stream` |
| Models | `with_models(fast_model, complex_model=None)` |
| Rate limiting | `with_rate_limiter(...)` · `add_api_key(provider, key)` · `add_fallback_chain(primary, fallbacks)` · `set_model_limits(model, rpm, tpm, ...)` · `load_rate_limiter_config(path)` |
| Context budget | `set_context_max_context_ratio(0.85)` · `set_context_immediate_offload_ratio(0.7)` · `set_context_displacement_threshold(0.4)` · `set_context_safety_margin_tokens(500)` · `set_context_heavy_hitter_min_tokens(1000)` |
| Tools | `add_tool(func, name, description, category, flags)` · `add_tools_from_module(module, prefix, exclude)` · `load_mcp_tools_from_config(path_or_dict)` |
| World model | `with_world_model(dict)` · `add_world_fact(key, value)` |
| Sandboxing | `with_docker_vfs(config)` · `with_docker(True)` · `with_lsp(True)` · `with_vfs_window_lines(n)` |
| Serving | `enable_mcp_server(host, port)` · `enable_a2a_server(host, port)` |
| Durability | `with_checkpointing(enabled, interval_seconds=300, max_checkpoints=10, max_age_hours=24)` |

<!-- verified: toolboxv2/mods/isaa/base/Agent/builder.py::FlowAgentBuilder @ 31a117e -->

---

# Execution Engine V3

`base/Agent/execution_engine.py` — "Intelligent Agent Orchestration". What actually runs when you call `a_run`:

| Component | Role |
|---|---|
| `ExecutionEngine` | Main orchestration loop for FlowAgent |
| `ExecutionContext` | Complete state of one execution run |
| `LoopDetector` | Detects when the agent is stuck repeating itself |
| `ContextBudgetConfig` | Dynamic context-budget management (the `set_context_*` builder knobs) |
| `HistoryCompressor` | Rule-based compression of working history |
| `ToolSlot` | Dynamically loaded tool slots with relevance tracking |
| `PersonaProfile` / `PersonaRouter` / `PersonaStats` | Runtime personas, selected per-query from skills + dreamer insights, with effectiveness tracking |
| `ToolValidationError` | Raised when the provider rejects a tool call as invalid |

<!-- verified: toolboxv2/mods/isaa/base/Agent/execution_engine.py class docstrings @ 31a117e -->

---

# Sessions

`AgentSessionV2` (`base/Agent/agent_session_v2.py`) — per-session isolation with **VFS V2** and Docker integration. The `session_id` you pass to `a_run`/`a_stream` selects the session; history, files, and state never leak across IDs.

---

# Skills

`base/Agent/skills.py` — learned behavioral patterns:

| Class | Role |
|---|---|
| `Skill` | A learned or predefined behavioral pattern |
| `SkillsManager` | Manages skills for one FlowAgent instance |
| `ToolGroup` | Groups multiple tools under one display name |
| `SkillIOAnthropicFormat` / `AnthropicSkillMetadata` | Import/export in Anthropic-compatible `SKILL.md` format |

<!-- verified: toolboxv2/mods/isaa/base/Agent/skills.py class docstrings @ 31a117e -->
