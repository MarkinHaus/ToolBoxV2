# ISAA Quickstart

###### 5 MINUTES · VERIFIED AGAINST `31a117e`

Requires `pip install toolboxv2[isaa]` and at least one model API key (configure via `tb -init config`, step 5, or environment variables).

## Get the module and initialize

```python
from toolboxv2 import get_app

app = get_app()
isaa = app.get_mod("isaa")
await isaa.init_isaa(name="self")          # loads config, prepares the default agent
```

## Run an agent

`run_agent` accepts a registered agent name or a `FlowAgent` instance:

```python
result = await isaa.run_agent(
    name="self",
    text="List the three largest files in the project and what they do",
    session_id="default",        # sessions isolate history + VFS
)
```

Signature (ground truth):

```python
async def run_agent(name: str | FlowAgent, text: str, verbose: bool = False,
                    session_id: str | None = "default",
                    progress_callback: Callable | None = None, **kwargs)
```

## One-shot completions — no agent loop

For single LLM calls without tool use, `mini_task_completion` is cheaper and faster:

```python
answer = await isaa.mini_task_completion(
    mini_task="Extract the version number",
    user_task="toolboxv2 0.1.28 released today",
)
```

## Structured output — Pydantic in, dict out

```python
from pydantic import BaseModel

class Ticket(BaseModel):
    title: str
    priority: int

data = await isaa.format_class(format_schema=Ticket,
                               task="Login page throws 500 after the deploy, blocks all users")
# → {"title": "...", "priority": ...}
```

## Build a custom agent

```python
builder = isaa.get_agent_builder("docs-bot", add_base_tools=True)
builder.with_models("gemini/gemini-2.5-flash") \
       .with_system_message("You answer only from the indexed docs.") \
       .with_temperature(0.2) \
       .add_tool(my_search_function, name="search_docs")

await isaa.register_agent(builder)
agent = await isaa.get_agent("docs-bot")
result = await agent.a_run("How do sessions work?", session_id="u1")
```

The builder is fluent — every `with_*` / `add_*` returns the builder. Full option surface in [Agents](agents.md#flowagentbuilder).

## Streaming

```python
async for chunk in agent.a_stream("Refactor utils/toolbox.py", session_id="dev"):
    handle(chunk)               # dicts: progress, tool events, text deltas

# Or pretty terminal output:
async for line in agent.a_stream_verbose("Explain the worker system"):
    print(line, end="")
```

<!-- verified: toolboxv2/mods/isaa/module.py::run_agent,mini_task_completion,format_class,get_agent_builder,register_agent,get_agent @ 31a117e -->
<!-- verified: toolboxv2/mods/isaa/base/Agent/flow_agent.py::a_run,a_stream,a_stream_verbose @ 31a117e -->
