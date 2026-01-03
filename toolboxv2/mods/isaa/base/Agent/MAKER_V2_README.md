# MAKER V2 - Virtual Workspace Architecture mit echtem Tool-Calling

## Kernproblem gelöst

**Problem:** `safe_tools: dict[str, Callable]` wurde erstellt aber nie verwendet. Der Agent konnte keine Tools ausführen.

**Lösung:** Neues `VirtualToolExecutor` System das:
1. Echte Tool-Calls über `agent.arun_function()` ausführt
2. Write-Operations zur `VirtualWorkspace` umleitet
3. Unsafe Tools blockiert

## Architektur

```
┌─────────────────────────────────────────────────────────────────┐
│                     AtomicConquerNodeV2                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              _execute_react_loop()                       │   │
│  │                                                          │   │
│  │  1. Prepare LiteLLM Tools Format                         │   │
│  │     └─> _prepare_tools_for_litellm()                     │   │
│  │                                                          │   │
│  │  2. Build Messages (System + User Prompt)                │   │
│  │                                                          │   │
│  │  3. ReAct Loop (max 10 iterations):                      │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  ┌────────────────────────────────┐                      │   │
│  │  │ _call_llm_with_tools()         │                      │   │
│  │  │  └─> litellm.acompletion()     │                      │   │
│  │  │      with tools + tool_choice  │                      │   │
│  │  └────────────────────────────────┘                      │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  ┌────────────────────────────────┐                      │   │
│  │  │ _extract_tool_calls()          │                      │   │
│  │  │  └─> Parse response.tool_calls │                      │   │
│  │  └────────────────────────────────┘                      │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │         VirtualToolExecutor.execute()              │  │   │
│  │  │                                                    │  │   │
│  │  │  if SAFE_READ/SEARCH/COMPUTE:                      │  │   │
│  │  │      └─> agent.arun_function() (echte Ausführung)  │  │   │
│  │  │                                                    │  │   │
│  │  │  if UNSAFE_WRITE:                                  │  │   │
│  │  │      └─> workspace.virtual_write_file() (staged)   │  │   │
│  │  │                                                    │  │   │
│  │  │  if UNSAFE_API/EXEC:                               │  │   │
│  │  │      └─> BLOCKED (unless in allowed_unsafe)        │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  Add tool_response to messages                           │   │
│  │  Loop until final_answer() called                        │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Nach Voting-Konsens:                                           │
│  └─> _commit_workspace() → Real FS Write                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Neue Komponenten

### 1. VirtualToolExecutor

```python
class VirtualToolExecutor:
    """Führt Tools mit Virtualisierung aus"""
    
    async def execute(self, tool_name: str, arguments: dict) -> dict:
        category = self.registry.classify(tool_name)
        
        if category == ToolCategory.UNSAFE_WRITE:
            # → Virtualisiert: Schreibt in Workspace Staging
            return await self._execute_virtual_write(tool_name, arguments)
        
        elif category in {UNSAFE_API, UNSAFE_EXEC}:
            # → Blockiert (außer in allowed_unsafe)
            return {"success": False, "error": "BLOCKED"}
        
        else:
            # → ECHTE Ausführung via agent.arun_function()
            result = await self.agent.arun_function(tool_name, **arguments)
            return {"success": True, "result": result}
```

### 2. _execute_react_loop

Echter ReAct-Loop mit:
- **LiteLLM Tool-Calling**: `litellm.acompletion()` mit `tools` Parameter
- **Message History**: Proper Assistant + Tool Response Management
- **final_answer Tool**: Spezielles Tool zum Beenden der Schleife
- **Max 10 Iterations**: Verhindert Endlosschleifen

```python
async def _execute_react_loop(self, task, context, agent, session_id, 
                               attempt, workspace, tool_executor):
    # 1. Prepare tools in LiteLLM format
    litellm_tools = self._prepare_tools_for_litellm(agent, safe_tool_names)
    
    # 2. Add final_answer tool
    litellm_tools.append({
        "function": {
            "name": "final_answer",
            "parameters": {"success": bool, "result": str, ...}
        }
    })
    
    # 3. ReAct Loop
    for iteration in range(10):
        response = await litellm.acompletion(
            model=agent.amd.fast_llm_model,
            messages=messages,
            tools=litellm_tools,
            tool_choice="auto"
        )
        
        tool_calls = self._extract_tool_calls(response)
        
        for tc in tool_calls:
            if tc.name == "final_answer":
                return AtomicResult(**tc.arguments)
            
            # Execute via VirtualToolExecutor
            result = await tool_executor.execute(tc.name, tc.arguments)
            messages.append({"role": "tool", "content": result})
```

### 3. Tool Classification

```python
class SafeToolRegistry:
    DEFAULT_CLASSIFICATIONS = {
        # Safe - echte Ausführung
        "read_file": SAFE_READ,
        "google_search": SAFE_SEARCH,
        "calculator": SAFE_COMPUTE,
        
        # Virtualisiert
        "write_file": UNSAFE_WRITE,  # → workspace.virtual_write_file()
        "create_file": UNSAFE_WRITE,
        
        # Blockiert
        "send_email": UNSAFE_API,
        "run_command": UNSAFE_EXEC,
    }
```

## Flow

```
1. Task kommt rein
2. DivideNode teilt in atomare Tasks
3. TaskTreeBuilder gruppiert parallel
4. Für jeden Task:
   a. VirtualWorkspace erstellen
   b. VirtualToolExecutor erstellen
   c. ReAct-Loop mit echten Tool-Calls
   d. Write-Ops landen in Staging
5. Voting auf (Text-Hash + Staging-Hash)
6. Winner-Workspace wird committed
```

## Usage

```python
from mda_accomplish_v2 import bind_accomplish_v2_to_agent

await bind_accomplish_v2_to_agent(agent, and_as_tool=True)

result = await agent.a_accomplish_v2(
    task="Lies config.json und aktualisiere die DB-Settings",
    context="Project: /home/user/myapp",
    min_complexity=3,
    enable_tools=True  # Jetzt wirklich mit Tool-Calls!
)

# Der Agent kann jetzt:
# - read_file aufrufen (echte Ausführung)
# - write_file aufrufen (virtuell, staged)
# - Nach Voting-Konsens werden Änderungen committed
```

## Unterschied zu V1

| Aspekt | V1 | V2 |
|--------|----|----|
| Tool-Ausführung | ❌ Nur Format erstellt | ✅ Echte Calls via arun_function |
| Write-Ops | Nicht möglich | ✅ Virtualisiert + Staged |
| ReAct-Loop | Fake (single LLM call) | ✅ Echter Loop mit litellm |
| Tool-Blocking | Nicht implementiert | ✅ Nach Kategorie |

## Debug

```python
# Tool Execution Log einsehen
executor = VirtualToolExecutor(agent, workspace, registry)
await executor.execute("read_file", {"path": "/test.txt"})
print(executor.get_execution_summary())
# Output: "Executed 1 tool calls:\n  ✓ read_file"

# Staged Changes prüfen
print(workspace.get_diff_summary())
# Output: "═══ STAGED CHANGES ═══\n+ CREATE: /new.txt (100 bytes)"
```
