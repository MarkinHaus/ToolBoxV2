# FlowAgent & FlowAgentBuilder Guide

## Table of Contents
1.  [Architecture Overview](#architecture-overview)
2.  [Core Concepts](#core-concepts)
    *   [LLM Reasoner: The Strategic Core](#llm-reasoner-the-strategic-core)
    *   [Unified Context Management](#unified-context-management)
    *   [Advanced Variable System](#advanced-variable-system)
3.  [FlowAgent API](#flowagent-api)
4.  [FlowAgentBuilder API](#flowagentbuilder-api)
5.  [Quick Start Guide](#quick-start-guide)
6.  [Configuration Management](#configuration-management)
7.  [Persona & Response Formatting](#persona--response-formatting)
8.  [Tool Integration](#tool-integration)
    *   [Custom & MCP Tools](#custom--mcp-tools)
9.  [Variable System In-Depth](#variable-system-in-depth)
10. [Context & Session Management](#context--session-management)
11. [Advanced Usage](#advanced-usage)
    *   [Checkpoint & Resume](#checkpoint--resume)
    *   [Performance Monitoring](#performance-monitoring)
12. [Production Deployment](#production-deployment)
13. [Best Practices](#best-practices)

---

## 1. Architecture Overview

The FlowAgent system has evolved into a hierarchical, reasoning-driven architecture. The central component is the **`LLMReasonerNode`**, which acts as the strategic core. It analyzes requests, creates execution outlines, and delegates tasks to specialized sub-systems.

This design moves from a linear pipeline to an intelligent, adaptive loop controlled by the reasoner.

```
                                      ┌────────────────────────┐
                                      │    FlowAgentBuilder    │
                                      │ (Configuration Engine) │
                                      └───────────┬────────────┘
                                                  │ (Builds)
                                                  ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                    FlowAgent                                      │
│ ┌───────────────────────────────────────────────────────────────────────────────┐ │
│ │                              LLMReasonerNode                                  │ │
│ │                           (The Strategic Core)                                │ │
│ │ ┌────────────────┐ ┌──────────────────┐ ┌────────────────┐ ┌────────────────┐ │ │
│ │ │ Outline Engine │ │ Meta-Tool Caller │ │ Context Manager│ │ Auto-Recovery  │ │ │
│ │ └────────────────┘ └──────────────────┘ └────────────────┘ └────────────────┘ │ │
│ └────────────────────────────────────┬──────────────────────────────────────────┘ │
│                                      │ (Delegates to Sub-Systems)                 │
│          ┌───────────────────────────┴───────────────────────────┐                │
│          ▼                           ▼                           ▼                │
│ ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────────────┐  │
│ │   LLMToolNode    │      │   TaskPlanner    │      │    Response Generation   │  │
│ │ (Simple Tool Use)│      │  (Complex Plans) │      │ (Formatting & Synthesis) │  │
│ └──────────────────┘      └─────────┬────────┘      └──────────────────────────┘  │
│                                     ▼                                             │
│                           ┌──────────────────┐                                    │
│                           │   TaskExecutor   │                                    │
│                           │(Parallel Execution)│                                  │
│                           └──────────────────┘                                    │
└───────────────────────────────────────────────────────────────────────────────────┘

```

---

## 2. Core Concepts

### LLM Reasoner: The Strategic Core

The `LLMReasonerNode` is the brain of the agent. Instead of following a fixed path, it performs these steps in a loop:

1.  **Outline Creation**: For any given query, it first generates a high-level strategic outline (e.g., "Step 1: Research data", "Step 2: Analyze findings", "Final Step: Synthesize response").
2.  **Task Stack Management**: It maintains an internal to-do list (`internal_task_stack`) based on the current outline step.
3.  **Meta-Tool Execution**: In each loop, it decides which "meta-tool" to use to make progress on its current task. These are not external tools but internal functions that control the agent's sub-systems:
    *   `internal_reasoning`: For thinking and analyzing the situation.
    *   `delegate_to_llm_tool_node`: For simple, self-contained tasks that require external tools (e.g., a web search).
    *   `create_and_execute_plan`: For complex, multi-step projects that require the full `TaskPlanner` and `TaskExecutor`.
    *   `read_from_variables`/`write_to_variables`: To interact with the stateful `VariableManager`.
    *   `direct_response`: To provide the final answer when the outline is complete.
4.  **Auto-Recovery**: It includes mechanisms to detect infinite loops and automatically attempt recovery, for example, by forcing an advance to the next outline step.

### Unified Context Management

The new `UnifiedContextManager` provides a single, authoritative source for all contextual information. It integrates:

*   **Chat History**: Persistent, session-aware conversation history via `ChatSession`.
*   **Variable System**: Access to all data in the `VariableManager`, including task results and world model facts.
*   **Execution State**: Real-time information about active, completed, and failed tasks.
*   **Intelligent Caching**: Reduces redundant context processing for better performance.

This eliminates the need for individual nodes to aggregate context manually, leading to a more streamlined and reliable system.

### Advanced Variable System

The `VariableManager` is a powerful state management system with the following features:

*   **Scoped Variables**: Organizes data into logical scopes like `world`, `results`, `user`, and `system`.
*   **Dot-Notation Access**: Access nested data in dictionaries and lists easily (e.g., `{{ results.task-123.data.some_key }}`).
*   **Multiple Syntaxes**: Use `{{ variable.path }}`, `{variable}`, or `$variable` for flexible text formatting.
*   **Dynamic Suggestions**: The system can suggest relevant variables to the LLM based on the current query.
*   **LLM-Friendly Documentation**: Can generate a comprehensive list of all available variables for the LLM to reference.

---

## 3. FlowAgent API

#### Basic Usage

```python
# Simple query execution
agent = await FlowAgentBuilder().with_assistant_persona().build()
response = await agent.a_run("Your query here")

# With session management
response = await agent.a_run(
    query="Follow up question",
    session_id="user_123",
    user_id="john_doe"
)

# Using variables
agent.set_variable("user.name", "John")
agent.set_variable("project.name", "FlowAgent Demo")
response = await agent.a_run("Hello {{ user.name }}! How is {{ project.name }} going?")

# Fast run mode - skips detailed outline creation for quick responses
response = await agent.a_run(
    query="What's the weather like?",
    fast_run=True  # Uses generic adaptive outline for faster execution
)

# Callback mode - inject real-time context for proactive responses
def my_callback():
    """Callback function that provides context"""
    pass

response = await agent.a_run(
    query="Process this event",
    as_callback=my_callback  # Injects callback context into agent execution
)
```

#### Advanced Features

```python
# Format-specific responses
agent.set_response_format(
    response_format="with-tables",
    text_length="detailed-indepth",
    custom_instructions="Focus on actionable insights"
)
response_with_format = await agent.a_run_with_format(
    query="Analyze sales data for Q3",
    response_format="with-tables"
)


# Checkpoint management
await agent.pause()  # Creates and saves a checkpoint
await agent.resume() # Resumes from the paused state

# Performance and status monitoring
summary = await agent.get_task_execution_summary()
reasoning = await agent.explain_reasoning_process()
status = agent.status(pretty_print=True)

# Context Management
await agent.save_context_to_session("user_123")
context_stats = agent.get_context_statistics()

# Lifecycle
await agent.close() # Saves a final checkpoint and shuts down gracefully
```

#### Fast Run Mode

The `fast_run` parameter allows the agent to skip the detailed outline creation phase and use a generic, adaptive outline instead. This is ideal for simple queries that need quick responses, especially in voice interfaces or real-time applications.

**When to use `fast_run=True`:**
- Simple, straightforward queries that don't require complex planning
- Voice interface interactions where speed is critical
- Real-time responses in chat applications
- Tool-based queries that can be answered with a single tool call

**How it works:**
1. Instead of creating a detailed, query-specific outline, the agent uses a pre-defined 2-step outline
2. Step 1: Immediate tool usage or direct analysis
3. Step 2: Synthesize and respond
4. This reduces latency by eliminating the outline creation LLM call

```python
# Example: Fast run for simple queries
response = await agent.a_run(
    query="What's 2+2?",
    fast_run=True
)

# Example: Fast run with tool usage
response = await agent.a_run(
    query="Search for the latest news on AI",
    fast_run=True  # Will use tools immediately without detailed planning
)
```

#### Callback Mode

The `as_callback` parameter enables the agent to be invoked within a callback context, providing real-time, context-specific information. This is useful for event-driven architectures where the agent needs to respond proactively to events.

**When to use `as_callback`:**
- Event-driven systems where the agent responds to external triggers
- Real-time monitoring and alerting systems
- Webhook handlers that need intelligent processing
- Proactive assistance based on system events

**How it works:**
1. When `as_callback` is provided, the agent injects callback context into the shared state
2. The context includes: callback timestamp, callback name, and the initial query
3. The LLMReasonerNode can access this context to tailor its responses
4. This enables the agent to understand it's operating in a reactive/proactive mode

```python
# Example: Using the agent in a callback
def on_file_uploaded(file_path: str):
    """Callback triggered when a file is uploaded"""
    pass

async def handle_upload_event(file_path: str):
    response = await agent.a_run(
        query=f"A new file was uploaded: {file_path}. Analyze and summarize it.",
        as_callback=on_file_uploaded,
        session_id="upload_handler"
    )
    return response

# Example: Webhook handler
async def webhook_handler(event_data: dict):
    response = await agent.a_run(
        query=f"Process this webhook event: {event_data}",
        as_callback=webhook_handler,
        fast_run=True  # Combine with fast_run for quick event processing
    )
    return response
```

**Callback Context Structure:**
```python
{
    'callback_timestamp': '2024-01-15T10:30:00.123456',
    'callback_name': 'on_file_uploaded',
    'initial_query': 'A new file was uploaded: /path/to/file.txt'
}
```

---

## 4. FlowAgentBuilder API

The `FlowAgentBuilder` is now a fluent, production-focused builder that relies on a structured `AgentConfig` model.

#### Builder Components

*   **Configuration Management**: `load_config()`, `save_config()`, `validate_config()`.
*   **Fluent API**: A chainable interface for programmatic configuration.
*   **Integration Systems**: Built-in support for MCP, A2A, and OpenTelemetry.

#### Fluent API Example

```python
builder = (FlowAgentBuilder()
    .with_name("MyProductionAgent")
    .with_models("openrouter/anthropic/claude-3-haiku", "openrouter/openai/gpt-4o")
    .with_system_message("You are a helpful production assistant.")
    .with_developer_persona()
    .enable_mcp_server(port=8001)
    .enable_a2a_server(port=5001)
    .enable_telemetry(service_name="prod-agent", console_export=True)
    .with_checkpointing(interval_seconds=600)
    .with_custom_variables({"env": "production"})
    .verbose(True)
)

# Validate before building
issues = builder.validate_config()
if not issues["errors"]:
    agent = await builder.build()
```

---

## 5. Quick Start Guide

### 1. Basic Agent Creation

```python
import asyncio
from builder import FlowAgentBuilder

async def basic_example():
    # Create a simple assistant agent using a pre-built factory method
    agent = await (FlowAgentBuilder
                  .create_general_assistant("MyAssistant")
                  .build())

    # Use the agent
    response = await agent.a_run("Hello! Can you help me write a Python function?")
    print(response)

    await agent.close()

# Run the example
asyncio.run(basic_example())
```

### 2. Pre-built Agent Types

The builder provides factory methods for common agent types, which configure the name, persona, and integrations.

```python
# Developer agent with code focus
developer = await FlowAgentBuilder.create_developer_agent("CodeHelper").build()

# Data analyst agent with visualization focus
analyst = await FlowAgentBuilder.create_analyst_agent("DataHelper").build()

# Creative assistant for content generation
creative = await FlowAgentBuilder.create_creative_agent("ContentCreator").build()

# Executive assistant for strategic tasks
executive = await FlowAgentBuilder.create_executive_agent("StrategyHelper").build()

# General assistant with full capabilities
assistant = await FlowAgentBuilder.create_general_assistant("GeneralHelper").build()
```

### 3. Custom Tool Integration

```python
async def custom_tool_example():
    # Define a custom tool
    def get_server_time() -> str:
        """Returns the current server time in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()

    # Build agent with the tool
    agent = await (FlowAgentBuilder()
                  .with_name("CustomToolAgent")
                  .with_assistant_persona()
                  .add_tool(get_server_time, "get_server_time")
                  .build())

    # Use the tool through natural language
    response = await agent.a_run("What is the current server time?")
    print(response)

    await agent.close()

asyncio.run(custom_tool_example())
```

---

## 6. Configuration Management

Configuration is managed through the `AgentConfig` Pydantic model, which can be loaded from or saved to YAML/JSON files.

### 1. Configuration Structure (`agent_config.yaml`)

```yaml
name: "ProductionAgent"
description: "Production-ready agent with full capabilities"
version: "2.0.0"

# LLM Configuration
fast_llm_model: "openrouter/anthropic/claude-3-haiku"
complex_llm_model: "openrouter/openai/gpt-4o"
temperature: 0.7
max_tokens_output: 2048
api_key_env_var: "OPENROUTER_API_KEY"

# Features
mcp:
  enabled: true
  host: "0.0.0.0"
  port: 8000
  config_path: "mcp_tools.json"

a2a:
  enabled: true
  host: "0.0.0.0"
  port: 5000
  agent_name: "ProductionAgent"

telemetry:
  enabled: true
  service_name: "production_agent"
  console_export: true

checkpoint:
  enabled: true
  interval_seconds: 300
  checkpoint_dir: "./checkpoints"

# Persona and Variables
active_persona: "developer"
persona_profiles:
  developer:
    name: "Senior Developer"
    style: "technical"
    # ... more persona settings
custom_variables:
  environment: "production"```

### 2. Loading and Saving

```python
# Load from a configuration file
builder = FlowAgentBuilder.from_config_file("agent_config.yaml")
agent = await builder.build()

# Save the current builder configuration to a file
builder.save_config("my_agent_config.yaml", format="yaml")
```

### 3. Configuration Validation

It's best practice to validate the configuration before building the agent.

```python
builder = FlowAgentBuilder.from_config_file("config.yaml")

# Validate configuration
issues = builder.validate_config()

if issues["errors"]:
    print("Configuration errors:", issues["errors"])
elif issues["warnings"]:
    print("Configuration warnings:", issues["warnings"])
else:
    agent = await builder.build()
```

---

## 7. Persona & Response Formatting

The persona system is now deeply integrated with response formatting to control the agent's output structure and style.

### 1. Persona and Format Structure

The `PersonaConfig` now includes an optional `FormatConfig` to define the desired output structure.

```python
@dataclass
class FormatConfig:
    response_format: ResponseFormat = ResponseFormat.FREI_TEXT
    text_length: TextLength = TextLength.CHAT_CONVERSATION
    # ... more settings

@dataclass
class PersonaConfig:
    name: str
    style: str = "professional"
    # ... other traits
    format_config: Optional[FormatConfig] = None
```

### 2. Pre-built Personas

The builder includes methods that set up personas with appropriate default formats.

```python
# Developer Persona -> Defaults to 'code-structure' format
builder.with_developer_persona()

# Analyst Persona -> Defaults to 'with-tables' format
builder.with_analyst_persona()
```

### 3. Dynamic Response Formatting

You can override the default persona format at runtime for a specific query.

```python
# Set a specific response format for the next call
agent.set_response_format(
    response_format="with-tables",      # Use tables for data
    text_length="detailed-indepth",     # Comprehensive responses
    custom_instructions="Include confidence scores"
)
response = await agent.a_run("Analyze this data: [1,2,3,4,5]")

# Or use the convenient run_with_format method
response_md = await agent.a_run_with_format(
    query="Explain this concept",
    response_format="md-text",
    text_length="detailed-indepth"
)

# Get available formats
formats = agent.get_available_formats()
print("Available formats:", formats["formats"])
```

---

## 8. Tool Integration

### Custom & MCP Tools

The system seamlessly integrates custom Python functions and tools loaded from MCP (Model Context Protocol) servers. The `FlowAgentBuilder` now features a robust MCP loader that automatically manages server processes and creates tool wrappers.

#### 1. Custom Functions

```python
def get_current_time():
    """Returns the current timestamp."""
    from datetime import datetime
    return datetime.now().isoformat()

# Add a custom tool to the agent
builder = FlowAgentBuilder().add_tool(get_current_time, "current_time")
```

#### 2. Module Integration

```python
import math

# Add all public functions from the math module with a prefix
builder.add_tools_from_module(module=math, prefix="math_")
```

#### 3. MCP Integration (via `mcp_servers.json`)

The builder can launch and integrate with MCP servers defined in a configuration file. It will automatically manage the server lifecycle and extract all its capabilities (tools, resources, prompts).

**`mcp_servers.json`:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["-m", "mcp_server_filesystem"],
      "env": { "FILESYSTEM_ROOT": "/home/user/documents" }
    },
    "sequential_thinking": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

**Loading in the builder:**
```python
agent = await (FlowAgentBuilder()
              .with_name("MCPAgent")
              .load_mcp_tools_from_config("mcp_servers.json")
              .build())

# The agent can now use tools like 'filesystem_read_file' or 'sequential_thinking_prompt_...'
response = await agent.a_run("Read the file 'report.txt' from my documents.")
```

---

## 9. Variable System In-Depth

The `VariableManager` provides a powerful way to manage state and create dynamic content.

### 1. Variable Scopes

*   **`world`**: Stores facts the agent has learned.
*   **`results`**: Holds the output of every executed task (e.g., `results.task-123.data`).
*   **`user`**: Contains information about the current user and session.
*   **`system`**: Provides system-level information like timestamps.
*   **Custom Scopes**: You can register your own scopes for better organization.

### 2. Variable Usage in Prompts

You can reference variables in prompts, system messages, and tool arguments using multiple syntaxes.

```python
# Double brace syntax (recommended for paths)
response = await agent.a_run("User: {{ user.name }}, Project: {{ project.details.name }}")

# Single brace syntax (for simple, top-level variables)
response = await agent.a_run("Welcome {user_name}!")

# Dollar syntax
response = await agent.a_run("Current time is $system_timestamp")
```

### 3. Variable Management API

```python
# Set a nested variable
agent.set_variable("project.details.version", "3.0")

# Get variable documentation for the LLM
docs = agent.get_variable_documentation()
print(docs)

# Get available variables as a dictionary
available_vars = agent.get_available_variables()
```

---

## 10. Context & Session Management

Context is now handled centrally by the `UnifiedContextManager`, ensuring consistency across the agent.

### 1. Session Initialization

Sessions are automatically created and managed. You just need to provide a `session_id`.

```python
# This will create or load the session for 'user_123'
await agent.a_run("My first question", session_id="user_123")

# The agent now has context from the first question
await agent.a_run("Follow-up question based on my first one", session_id="user_123")
```

### 2. Unified Context

The context provided to the LLM reasoner is a rich, unified view of:
*   Recent conversation history.
*   The current execution state (active and completed tasks).
*   Available results from the variable system.
*   Relevant facts from the world model.

### 3. Context API

```python
# Initialize a session explicitly (optional)
await agent.initialize_session_context(session_id="user_456", max_history=300)

# Get a snapshot of the current unified context
context_data = await agent.get_context(session_id="user_456", format_for_llm=False)

# Save a snapshot of the context to the persistent session history
await agent.save_context_to_session("user_456")

# Get context statistics
stats = agent.get_context_statistics()
```

---

## 11. Advanced Usage

### Checkpoint & Resume

The agent can automatically save its state and be restored later, making long-running tasks more reliable.

```python
# Enable checkpointing in the builder
builder.with_checkpointing(enabled=True, interval_seconds=300)

# Manually pause the agent (this also saves a checkpoint)
await agent.pause()

# Later, you can resume
resumed_agent = await FlowAgentBuilder.from_config_file("config.yaml").build()
await resumed_agent.load_latest_checkpoint()
await resumed_agent.resume()
```

### Performance Monitoring

The agent exposes detailed status and performance metrics.

```python
# Enable telemetry for distributed tracing (e.g., with Jaeger)
builder.enable_telemetry(service_name="my_agent", endpoint="http://localhost:14268/api/traces")

# Get a comprehensive status report
agent.status(pretty_print=True)

# Get a summary of the reasoning and execution process
reasoning_explanation = await agent.explain_reasoning_process()
print(reasoning_explanation)

# Get detailed statistics from the task executor
if hasattr(agent.task_flow, 'executor_node'):
    stats = agent.task_flow.executor_node.get_execution_statistics()
    print("Execution stats:", stats)
```

---

## 12. Production Deployment

### 1. Production Configuration

Use a dedicated YAML configuration file for production environments to manage settings without code changes. Disable verbose logging.

```python
# production_agent.py
import asyncio
from builder import FlowAgentBuilder

async def main():
    agent = await (FlowAgentBuilder
                  .from_config_file("production_config.yaml")
                  .verbose(False)
                  .build())

    try:
        await agent.start_servers() # Starts MCP/A2A if enabled
        print(f"Production agent '{agent.amd.name}' is ready.")
        # Keep the agent running
        while True:
            await asyncio.sleep(3600)
    finally:
        await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Expose ports for MCP and A2A
EXPOSE 8000 5000

# Run the agent
CMD ["python", "production_agent.py"]
```

### 3. Health Monitoring

Implement a health check endpoint to monitor the agent's status in production.

```python
async def health_check(agent: FlowAgent):
    status = agent.status()
    is_healthy = status["runtime_status"]["status"] in ["idle", "running"]
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "agent_status": status["runtime_status"]["status"],
        "total_cost": status["performance"]["total_cost"],
        "active_tasks": status["task_execution"]["active_tasks"]
    }
```

---

## 13. Best Practices

*   **Use Configuration Files**: Manage agent settings in YAML files (`from_config_file`) instead of hard-coding them in the builder for better maintainability.
*   **Validate Configuration**: Always run `builder.validate_config()` before `build()` to catch issues early.
*   **Leverage Pre-built Personas**: Start with pre-built personas (`.with_developer_persona()`) and customize from there.
*   **Use Sessions**: Pass a unique `session_id` to `a_run()` for each user or conversation to maintain context.
*   **Manage Resources**: Use `await agent.close()` for a graceful shutdown, which saves a final checkpoint and cleans up server processes.
*   **Enable Checkpointing**: For any long-running or critical tasks, enable checkpointing to ensure reliability.
*   **Monitor Performance**: Regularly check `agent.status()` and enable telemetry in production to monitor costs and performance.
*   **Secure API Keys**: Always load API keys from environment variables (`.with_api_config(api_key_env_var=...)`) and never hard-code them.
*   **Use `fast_run` Wisely**: Enable `fast_run=True` for simple queries in voice interfaces or real-time applications, but use the default detailed planning for complex, multi-step tasks.
*   **Combine Features**: You can combine `fast_run=True` with `as_callback` for ultra-fast event processing in reactive systems.
*   **Callback Context**: When using `as_callback`, ensure the callback function has a meaningful `__name__` attribute for better debugging and context tracking.
