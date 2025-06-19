# ISAA (Intelligent System Agent Architecture) Module Documentation

**Version:** 0.2.0

## 1. Overview

The ISAA module provides a comprehensive framework for building, managing, and orchestrating AI agents. It leverages the `EnhancedAgent` and `EnhancedAgentBuilder` for creating sophisticated agents with capabilities like tool use, code execution, web interaction, and persistent memory. The module is designed to be extensible and configurable, allowing developers to create complex multi-agent systems and automated workflows.

Key features include:
*   **Advanced Agent System**: Based on `EnhancedAgent` for robust and production-ready agents.
*   **Flexible Agent Configuration**: Uses `EnhancedAgentBuilder` for fluent and detailed agent setup.
*   **Task Chain Management**: Define and execute sequences of agent actions or tool uses.
*   **Interactive Code Execution Pipelines**: Stateful Python execution environments for agents.
*   **Semantic Memory**: Persistent knowledge storage and retrieval using `AISemanticMemory`.
*   **Tool Integration**: Supports ADK-compatible tools, custom functions, and has provisions for LangChain tools.
*   **Asynchronous Operations**: Many core functionalities are `async` for better performance.

## 2. Core Concepts

### 2.1. `EnhancedAgent`
The `EnhancedAgent` is the primary agent class in ISAA. It integrates:
*   **LiteLLM**: For interaction with a wide range of LLMs.
*   **ADK (Agent Development Kit)**: For structured tool use, planning, and code execution (if ADK is available and configured).
*   **A2A (Agent-to-Agent)**: For inter-agent communication (if A2A is available).
*   **MCP (Model Context Protocol)**: For exposing agent capabilities (if MCP is available).
*   **World Model**: A dictionary-like persistent state for the agent.
*   **Cost Tracking**: Built-in user cost tracking.
*   **Callbacks**: For streaming, progress, and post-run actions.

### 2.2. `EnhancedAgentBuilder`
The `EnhancedAgentBuilder` provides a fluent API to configure and construct `EnhancedAgent` instances. Key aspects:
*   **Configuration Model (`BuilderConfig`)**: A Pydantic model that holds all serializable configurations for an agent. This can be saved to and loaded from JSON.
*   **Model Configuration**: Specify LLM model, API keys (via env vars), temperature, etc.
*   **Behavior**: Streaming, logging, initial world model data.
*   **Framework Integrations**: Enable and configure ADK, A2A, MCP.
*   **Tool Management**: Add ADK tools (including wrapped functions).
*   **Cost Tracking**: Configure persistence for user costs.
*   **Telemetry**: Configure OpenTelemetry.

## 3. Initialization and Configuration (`Tools` Class)

The `Tools` class is the main entry point for interacting with the ISAA module.

```python
from toolboxv2 import get_app
from toolboxv2.mods.isaa.module import Tools

# Get the application instance (if using toolboxv2 framework)
app = get_app("my_application")
isaa = Tools(app=app) # or isaa = app.get_mod("isaa") if registered

# Initialize ISAA (loads configs, sets up defaults)
# This is now an async operation if it involves building default agents
async def initialize_isaa():
    await isaa.init_isaa()
    print("ISAA initialized.")

# asyncio.run(initialize_isaa())
```

### 3.1. Configuration (`isaa.config`)
The `isaa.config` dictionary holds various settings:
*   `DEFAULTMODEL*`: Default LLM model identifiers for different agent types (e.g., `DEFAULTMODEL0`, `DEFAULTMODELCODE`). These can be overridden by environment variables.
*   `agents-name-list`: A list of registered agent names.
*   `controller_file`: Path to the JSON file for `ControllerManager` (LLM modes).
*   Other internal states and paths.

API keys (like `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.) are typically loaded from environment variables by LiteLLM or explicitly set in the `EnhancedAgentBuilder` via `with_api_key_from_env()`.

### 3.2. `isaa.on_exit()`
Called when the application or module shuts down. It saves:
*   Agent builder configurations (`BuilderConfig` dicts from `isaa.agent_data`).
*   `ControllerManager` state.
*   `AgentChain` definitions.
*   `Scripts` definitions.
*   ISAA `Tools` class configuration.

## 4. Agent Management

All agent management functions are now primarily `async`.

### 4.1. Getting an Agent Builder (`async get_agent_builder`)
This method returns a pre-configured `EnhancedAgentBuilder` instance.

```python
async def manage_agent_builder():
    # Get a default builder for an agent named "coder_agent"
    coder_builder = await isaa.get_agent_builder("coder_agent")

    # Further configure the builder
    coder_builder.with_model("anthropic/claude-3-haiku-20240229")
    coder_builder.with_system_message("You are a master Python programmer.")
    coder_builder.enable_adk_code_executor("adk_builtin") # If ADK is available

    # ... other configurations ...
    return coder_builder
```
Default builders come with common ISAA tools like `runAgent`, `memorySearch`, `searchWeb`, and `shell`.

### 4.2. Registering an Agent (`async register_agent`)
Once an `EnhancedAgentBuilder` is configured, its configuration can be registered with ISAA. The agent instance itself will be built on demand.

```python
async def register_my_agent():
    builder = await isaa.get_agent_builder("my_query_agent")
    builder.with_system_message("You answer questions based on internal memory.")

    await isaa.register_agent(builder)
    print("Agent 'my_query_agent' configuration registered.")
```
This saves the builder's configuration to a JSON file (e.g., `.data/app_id/Agents/my_query_agent.agent.json`) and stores the config dictionary in `isaa.agent_data`.

### 4.3. Retrieving an Agent Instance (`async get_agent`)
This `async` method retrieves (and builds if necessary) an `EnhancedAgent` instance.

```python
async def retrieve_and_use_agent():
    my_agent = await isaa.get_agent("my_query_agent")
    # my_agent is now an instance of EnhancedAgent

    # If you need to override the model for this instance (will rebuild if different)
    # my_agent_gpt4 = await isaa.get_agent("my_query_agent", model_override="openai/gpt-4-turbo")

    response = await my_agent.a_run("What is the capital of France?")
    print(response)
```
If an agent configuration exists, it's loaded. Otherwise, a default builder is used. The agent instance is cached in `isaa.config['agent-instance-{name}']`.

## 5. Running Agents and Tasks

### 5.1. Running an Agent (`async run_agent`)
This is the primary method to interact with a registered agent.

```python
async def run_specific_agent():
    # Ensure agent is registered (e.g., during init_isaa or manually)
    # For example, register a 'self' agent if not already present
    if "self" not in isaa.config.get("agents-name-list", []):
        self_builder = await isaa.get_agent_builder("self")
        await isaa.register_agent(self_builder)

    response = await isaa.run_agent("self", "Tell me a joke.")
    print(f"Self agent says: {response}")

    # Example with session_id for persistent history with EnhancedAgent
    session_id = "user123_chat_session"
    response1 = await isaa.run_agent("self", "My name is Bob.", session_id=session_id)
    response2 = await isaa.run_agent("self", "What is my name?", session_id=session_id)
    print(f"Agent remembers: {response2}")
```
The `run_agent` method now calls the `EnhancedAgent.a_run()` method, which supports features like session-based history, world model updates, ADK tool execution, etc.

### 5.2. Mini Task Completion (`async mini_task_completion`)
For quick, one-off LLM calls without full agent capabilities.

```python
async def run_mini_task():
    translation = await isaa.mini_task_completion(
        mini_task="Translate to German.",
        user_task="Hello, how are you?"
    )
    print(f"Translation: {translation}")
```

### 5.3. Structured Output (`async format_class`)
To get LLM output structured according to a Pydantic model.

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int
    city: Optional[str] = None

async def get_structured_info():
    user_data_dict = await isaa.format_class(
        UserInfo,
        "The user is Alice, 30 years old, from New York."
    )
    if user_data_dict:
        user_info = UserInfo(**user_data_dict)
        print(f"Parsed User: {user_info.name}, Age: {user_info.age}")
```

## 6. Task Chains

Task chains allow defining a sequence of operations involving agents or tools.

### 6.1. Defining a Task Chain
Task chains are defined as lists of dictionaries, where each dictionary represents a task.
The `TaskChain` Pydantic model from `toolboxv2.mods.isaa.types` can be used for validation.

Example structure for a task in the list:
```json
{
  "use": "agent", // "agent", "tool", "chain"
  "name": "agent_name_or_tool_name_or_chain_name",
  "args": "Prompt or arguments, can use $variable or $user-input",
  "return_key": "my_result_variable" // Result stored under this key
}
```

### 6.2. Creating a Task Chain (`async crate_task_chain`)
Uses an LLM (typically "TaskChainAgent") to generate a task chain definition from a natural language prompt.

```python
async def create_my_chain():
    chain_name = await isaa.crate_task_chain(
        "Create a plan to research a topic: first search the web, then summarize findings."
    )
    if chain_name:
        print(f"Task chain '{chain_name}' created.")
        isaa.save_task(chain_name) # Save it
    else:
        print("Failed to create task chain.")
```

### 6.3. Managing Task Chains
*   `isaa.add_task(chain_name, task_definition_list)`: Manually add/update a chain.
*   `isaa.get_task(chain_name)`: Get the definition of a chain.
*   `isaa.list_task()`: List names of all available chains.
*   `isaa.save_task(chain_name=None)`: Save one or all chains to file.
*   `isaa.load_task(chain_name=None)`: Load one or all chains from file.

### 6.4. Running a Task Chain (`async run_task`)
Executes a defined task chain.

```python
async def execute_my_chain():
    # Assume "research_topic_chain" was created and saved earlier
    isaa.load_task("research_topic_chain") # Load if not already in memory
    results = await isaa.run_task(
        task_input="Quantum computing advancements in 2024",
        chain_name="research_topic_chain"
    )
    print("Task chain execution results:", results)
```
The `ChainTreeExecutor` handles variable injection (`$variable_name`, `$user-input`) and result passing between tasks.

## 7. Pipelines for Code Execution (`Pipeline` Class)

The `Pipeline` class (from `toolboxv2.mods.isaa.CodingAgent.live`) provides a stateful environment for agents to execute Python code iteratively. It uses a mock IPython interface.

### 7.1. Getting a Pipeline Instance (`async get_pipe`)
Retrieves or creates a `Pipeline` instance associated with a specific ISAA agent. The agent's context (variables, potentially its LLM for thinking within the pipeline) can influence the pipeline's behavior.

```python
async def setup_pipeline():
    # Get a pipeline associated with the 'coder_agent'
    # Ensure 'coder_agent' is registered
    coder_agent_builder = await isaa.get_agent_builder("coder_agent")
    coder_agent_builder.with_system_message("You write and execute Python code to solve problems.")
    await isaa.register_agent(coder_agent_builder)

    coder_pipeline = await isaa.get_pipe("coder_agent", verbose=True)
    return coder_pipeline
```

### 7.2. Running a Pipeline (`async run_pipe`)
Executes a task within the pipeline. The agent associated with the pipeline will "think" and generate code or actions to be executed by the pipeline's IPython environment.

```python
async def execute_pipeline_task():
    coder_pipeline = await isaa.get_pipe("coder_agent") # Assumes "coder_agent" is set up

    task_description = "Define a function to calculate factorial and test it with n=5."
    pipeline_result = await coder_pipeline.run(task_description)

    print(f"Pipeline Final Result: {pipeline_result.result}")
    print("Execution History:")
    for record in pipeline_result.execution_history:
        print(f"  Code: {record.code[:50]}... -> Result: {record.result}, Error: {record.error}")
    print("Final Variables in Pipeline:", pipeline_result.variables)
```
The `Pipeline.run` method involves multiple turns of the agent thinking, generating code/actions, and the pipeline executing them, until the task is marked "done" or iterations are exhausted.

## 8. Semantic Memory (`AISemanticMemory`)

ISAA uses `AISemanticMemory` for persistent, semantic storage and retrieval of information.

### 8.1. Accessing Memory (`isaa.get_memory`)
```python
# Get the global AISemanticMemory instance
semantic_memory = isaa.get_memory()

# AISemanticMemory can manage multiple named "memory spaces" (KnowledgeBase instances)
# To get a specific KnowledgeBase instance (e.g., for an agent):
agent_kb = isaa.get_memory(name="my_agent_context") # This will create if not exists
```

### 8.2. Interacting with Memory
The `AISemanticMemory` class (and its underlying `KnowledgeBase` instances) provides methods like:
*   `async add_data(memory_name: str, data: ..., metadata: ...)`: Adds data to a specific memory space.
*   `async query(query: str, memory_names: ..., to_str: bool)`: Queries one or more memory spaces.
*   `async unified_retrieve(...)`: A more comprehensive retrieval method.

```python
async def use_semantic_memory():
    mem = isaa.get_memory()
    agent_name = "researcher"

    # Ensure the agent's memory space exists (usually handled by agent init)
    await mem.create_memory(agent_name) # Or rely on get_memory(name=...)

    # Add data
    await mem.add_data(
        memory_name=agent_name,
        data="Photosynthesis is a process used by plants to convert light energy into chemical energy.",
        metadata={"source": "biology_notes"}
    )

    # Query data
    results = await mem.query(
        query="How do plants get energy?",
        memory_names=[agent_name],
        to_str=True
    )
    print(f"Memory search results: {results}")
```
`EnhancedAgent` instances often have their world model, but can also interact with `AISemanticMemory` via tools for broader knowledge. `ChatSession` (used by `Pipeline`) also uses `AISemanticMemory`.

## 9. Tool Integration

### 9.1. Default ISAA Tools
Agents created with `get_agent_builder` automatically get several tools:
*   `runAgent`: To call other registered ISAA agents.
*   `memorySearch`: To query the `AISemanticMemory`.
*   `saveDataToMemory`: To save data into the agent's context in `AISemanticMemory`.
*   `searchWeb`: Uses `WebScraper` to search the internet.
*   `shell`: Executes shell commands using `shell_tool_function`.
*   `runCodePipeline` (for agents like "self", "code"): To invoke a `Pipeline` task.

These are added as ADK-compatible function tools to the `EnhancedAgentBuilder`.

### 9.2. Adding Custom and LangChain Tools (`async init_tools`)
The `init_tools` method is intended for adding external tools, particularly LangChain tools, to an agent *builder*.

```python
# This is a conceptual example, as init_tools itself needs to be fully async
# and adapt to how EnhancedAgentBuilder handles external tool definitions.

async def add_external_tools():
    builder = await isaa.get_agent_builder("tool_user_agent")

    # Configuration for tools (example)
    tools_config = {
        "lagChinTools": ["wikipedia", "ddg-search"], # Example LangChain tool names
        # "huggingTools": [], # HF tools are also LC tools
        # "Plugins": [] # AIPluginTool
    }

    # init_tools would modify the builder by adding wrapped LangChain tools
    await isaa.init_tools(tools_config, builder) # Pass the builder instance

    await isaa.register_agent(builder)

    agent_with_tools = await isaa.get_agent("tool_user_agent")
    response = await agent_with_tools.a_run("Search Wikipedia for 'Large Language Models'")
    print(response)
```
**Note**: Wrapping arbitrary LangChain tools (which can have complex Pydantic `args_schema`) into ADK `FunctionTool`s (which prefer simpler schemas or Pydantic models for arguments) can be non-trivial. `init_tools` will need careful implementation to handle schema mapping or require tools to be provided as simple callables.

## 10. Example Usage Flow

```python
import asyncio
from toolboxv2 import get_app
from toolboxv2.mods.isaa.module import Tools
from pydantic import BaseModel

# --- Setup ---
app = get_app("isaa_demo_app")
isaa = Tools(app=app)

# --- Pydantic Model for Structured Output ---
class AnalysisResult(BaseModel):
    topic: str
    key_points: List[str]
    sentiment: Optional[str] = None

async def main_demo():
    # 1. Initialize ISAA (loads configs, ControllerManager, etc.)
    await isaa.init_isaa()
    print("ISAA Initialized.")

    # 2. Create and Register a "Researcher" Agent
    researcher_builder = await isaa.get_agent_builder("Researcher")
    researcher_builder.with_system_message(
        "You are a research assistant. Your job is to find information using web search and summarize it."
    )
    researcher_builder.with_model("openai/gpt-3.5-turbo") # Example model
    # The default builder already adds searchWeb and memory tools.
    await isaa.register_agent(researcher_builder)
    print("Researcher agent registered.")

    # 3. Create and Register a "Summarizer" Agent for structured output
    summarizer_builder = await isaa.get_agent_builder("Summarizer")
    summarizer_builder.with_system_message(
        "You summarize text and extract key information into a structured format."
    )
    summarizer_builder.with_model("openai/gpt-4-turbo") # Example model, good for JSON
    await isaa.register_agent(summarizer_builder)
    print("Summarizer agent registered.")

    # 4. Define a Task Chain
    chain_name = "WebResearchAndAnalyze"
    research_tasks = [
        {
            "use": "agent",
            "name": "Researcher",
            "args": "Find recent news about AI in healthcare. Focus on the top 3 articles. User input: $user-input", # $user-input will be main query
            "return_key": "research_findings"
        },
        {
            "use": "agent",
            "name": "Summarizer",
            # The 'Summarizer' agent will use its format_class capability internally if prompted correctly
            # For this example, we assume 'Summarizer' is prompted to produce AnalysisResult
            # A more robust way is to have a specific tool/agent that *only* does formatting.
            "args": "Analyze the following research findings and structure them: $research_findings. Extract topic, key points, and sentiment.",
            "return_key": "structured_analysis"
        }
    ]
    isaa.agent_chain.add(chain_name, research_tasks)
    isaa.save_task(chain_name)
    print(f"Task chain '{chain_name}' created and saved.")

    # 5. Run the Task Chain
    user_query = "Latest breakthroughs in AI-driven drug discovery"
    print(f"\nRunning task chain '{chain_name}' for query: '{user_query}'")
    chain_results = await isaa.run_task(user_query, chain_name)
    print("\n--- Task Chain Results ---")
    if chain_results and "structured_analysis" in chain_results:
        analysis_output = chain_results["structured_analysis"]
        # If the summarizer agent directly returned a dict matching AnalysisResult:
        try:
            # The result from an agent run is typically a string.
            # If the 'Summarizer' was specifically designed to output JSON string for AnalysisResult:
            analysis_dict = json.loads(analysis_output) # Agent must output valid JSON string
            structured_data = AnalysisResult(**analysis_dict)
            print(f"Topic: {structured_data.topic}")
            print("Key Points:")
            for point in structured_data.key_points:
                print(f"  - {point}")
            if structured_data.sentiment:
                print(f"Sentiment: {structured_data.sentiment}")
        except Exception as e:
            print(f"Could not parse structured analysis: {e}")
            print("Raw analysis output:", analysis_output)
    else:
        print("Chain did not produce expected 'structured_analysis'. Full results:", chain_results)

    # 6. Example of using a Pipeline with a "Coder" agent
    coder_builder = await isaa.get_agent_builder("PyCoder")
    coder_builder.with_system_message("You are a Python coding assistant. You write and execute Python code to solve problems. Ensure your code prints results or returns values.")
    coder_builder.enable_adk_code_executor("unsafe_simple") # Or "adk_builtin" if model supports
    await isaa.register_agent(coder_builder)

    py_coder_pipeline = await isaa.get_pipe("PyCoder", verbose=True) # Get pipeline for this agent
    pipeline_task = "Write a Python function that takes a list of numbers and returns their sum. Then, call this function with the list [1, 2, 3, 4, 5] and print the result."
    print(f"\nRunning Pipeline for task: '{pipeline_task}'")
    pipeline_result = await py_coder_pipeline.run(pipeline_task)
    print("\n--- Pipeline Final Output ---")
    print(pipeline_result.result)
    print("\n--- Pipeline Variables ---")
    # Filter out internal IPython variables for clarity
    final_vars = {k: v for k, v in pipeline_result.variables.items() if not k.startswith('_') and k not in ['In', 'Out', 'exit', 'quit', 'get_ipython', 'open']}
    print(json.dumps(final_vars, default=str, indent=2))

    # 7. Clean up (optional, as on_exit handles saving)
    # isaa.on_exit()

if __name__ == "__main__":
    asyncio.run(main_demo())
```

## 11. Important Notes
*   **Asynchronous Nature**: Most core methods of the `Tools` class are now `async`. Ensure your calling code uses `await` appropriately or runs within an asyncio event loop.
*   **Agent Configuration**: Agent capabilities are primarily defined by their system message, the tools provided to them via the `EnhancedAgentBuilder`, and their underlying LLM model.
*   **Error Handling**: Robust error handling should be implemented around `async` calls, especially for network-dependent operations like LLM calls or web interactions.
*   **ADK Integration**: Full ADK functionality (planning, advanced tool schemas, long-running operations) requires Google ADK to be installed and properly configured. The `EnhancedAgentBuilder` provides methods like `with_adk_code_executor`, `with_adk_tool_instance`, etc.
*   **Security**: Be cautious when enabling code execution (`unsafe_simple` executor is for development only) or shell access.

This documentation provides a starting point for using the refactored ISAA module. As the module evolves, further details on specific component interactions (e.g., advanced ADK planner configurations, A2A/MCP server setup via builder) will be crucial.
