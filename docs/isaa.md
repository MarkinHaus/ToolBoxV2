# ISAA (Intelligent System Agent Architecture) Documentation

## 1. Overview

The ISAA (Intelligent System Agent Architecture) module is a sophisticated framework for building, configuring, and orchestrating advanced AI agents. At its core, ISAA now leverages the powerful **`FlowAgent`**, a next-generation agent designed for complex reasoning, dynamic planning, and robust tool use.

The module provides a high-level API to manage the lifecycle of these agents, from creation and configuration to execution and state management. It is built for asynchronous operations, ensuring high performance in complex, multi-agent workflows.

**Key Features:**

*   **Advanced Agent Core**: Powered by the new **[FlowAgent](./agent.md)**, which features an outline-driven reasoning loop, sub-system orchestration, and auto-recovery.
*   **Builder-Centric Configuration**: Uses the `FlowAgentBuilder` for a fluent, declarative, and serializable approach to agent setup.
*   **Stateful Code & File Execution**: Each agent is equipped with a `ToolsInterface`, providing a sandboxed environment for code execution, file system operations, and web browsing.
*   **Semantic Memory**: Integrates with `AISemanticMemory` for long-term, persistent knowledge storage and retrieval.
*   **Unified Tool System**: Seamlessly integrates custom functions, core ISAA utilities, and agent-specific `ToolsInterface` capabilities.
*   **Asynchronous by Design**: All primary operations are `async`, making the system scalable and efficient.

## 2. Core Concepts

### 2.1. `FlowAgent`

The `FlowAgent` is the heart of the ISAA module. It's a highly advanced agent capable of autonomous reasoning and complex task execution.

For a comprehensive guide on the internal architecture, capabilities, and API of the `FlowAgent` itself, please refer to the **[FlowAgent Documentation](./agent.md)**.

Key characteristics relevant to ISAA include:

*   **Reasoning Core:** Makes strategic decisions instead of following rigid plans.
*   **Unified Context:** Manages conversation history, task status, and variables through a `UnifiedContextManager`.
*   **Dynamic Tool Use:** Intelligently analyzes and selects tools based on the immediate context.
*   **State Management:** Utilizes a powerful, scoped `VariableManager` to maintain state during execution.

### 2.2. `FlowAgentBuilder`

The `FlowAgentBuilder` is the exclusive method for configuring and creating `FlowAgent` instances. It provides a fluent API that promotes clear and maintainable agent definitions.

*   **Configuration as Code:** Define every aspect of an agent—models, persona, tools, and features—programmatically.
*   **Serializable Config:** The builder's state is backed by a Pydantic model (`AgentConfig`), allowing any agent's configuration to be saved to and loaded from a JSON or YAML file.
*   **Centralized Setup:** ISAA uses the builder to inject core tools and system-wide configurations automatically.

### 2.3. `ToolsInterface`

Replacing the previous `Pipeline` system, the `ToolsInterface` is a stateful execution environment that is **directly tied to an agent instance**. It provides a powerful set of tools for interacting with the outside world.

*   **Agent-Specific Environment:** Each agent gets its own sandboxed `ToolsInterface` to manage files, execute code, and maintain state without interfering with other agents.
*   **Rich Toolset:** Provides tools for file I/O (`read_file`, `write_file`, `list_directory`), multi-language code execution (`execute_python`), and web browsing.
*   **Automatic Integration:** When an agent is created via `isaa.get_agent_builder()`, the relevant tools from its `ToolsInterface` are automatically added to the agent's capabilities.

### 2.4. `AISemanticMemory`

`AISemanticMemory` serves as the long-term memory for the entire system. Agents can query this memory to retrieve past knowledge or save new information for future use.

## 3. Initialization and Configuration (`Tools` Class)

The `Tools` class is the main entry point for using the ISAA module.

```python
from toolboxv2 import get_app
from toolboxv2.mods.isaa.module import Tools

# Get the application instance
app = get_app("my_application")
isaa = app.get_mod("isaa") # Assumes ISAA is registered with the app

# Initialize ISAA (loads configs, sets up defaults)
async def initialize_isaa():
    await isaa.init_isaa()
    print("ISAA initialized.")

# asyncio.run(initialize_isaa())
```

### `isaa.on_exit()`

This method ensures that all agent configurations and other states are saved gracefully when the application shuts down.

## 4. Agent Management

All agent management is now fully asynchronous and builder-oriented.

### 4.1. Getting an Agent Builder (`get_agent_builder`)

This is the starting point for creating any new agent. It returns a `FlowAgentBuilder` pre-configured with ISAA's core tools.

```python
async def manage_agent_builder():
    # Get a builder for an agent named "coder_agent"
    coder_builder = isaa.get_agent_builder("coder_agent")

    # Further configure the builder
    coder_builder.with_models(
        fast_llm_model="openrouter/anthropic/claude-3-haiku",
        complex_llm_model="openrouter/openai/gpt-4o"
    )
    coder_builder.with_system_message("You are a master Python programmer.")

    return coder_builder
```

### 4.2. Registering an Agent (`async register_agent`)

Once a builder is configured, you register its configuration with ISAA. This saves the agent's definition and makes it available for use.

```python
async def register_my_agent():
    builder = isaa.get_agent_builder("my_query_agent")
    builder.with_system_message("You answer questions based on internal memory.")

    await isaa.register_agent(builder)
    print("Agent 'my_query_agent' configuration registered.")
```

### 4.3. Retrieving an Agent Instance (`async get_agent`)

This method builds (or retrieves from a cache) a fully operational `FlowAgent` instance from a registered configuration.

```python
async def retrieve_and_use_agent():
    # This will build the agent if it's the first time it's requested
    my_agent = await isaa.get_agent("my_query_agent")

    response = await my_agent.a_run("What is the capital of France?")
    print(response)
```

## 5. Running Agents and Tasks

### 5.1. Running an Agent (`async run_agent`)

The primary method for interacting with a registered agent by name.

```python
async def run_specific_agent():
    # Register a simple agent if it doesn't exist
    if "responder" not in isaa.config.get("agents-name-list", []):
        builder = isaa.get_agent_builder("responder")
        await isaa.register_agent(builder)

    # Use a session_id for conversations with persistent history
    session_id = "user123_chat"
    response1 = await isaa.run_agent("responder", "My favorite color is blue.", session_id=session_id)
    response2 = await isaa.run_agent("responder", "What is my favorite color?", session_id=session_id)
    print(f"Agent remembers: {response2}")
```

### 5.2. Structured Output (`async format_class`)

Leverage an agent's reasoning to structure output according to a Pydantic model.

```python
from pydantic import BaseModel, Field
from typing import List

class UserProfile(BaseModel):
    name: str = Field(description="The user's full name.")
    age: int = Field(description="The user's age.")
    interests: List[str] = Field(description="A list of the user's interests.")

async def get_structured_info():
    profile_dict = await isaa.format_class(
        UserProfile,
        "The user is Alice Smith. She is 30 years old and enjoys hiking and photography."
    )
    if profile_dict:
        profile = UserProfile(**profile_dict)
        print(f"Parsed User: {profile.name}, Interests: {profile.interests}")
```

## 6. Code Execution with `ToolsInterface`

The `ToolsInterface` provides a powerful, stateful environment for each agent to execute code, manage files, and interact with the web. **You do not interact with the `ToolsInterface` directly.** Instead, you instruct the agent to use the tools that ISAA has automatically provided from the interface.

The agent's reasoner is aware of these tools and will use them when a task requires it.

**Example: Instructing an agent to use its file and code tools.**
```python
async def execute_code_task():
    # 1. Get a builder for a coding agent. ISAA will automatically add
    #    code and file tools from the ToolsInterface.
    coder_builder = isaa.get_agent_builder("PyCoder")
    coder_builder.with_system_message(
        "You are a Python coding assistant. You write and execute Python code to solve problems."
    )
    await isaa.register_agent(coder_builder)

    # 2. Get the agent instance
    coder_agent = await isaa.get_agent("PyCoder")

    # 3. Give the agent a multi-step task involving file I/O and code execution
    task_prompt = (
        "First, create a Python script named 'hello.py' that prints 'Hello from ISAA!'. "
        "Then, execute that script and show me the output."
    )

    # The agent's reasoner will create an outline:
    # - Step 1: Use the `createScript` tool to write the file.
    # - Step 2: Use the `runScript` tool to execute it.
    # - Step 3: Return the captured output.
    response = await coder_agent.a_run(task_prompt)
    print("--- Agent Response ---")
    print(response)
```

## 7. Semantic Memory (`AISemanticMemory`)

Agents can interact with the shared semantic memory through the tools provided by ISAA.

```python
async def use_semantic_memory():
    # Get a general-purpose agent
    agent = await isaa.get_agent("self")

    # Instruct the agent to save information
    await agent.a_run("Please remember that the project codename is 'Orion'.", session_id="project_orion")

    # Later, in the same or a different session, instruct it to retrieve the information
    response = await agent.a_run("What is the project codename?", session_id="project_orion")

    print(response) # The agent will use its `memorySearch` tool to find the answer.
```

## 8. Tool Integration

The `FlowAgentBuilder` is the central point for tool management. ISAA's `get_agent_builder` method automatically equips new builders with a powerful set of default tools.

### Default ISAA Tools
*   **Core Tools**: `searchWeb`, `shell`.
*   **Memory Tools**: `memorySearch`, `saveDataToMemory`.
*   **Scripting Tools**: `createScript`, `runScript`, `listScripts`, `deleteScript`.
*   **`ToolsInterface` Tools**: A suite of tools for file management (`write_file`, `read_file`, `list_directory`), code execution (`execute_python`), and more, tailored to the agent's purpose.

### Adding Your Own Tools
You can easily add your own custom functions to any agent builder.
```python
# 1. Define your custom async function
async def get_database_user_count() -> int:
    """Returns the current number of users in the database."""
    # In a real scenario, this would connect to a database
    return 1337

# 2. Get a builder and add your tool
my_builder = isaa.get_agent_builder("db_agent")
my_builder.add_tool(
    get_database_user_count,
    name="getUserCount",
    description="Fetches the total number of users from the main database."
)

# 3. Register and use the agent
await isaa.register_agent(my_builder)
db_agent = await isaa.get_agent("db_agent")
response = await db_agent.a_run("How many users are in the system?")
print(response)
```

## 9. Example Usage Flow

This example demonstrates a complete workflow: creating an agent, giving it a complex task that requires file I/O and code execution, and getting the final result.

```python
import asyncio
from toolboxv2 import get_app
from toolboxv2.mods.isaa.module import Tools

# --- Setup ---
app = get_app("isaa_demo_app")
isaa = app.get_mod("isaa")

async def main_demo():
    # 1. Initialize ISAA
    await isaa.init_isaa()
    print("ISAA Initialized.")

    # 2. Get a builder for a "DataProcessor" agent.
    #    ISAA will automatically add file and code execution tools.
    data_builder = isaa.get_agent_builder("DataProcessor")
    data_builder.with_system_message(
        "You are a data processing specialist. You write Python scripts to "
        "manipulate data, save it to files, and read it back."
    )
    data_builder.with_models(
        fast_llm_model="openrouter/anthropic/claude-3-haiku",
        complex_llm_model="openrouter/openai/gpt-4o"
    )

    # 3. Register the agent's configuration
    await isaa.register_agent(data_builder)
    print("DataProcessor agent registered.")

    # 4. Get the agent instance
    data_agent = await isaa.get_agent("DataProcessor")

    # 5. Define a multi-step task for the agent
    user_task = (
        "I need you to perform a data processing task. Here are the steps:\n"
        "1. Create a Python script called 'process.py'.\n"
        "2. The script should create a list of numbers from 1 to 10 and calculate their sum.\n"
        "3. It must then write the result to a file named 'result.txt' in the format: 'The sum is: [sum]'.\n"
        "4. After creating the script, execute it.\n"
        "5. Finally, read the content of 'result.txt' and tell me what it says."
    )

    print(f"\n--- Running Agent for Task ---\n{user_task}\n")

    # 6. Run the agent
    # The agent will use its internal reasoner to create an outline and use its
    # `createScript`, `runScript`, and `read_file` tools to complete the task.
    final_response = await data_agent.a_run(user_task, session_id="data_processing_task_01")

    print("\n--- Final Agent Response ---")
    print(final_response)

    # 7. Check the agent's status
    print("\n--- Agent Status ---")
    data_agent.status(pretty_print=True)

    await data_agent.close()

# --- Run the Demo ---
asyncio.run(main_demo())
```

## 10. Important Notes
*   **Asynchronous First**: The entire ISAA module is built around `asyncio`. Ensure your code is running in an async context.
*   **Configuration is Key**: An agent's performance is highly dependent on its configuration, especially its system message, persona, and the tools it has access to.
*   **Security**: Be extremely cautious when using tools that can execute code (`execute_python`) or shell commands (`shell`). The default `ToolsInterface` is sandboxed to a specific directory, but care should always be taken. Avoid exposing these agents to untrusted inputs.
*   **State and Sessions**: Use `session_id` to maintain distinct conversational contexts for different users or tasks. The agent's memory and state are tied to this ID.

## Overview

The Chain system provides a powerful way to orchestrate multiple ISAA agents in complex workflows. Chains allow you to create sophisticated AI pipelines with sequential execution, parallel processing, conditional branching, error handling, and automatic data formatting between agents.

## Core Concepts

### 1. Basic Chain Operations

The chain system uses intuitive operators to define workflows:

- `>>` : Sequential execution (pipe operator)
- `+` or `&` : Parallel execution
- `%` : Conditional branching
- `|` : Error handling (try/catch)
- `-` : Data extraction and formatting

### 2. Chain Components

#### **CF (Chain Format)**
Handles data transformation and extraction between agents using Pydantic models.

```python
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    interests: list[str]

# Create a formatter
profile_format = CF(UserProfile)
```

#### **IS (Conditional Check)**
Creates conditional logic based on data values.

```python
# Check if age is over 18
adult_check = IS("age", 18)
```

## Basic Usage

### 1. Setting Up Agents for Chaining

```python
import asyncio
from toolboxv2 import get_app

# Initialize ISAA
app = get_app("chain_demo")
isaa = app.get_mod("isaa")
await isaa.init_isaa()

# Create specialized agents
async def setup_agents():
    # Data extractor agent
    extractor_builder = isaa.get_agent_builder("data_extractor")
    extractor_builder.with_system_message(
        "You extract structured information from text. Always provide complete, accurate data."
    )
    await isaa.register_agent(extractor_builder)

    # Analyzer agent
    analyzer_builder = isaa.get_agent_builder("analyzer")
    analyzer_builder.with_system_message(
        "You analyze data and provide insights and recommendations."
    )
    await isaa.register_agent(analyzer_builder)

    # Report generator
    reporter_builder = isaa.get_agent_builder("reporter")
    reporter_builder.with_system_message(
        "You create detailed reports from analysis data."
    )
    await isaa.register_agent(reporter_builder)

await setup_agents()
```

### 2. Simple Sequential Chain

```python
async def simple_sequential_chain():
    # Get agent instances
    extractor = await isaa.get_agent("data_extractor")
    analyzer = await isaa.get_agent("analyzer")
    reporter = await isaa.get_agent("reporter")

    # Create a sequential chain
    chain = extractor >> analyzer >> reporter

    # Execute the chain
    user_text = "John Smith is 25 years old and loves hiking, photography, and cooking."
    result = await chain.a_run(user_text)

    print("Final Report:", result)

# Run the chain
await simple_sequential_chain()
```

## Advanced Chain Features

### 1. Data Formatting with CF

```python
from pydantic import BaseModel
from typing import List

class PersonData(BaseModel):
    name: str
    age: int
    interests: List[str]
    location: str = "Unknown"

class Analysis(BaseModel):
    profile_summary: str
    recommendations: List[str]
    risk_score: int

async def formatted_chain():
    extractor = await isaa.get_agent("data_extractor")
    analyzer = await isaa.get_agent("analyzer")

    # Chain with structured data formatting
    chain = (
        extractor >>
        CF(PersonData) >>
        analyzer >>
        CF(Analysis)
    )

    user_input = "Sarah Johnson, 28, lives in Seattle. Enjoys rock climbing, programming, and yoga."

    result = await chain.a_run(user_input)
    print("Structured Result:", result)

await formatted_chain()
```

### 2. Data Extraction with Key Selection

```python
async def extraction_chain():
    extractor = await isaa.get_agent("data_extractor")
    analyzer = await isaa.get_agent("analyzer")

    # Extract specific fields using the - operator
    chain = (
        extractor >>
        CF(PersonData) - "interests" >>  # Extract only interests
        analyzer
    )

    # Extract multiple fields
    multi_extract_chain = (
        extractor >>
        CF(PersonData) - ("name", "age") >>  # Extract name and age
        analyzer
    )

    # Extract all fields
    all_extract_chain = (
        extractor >>
        CF(PersonData) - "*" >>  # Extract everything
        analyzer
    )

    user_input = "Mike loves surfing and coding. He's 30 years old."

    result1 = await chain.a_run(user_input)
    result2 = await multi_extract_chain.a_run(user_input)
    result3 = await all_extract_chain.a_run(user_input)

    print("Interests only:", result1)
    print("Name and age:", result2)
    print("All data:", result3)

await extraction_chain()
```

### 3. Parallel Processing

```python
async def parallel_chain():
    analyzer = await isaa.get_agent("analyzer")
    reporter = await isaa.get_agent("reporter")

    # Create a specialized summarizer
    summarizer_builder = isaa.get_agent_builder("summarizer")
    summarizer_builder.with_system_message("You create concise summaries.")
    await isaa.register_agent(summarizer_builder)

    summarizer = await isaa.get_agent("summarizer")

    # Parallel execution using + operator
    parallel_chain = analyzer + reporter + summarizer

    # Sequential then parallel
    extractor = await isaa.get_agent("data_extractor")
    complex_chain = extractor >> (analyzer + reporter)

    data = "Complex project data with multiple stakeholders and requirements..."

    # This will run analyzer, reporter, and summarizer simultaneously
    parallel_result = await parallel_chain.a_run(data)
    print("Parallel Results:", parallel_result)

    # This will run extractor first, then analyzer and reporter in parallel
    complex_result = await complex_chain.a_run(data)
    print("Complex Chain Result:", complex_result)

await parallel_chain()
```

### 4. Auto-Parallel Processing

```python
class TaskList(BaseModel):
    tasks: List[str]
    priority: str

async def auto_parallel_chain():
    extractor = await isaa.get_agent("data_extractor")
    analyzer = await isaa.get_agent("analyzer")

    # Auto-parallel: Process each task in the list simultaneously
    chain = (
        extractor >>
        CF(TaskList) - "tasks[n]" >>  # [n] creates auto-parallel
        analyzer
    )

    input_text = """
    I have these tasks to analyze:
    - Review quarterly reports
    - Plan next sprint
    - Update documentation
    - Conduct team interviews
    """

    # The analyzer will process each task in parallel automatically
    results = await chain.a_run(input_text)
    print("Auto-parallel Results:", results)

await auto_parallel_chain()
```

### 5. Conditional Chains

```python
async def conditional_chain():
    extractor = await isaa.get_agent("data_extractor")
    analyzer = await isaa.get_agent("analyzer")
    reporter = await isaa.get_agent("reporter")

    # Create a simple approval agent
    approver_builder = isaa.get_agent_builder("approver")
    approver_builder.with_system_message(
        "You approve or reject based on criteria. Return 'approved' or 'rejected'."
    )
    await isaa.register_agent(approver_builder)
    approver = await isaa.get_agent("approver")

    # Conditional chain: different paths based on approval
    chain = (
        extractor >>
        approver >>
        IS("status", "approved") >> reporter %  # If approved, generate report
        analyzer  # If not approved, send to analyzer for revision
    )

    approved_request = "This is a well-structured, reasonable business proposal."
    rejected_request = "This request lacks proper documentation and justification."

    result1 = await chain.a_run(approved_request)  # Goes to reporter
    result2 = await chain.a_run(rejected_request)  # Goes to analyzer

    print("Approved path result:", result1)
    print("Rejected path result:", result2)

await conditional_chain()
```

### 6. Error Handling Chains

```python
async def error_handling_chain():
    # Create a potentially failing agent
    risky_builder = isaa.get_agent_builder("risky_processor")
    risky_builder.with_system_message(
        "You process data but sometimes fail. Randomly throw errors for demonstration."
    )
    await isaa.register_agent(risky_builder)

    risky_agent = await isaa.get_agent("risky_processor")
    safe_agent = await isaa.get_agent("analyzer")  # Fallback

    # Error handling chain using | operator
    chain = risky_agent | safe_agent

    # If risky_agent fails, safe_agent will handle it
    try:
        result = await chain.a_run("Process this potentially problematic data")
        print("Chain completed:", result)
    except Exception as e:
        print("Chain failed despite fallback:", e)

await error_handling_chain()
```

## Complex Workflow Examples

### 1. Multi-Stage Document Processing

```python
class DocumentMetadata(BaseModel):
    title: str
    author: str
    document_type: str
    complexity_score: int

class ProcessingResult(BaseModel):
    summary: str
    key_points: List[str]
    action_items: List[str]

async def document_processing_workflow():
    # Set up specialized agents
    metadata_extractor = await isaa.get_agent("data_extractor")
    content_analyzer = await isaa.get_agent("analyzer")
    summarizer = await isaa.get_agent("summarizer")
    action_extractor = await isaa.get_agent("reporter")

    # Complex multi-path workflow
    workflow = (
        metadata_extractor >>
        CF(DocumentMetadata) >>
        # Branch based on complexity
        (IS("complexity_score", "high") >>
         (content_analyzer + summarizer) >>  # Parallel processing for complex docs
         action_extractor) %
        (summarizer >> action_extractor) >>  # Simple path for easy docs
        CF(ProcessingResult)
    )

    complex_doc = """
    Title: Advanced AI Architecture Proposal
    Author: Dr. Sarah Chen
    Type: Technical Specification

    This document outlines a comprehensive approach to implementing
    next-generation AI systems with distributed processing capabilities...
    [Complex technical content continues...]
    """

    simple_doc = """
    Title: Meeting Notes
    Author: John Smith
    Type: Meeting Minutes

    Brief discussion about quarterly goals and upcoming deadlines.
    """

    complex_result = await workflow.a_run(complex_doc)
    simple_result = await workflow.a_run(simple_doc)

    print("Complex Document Result:", complex_result)
    print("Simple Document Result:", simple_result)

await document_processing_workflow()
```

### 2. Customer Service Chain

```python
class CustomerInquiry(BaseModel):
    customer_id: str
    inquiry_type: str
    priority: str
    issue_description: str

class ServiceResponse(BaseModel):
    response_text: str
    escalation_needed: bool
    follow_up_required: bool
    estimated_resolution_time: str

async def customer_service_chain():
    # Specialized customer service agents
    classifier = await isaa.get_agent("data_extractor")  # Classifies inquiries
    support_agent = await isaa.get_agent("analyzer")     # Handles standard issues
    specialist = await isaa.get_agent("reporter")        # Handles complex issues

    # Customer service workflow
    service_chain = (
        classifier >>
        CF(CustomerInquiry) >>
        # Route based on priority
        (IS("priority", "high") >> specialist) %      # High priority to specialist
        (IS("priority", "medium") >> support_agent) %  # Medium to support
        support_agent >>                               # Low to standard support
        CF(ServiceResponse) |
        # Fallback for any errors
        "I apologize, but I'm unable to process your request right now. Please contact our support team directly."
    )

    high_priority = """
    Customer ID: CUST001
    Issue: Critical system outage affecting production
    Priority: HIGH
    Description: Our entire payment system is down, affecting thousands of transactions.
    """

    low_priority = """
    Customer ID: CUST002
    Issue: Question about account settings
    Priority: LOW
    Description: How do I change my email preferences?
    """

    high_result = await service_chain.a_run(high_priority)
    low_result = await service_chain.a_run(low_priority)

    print("High Priority Response:", high_result)
    print("Low Priority Response:", low_result)

await customer_service_chain()
```

## Chain Visualization and Debugging

### 1. Visualize Chain Structure

```python
async def visualize_chain():
    extractor = await isaa.get_agent("data_extractor")
    analyzer = await isaa.get_agent("analyzer")
    reporter = await isaa.get_agent("reporter")

    # Create a complex chain
    complex_chain = (
        extractor >>
        CF(PersonData) - "interests[n]" >>  # Auto-parallel
        (analyzer + reporter) >>            # Parallel processing
        CF(Analysis) |                      # Error handling
        "Fallback response"
    )

    # Visualize the chain structure
    complex_chain.print_graph()

await visualize_chain()
```

### 2. Progress Tracking

```python
from toolboxv2.mods.isaa.base.Agent.types import ProgressEvent

class ChainProgressTracker:
    def __init__(self):
        self.events = []

    async def emit_event(self, event: ProgressEvent):
        self.events.append(event)
        print(f"[{event.event_type}] {event.node_name}: {event.status}")

async def tracked_chain():
    extractor = await isaa.get_agent("data_extractor")
    analyzer = await isaa.get_agent("analyzer")

    chain = extractor >> analyzer

    # Set up progress tracking
    tracker = ChainProgressTracker()
    chain.set_progress_callback(tracker)

    result = await chain.a_run("Process this data with tracking")

    print("\nProgress Events:")
    for event in tracker.events:
        print(f"  {event.timestamp}: {event.event_type} - {event.node_name}")

await tracked_chain()
```

## Best Practices

### 1. Agent Specialization

```python
async def specialized_agents_example():
    # Create highly specialized agents for better chain performance

    # JSON extractor
    json_extractor_builder = isaa.get_agent_builder("json_extractor")
    json_extractor_builder.with_system_message(
        "You extract data and return it in valid JSON format. Always use proper JSON syntax."
    )
    await isaa.register_agent(json_extractor_builder)

    # Validator agent
    validator_builder = isaa.get_agent_builder("validator")
    validator_builder.with_system_message(
        "You validate data for completeness and accuracy. Return 'valid' or 'invalid' with reasons."
    )
    await isaa.register_agent(validator_builder)

    # Clean chain with specialized agents
    extraction_chain = (
        await isaa.get_agent("json_extractor") >>
        CF(PersonData) >>
        await isaa.get_agent("validator") >>
        await isaa.get_agent("analyzer")
    )

    return extraction_chain
```

### 2. Error Recovery Patterns

```python
async def robust_chain_pattern():
    primary_agent = await isaa.get_agent("analyzer")
    backup_agent = await isaa.get_agent("data_extractor")

    # Multi-layer error recovery
    robust_chain = (
        primary_agent |                    # Try primary
        (backup_agent >> primary_agent) |  # Try backup + primary
        "Unable to process request"        # Final fallback
    )

    return robust_chain
```

### 3. Performance Optimization

```python
async def optimized_chain():
    # Use parallel processing for independent operations
    extractor = await isaa.get_agent("data_extractor")
    analyzer1 = await isaa.get_agent("analyzer")
    analyzer2 = await isaa.get_agent("reporter")  # Different analysis

    # Optimize: parallel analysis, then combine
    optimized = (
        extractor >>
        CF(PersonData) >>
        (analyzer1 + analyzer2) >>  # Parallel analysis
        await isaa.get_agent("summarizer")  # Combine results
    )

    return optimized
```

### Direckt call Support

- chain(...) → ruft sofort run
- await chain(...) → ruft automatisch a_run

## Chain Operators Reference

| Operator | Purpose | Example |
|----------|---------|---------|
| `>>` | Sequential execution | `agent1 >> agent2` |
| `+` | Parallel execution | `agent1 + agent2` |
| `&` | Parallel execution (alias) | `agent1 & agent2` |
| `%` | False branch in conditional | `condition >> true_branch % false_branch` |
| `\|` | Error handling fallback | `risky_agent \| safe_agent` |
| `-` | Data extraction | `CF(Model) - "field"` |

## CF (Chain Format) Reference

| Pattern | Purpose | Example |
|---------|---------|---------|
| `CF(Model)` | Format to Pydantic model | `CF(UserProfile)` |
| `CF(Model) - "field"` | Extract single field | `CF(User) - "name"` |
| `CF(Model) - ("f1", "f2")` | Extract multiple fields | `CF(User) - ("name", "age")` |
| `CF(Model) - "*"` | Extract all fields | `CF(User) - "*"` |
| `CF(Model) - "field[n]"` | Auto-parallel extraction | `CF(Tasks) - "tasks[n]"` |

This documentation provides comprehensive guidance for using the ISAA Chain system to create sophisticated AI agent workflows with powerful orchestration capabilities.
