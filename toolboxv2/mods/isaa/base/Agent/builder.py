import asyncio
import contextlib
import json
import logging
import os
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
    TypeVar,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# Framework Imports & Availability Checks (mirrored from agent.py)
try: from google.adk.agents import LlmAgent; ADK_AVAILABLE_BLD = True
except ImportError: LlmAgent = object; ADK_AVAILABLE_BLD = False # Need LlmAgent for isinstance check
try: from google.adk.tools import BaseTool, FunctionTool, AgentTool; from google.adk.tools.mcp_tool import MCPToolset, StdioServerParameters, SseServerParams; from google.adk.runners import Runner, InMemoryRunner, AsyncWebRunner; from google.adk.sessions import SessionService, InMemorySessionService; from google.adk.code_executors import BaseCodeExecutor as ADKBaseCodeExecutor; from google.adk.planners import BasePlanner; from google.adk.examples import Example
except ImportError: BaseTool = object; FunctionTool = object; AgentTool = object; MCPToolset = object; Runner = object; InMemoryRunner = object; AsyncWebRunner = object; SessionService = object; InMemorySessionService = object; ADKBaseCodeExecutor = object; BasePlanner = object; Example = object; StdioServerParameters = object; SseServerParams = object
try: from python_a2a.server import A2AServer; from python_a2a.models import AgentCard; A2A_AVAILABLE_BLD = True
except ImportError: A2AServer = object; AgentCard = object; A2A_AVAILABLE_BLD = False
try: from mcp.server.fastmcp import FastMCP; MCP_AVAILABLE_BLD = True
except ImportError: FastMCP = object; MCP_AVAILABLE_BLD = False
try: from litellm import BudgetManager; LITELLM_AVAILABLE_BLD = True
except ImportError: BudgetManager = object; LITELLM_AVAILABLE_BLD = False

# --- Framework Imports & Availability Checks (Copied from EnhancedAgent) ---
# Google ADK
try:
    from google.adk.agents import BaseAgent, LlmAgent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.code_executors import BaseCodeExecutor
    from google.adk.code_executors.code_execution_utils import (
        CodeExecutionInput,
        CodeExecutionResult,
    )
    from google.adk.events import Event
    from google.adk.examples import Example  # For few-shot
    from google.adk.models import BaseLlm, Gemini
    from google.adk.models.lite_llm import LiteLlm  # ADK Wrapper for LiteLLM
    from google.adk.planners import BasePlanner
    from google.adk.runners import (  # Base SessionService
        BaseSessionService,
        InMemoryRunner,
        Runner,
    )
    from google.adk.sessions import Session, State
    from google.adk.tools import (
        BaseTool,
        FunctionTool,
        LongRunningFunctionTool,
        ToolContext,
    )
    from google.adk.tools import VertexAiSearchTool as AdkVertexAiSearchTool
    from google.adk.tools import (
        built_in_code_execution as adk_built_in_code_execution,  # Secure option
    )
    from google.adk.tools import google_search as adk_google_search
    from google.adk.tools.agent_tool import AgentTool
    from google.adk.tools.mcp_tool.mcp_toolset import (
        MCPToolset,
        SseServerParams,
        StdioServerParameters,
    )
    from google.genai.types import Content, FunctionCall, FunctionResponse, Part

    ADK_AVAILABLE = True
    ADKBaseCodeExecutor = BaseCodeExecutor # Alias for clarity in builder
    ADKRunner = Runner
    ADKSessionService = BaseSessionService
    ADKBaseTool = BaseTool
    ADKFunctionTool = FunctionTool
    ADKExample = Example
    ADKPlanner = BasePlanner
    ADKLlmAgent = LlmAgent
except ImportError:
    ADK_AVAILABLE = False
    # Define dummy types for type hinting if ADK is not installed
    class ADKBaseCodeExecutor: pass
    class ADKRunner: pass
    class ADKSessionService: pass
    class ADKBaseTool: pass
    class ADKFunctionTool: pass
    class ADKExample: pass
    class ADKPlanner: pass
    class ADKLlmAgent: pass # Use basic object if LlmAgent isn't available for inheritance checks
    StdioServerParameters = object
    SseServerParams = object
    MCPToolset = object # Dummy
    LlmAgent = object # Dummy for isinstance checks if EnhancedAgent itself needs it


# python-a2a
try:
    from python_a2a import A2AClient, A2AServer, AgentCard
    from python_a2a import run_server as run_a2a_server_func
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    class A2AServer: pass
    class A2AClient: pass
    class AgentCard: pass
    def run_a2a_server_func(*a, **kw):
        return None


# MCP
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    class FastMCP: pass


# LiteLLM
try:
    import litellm
    from litellm import BudgetManager
    from litellm.utils import get_max_tokens
    LITELLM_AVAILABLE = True
except ImportError:
    print("CRITICAL ERROR: LiteLLM not found. Agent functionality will be severely limited.")
    LITELLM_AVAILABLE = False
    class BudgetManager: pass
    def get_max_tokens(*a, **kw):
        return 4096 # Dummy fallback

# OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    class TracerProvider: pass # Dummy

# --- Local Imports ---
# Assume EnhancedAgent and supporting classes (WorldModel, AgentModelData, etc.)
# are in the same directory or properly importable
from toolboxv2.mods.isaa.base.Agent.agent import (
    A2A_AVAILABLE as AGENT_A2A_AVAILABLE,  # Check agent's view
)
from toolboxv2.mods.isaa.base.Agent.agent import (
    MCP_AVAILABLE as AGENT_MCP_AVAILABLE,  # Check agent's view
)
from toolboxv2.mods.isaa.base.Agent.agent import (  # Relative import assuming builder is in same dir/package
    AgentModelData,
    EnhancedAgent,
    SecureCodeExecutorPlaceholder,
    UnsafeSimplePythonExecutor,
)

# Local Imports

logger = logging.getLogger("EnhancedAgentBuilder")

T = TypeVar('T', bound='EnhancedAgent') # Type variable for the agent being built


# --- User Cost Tracking ---

class UserCostTracker(Protocol):
    """Protocol for tracking costs per user."""
    def get_cost(self, user_id: str) -> float: ...
    def add_cost(self, user_id: str, cost: float) -> None: ...
    def get_all_costs(self) -> dict[str, float]: ...
    def save(self) -> None: ...
    def load(self) -> None: ...

class JsonFileUserCostTracker:
    """Stores user costs persistently in a JSON file."""
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self._costs: dict[str, float] = {}
        self._lock = threading.Lock()
        self.load() # Load costs on initialization

    def get_cost(self, user_id: str) -> float:
        with self._lock:
            return self._costs.get(user_id, 0.0)

    def add_cost(self, user_id: str, cost: float) -> None:
        if not user_id:
            logger.warning("Cost tracking skipped: user_id is missing.")
            return
        if cost > 0:
            with self._lock:
                self._costs[user_id] = self._costs.get(user_id, 0.0) + cost
                logger.debug(f"Cost added for user '{user_id}': +{cost:.6f}. New total: {self._costs[user_id]:.6f}")
            # Optional: Auto-save periodically or based on number of updates
            # For simplicity, we rely on explicit save() or agent close

    def get_all_costs(self) -> dict[str, float]:
        with self._lock:
            return self._costs.copy()

    def save(self) -> None:
        with self._lock:
            try:
                self.filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(self.filepath, 'w') as f:
                    json.dump(self._costs, f, indent=2)
                logger.info(f"User costs saved to {self.filepath}")
            except OSError as e:
                logger.error(f"Failed to save user costs to {self.filepath}: {e}")

    def load(self) -> None:
        with self._lock:
            if self.filepath.exists():
                try:
                    with open(self.filepath) as f:
                        self._costs = json.load(f)
                    logger.info(f"User costs loaded from {self.filepath}")
                except (OSError, json.JSONDecodeError) as e:
                    logger.error(f"Failed to load user costs from {self.filepath}: {e}. Starting fresh.")
                    self._costs = {}
            else:
                logger.info(f"User cost file not found ({self.filepath}). Starting fresh.")
                self._costs = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()


# --- Builder Configuration Model ---

class BuilderADKConfig(BaseModel):
    enabled: bool = False
    description: str | None = None
    runner_class_name: str | None = Field(default="InMemoryRunner", description="ADK Runner class name (e.g., 'InMemoryRunner', 'AsyncWebRunner')")
    runner_options: dict[str, Any] = Field(default_factory=dict)
    code_executor_config: Literal["adk_builtin", "unsafe_simple", "secure_placeholder", "custom_instance", "none"] | dict = "none"
    # Custom instance requires passing instance during build, dict allows config for future executors
    sync_state: bool = Field(default=False, description="Sync WorldModel <-> ADK Session State")
    mcp_toolset_configs: list[dict[str, Any]] = Field(default_factory=list, description="Configs for ADK MCPToolset (e.g., {'type': 'stdio', 'command': '...', 'args': []})")
    planner_config: dict[str, Any] | None = None # For future planner config
    examples: list[dict[str, Any]] | None = None # For few-shot examples (ADK Example format)
    output_schema: dict[str, Any] | None = None # For structured output hints

    model_config = ConfigDict(extra='ignore')

    @field_validator('runner_class_name')
    def check_runner_class(cls, v):
        # Basic check for known runner types
        known_runners = {"InMemoryRunner","Runner", "AsyncWebRunner"} # Add more as needed
        if v not in known_runners:
            logger.warning(f"ADK Runner class '{v}' not in known list {known_runners}. Ensure it's importable.")
        return v

class BuilderServerConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 0 # Placeholder, specific defaults below
    extra_options: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra='ignore')

class BuilderA2AConfig(BuilderServerConfig):
    port: int = 5000 # Default A2A port
    known_clients: dict[str, str] = Field(default_factory=dict, description="Map name -> URL for A2A clients")

class BuilderMCPConfig(BuilderServerConfig):
    port: int = 8000 # Default MCP port
    server_name: str | None = None

class BuilderHistoryConfig(BaseModel):
    max_turns: int | None = 20
    max_tokens: int | None = None # Takes precedence over turns if set
    trim_strategy: Literal["litellm", "basic"] = "litellm"

    model_config = ConfigDict(extra='ignore')

class BuilderConfig(BaseModel):
    """Serializable configuration state for the EnhancedAgentBuilder."""
    agent_name: str = "UnnamedEnhancedAgent"
    agent_version: str = "0.1.0"

    # Core Model Config (Subset of AgentModelData, as some are instance-specific like BudgetManager)
    model_identifier: str | None = None
    formatter_llm_model: str | None = None
    system_message: str = "You are a helpful AI assistant."
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_tokens_output: int | None = None # Max tokens for LLM *generation*
    max_tokens_input: int | None = None # Max context window (for trimming)
    api_key_env_var: str | None = None # Store env var name, not the key itself
    api_base: str | None = None
    api_version: str | None = None
    stop_sequence: list[str] | None = None
    llm_user_id: str | None = None # 'user' param for LLM calls
    enable_litellm_caching: bool = True

    # Agent Behavior
    enable_streaming: bool = False
    verbose_logging: bool = False
    world_model_initial_data: dict[str, Any] | None = None
    history: BuilderHistoryConfig = Field(default_factory=BuilderHistoryConfig)

    # Framework Integrations
    adk: BuilderADKConfig = Field(default_factory=BuilderADKConfig)
    a2a: BuilderA2AConfig = Field(default_factory=BuilderA2AConfig)
    mcp: BuilderMCPConfig = Field(default_factory=BuilderMCPConfig)

    # Cost Tracking (Configuration for persistence)
    cost_tracker_config: dict[str, Any] | None = Field(default={'type': 'json', 'filepath': './user_costs.json'}, description="Config for UserCostTracker (e.g., type, path)")

    # Observability (Configuration)
    telemetry_config: dict[str, Any] | None = Field(default={'enabled': False, 'service_name': None, 'endpoint': None}, description="Basic OTel config hints")

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode='after')
    def _resolve_names(self) -> 'BuilderConfig':
        # Ensure service name defaults to agent name if not set
        if self.telemetry_config and self.telemetry_config.get('enabled') and not self.telemetry_config.get('service_name'):
            self.telemetry_config['service_name'] = self.agent_name
        # Ensure MCP server name defaults if not set
        if self.mcp.enabled and not self.mcp.server_name:
             self.mcp.server_name = f"{self.agent_name}_MCPServer"
        return self


# --- Production Builder Class ---


class EnhancedAgentBuilder:
    """
    Fluent builder for configuring and constructing production-ready EnhancedAgent instances.
    Supports loading configuration from files and provides methods for detailed setup.
    """

    def __init__(self,agent_name: str = "DefaultAgent", config: BuilderConfig | None = None, config_path: str | Path | None = None):
        """
        Initialize the builder. Can start with a config object, path, or blank.

        Args:
            config: An existing BuilderConfig object.
            config_path: Path to a YAML/JSON configuration file for the builder.
        """
        if config and config_path:
            raise ValueError("Provide either config object or config_path, not both.")

        if config_path:
            self.load_config(config_path) # Sets self._config
        elif config:
            self._config = config.copy(deep=True)
        else:
            self._config = BuilderConfig() # Start with defaults

        # --- Transient fields (not saved/loaded directly via BuilderConfig JSON) ---
        # Instances or non-serializable objects provided programmatically.
        self._adk_tools_transient: list[ADKBaseTool | Callable] = []
        self._adk_code_executor_instance: ADKBaseCodeExecutor | None = None
        self._adk_runner_instance: ADKRunner | None = None
        self._adk_session_service_instance: ADKSessionService | None = None
        self._adk_planner_instance: ADKPlanner | None = None
        self._litellm_budget_manager_instance: BudgetManager | None = None
        self._user_cost_tracker_instance: UserCostTracker | None = None
        self._otel_trace_provider_instance: TracerProvider | None = None
        self._callbacks_transient: dict[str, Callable] = {}
        # Pre-initialized server instances (less common, but possible)
        self._a2a_server_instance: A2AServer | None = None
        self._mcp_server_instance: FastMCP | None = None

        # Set initial log level based on loaded config
        logger.setLevel(logging.DEBUG if self._config.verbose_logging else logging.INFO)
        self.with_agent_name(agent_name)

    # --- Configuration Save/Load ---

    def save_config(self, path: str | Path, indent: int = 2):
        """Saves the current builder configuration to a JSON file."""
        filepath = Path(path)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            config_json = self._config.model_dump_json(indent=indent)
            with open(filepath, 'w') as f:
                f.write(config_json)
            logger.info(f"Builder configuration saved to {filepath}")
        except OSError as e:
            logger.error(f"Failed to save builder configuration to {filepath}: {e}")
        except ValidationError as e:
             logger.error(f"Configuration is invalid, cannot save: {e}")
        except Exception as e:
             logger.error(f"An unexpected error occurred during config save: {e}")


    def load_config(self, path: str | Path) -> 'EnhancedAgentBuilder':
        """Loads builder configuration from a JSON file, overwriting current settings."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Builder configuration file not found: {filepath}")
        try:
            with open(filepath) as f:
                config_data = json.load(f)
            self._config = BuilderConfig.model_validate(config_data)
            logger.info(f"Builder configuration loaded from {filepath}")
            # Reset transient fields, as they are not saved
            self._reset_transient_fields()
            logger.warning("Transient fields (callbacks, tool instances, tracker instance, etc.) reset. Re-add them if needed.")
            # Update logger level based on loaded config
            logger.setLevel(logging.DEBUG if self._config.verbose_logging else logging.INFO)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load or parse builder configuration from {filepath}: {e}")
            raise
        except ValidationError as e:
             logger.error(f"Loaded configuration data is invalid: {e}")
             raise
        return self

    def _reset_transient_fields(self):
        """Clears fields that are not part of the saved BuilderConfig."""
        self._adk_tools_transient = []
        self._adk_code_executor_instance = None
        self._adk_runner_instance = None
        self._adk_session_service_instance = None
        self._adk_planner_instance = None
        self._litellm_budget_manager_instance = None
        self._user_cost_tracker_instance = None
        self._otel_trace_provider_instance = None
        self._callbacks_transient = {}
        self._a2a_server_instance = None
        self._mcp_server_instance = None

    # --- Fluent Configuration Methods (Modify self._config) ---

    def with_agent_name(self, name: str) -> 'EnhancedAgentBuilder':
        self._config.agent_name = name
        # Update dependent defaults
        self._config = BuilderConfig.model_validate(self._config.model_dump())
        return self

    def with_agent_version(self, version: str) -> 'EnhancedAgentBuilder':
        self._config.agent_version = version
        return self

    def with_model(self, model_identifier: str) -> 'EnhancedAgentBuilder':
        self._config.model_identifier = model_identifier
        # Auto-detect context window if not set
        if not self._config.max_tokens_input:
            try:
                max_input = get_max_tokens(model_identifier)
                if max_input:
                    self._config.max_tokens_input = max_input
                    logger.info(f"Auto-detected max_input_tokens for {model_identifier}: {max_input}")
                else:
                     # Default fallback if detection fails
                    self._config.max_tokens_input = 4096
                    logger.warning(f"Could not auto-detect max_input_tokens for {model_identifier}, defaulting to 4096.")
            except Exception as e:
                 self._config.max_tokens_input = 4096
                 logger.warning(f"Error auto-detecting max_input_tokens ({e}), defaulting to 4096.")
        # Auto-configure Ollama base URL
        if 'ollama/' in model_identifier and not self._config.api_base:
            self.with_api_base("http://localhost:11434") # Uses the method to log
        return self

    def with_system_message(self, message: str) -> 'EnhancedAgentBuilder':
        self._config.system_message = message
        return self

    def with_temperature(self, temp: float) -> 'EnhancedAgentBuilder':
        self._config.temperature = temp
        return self

    def with_max_output_tokens(self, tokens: int) -> 'EnhancedAgentBuilder':
        self._config.max_tokens_output = tokens
        return self

    def with_max_input_tokens(self, tokens: int) -> 'EnhancedAgentBuilder':
        self._config.max_tokens_input = tokens
        return self

    def with_stop_sequence(self, stop: list[str]) -> 'EnhancedAgentBuilder':
        self._config.stop_sequence = stop
        return self

    def with_api_key_from_env(self, env_var_name: str) -> 'EnhancedAgentBuilder':
        self._config.api_key_env_var = env_var_name
        # Quick check if env var exists
        if not os.getenv(env_var_name):
            logger.warning(f"API key environment variable '{env_var_name}' is not set.")
        return self

    def with_api_base(self, base_url: str | None) -> 'EnhancedAgentBuilder':
        self._config.api_base = base_url
        logger.info(f"API base set to: {base_url}")
        return self

    def with_api_version(self, version: str | None) -> 'EnhancedAgentBuilder':
        self._config.api_version = version
        return self

    def with_llm_user_id(self, user_id: str) -> 'EnhancedAgentBuilder':
        self._config.llm_user_id = user_id
        return self

    def enable_litellm_caching(self, enable: bool = True) -> 'EnhancedAgentBuilder':
        self._config.enable_litellm_caching = enable
        return self

    def enable_streaming(self, enable: bool = True) -> 'EnhancedAgentBuilder':
        self._config.enable_streaming = enable
        return self

    def verbose(self, enable: bool = True) -> 'EnhancedAgentBuilder':
        self._config.verbose_logging = enable
        logger.setLevel(logging.DEBUG if enable else logging.INFO)
        os.environ['LITELLM_LOG'] = 'DEBUG' if enable else 'NONE' # Control LiteLLM verbosity too
        return self
    def formatter_llm_model(self, model: str) -> 'EnhancedAgentBuilder':
        self._config.formatter_llm_model = model
        return self

    def with_initial_world_data(self, data: dict[str, Any]) -> 'EnhancedAgentBuilder':
        self._config.world_model_initial_data = data
        return self

    def with_history_options(self, max_turns: int | None = 20, max_tokens: int | None = None, trim_strategy: Literal["litellm", "basic"] = "litellm") -> 'EnhancedAgentBuilder':
        self._config.history = BuilderHistoryConfig(max_turns=max_turns, max_tokens=max_tokens, trim_strategy=trim_strategy)
        return self

    # --- ADK Configuration Methods ---
    def _ensure_adk(self, feature: str):
        if not ADK_AVAILABLE:
            logger.warning(f"ADK not available. Cannot configure ADK feature: {feature}.")
            return False
        self._config.adk.enabled = True # Mark ADK as enabled if any ADK feature is used
        return True

    def enable_adk(self, runner_class: type[ADKRunner] = InMemoryRunner, runner_options: dict[str, Any] | None = None) -> 'EnhancedAgentBuilder':
        """Enables ADK integration with a specified runner."""
        if not self._ensure_adk("Runner"): return self
        self._config.adk.runner_class_name = runner_class.__name__
        self._config.adk.runner_options = runner_options or {}
        logger.info(f"ADK integration enabled with runner: {self._config.adk.runner_class_name}")
        return self

    def with_adk_description(self, description: str) -> 'EnhancedAgentBuilder':
        if not self._ensure_adk("Description"): return self
        self._config.adk.description = description
        return self

    def with_adk_tool_instance(self, tool: ADKBaseTool) -> 'EnhancedAgentBuilder':
        """Adds a pre-initialized ADK Tool instance (transient)."""
        if not self._ensure_adk("Tool Instance"): return self
        if not isinstance(tool, ADKBaseTool):
            raise TypeError(f"Expected ADK BaseTool instance, got {type(tool)}")
        self._adk_tools_transient.append(tool)
        return self

    def with_adk_tool_function(self, func: Callable) -> 'EnhancedAgentBuilder':
        """Adds a callable function as an ADK tool (transient)."""
        if not self._ensure_adk("Tool Function"): return self
        if not callable(func):
            raise TypeError(f"Expected callable function for ADK tool, got {type(func)}")
        self._adk_tools_transient.append(func)
        return self

    def with_adk_mcp_toolset(self, connection_type: Literal["stdio", "sse"], **kwargs) -> 'EnhancedAgentBuilder':
        """Configures an ADK MCP Toolset connection (saved in config)."""
        if not self._ensure_adk("MCP Toolset"): return self
        if connection_type == "stdio":
            if "command" not in kwargs: raise ValueError("Stdio MCP toolset requires 'command' argument.")
            config = {"type": "stdio", "command": kwargs["command"], "args": kwargs.get("args", [])}
        elif connection_type == "sse":
            if "url" not in kwargs: raise ValueError("SSE MCP toolset requires 'url' argument.")
            config = {"type": "sse", "url": kwargs["url"]}
        else:
            raise ValueError(f"Unknown MCP toolset connection type: {connection_type}")
        self._config.adk.mcp_toolset_configs.append(config)
        logger.info(f"Configured ADK MCP Toolset: {config}")
        return self

    def with_adk_code_executor(self, executor_type: Literal["adk_builtin", "unsafe_simple", "secure_placeholder", "none"]) -> 'EnhancedAgentBuilder':
        """Configures the type of ADK code executor to use (saved in config)."""
        if not self._ensure_adk("Code Executor Type"): return self
        if executor_type == "unsafe_simple":
            logger.critical("***********************************************************")
            logger.critical("*** WARNING: Configuring UNSAFE SimplePythonExecutor!   ***")
            logger.critical("***********************************************************")
        elif executor_type == "secure_placeholder":
            logger.warning("Configuring SecureCodeExecutorPlaceholder. Implement actual sandboxing!")
        elif executor_type == "adk_builtin":
            if self._config.model_identifier and ("gemini-1.5" not in self._config.model_identifier and "gemini-2" not in self._config.model_identifier) :
                logger.warning(f"ADK built-in code execution selected, but model '{self._config.model_identifier}' might not support it. Ensure model compatibility.")
            logger.info("Configuring ADK built-in code execution (tool-based, requires compatible model).")

        self._config.adk.code_executor_config = executor_type
        self._adk_code_executor_instance = None # Clear any previously set instance
        return self

    def with_adk_code_executor_instance(self, executor: ADKBaseCodeExecutor) -> 'EnhancedAgentBuilder':
        """Provides a pre-initialized ADK code executor instance (transient)."""
        if not self._ensure_adk("Code Executor Instance"): return self
        if not isinstance(executor, ADKBaseCodeExecutor):
            raise TypeError(f"Expected ADKBaseCodeExecutor instance, got {type(executor)}")
        self._adk_code_executor_instance = executor
        self._config.adk.code_executor_config = "custom_instance" # Mark config
        logger.info(f"Using custom ADK code executor instance: {type(executor).__name__}")
        return self

    def enable_adk_state_sync(self, enable: bool = True) -> 'EnhancedAgentBuilder':
        if not self._ensure_adk("State Sync"): return self
        self._config.adk.sync_state = enable
        return self

    # --- Server Configuration Methods ---
    def enable_a2a_server(self, host: str = "0.0.0.0", port: int = 5000, **extra_options) -> 'EnhancedAgentBuilder':
        if not A2A_AVAILABLE:
            logger.warning("python-a2a library not available. Cannot enable A2A server.")
            self._config.a2a.enabled = False
            return self
        self._config.a2a.enabled = True
        self._config.a2a.host = host
        self._config.a2a.port = port
        self._config.a2a.extra_options = extra_options
        return self

    def add_a2a_known_client(self, name: str, url: str) -> 'EnhancedAgentBuilder':
        if not A2A_AVAILABLE:
            logger.warning("python-a2a library not available. Cannot add known A2A client.")
            return self
        # A2A client setup is handled by the agent itself, we just store the config
        self._config.a2a.known_clients[name] = url
        logger.info(f"Added known A2A client config: '{name}' -> {url}")
        return self

    def enable_mcp_server(self, host: str = "0.0.0.0", port: int = 8000, server_name: str | None = None, **extra_options) -> 'EnhancedAgentBuilder':
         if not MCP_AVAILABLE:
             logger.warning("MCP library (FastMCP) not available. Cannot enable MCP server.")
             self._config.mcp.enabled = False
             return self
         self._config.mcp.enabled = True
         self._config.mcp.host = host
         self._config.mcp.port = port
         self._config.mcp.server_name = server_name # Will default later if None
         self._config.mcp.extra_options = extra_options
         # Re-validate to update default name if needed
         self._config = BuilderConfig.model_validate(self._config.model_dump())
         return self

    # --- Cost Tracking & Budgeting Methods ---
    def with_cost_tracker(self, tracker: UserCostTracker) -> 'EnhancedAgentBuilder':
        """Provides a pre-initialized UserCostTracker instance (transient)."""
        if not hasattr(tracker, "get_all_costs"): # Check protocol using isinstance
             raise TypeError("Cost tracker must implement the UserCostTracker protocol.")
        self._user_cost_tracker_instance = tracker
        # Clear file config if instance is provided
        self._config.cost_tracker_config = {'type': 'custom_instance'}
        logger.info(f"Using custom UserCostTracker instance: {type(tracker).__name__}")
        return self

    def with_json_cost_tracker(self, filepath: str | Path) -> 'EnhancedAgentBuilder':
        """Configures the builder to use the JsonFileUserCostTracker (saved in config)."""
        self._config.cost_tracker_config = {'type': 'json', 'filepath': str(filepath)}
        self._user_cost_tracker_instance = None # Clear any instance
        logger.info(f"Configured JsonFileUserCostTracker: {filepath}")
        return self

    def with_litellm_budget_manager(self, manager: BudgetManager) -> 'EnhancedAgentBuilder':
        """Provides a pre-initialized LiteLLM BudgetManager instance (transient)."""
        if not LITELLM_AVAILABLE:
             logger.warning("LiteLLM not available, cannot set BudgetManager.")
             return self
        if not isinstance(manager, BudgetManager):
            raise TypeError("Expected litellm.BudgetManager instance.")
        self._litellm_budget_manager_instance = manager
        return self

    # --- Observability Methods ---
    def enable_telemetry(self, service_name: str | None = None, endpoint: str | None = None) -> 'EnhancedAgentBuilder':
         if not OTEL_AVAILABLE:
              logger.warning("OpenTelemetry SDK not available. Cannot enable telemetry.")
              self._config.telemetry_config = {'enabled': False}
              return self
         self._config.telemetry_config = {
             'enabled': True,
             'service_name': service_name, # Defaults to agent name later
             'endpoint': endpoint # For OTLP exporter, e.g. "http://localhost:4317"
         }
         # Re-validate to update default name if needed
         self._config = BuilderConfig.model_validate(self._config.model_dump())
         return self

    def with_telemetry_provider_instance(self, provider: TracerProvider) -> 'EnhancedAgentBuilder':
        """Provides a pre-initialized OpenTelemetry TracerProvider instance (transient)."""
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry SDK not available. Cannot set TracerProvider.")
            return self
        if not isinstance(provider, TracerProvider):
             raise TypeError("Expected opentelemetry.sdk.trace.TracerProvider instance.")
        self._otel_trace_provider_instance = provider
        # Mark telemetry as enabled, but using custom instance
        self._config.telemetry_config = {'enabled': True, 'type': 'custom_instance'}
        logger.info("Using custom OpenTelemetry TracerProvider instance.")
        return self

    # --- Callback Methods (Transient) ---
    def with_stream_callback(self, func: Callable[[str], None | Awaitable[None]]) -> 'EnhancedAgentBuilder':
        self._callbacks_transient['stream_callback'] = func; return self
    def with_post_run_callback(self, func: Callable[[str, str, float, str | None], None | Awaitable[None]]) -> 'EnhancedAgentBuilder':
        self._callbacks_transient['post_run_callback'] = func; return self # Added user_id
    def with_progress_callback(self, func: Callable[[Any], None | Awaitable[None]]) -> 'EnhancedAgentBuilder':
        self._callbacks_transient['progress_callback'] = func; return self
    def with_human_in_loop_callback(self, func: Callable[[dict], str | Awaitable[str]]) -> 'EnhancedAgentBuilder':
        self._callbacks_transient['human_in_loop_callback'] = func; return self

    # --- Build Method ---
    async def build(self) -> EnhancedAgent:
        """
        Constructs and returns the configured EnhancedAgent instance.
        Handles asynchronous setup like fetching ADK MCP tools.
        """
        logger.info(f"--- Building EnhancedAgent: {self._config.agent_name} v{self._config.agent_version} ---")

        # 1. Final Config Validation (Pydantic model handles most)
        if not self._config.model_identifier:
            raise ValueError("LLM model identifier is required. Use .with_model()")

        # 2. Resolve API Key
        api_key = None
        if self._config.api_key_env_var:
            api_key = os.getenv(self._config.api_key_env_var)
            if not api_key:
                logger.warning(f"API key environment variable '{self._config.api_key_env_var}' is set in config but not found in environment.")
            # else: logger.debug("API key loaded from environment variable.") # Avoid logging key presence

        # 3. Setup Telemetry Provider (if instance provided)
        if self._otel_trace_provider_instance and OTEL_AVAILABLE:
            trace.set_tracer_provider(self._otel_trace_provider_instance)
            logger.info("Global OpenTelemetry TracerProvider set from provided instance.")
        elif self._config.telemetry_config.get('enabled') and self._config.telemetry_config.get('type') != 'custom_instance' and OTEL_AVAILABLE:
             # Basic provider setup from config (can be expanded)
             logger.info("Setting up basic OpenTelemetry based on config (ConsoleExporter example).")
             from opentelemetry.sdk.trace.export import (
                 BatchSpanProcessor,
                 ConsoleSpanExporter,
             )
             provider = TracerProvider()
             provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
             #: Add OTLP exporter based on self._config.telemetry_config['endpoint']
             trace.set_tracer_provider(provider)
             self._otel_trace_provider_instance = provider # Store for potential access?

        # 4. Prepare Core Components
        # Agent Model Data
        try:
            amd = AgentModelData(
                name=self._config.agent_name,
                model=self._config.model_identifier,
                system_message=self._config.system_message,
                temperature=self._config.temperature,
                top_k=self._config.top_k,
                top_p=self._config.top_p,
                max_tokens=self._config.max_tokens_output,
                max_input_tokens=self._config.max_tokens_input,
                api_key=api_key,
                api_base=self._config.api_base,
                api_version=self._config.api_version,
                stop_sequence=self._config.stop_sequence,
                user_id=self._config.llm_user_id,
                budget_manager=self._litellm_budget_manager_instance,
                caching=self._config.enable_litellm_caching
            )
        except ValidationError as e:
            logger.error(f"Validation error creating AgentModelData: {e}")
            raise

        # World Model
        world_model = self._config.world_model_initial_data or {}

        # User Cost Tracker
        cost_tracker = self._user_cost_tracker_instance # Use provided instance if available
        if not cost_tracker and self._config.cost_tracker_config:
            tracker_type = self._config.cost_tracker_config.get('type')
            if tracker_type == 'json':
                filepath = self._config.cost_tracker_config.get('filepath')
                if filepath:
                    cost_tracker = JsonFileUserCostTracker(filepath)
                    logger.info(f"Initialized JsonFileUserCostTracker ({filepath})")
                else:
                    logger.warning("JSON cost tracker configured but filepath missing.")
            elif tracker_type == 'custom_instance':
                 logger.warning("Cost tracker configured as 'custom_instance' but no instance was provided via .with_cost_tracker().")
            # Add other tracker types (DB, InMemory) here

        # 5. Prepare ADK Components
        adk_runner_instance = self._adk_runner_instance
        adk_session_service = self._adk_session_service_instance
        adk_planner_instance = self._adk_planner_instance
        adk_code_executor = self._adk_code_executor_instance # Use provided instance first
        adk_exit_stack = None
        processed_adk_tools = list(self._adk_tools_transient) # Start with transient tools

        if ADK_AVAILABLE and self._config.adk.enabled:
            logger.info("Configuring ADK components...")
            adk_exit_stack = contextlib.AsyncExitStack()

            # --- ADK Runner & Session Service ---
            if not adk_runner_instance:
                runner_cls_name = self._config.adk.runner_class_name
                runner_opts = self._config.adk.runner_options
                try:
                    # Dynamically import/get runner class
                    if runner_cls_name == "InMemoryRunner": runner_class = InMemoryRunner
                    elif runner_cls_name == "Runner": runner_class = Runner
                    elif runner_cls_name == "AsyncWebRunner": runner_class = AsyncWebRunner # If available
                    else: raise ValueError(f"Unsupported ADK Runner class name: {runner_cls_name}")

                    # Special handling: InMemoryRunner needs agent instance *later*
                    if runner_class is InMemoryRunner or runner_class is Runner:
                         logger.debug("Deferring InMemoryRunner creation until after agent instantiation.")
                         # Store config to create it later
                         adk_runner_config_for_later = {
                             "runner_class": runner_class,
                             "app_name": runner_opts.get("app_name", f"{self._config.agent_name}_ADKApp"),
                             "session_service": adk_session_service, # Pass service if already created
                             **runner_opts # Pass other options
                         }
                         adk_runner_instance = None # Ensure it's None for now
                    else: # Other runners might be creatable now
                         # Need to ensure session service is handled correctly if runner needs it
                         if not adk_session_service:
                             # Create default session service if needed by runner
                             # This part is complex as runners might create their own
                             logger.info("Using default ADK InMemorySessionService for runner.")
                             adk_session_service = InMemorySessionService()

                         adk_runner_instance = runner_class(
                             session_service=adk_session_service,
                             app_name=runner_opts.get("app_name", f"{self._config.agent_name}_ADKApp"),
                             **runner_opts # Pass other options
                         )
                         logger.info(f"Created ADK Runner instance: {runner_cls_name}")

                except (ImportError, ValueError, TypeError) as e:
                    logger.error(f"Failed to configure ADK Runner '{runner_cls_name}': {e}", exc_info=True)
                    raise ValueError(f"Failed to setup ADK Runner: {e}") from e

            # Ensure session service exists if runner created one
            if adk_runner_instance and hasattr(adk_runner_instance, 'session_service'):
                 if not adk_session_service:
                     adk_session_service = adk_runner_instance.session_service
                 elif adk_session_service is not adk_runner_instance.session_service:
                     logger.warning("Provided ADK SessionService differs from the one in the provided ADK Runner. Using the runner's service.")
                     adk_session_service = adk_runner_instance.session_service

            # Fallback: create default session service if none exists by now
            if not adk_session_service:
                  logger.info("Using default ADK InMemorySessionService.")
                  adk_session_service = InMemorySessionService()


            # --- ADK Code Executor ---
            if not adk_code_executor: # If instance wasn't provided directly
                executor_config = self._config.adk.code_executor_config
                if executor_config == "unsafe_simple":
                    adk_code_executor = UnsafeSimplePythonExecutor()
                    logger.critical("UNSAFE code executor instance created!")
                elif executor_config == "secure_placeholder":
                    adk_code_executor = SecureCodeExecutorPlaceholder()
                    logger.warning("SecureCodeExecutorPlaceholder instance created.")
                elif executor_config == "adk_builtin":
                    # This type uses the TOOL, not an executor instance passed to LlmAgent init
                    adk_code_executor = adk_built_in_code_execution
                    #if not any(getattr(t, 'func', None) == tool_func for t in processed_adk_tools if isinstance(t, FunctionTool)):
                    #     tool_func.__name__ = "code_execution"
                    # processed_adk_tools.append(tool_func)
                    #     logger.info("Added ADK built-in code execution tool.")
                    adk_code_executor = None # Ensure no executor instance is passed for this case
                elif executor_config == "none":
                    adk_code_executor = None
                elif executor_config == "custom_instance":
                    # Should have been provided via .with_adk_code_executor_instance()
                    logger.error("ADK code executor configured as 'custom_instance' but no instance was provided.")
                    adk_code_executor = None
                # Add handling for dict config if needed in the future

            # --- ADK Tools (Wrap callables) ---
            temp_tools = []
            for tool_input in processed_adk_tools:
                 if isinstance(tool_input, ADKBaseTool):
                     temp_tools.append(tool_input)
                 elif callable(tool_input):
                     try:
                         wrapped = ADKFunctionTool(func=tool_input)
                         temp_tools.append(wrapped)
                     except Exception as e: logger.warning(f"Could not wrap callable '{getattr(tool_input, '__name__', 'unknown')}' as ADK tool: {e}")
                 else: logger.warning(f"Skipping invalid ADK tool input: {type(tool_input)}")
            processed_adk_tools = temp_tools

            # --- ADK MCP Toolsets ---
            for mcp_conf in self._config.adk.mcp_toolset_configs:
                 logger.info(f"Fetching tools from configured MCP Server: {mcp_conf}...")
                 try:
                      params = None
                      if mcp_conf.get("type") == "stdio":
                          params = StdioServerParameters(command=mcp_conf["command"], args=mcp_conf.get("args", []))
                      elif mcp_conf.get("type") == "sse":
                           params = SseServerParams(url=mcp_conf["url"])

                      if params:
                          mcp_tools, _ = await MCPToolset.from_server(
                              connection_params=params,
                              async_exit_stack=adk_exit_stack
                          )
                          for tool in mcp_tools: tool._is_mcp_tool = True
                          processed_adk_tools.extend(mcp_tools)
                          logger.info(f"Fetched {len(mcp_tools)} tools via ADK MCPToolset ({mcp_conf.get('type')}).")
                      else:
                           logger.warning(f"Unsupported MCP config type: {mcp_conf.get('type')}")

                 except Exception as e:
                      logger.error(f"Failed to fetch tools from MCP server {mcp_conf}: {e}", exc_info=True)
                      # Decide whether to raise or continue

            # --- ADK Planner, Examples, Output Schema ---



        # 6. Instantiate EnhancedAgent
        try:
            # Base arguments for EnhancedAgent
            agent_init_kwargs = {
                'amd': amd,
                'world_model': world_model,
                'format_model': self._config.formatter_llm_model if self._config.formatter_llm_model else None, # Example passing extra config
                'verbose': self._config.verbose_logging,
                'stream': self._config.enable_streaming,
                'max_history_turns': self._config.history.max_turns,
                'max_history_tokens': self._config.history.max_tokens,
                'trim_strategy': self._config.history.trim_strategy,
                'sync_adk_state': self._config.adk.sync_state if ADK_AVAILABLE else False,
                'adk_exit_stack': adk_exit_stack, # Pass stack for cleanup
                'user_cost_tracker': cost_tracker, # Pass the tracker instance
                **self._callbacks_transient, # Pass configured callbacks
                # Pass server instances if provided (less common)
                'a2a_server': self._a2a_server_instance,
                'mcp_server': self._mcp_server_instance,
            }

            # Add ADK-specific arguments if inheriting from LlmAgent
            agent_class = EnhancedAgent
            if ADK_AVAILABLE and issubclass(EnhancedAgent, ADKLlmAgent):
                 logger.debug("Adding ADK LlmAgent specific arguments to init.")
                 adk_specific_kwargs = {
                     'name': self._config.agent_name, # Required by LlmAgent
                     'model': LiteLlm(model=self._config.model_identifier), # LlmAgent needs BaseLlm instance
                     'description': self._config.adk.description or self._config.system_message,
                     'instruction': self._config.system_message, # Or dedicated instruction field?
                     'tools': processed_adk_tools,
                     'code_executor': adk_code_executor, # Pass the *instance*
                     'planner': adk_planner_instance,
                     # Process examples/schema if needed
                     'examples': [ADKExample(**ex) for ex in self._config.adk.examples] if self._config.adk.examples else None,
                     'output_schema': self._config.adk.output_schema,
                     # Pass runner/session service if NOT using InMemoryRunner deferred creation
                     # If runner is created later, it's assigned post-init
                     'runner': adk_runner_instance if adk_runner_instance else None, # Pass runner if created now
                     'session_service': adk_session_service, # Pass session service
                 }
                 # Merge, ensuring agent_init_kwargs takes precedence for overlapping basic fields if necessary
                 # but allow ADK specifics to be added. Be careful with overlaps like 'name'.
                 # EnhancedAgent init should handle reconciling these if needed.
                 # A safer merge:
                 final_kwargs = agent_init_kwargs.copy()
                 for k, v in adk_specific_kwargs.items():
                      if k not in final_kwargs: # Only add ADK specifics not already handled
                          final_kwargs[k] = v
                      # Handle specific overrides/merges needed for LlmAgent base
                      elif k == 'tools' and v: # Merge tools
                          final_kwargs['tools'] = (final_kwargs.get('tools') or []) + v
                      # Overwrite description/instruction from ADK config if set
                      elif k in ['description', 'instruction'] and v or k == 'code_executor' or k == 'model':
                           final_kwargs[k] = v

                 agent_init_kwargs = final_kwargs


            logger.debug(f"Final keys for EnhancedAgent init: {list(agent_init_kwargs.keys())}")

            # --- Instantiate the Agent ---
            agent = agent_class(**agent_init_kwargs)
            # --- Agent Instantiated ---

            # If ADK InMemoryRunner creation was deferred, create and assign now
            if ADK_AVAILABLE and 'adk_runner_config_for_later' in locals():
                 cfg = locals()['adk_runner_config_for_later']
                 if not isinstance(cfg['runner_class'], InMemoryRunner) and cfg.get('session_service') is None: cfg['session_service'] = agent.adk_session_service # Ensure service is passed
                 agent.setup_adk_runner(cfg)
                 logger.info(f"Created and assigned deferred ADK Runner instance: {agent.adk_runner.__class__.__name__}")
                 # Ensure agent has runner's session service if it differs
                 if agent.adk_runner and agent.adk_session_service is not agent.adk_runner.session_service:
                      logger.warning("Agent session service differs from deferred runner's service. Updating agent's reference.")
                      agent.adk_session_service = agent.adk_runner.session_service
            elif ADK_AVAILABLE and adk_runner_instance and not agent.adk_runner:
                # If runner was created earlier but not passed via LlmAgent init (e.g. non-LlmAgent base)
                # Or if we want to explicitly assign it
                 agent.adk_runner = adk_runner_instance
                 # Ensure session service consistency
                 if agent.adk_session_service is not agent.adk_runner.session_service:
                      agent.adk_session_service = agent.adk_runner.session_service


        except ValidationError as e:
            logger.error(f"Pydantic validation error Instantiating EnhancedAgent: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error Instantiating EnhancedAgent: {e}", exc_info=True)
            raise

        # 7. Setup Agent's Internal Server Capabilities (if enabled and not pre-initialized)
        if self._config.a2a.enabled and not agent.a2a_server:
            if AGENT_A2A_AVAILABLE:
                logger.info("Setting up A2A server on agent instance...")
                agent.setup_a2a_server(
                    host=self._config.a2a.host,
                    port=self._config.a2a.port,
                    **self._config.a2a.extra_options
                )
            else: logger.warning("A2A server configured in builder, but A2A not available in agent environment.")

        if self._config.mcp.enabled and not agent.mcp_server:
            if AGENT_MCP_AVAILABLE:
                logger.info("Setting up MCP server on agent instance...")
                agent.setup_mcp_server(
                    host=self._config.mcp.host,
                    port=self._config.mcp.port,
                    name=self._config.mcp.server_name, # Already defaulted
                    **self._config.mcp.extra_options
                )
            else: logger.warning("MCP server configured in builder, but MCP not available in agent environment.")

        # 8. Setup A2A known clients configuration on the agent
        if self._config.a2a.known_clients:
             if AGENT_A2A_AVAILABLE:
                 # The agent likely handles client creation on demand,
                 # but we can pass the config for it to use.
                 # Assuming agent has a way to receive this, e.g., during init or a setter
                 if hasattr(agent, 'set_known_a2a_clients'):
                     agent.set_known_a2a_clients(self._config.a2a.known_clients)
                 else:
                      # Fallback: store on a generic config dict? Less ideal.
                      # agent.config.a2a_known_clients = self._config.a2a.known_clients
                      logger.warning("Agent does not have 'set_known_a2a_clients' method. Known client config stored raw.")
             else:
                  logger.warning("A2A known clients configured, but A2A not available in agent env.")


        logger.info(f"--- EnhancedAgent Build Complete: {agent.amd.name} ---")
        return agent


# Example Usage (Illustrative)
async def main_example():

    # --- Configure Telemetry (Example: Console Exporter) ---
    # This should typically happen once at application startup
    provider = None
    if OTEL_AVAILABLE:
        provider = TracerProvider()
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        # Set the global provider
        # trace.set_tracer_provider(provider) # Builder will do this if passed

    # --- Build the Agent ---
    builder = EnhancedAgentBuilder(agent_name="ProdAgent007")

    agent = await (
        builder
        .with_model("gemini/gemini-1.5-flash-latest") # Or "ollama/mistral", "gpt-4o", etc.
        # .with_api_key(os.getenv("GEMINI_API_KEY")) # Use env vars
        .with_system_message("You are ProdAgent007, an advanced AI assistant integrating multiple frameworks.")
        .verbose(True)
        .enable_streaming(False)
        .with_history_options(max_tokens=8000, trim_strategy="litellm") # Token-based history
        .with_initial_world_data({"user_prefs": {"theme": "dark"}, "last_location": None})

        # --- ADK Features ---
        .with_adk_code_executor("adk_builtin") # Use ADK's secure executor if possible
        # .with_unsafe_code_executor() # Uncomment ONLY if you understand the risks
        .enable_adk_state_sync() # Sync world model with ADK state
        # Add MCP tools via ADK Toolset
        # .with_adk_mcp_toolset(StdioServerParameters(command='npx', args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/your/folder"]))

        # --- A2A Features ---
        .enable_a2a_server(host="127.0.0.1", port=5001) # Agent can receive A2A tasks
        # .with_a2a_client("http://other-agent:5002") # Pre-configure client for delegation

         # --- MCP Server Feature ---
        .enable_mcp_server(host="127.0.0.1", port=8001) # Agent can expose capabilities via MCP

        # --- Callbacks & Observability ---
        .with_post_run_callback(lambda sid, resp, cost, *s: print(f"[Callback] Run Done (Session: {sid}): Cost ${cost:.6f}, s: {s}, Resp: {resp[:50]}..."))
        .with_telemetry_provider_instance(provider) # Pass configured OTel provider

        .build() # Asynchronous build step
    )

    # --- Interact with the Agent ---
    print("\n--- Interacting with Agent ---")
    response1 = await agent.a_run("Hello there! What can you do?")
    print("Response 1:", response1)

    response2 = await agent.a_run("What is the square root of 289?") # Test code execution (if enabled)
    print("Response 2:", response2)

    response3 = await agent.a_run("My location is London.", session_id="user123") # Test world model update
    print("Response 3:", response3)

    response4 = await agent.a_run("What was the location I mentioned?", session_id="user123") # Test world model retrieval
    print("Response 4:", response4)

    # If A2A client configured:
    # response5 = await agent.a_run("Ask the agent at http://other-agent:5002 about the weather in London.")
    # print("Response 5:", response5)

    print(agent.total_cost)


    # --- Cleanup ---
    await agent.close()

    # --- Run Servers (Example - requires separate processes/tasks usually) ---
    # These are blocking calls, typically run separately
    # if agent.a2a_server:
    #     print("\nStarting A2A Server (Blocking)... Press Ctrl+C to stop.")
    #     # agent.run_a2a_server() # Run in separate thread/process
    # if agent.mcp_server:
    #      print("\nStarting MCP Server (Blocking)... Press Ctrl+C to stop.")
         # agent.run_mcp_server() # Run in separate thread/process


if __name__ == "__main__":
    # Note: Running the example might require setting up environment variables (API keys)
    # and potentially running dependent services (like other A2A/MCP agents).
    try:
        asyncio.run(main_example())
    except Exception as main_e:
         print(f"\n--- Example execution failed: {main_e} ---")
         # Print traceback for debugging
         import traceback
         traceback.print_exc()

