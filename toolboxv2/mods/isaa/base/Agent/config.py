# core/config.py
import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from toolboxv2 import get_logger

# Import dummy types if ADK/A2A/MCP are not available (ensure these align with agent.py)
try:
    from google.adk.agents import BaseAgent
    from google.adk.code_executors import BaseCodeExecutor
    from google.adk.examples import Example
    from google.adk.planners import BasePlanner
    from google.adk.runners import Runner
    from google.adk.tools import BaseTool
    ADK_AVAILABLE_CONF = True
except ImportError:
    ADK_AVAILABLE_CONF = False
    BaseAgent = object
    BaseTool = object
    BaseCodeExecutor = object
    BasePlanner = object
    Example = object
    Runner = object

try:
    from python_a2a.client import A2AClient
    from python_a2a.models import AgentCard
    from python_a2a.server import A2AServer
    A2A_AVAILABLE_CONF = True
except ImportError:
    A2A_AVAILABLE_CONF = False
    A2AServer = object
    A2AClient = object
    AgentCard = object

try:
    from mcp.client import ClientSession
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE_CONF = True
except ImportError:
    MCP_AVAILABLE_CONF = False
    FastMCP = object
    ClientSession = object

try:
    from litellm import BudgetManager
    LITELLM_AVAILABLE_CONF = True
except ImportError:
    LITELLM_AVAILABLE_CONF = False
    BudgetManager = object


logger = logging.getLogger(__name__)
logger.setLevel(get_logger().level)

# --- Configuration Models ---

class ModelConfig(BaseModel):
    """Configuration specific to an LLM model via LiteLLM."""
    # Used as key for model selection
    name: str = Field(..., description="Unique identifier/alias for this model configuration (e.g., 'fast_formatter', 'main_reasoner').")
    model: str = Field(..., description="LiteLLM model string (e.g., 'gemini/gemini-1.5-pro-latest', 'ollama/mistral').")
    provider: str | None = Field(default=None, description="LiteLLM provider override if needed.")
    api_key: str | None = Field(default=None, description="API Key (consider using environment variables).")
    api_base: str | None = Field(default=None, description="API Base URL (for local models, proxies).")
    api_version: str | None = Field(default=None, description="API Version (e.g., for Azure).")

    # Common LLM Parameters
    temperature: float | None = Field(default=0.7)
    top_p: float | None = Field(default=None)
    top_k: int | None = Field(default=None)
    max_tokens: int | None = Field(default=2048, description="Max tokens for generation.")
    max_input_tokens: int | None = Field(default=None, description="Max input context window (autodetected if None).")
    stop_sequence: list[str] | None = Field(default=None)
    presence_penalty: float | None = Field(default=None)
    frequency_penalty: float | None = Field(default=None)
    system_message: str | None = Field(default=None, description="Default system message for this model.")

    # LiteLLM Specific
    caching: bool = Field(default=True, description="Enable LiteLLM caching for this model.")
    # budget_manager: Optional[BudgetManager] = Field(default=None) # Budget manager applied globally or per-agent

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow') # Allow extra LiteLLM params


class ADKConfig(BaseModel):
    """Configuration for ADK integration."""
    enabled: bool = Field(default=True, description="Enable ADK features if ADK is installed.")
    description: str | None = Field(default=None, description="ADK LlmAgent description.")
    instruction_override: str | None = Field(default=None, description="Override agent's system message for ADK.")
    # Tools added via builder or auto-discovery
    code_executor: str | BaseCodeExecutor | None = Field(default=None, description="Reference name or instance of ADK code executor.")
    planner: str | BasePlanner | None = Field(default=None, description="Reference name or instance of ADK planner.")
    examples: list[Example] | None = Field(default=None, description="Few-shot examples for ADK.")
    output_schema: type[BaseModel] | None = Field(default=None, description="Pydantic model for structured output.")
    # MCP Toolset config handled separately if ADK is enabled
    use_mcp_toolset: bool = Field(default=True, description="Use ADK's MCPToolset for MCP client connections if ADK is enabled.")
    # Runner config handled separately

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MCPConfig(BaseModel):
    """Configuration for MCP integration."""
    server: dict[str, Any] | None = Field(default=None, description="Configuration to run an MCP server (host, port, etc.).")
    client_connections: dict[str, str] = Field(default_factory=dict, description="Named MCP server URLs to connect to as a client (e.g., {'files': 'stdio:npx @mcp/server-filesystem /data'}).")
    # ADK's MCPToolset handles client connections if ADKConfig.use_mcp_toolset is True

    model_config = ConfigDict(arbitrary_types_allowed=True)


class A2AConfig(BaseModel):
    """Configuration for A2A integration."""
    server: dict[str, Any] | None = Field(default=None, description="Configuration to run an A2A server (host, port, etc.).")
    known_agents: dict[str, str] = Field(default_factory=dict, description="Named A2A agent URLs to interact with (e.g., {'weather_agent': 'http://weather:5000'}).")
    default_task_timeout: int = Field(default=120, description="Default timeout in seconds for waiting on A2A task results.")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ObservabilityConfig(BaseModel):
    """Configuration for observability (OpenTelemetry)."""
    enabled: bool = Field(default=True)
    endpoint: str | None = Field(default=None, description="OTLP endpoint URL (e.g., http://jaeger:4317).")
    service_name: str | None = Field(default=None, description="Service name for traces/metrics (defaults to agent name).")
    # Add more OTel config options as needed (headers, certs, resource attributes)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentConfig(BaseModel):
    """Main configuration schema for an EnhancedAgent."""
    agent_name: str = Field(..., description="Unique name for this agent instance.")
    version: str = Field(default="0.1.0")

    agent_instruction: str = Field(default="You are a helpful AI assistant. Answer user questions to the best of your knowledge. Respond concisely. use tools when needed")
    agent_description: str = Field(default="An configurable, production-ready agent with integrated capabilities.")

    # Model Selection
    models: list[ModelConfig] = Field(..., description="List of available LLM configurations.")
    default_llm_model: str = Field(..., description="Name of the ModelConfig to use for general LLM calls.")
    formatter_llm_model: str | None = Field(default=None, description="Optional: Name of a faster/cheaper ModelConfig for a_format_class calls.")

    # Core Agent Settings
    world_model_initial_data: dict[str, Any] | None = Field(default=None)
    enable_streaming: bool = Field(default=False)
    verbose: bool = Field(default=False)
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR).")
    max_history_length: int = Field(default=20, description="Max conversation turns for LiteLLM history.")
    trim_strategy: Literal["litellm", "basic"] = Field(default="litellm")
    persist_history: bool = Field(default=True, description="Persist conversation history (requires persistent ChatSession).")
    user_id_default: str | None = Field(default=None, description="Default user ID for interactions.")

    # Secure Code Execution
    code_executor_type: Literal["restricted", "docker", "none"] = Field(default="restricted", description="Type of code executor to use.")
    code_executor_config: dict[str, Any] = Field(default_factory=dict, description="Configuration specific to the chosen code executor.")
    enable_adk_code_execution_tool: bool = Field(default=True, description="Expose code execution as an ADK tool if ADK is enabled.")

    # Framework Integrations
    adk: ADKConfig | None = Field(default_factory=ADKConfig if ADK_AVAILABLE_CONF else lambda: None)
    mcp: MCPConfig | None = Field(default_factory=MCPConfig if MCP_AVAILABLE_CONF else lambda: None)
    a2a: A2AConfig | None = Field(default_factory=A2AConfig if A2A_AVAILABLE_CONF else lambda: None)

    # Observability & Cost
    observability: ObservabilityConfig | None = Field(default_factory=ObservabilityConfig)
    budget_manager: BudgetManager | None = Field(default=None, description="Global LiteLLM budget manager instance.") # Needs to be passed in

    # Human-in-the-Loop
    enable_hitl: bool = Field(default=False, description="Enable basic Human-in-the-Loop hooks.")

    # Add other global settings as needed

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def validate_model_references(self) -> 'AgentConfig':
        model_names = {m.name for m in self.models}
        if self.default_llm_model not in model_names:
            raise ValueError(f"default_llm_model '{self.default_llm_model}' not found in defined models.")
        if self.formatter_llm_model and self.formatter_llm_model not in model_names:
            raise ValueError(f"formatter_llm_model '{self.formatter_llm_model}' not found in defined models.")
        return self

    @model_validator(mode='after')
    def validate_framework_availability(self) -> 'AgentConfig':
        if self.adk and self.adk.enabled and not ADK_AVAILABLE_CONF:
            logger.warning("ADK configuration provided but ADK library not installed. Disabling ADK features.")
            self.adk.enabled = False
        if self.mcp and (self.mcp.server or self.mcp.client_connections) and not MCP_AVAILABLE_CONF:
             logger.warning("MCP configuration provided but MCP library not installed. Disabling MCP features.")
             self.mcp = None # Or disable specific parts
        if self.a2a and (self.a2a.server or self.a2a.known_agents) and not A2A_AVAILABLE_CONF:
             logger.warning("A2A configuration provided but A2A library not installed. Disabling A2A features.")
             self.a2a = None # Or disable specific parts
        return self

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> 'AgentConfig':
        """Loads configuration from a YAML file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(file_path) as f:
            config_data = yaml.safe_load(f)
        logger.info(f"Loaded agent configuration from {path}")
        return cls(**config_data)

    def save_to_yaml(self, path: str | Path):
        """Saves the current configuration to a YAML file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            # Use Pydantic's model_dump for clean serialization
            yaml.dump(self.model_dump(mode='python'), f, sort_keys=False)
        logger.info(f"Saved agent configuration to {path}")
