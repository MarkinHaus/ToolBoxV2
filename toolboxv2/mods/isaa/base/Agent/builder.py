"""
FlowAgentBuilder V2 - Production-ready Builder for FlowAgent

Refactored to work with the new FlowAgent architecture:
- VFS-based context management
- RuleSet integration for persona/behavior
- Unified ToolManager
- IntelligentRateLimiter integration
- MCP Session Management

Author: FlowAgent V2
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from toolboxv2 import Spinner, get_logger
from toolboxv2.mods.isaa.base.Agent.docker_vfs import DockerConfig

# Framework imports with graceful degradation
try:
    import litellm
    from litellm import BudgetManager
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    class BudgetManager: pass

try:
    from python_a2a import A2AServer, AgentCard
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    class A2AServer: pass
    class AgentCard: pass

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    class FastMCP: pass

from toolboxv2.mods.isaa.base.Agent.types import (
    AgentModelData,
    CheckpointConfig,
    FormatConfig,
    PersonaConfig,
    ResponseFormat,
    TextLength,
)
from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent

logger = get_logger()
AGENT_VERBOSE = os.environ.get("AGENT_VERBOSE", "false").lower() == "true"

rprint = print if AGENT_VERBOSE else lambda *a, **k: None
iprint = print if AGENT_VERBOSE else lambda *a, **k: None
wprint = print if AGENT_VERBOSE else lambda *a, **k: None
eprint = print if AGENT_VERBOSE else lambda *a, **k: None


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class MCPConfig(BaseModel):
    """MCP server and tools configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = False
    config_path: Optional[str] = None
    server_name: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8000
    auto_expose_tools: bool = True


class A2AConfig(BaseModel):
    """A2A server configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 5000
    agent_name: Optional[str] = None
    agent_description: Optional[str] = None
    agent_version: str = "1.0.0"
    expose_tools_as_skills: bool = True


class RateLimiterConfig(BaseModel):
    """Rate Limiter configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Feature toggles
    enable_rate_limiting: bool = True
    enable_model_fallback: bool = True
    enable_key_rotation: bool = True
    key_rotation_mode: str = "balance"  # "drain" or "balance"

    # API Keys: provider -> list of keys
    api_keys: dict[str, list[str]] = Field(default_factory=dict)

    # Fallback chains: primary_model -> [fallback_models]
    fallback_chains: dict[str, list[str]] = Field(default_factory=dict)

    # Custom limits: model -> {rpm, tpm, input_tpm, ...}
    custom_limits: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Behavior
    max_retries: int = 3
    wait_if_all_exhausted: bool = True


class AgentConfig(BaseModel):
    """Complete agent configuration for loading/saving"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Basic settings
    name: str = "FlowAgent"
    description: str = "Production-ready FlowAgent"
    version: str = "2.0.0"

    # LLM settings
    fast_llm_model: str = "openrouter/anthropic/claude-3-haiku"
    complex_llm_model: str = "openrouter/openai/gpt-4o"
    system_message: str = """You are a production-ready autonomous agent with advanced capabilities."""

    temperature: float = 0.7
    max_tokens_output: int = 2048
    max_tokens_input: int = 32768
    vfs_max_window_lines: int = 250
    api_key_env_var: Optional[str] = "OPENROUTER_API_KEY"
    use_fast_response: bool = True

    # Features
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    rate_limiter: RateLimiterConfig = Field(default_factory=RateLimiterConfig)

    # Agent behavior
    max_parallel_tasks: int = 3
    verbose_logging: bool = False
    stream: bool = True

    # Persona and formatting
    active_persona: Optional[str] = None
    persona_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # World Model (initial VFS content)
    world_model: dict[str, Any] = Field(default_factory=dict)

    # Rule config path
    rule_config_path: Optional[str] = None


# =============================================================================
# FLOWAGENT BUILDER V2
# =============================================================================

class FlowAgentBuilder:
    """
    Production-ready FlowAgent builder for the V2 architecture.

    Features:
    - Fluent API for configuration
    - MCP integration with automatic tool categorization
    - Persona → RuleSet integration
    - World Model → VFS file
    - IntelligentRateLimiter configuration
    - Checkpoint management
    """

    def __init__(self, config: AgentConfig = None, config_path: str = None):
        """
        Initialize builder with configuration.

        Args:
            config: AgentConfig object
            config_path: Path to YAML/JSON config file
        """
        if config and config_path:
            raise ValueError("Provide either config object or config_path, not both")

        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = AgentConfig()

        # Runtime components
        self._custom_tools: dict[str, tuple[Callable, str, list[str] | None, dict[str, Any] | None]] = {}
        self._mcp_tools: dict[str, dict] = {}
        self._mcp_session_manager = None
        self._mcp_config_data: dict = {}
        self._mcp_needs_loading: bool = False
        self._budget_manager: BudgetManager = None

        # Persona patterns for RuleSet
        self._persona_patterns: list[dict] = []

        if self.config.verbose_logging:
            logging.getLogger().setLevel(logging.DEBUG)

        iprint(f"FlowAgentBuilder initialized: {self.config.name}")

    # =========================================================================
    # CONFIGURATION MANAGEMENT
    # =========================================================================

    def _load_config(self, config_path: str) -> AgentConfig:
        """Load agent configuration from file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return AgentConfig(**data)

    def save_config(self, config_path: str, format: str = 'yaml'):
        """Save current configuration to file"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.config.model_dump()

        with open(path, 'w', encoding='utf-8') as f:
            if format.lower() == 'yaml':
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2)

        iprint(f"Configuration saved to {config_path}")

    @classmethod
    def from_config_file(cls, config_path: str) -> 'FlowAgentBuilder':
        """Create builder from configuration file"""
        return cls(config_path=config_path)

    # =========================================================================
    # FLUENT BUILDER API - Basic Settings
    # =========================================================================

    def with_name(self, name: str) -> 'FlowAgentBuilder':
        """Set agent name"""
        self.config.name = name
        return self

    def with_models(self, fast_model: str, complex_model: str = None) -> 'FlowAgentBuilder':
        """Set LLM models"""
        self.config.fast_llm_model = fast_model
        if complex_model:
            self.config.complex_llm_model = complex_model
        return self

    def with_system_message(self, message: str) -> 'FlowAgentBuilder':
        """
        Set system message.
        Stored in AgentModelData for all LLM calls.
        """
        self.config.system_message = message
        return self

    def with_temperature(self, temp: float) -> 'FlowAgentBuilder':
        """Set temperature"""
        self.config.temperature = temp
        return self

    def with_budget_manager(self, max_cost: float = 10.0) -> 'FlowAgentBuilder':
        """Enable budget management"""
        if LITELLM_AVAILABLE:
            self._budget_manager = BudgetManager("agent")
            iprint(f"Budget manager enabled: ${max_cost}")
        else:
            wprint("LiteLLM not available, budget manager disabled")
        return self

    def verbose(self, enable: bool = True) -> 'FlowAgentBuilder':
        """Enable verbose logging"""
        self.config.verbose_logging = enable
        if enable:
            logging.getLogger().setLevel(logging.DEBUG)
        return self

    def with_stream(self, enable: bool = True) -> 'FlowAgentBuilder':
        """Enable/disable streaming"""
        self.config.stream = enable
        return self

    def with_vfs_window_lines(self, lines: int) -> 'FlowAgentBuilder':
        """Set max VFS window lines"""
        self.config.vfs_max_window_lines = lines
        return self

    # =========================================================================
    # WORLD MODEL (VFS File)
    # =========================================================================

    def with_world_model(self, world_model: dict[str, Any]) -> 'FlowAgentBuilder':
        """
        Set initial world model.

        This creates a read-write VFS file 'world_model' that the agent can
        read and update with world facts during execution.

        Args:
            world_model: Dict with initial world facts

        Example:
            .with_world_model({
                "project_name": "MyProject",
                "environment": "production",
                "user_preferences": {"language": "de"}
            })
        """
        self.config.world_model.update(world_model)
        return self

    def add_world_fact(self, key: str, value: Any) -> 'FlowAgentBuilder':
        """Add a single world fact"""
        self.config.world_model[key] = value
        return self

    # =========================================================================
    # RATE LIMITER CONFIGURATION
    # =========================================================================

    def with_rate_limiter(
        self,
        enable_rate_limiting: bool = True,
        enable_model_fallback: bool = True,
        enable_key_rotation: bool = True,
        key_rotation_mode: str = "balance",
        max_retries: int = 3
    ) -> 'FlowAgentBuilder':
        """
        Configure rate limiter settings.

        Args:
            enable_rate_limiting: Enable rate limiting
            enable_model_fallback: Enable automatic model fallback
            enable_key_rotation: Enable multi-key rotation
            key_rotation_mode: "drain" (one key until limit) or "balance" (round-robin)
            max_retries: Max retry attempts
        """
        self.config.rate_limiter.enable_rate_limiting = enable_rate_limiting
        self.config.rate_limiter.enable_model_fallback = enable_model_fallback
        self.config.rate_limiter.enable_key_rotation = enable_key_rotation
        self.config.rate_limiter.key_rotation_mode = key_rotation_mode
        self.config.rate_limiter.max_retries = max_retries
        return self

    def add_api_key(
        self,
        provider: str,
        key: str
    ) -> 'FlowAgentBuilder':
        """
        Add an API key for rate limiter key rotation.

        Args:
            provider: Provider name (e.g., "vertex_ai", "openai", "anthropic")
            key: The API key

        Example:
            .add_api_key("vertex_ai", "AIza...")
            .add_api_key("openai", "sk-...")
        """
        if provider not in self.config.rate_limiter.api_keys:
            self.config.rate_limiter.api_keys[provider] = []
        self.config.rate_limiter.api_keys[provider].append(key)
        return self

    def add_fallback_chain(
        self,
        primary_model: str,
        fallback_models: list[str]
    ) -> 'FlowAgentBuilder':
        """
        Add a model fallback chain.

        Args:
            primary_model: Primary model (e.g., "vertex_ai/gemini-2.5-pro")
            fallback_models: List of fallback models in priority order

        Example:
            .add_fallback_chain(
                "vertex_ai/gemini-2.5-pro",
                ["vertex_ai/gemini-2.5-flash", "vertex_ai/gemini-2.0-flash"]
            )
        """
        self.config.rate_limiter.fallback_chains[primary_model] = fallback_models
        return self

    def set_model_limits(
        self,
        model: str,
        rpm: int = None,
        tpm: int = None,
        input_tpm: int = None,
        is_free_tier: bool = False
    ) -> 'FlowAgentBuilder':
        """
        Set custom rate limits for a model.

        Args:
            model: Model string (e.g., "vertex_ai/gemini-2.5-pro")
            rpm: Requests per minute
            tpm: Tokens per minute
            input_tpm: Input tokens per minute
            is_free_tier: Whether this is a free tier model
        """
        limits = {}
        if rpm is not None:
            limits['rpm'] = rpm
        if tpm is not None:
            limits['tpm'] = tpm
        if input_tpm is not None:
            limits['input_tpm'] = input_tpm
        limits['is_free_tier'] = is_free_tier

        self.config.rate_limiter.custom_limits[model] = limits
        return self

    def load_rate_limiter_config(self, config_path: str) -> 'FlowAgentBuilder':
        """
        Load rate limiter configuration from file.

        Expected format:
        ```json
        {
            "features": {
                "rate_limiting": true,
                "model_fallback": true,
                "key_rotation": true,
                "key_rotation_mode": "balance"
            },
            "api_keys": {
                "vertex_ai": ["key1", "key2"],
                "openai": ["sk-..."]
            },
            "fallback_chains": {
                "vertex_ai/gemini-2.5-pro": ["vertex_ai/gemini-2.5-flash"]
            },
            "limits": {
                "vertex_ai/gemini-2.5-pro": {"rpm": 2, "input_tpm": 32000}
            }
        }
        ```
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Rate limiter config not found: {config_path}")

        with open(path, encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # Apply features
        features = data.get('features', {})
        self.config.rate_limiter.enable_rate_limiting = features.get('rate_limiting', True)
        self.config.rate_limiter.enable_model_fallback = features.get('model_fallback', True)
        self.config.rate_limiter.enable_key_rotation = features.get('key_rotation', True)
        self.config.rate_limiter.key_rotation_mode = features.get('key_rotation_mode', 'balance')

        # Apply API keys
        for provider, keys in data.get('api_keys', {}).items():
            self.config.rate_limiter.api_keys[provider] = keys

        # Apply fallback chains
        for primary, fallbacks in data.get('fallback_chains', {}).items():
            self.config.rate_limiter.fallback_chains[primary] = fallbacks

        # Apply limits
        for model, limits in data.get('limits', {}).items():
            self.config.rate_limiter.custom_limits[model] = limits

        iprint(f"Loaded rate limiter config from {config_path}")
        return self

    # =========================================================================
    # MCP INTEGRATION
    # =========================================================================

    def enable_mcp_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        server_name: str = None
    ) -> 'FlowAgentBuilder':
        """Enable MCP server"""
        if not MCP_AVAILABLE:
            wprint("MCP not available, cannot enable server")
            return self

        self.config.mcp.enabled = True
        self.config.mcp.host = host
        self.config.mcp.port = port
        self.config.mcp.server_name = server_name or f"{self.config.name}_MCP"

        iprint(f"MCP server enabled: {host}:{port}")
        return self

    def load_mcp_tools_from_config(self, config_path: str | dict) -> 'FlowAgentBuilder':
        """
        Load MCP tools from configuration.

        The builder will:
        1. Initialize MCPSessionManager
        2. Connect to MCP servers
        3. Extract all capabilities (tools, resources, prompts)
        4. Register tools in ToolManager with categories
        5. Create RuleSet tool groups automatically

        Args:
            config_path: Path to MCP config file or config dict
        """
        if not MCP_AVAILABLE:
            wprint("MCP not available, skipping tool loading")
            return self

        if isinstance(config_path, dict):
            self._mcp_config_data = config_path
        else:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"MCP config not found: {config_path}")

            with open(path, encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    self._mcp_config_data = yaml.safe_load(f)
                else:
                    self._mcp_config_data = json.load(f)

        self.config.mcp.config_path = str(config_path) if isinstance(config_path, str) else None
        self._mcp_needs_loading = True

        iprint(f"MCP config loaded, will process during build")
        return self

    # =========================================================================
    # A2A INTEGRATION
    # =========================================================================

    def enable_a2a_server(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        agent_name: str = None,
        agent_description: str = None
    ) -> 'FlowAgentBuilder':
        """Enable A2A server for agent-to-agent communication"""
        if not A2A_AVAILABLE:
            wprint("A2A not available, cannot enable server")
            return self

        self.config.a2a.enabled = True
        self.config.a2a.host = host
        self.config.a2a.port = port
        self.config.a2a.agent_name = agent_name or self.config.name
        self.config.a2a.agent_description = agent_description or self.config.description

        iprint(f"A2A server enabled: {host}:{port}")
        return self

    # =========================================================================
    # CHECKPOINT CONFIGURATION
    # =========================================================================

    def with_checkpointing(
        self,
        enabled: bool = True,
        interval_seconds: int = 300,
        max_checkpoints: int = 10,
        max_age_hours: int = 24
    ) -> 'FlowAgentBuilder':
        """Configure checkpointing (minimal - agent handles heavy lifting)"""
        self.config.checkpoint.enabled = enabled
        self.config.checkpoint.interval_seconds = interval_seconds
        self.config.checkpoint.max_checkpoints = max_checkpoints
        self.config.checkpoint.max_age_hours = max_age_hours
        self.config.checkpoint.auto_load_on_start = enabled

        if enabled:
            iprint(f"Checkpointing enabled (max {max_age_hours}h)")

        return self

    # =========================================================================
    # TOOL MANAGEMENT
    # =========================================================================

    def add_tool(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        category: list[str] | str = None,
        flags: dict[str, bool] = None,
    ) -> 'FlowAgentBuilder':
        """
        Add custom tool function.

        Args:
            func: The tool function
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            category: Category or list of categories for RuleSet grouping
            flags: Dictionary of flags (e.g., {"dangerous": True})
        """
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"

        # Normalize category
        if category is None:
            categories = ["local"]
        elif isinstance(category, str):
            categories = [category]
        else:
            categories = category

        self._custom_tools[tool_name] = (func, tool_desc, categories, flags)

        iprint(f"Tool added: {tool_name} (categories: {categories})")
        return self

    def add_tools_from_module(
        self,
        module,
        prefix: str = "",
        category: str = None,
        exclude: list[str] = None
    ) -> 'FlowAgentBuilder':
        """Add all functions from a module as tools"""
        import inspect

        exclude = exclude or []

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name in exclude or name.startswith('_'):
                continue

            tool_name = f"{prefix}{name}" if prefix else name
            self.add_tool(obj, name=tool_name, category=category or module.__name__)

        iprint(f"Added tools from module {module.__name__}")
        return self

    def with_docker_vfs(self, config: DockerConfig | None = None) -> 'FlowAgentBuilder':
        """Enable Docker VFS"""
        self.config.docker_config = config or DockerConfig()
        iprint(f"Docker VFS enabled")
        return self.with_docker(True)

    def with_lsp(self, enabled: bool = True) -> 'FlowAgentBuilder':
        """Enable LSP"""
        self.config.enable_lsp = enabled
        iprint(f"LSP enabled: {enabled}")
        return self

    def with_docker(self, enabled: bool = True) -> 'FlowAgentBuilder':
        """Enable Docker"""
        self.config.enable_docker = enabled
        iprint(f"Docker enabled: {enabled}")
        return self

    # =========================================================================
    # PERSONA MANAGEMENT (→ RuleSet)
    # =========================================================================

    def add_persona_profile(
        self,
        profile_name: str,
        name: str,
        style: str = "professional",
        tone: str = "friendly",
        personality_traits: list[str] = None,
        custom_instructions: str = "",
        response_format: str = None,
        text_length: str = None
    ) -> 'FlowAgentBuilder':
        """
        Add a persona profile.

        Persona information is written to VFS system_context and
        creates learned patterns in RuleSet.

        Args:
            profile_name: Internal profile name
            name: Display name for the persona
            style: Communication style
            tone: Tone of responses
            personality_traits: List of personality traits
            custom_instructions: Additional instructions
            response_format: Response format preference
            text_length: Text length preference
        """
        if personality_traits is None:
            personality_traits = ["helpful", "concise"]

        persona_data = {
            "name": name,
            "style": style,
            "tone": tone,
            "personality_traits": personality_traits,
            "custom_instructions": custom_instructions,
        }

        # Add format config if specified
        if response_format or text_length:
            persona_data["format_config"] = {
                "response_format": response_format or "free-text",
                "text_length": text_length or "chat-conversation",
            }

        self.config.persona_profiles[profile_name] = persona_data

        # Create patterns for RuleSet learning
        self._persona_patterns.append({
            "profile_name": profile_name,
            "pattern": f"Communication style: {style}, tone: {tone}",
            "traits": personality_traits,
            "instructions": custom_instructions
        })

        iprint(f"Persona profile added: {profile_name}")
        return self

    def set_active_persona(self, profile_name: str) -> 'FlowAgentBuilder':
        """Set active persona profile"""
        if profile_name in self.config.persona_profiles:
            self.config.active_persona = profile_name
            iprint(f"Active persona set: {profile_name}")
        else:
            wprint(f"Persona profile not found: {profile_name}")
        return self

    # Preset personas
    def with_developer_persona(self, name: str = "Senior Developer") -> 'FlowAgentBuilder':
        """Add and set pre-built developer persona"""
        return (self
                .add_persona_profile(
                    "developer",
                    name=name,
                    style="technical",
                    tone="professional",
                    personality_traits=["precise", "thorough", "security_conscious", "best_practices"],
                    custom_instructions="Focus on code quality, maintainability, and security. Always consider edge cases.",
                    response_format="code-structure",
                    text_length="detailed-indepth"
                )
                .set_active_persona("developer"))

    def with_analyst_persona(self, name: str = "Data Analyst") -> 'FlowAgentBuilder':
        """Add and set pre-built analyst persona"""
        return (self
                .add_persona_profile(
                    "analyst",
                    name=name,
                    style="analytical",
                    tone="objective",
                    personality_traits=["methodical", "insight_driven", "evidence_based"],
                    custom_instructions="Focus on statistical rigor and actionable recommendations.",
                    response_format="with-tables",
                    text_length="detailed-indepth"
                )
                .set_active_persona("analyst"))

    def with_assistant_persona(self, name: str = "AI Assistant") -> 'FlowAgentBuilder':
        """Add and set pre-built general assistant persona"""
        return (self
                .add_persona_profile(
                    "assistant",
                    name=name,
                    style="friendly",
                    tone="helpful",
                    personality_traits=["helpful", "patient", "clear", "adaptive"],
                    custom_instructions="Be helpful and adapt communication to user expertise level.",
                    response_format="with-bullet-points",
                    text_length="chat-conversation"
                )
                .set_active_persona("assistant"))

    def with_creative_persona(self, name: str = "Creative Assistant") -> 'FlowAgentBuilder':
        """Add and set pre-built creative persona"""
        return (self
                .add_persona_profile(
                    "creative",
                    name=name,
                    style="creative",
                    tone="inspiring",
                    personality_traits=["imaginative", "expressive", "innovative", "engaging"],
                    custom_instructions="Think outside the box and provide creative, inspiring solutions.",
                    response_format="md-text",
                    text_length="detailed-indepth"
                )
                .set_active_persona("creative"))

    def with_executive_persona(self, name: str = "Executive Assistant") -> 'FlowAgentBuilder':
        """Add and set pre-built executive persona"""
        return (self
                .add_persona_profile(
                    "executive",
                    name=name,
                    style="professional",
                    tone="authoritative",
                    personality_traits=["strategic", "decisive", "results_oriented", "efficient"],
                    custom_instructions="Provide strategic insights with executive-level clarity and focus on outcomes.",
                    response_format="with-bullet-points",
                    text_length="table-conversation"
                )
                .set_active_persona("executive"))

    # =========================================================================
    # RULE CONFIG
    # =========================================================================

    def with_rule_config(self, config_path: str) -> 'FlowAgentBuilder':
        """Set path to RuleSet configuration file"""
        self.config.rule_config_path = config_path
        return self

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_config(self) -> dict[str, list[str]]:
        """Validate the current configuration"""
        issues = {"errors": [], "warnings": []}

        if not self.config.fast_llm_model:
            issues["errors"].append("Fast LLM model not specified")
        if not self.config.complex_llm_model:
            issues["errors"].append("Complex LLM model not specified")

        if self.config.mcp.enabled and not MCP_AVAILABLE:
            issues["errors"].append("MCP enabled but MCP not available")

        if self.config.a2a.enabled and not A2A_AVAILABLE:
            issues["errors"].append("A2A enabled but A2A not available")

        if self.config.active_persona and self.config.active_persona not in self.config.persona_profiles:
            issues["errors"].append(f"Active persona '{self.config.active_persona}' not found in profiles")

        return issues

    # =========================================================================
    # BUILD - Main Method
    # =========================================================================

    async def build(self) -> FlowAgent:
        """
        Build the production-ready FlowAgent.

        Steps:
        1. Setup API configuration
        2. Create PersonaConfig
        3. Create AgentModelData
        4. Create FlowAgent instance
        5. Add custom variables to VFS
        6. Add custom tools
        7. Process MCP configuration (load tools, categorize)
        8. Add MCP tools to ToolManager
        9. Setup MCP server
        10. Setup A2A server
        11. Apply persona to RuleSet
        12. Restore checkpoint if enabled

        Returns:
            Configured FlowAgent instance
        """
        from toolboxv2 import get_app

        info_print = logger.info

        with Spinner(message=f"Building Agent {self.config.name}", symbols='c'):
            iprint(f"Building FlowAgent: {self.config.name}")

            # Validate configuration
            validation_issues = self.validate_config()
            if validation_issues["errors"]:
                error_msg = f"Configuration validation failed: {', '.join(validation_issues['errors'])}"
                eprint(error_msg)
                raise ValueError(error_msg)

            for warning in validation_issues["warnings"]:
                wprint(f"Configuration warning: {warning}")

            try:
                # Step 1: API configuration
                api_key = None
                if self.config.api_key_env_var:
                    api_key = os.getenv(self.config.api_key_env_var)
                    if not api_key:
                        wprint(f"API key env var {self.config.api_key_env_var} not set")

                # Step 2: Create PersonaConfig if configured
                active_persona = None
                if self.config.active_persona and self.config.active_persona in self.config.persona_profiles:
                    persona_data = self.config.persona_profiles[self.config.active_persona].copy()

                    # Create FormatConfig if present
                    format_config = None
                    if "format_config" in persona_data:
                        fc_data = persona_data.pop("format_config")
                        format_config = FormatConfig(
                            response_format=ResponseFormat(fc_data.get("response_format", "free-text")),
                            text_length=TextLength(fc_data.get("text_length", "chat-conversation")),
                        )

                    active_persona = PersonaConfig(**persona_data)
                    active_persona.format_config = format_config

                    iprint(f"Using persona: {active_persona.name}")

                # Step 3: Create AgentModelData
                # Build rate limiter handler config
                handler_config = self._build_rate_limiter_config()

                amd = AgentModelData(
                    name=self.config.name,
                    fast_llm_model=self.config.fast_llm_model,
                    complex_llm_model=self.config.complex_llm_model,
                    system_message=self.config.system_message,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens_output,
                    max_input_tokens=self.config.max_tokens_input,
                    vfs_max_window_lines=self.config.vfs_max_window_lines,
                    api_key=api_key,
                    budget_manager=self._budget_manager,
                    persona=active_persona,
                    use_fast_response=self.config.use_fast_response,
                    handler_path_or_dict=handler_config
                )

                # Step 4: Create FlowAgent
                agent = FlowAgent(
                    amd=amd,
                    verbose=self.config.verbose_logging,
                    max_parallel_tasks=self.config.max_parallel_tasks,
                    auto_load_checkpoint=self.config.checkpoint.enabled,
                    rule_config_path=self.config.rule_config_path,
                    stream=self.config.stream
                )

                # Step 5: Initialize world model in VFS
                if self.config.world_model:
                    await self._init_world_model(agent)

                # Step 6: Add custom tools
                tools_added = 0
                for tool_name, (tool_func, tool_desc, categories, flags) in self._custom_tools.items():
                    try:
                        await agent.add_tool(
                            tool_func,
                            name=tool_name,
                            description=tool_desc,
                            category=categories,
                            flags=flags
                        )
                        tools_added += 1
                    except Exception as e:
                        eprint(f"Failed to add tool {tool_name}: {e}")

                # Step 7: Process MCP configuration
                with Spinner(message="Loading MCP", symbols='w'):
                    if self._mcp_needs_loading:
                        await self._process_mcp_config(agent)

                # Step 8: Add MCP tools (already done in _process_mcp_config)
                mcp_tools_count = len(self._mcp_tools)

                # Step 9: Setup MCP server
                if self.config.mcp.enabled and MCP_AVAILABLE:
                    try:
                        agent.setup_mcp_server(name=self.config.mcp.server_name)
                        iprint("MCP server configured")
                    except Exception as e:
                        eprint(f"Failed to setup MCP server: {e}")

                # Step 10: Setup A2A server
                if self.config.a2a.enabled and A2A_AVAILABLE:
                    try:
                        agent.setup_a2a_server(
                            host=self.config.a2a.host,
                            port=self.config.a2a.port
                        )
                        iprint("A2A server configured")
                    except Exception as e:
                        eprint(f"Failed to setup A2A server: {e}")

                # Step 11: Apply persona patterns to RuleSet
                await self._apply_persona_to_ruleset(agent)

                # Step 12: Checkpoint loding
                if self.config.checkpoint.enabled:
                    res = await agent.checkpoint_manager.auto_restore()
                    print(
                        f"Auto-restore result: {res.get('success')} - {res.get('error')} - {res.get('restored_components')=}")

                # Final summary
                iprint("✓ FlowAgent built successfully!")
                iprint(f"   Agent: {agent.amd.name}")
                iprint(f"   Custom Tools: {tools_added}")
                iprint(f"   MCP Tools: {mcp_tools_count}")
                iprint(f"   MCP Server: {'✓' if self.config.mcp.enabled else '✗'}")
                iprint(f"   A2A Server: {'✓' if self.config.a2a.enabled else '✗'}")
                iprint(f"   Checkpoints: {'✓' if self.config.checkpoint.enabled else '✗'}")
                iprint(f"   Persona: {active_persona.name if active_persona else 'Default'}")

                return agent

            except Exception as e:
                eprint(f"Failed to build FlowAgent: {e}")
                import traceback
                traceback.print_exc()
                raise

    def _build_rate_limiter_config(self) -> dict:
        """Build rate limiter configuration dict"""
        rl = self.config.rate_limiter

        return {
            "features": {
                "rate_limiting": rl.enable_rate_limiting,
                "model_fallback": rl.enable_model_fallback,
                "key_rotation": rl.enable_key_rotation,
                "key_rotation_mode": rl.key_rotation_mode,
                "wait_if_all_exhausted": rl.wait_if_all_exhausted,
            },
            "api_keys": rl.api_keys,
            "fallback_chains": rl.fallback_chains,
            "limits": rl.custom_limits,
        }

    async def _init_world_model(self, agent: FlowAgent):
        """Initialize world model in VFS"""
        session = await agent.session_manager.get_or_create("default")

        # Format world model as YAML for readability
        content_lines = ["# World Model", "# Agent can read and update these facts", ""]

        for key, value in self.config.world_model.items():
            if isinstance(value, dict):
                content_lines.append(f"{key}:")
                for k, v in value.items():
                    content_lines.append(f"  {k}: {v}")
            else:
                content_lines.append(f"{key}: {value}")

        content = "\n".join(content_lines)

        # Create as writable file
        session.vfs.create("world_model", content)
        iprint("World model initialized in VFS")

    async def _process_mcp_config(self, agent: FlowAgent):
        """Process MCP configuration with automatic categorization"""
        if not self._mcp_config_data:
            return

        # Initialize MCP Session Manager
        from toolboxv2.mods.isaa.extras.mcp_session_manager import MCPSessionManager
        self._mcp_session_manager = MCPSessionManager()

        mcp_config = self._mcp_config_data

        if 'mcpServers' not in mcp_config:
            return

        for server_name, server_config in mcp_config['mcpServers'].items():
            try:
                iprint(f"🔄 Processing MCP server: {server_name}")

                # Get session
                session = await self._mcp_session_manager.get_session(server_name, server_config)
                if not session:
                    eprint(f"✗ Failed to create session for: {server_name}")
                    continue

                # Extract capabilities
                capabilities = await self._mcp_session_manager.extract_capabilities(session, server_name)

                # Register tools with categories
                for tool_name, tool_info in capabilities.get('tools', {}).items():
                    wrapper_name = f"{server_name}_{tool_name}"

                    # Create tool wrapper
                    tool_wrapper = self._create_mcp_tool_wrapper(
                        server_name, tool_name, tool_info, session
                    )

                    # Determine categories
                    categories = [
                        f"mcp_{server_name}",
                        "mcp",
                        server_name
                    ]

                    # Register in agent's ToolManager
                    await agent.add_tool(
                        tool_wrapper,
                        name=wrapper_name,
                        description=tool_info.get('description', f"MCP tool: {tool_name}"),
                        category=categories
                    )

                    self._mcp_tools[wrapper_name] = tool_info

                total_tools = len(capabilities.get('tools', {}))
                iprint(f"✓ Loaded {total_tools} tools from {server_name}")

                # Register tool group in RuleSet
                session_obj = await agent.session_manager.get_or_create("default")
                if session_obj.rule_set:
                    tool_names = [f"{server_name}_{t}" for t in capabilities.get('tools', {}).keys()]
                    session_obj.rule_set.register_tool_group(
                        name=f"{server_name}_tools",
                        display_name=f"{server_name.replace('_', ' ').title()} Tools",
                        tool_names=tool_names,
                        trigger_keywords=[server_name.lower()],
                        auto_generated=True
                    )

            except Exception as e:
                eprint(f"✗ Failed to load MCP server {server_name}: {e}")

        # Pass MCP session manager to agent
        agent._mcp_session_manager = self._mcp_session_manager

    def _create_mcp_tool_wrapper(self, server_name: str, tool_name: str, tool_info: dict, session):
        """Create wrapper function for MCP tool"""
        import inspect

        input_schema = tool_info.get('input_schema', {})
        properties = input_schema.get('properties', {})
        required_params = set(input_schema.get('required', []))

        # Build parameters
        parameters = []
        for param_name, param_info in properties.items():
            param_type = param_info.get('type', 'string')
            python_type = {
                'string': str, 'integer': int, 'number': float,
                'boolean': bool, 'array': list, 'object': dict
            }.get(param_type, str)

            if param_name in required_params:
                param = inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=python_type)
            else:
                param = inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=python_type, default=None)
            parameters.append(param)

        async def tool_wrapper(**kwargs):
            try:
                # Filter None values for optional params
                arguments = {k: v for k, v in kwargs.items() if v is not None or k in required_params}

                # Validate required
                missing = required_params - set(arguments.keys())
                if missing:
                    raise ValueError(f"Missing required parameters: {missing}")

                result = await session.call_tool(tool_name, arguments)

                if hasattr(result, 'content') and result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        return content.text
                    elif hasattr(content, 'data'):
                        return content.data
                    return str(content)

                return "No content returned"

            except Exception as e:
                raise RuntimeError(f"Error executing {tool_name}: {str(e)}")

        # Set metadata
        tool_wrapper.__name__ = f"{server_name}_{tool_name}"
        tool_wrapper.__doc__ = tool_info.get('description', f"MCP tool: {tool_name}")

        if parameters:
            tool_wrapper.__signature__ = inspect.Signature(parameters)

        return tool_wrapper

    async def _apply_persona_to_ruleset(self, agent: FlowAgent):
        """Apply persona patterns to RuleSet"""
        if not self._persona_patterns:
            return

        session = await agent.session_manager.get_or_create("default")
        if not session.rule_set:
            return

        for pattern_info in self._persona_patterns:
            # Add as learned pattern
            session.rule_set.learn_pattern(
                pattern=pattern_info["pattern"],
                source_situation="persona_config",
                confidence=0.9,
                category="persona",
                tags=pattern_info.get("traits", [])
            )

            # If has custom instructions, add as rule
            if pattern_info.get("instructions"):
                session.rule_set.add_rule(
                    situation="general",
                    intent="respond",
                    instructions=[pattern_info["instructions"]],
                    rule_id=f"persona_{pattern_info['profile_name']}",
                    confidence=0.95
                )

        # Update VFS system_context with persona info
        if self.config.active_persona:
            persona_data = self.config.persona_profiles.get(self.config.active_persona, {})
            persona_context = self._build_persona_context(persona_data)

            # Append to system_context
            if "system_context" in session.vfs.files:
                current = session.vfs.files["system_context"].content
                session.vfs.files["system_context"].content = current + "\n" + persona_context

        iprint("Persona patterns applied to RuleSet")

    def _build_persona_context(self, persona_data: dict) -> str:
        """Build persona context string for VFS"""
        lines = [
            "",
            "# Active Persona",
            f"Name: {persona_data.get('name', 'Default')}",
            f"Style: {persona_data.get('style', 'professional')}",
            f"Tone: {persona_data.get('tone', 'friendly')}",
        ]

        traits = persona_data.get('personality_traits', [])
        if traits:
            lines.append(f"Traits: {', '.join(traits)}")

        instructions = persona_data.get('custom_instructions', '')
        if instructions:
            lines.append(f"Instructions: {instructions}")

        return "\n".join(lines)

    # =========================================================================
    # FACTORY METHODS
    # =========================================================================

    @classmethod
    def create_developer_agent(
        cls,
        name: str = "DeveloperAgent",
        with_mcp: bool = True,
        with_a2a: bool = False
    ) -> 'FlowAgentBuilder':
        """Create a pre-configured developer agent"""
        builder = (cls()
                   .with_name(name)
                   .with_developer_persona()
                   .with_checkpointing(enabled=True, interval_seconds=300)
                   .verbose(True))

        if with_mcp:
            builder.enable_mcp_server(port=8001)
        if with_a2a:
            builder.enable_a2a_server(port=5001)

        return builder

    @classmethod
    def create_analyst_agent(cls, name: str = "AnalystAgent") -> 'FlowAgentBuilder':
        """Create a pre-configured data analyst agent"""
        return (cls()
                .with_name(name)
                .with_analyst_persona()
                .with_checkpointing(enabled=True)
                .verbose(False))

    @classmethod
    def create_general_assistant(
        cls,
        name: str = "AssistantAgent",
        full_integration: bool = True
    ) -> 'FlowAgentBuilder':
        """Create a general-purpose assistant with full integration"""
        builder = (cls()
                   .with_name(name)
                   .with_assistant_persona()
                   .with_checkpointing(enabled=True))

        if full_integration:
            builder.enable_mcp_server()
            builder.enable_a2a_server()

        return builder

    @classmethod
    def create_creative_agent(cls, name: str = "CreativeAgent") -> 'FlowAgentBuilder':
        """Create a creative assistant agent"""
        return (cls()
                .with_name(name)
                .with_creative_persona()
                .with_temperature(0.8)
                .with_checkpointing(enabled=True))

    @classmethod
    def create_executive_agent(
        cls,
        name: str = "ExecutiveAgent",
        with_integrations: bool = True
    ) -> 'FlowAgentBuilder':
        """Create an executive assistant agent"""
        builder = (cls()
                   .with_name(name)
                   .with_executive_persona()
                   .with_checkpointing(enabled=True))

        if with_integrations:
            builder.enable_a2a_server()

        return builder


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example usage of the new FlowAgentBuilder"""

    # Example 1: Simple developer agent
    agent = await (FlowAgentBuilder()
                   .with_name("MyDev")
                   .with_developer_persona()
                   .with_models("openrouter/anthropic/claude-3-haiku", "openrouter/openai/gpt-4o")
                   .with_system_message("You are an expert Python developer.")
                   .with_temperature(0.7)
                   .with_checkpointing(enabled=True)
                   .build())

    # Example 2: Agent with rate limiter configuration
    agent2 = await (FlowAgentBuilder()
                    .with_name("RateLimitedAgent")
                    .with_rate_limiter(
                        enable_model_fallback=True,
                        key_rotation_mode="drain"
                    )
                    .add_api_key("vertex_ai", "AIza_KEY_1")
                    .add_api_key("vertex_ai", "AIza_KEY_2")
                    .add_fallback_chain(
                        "vertex_ai/gemini-2.5-pro",
                        ["vertex_ai/gemini-2.5-flash", "vertex_ai/gemini-2.0-flash"]
                    )
                    .set_model_limits("vertex_ai/gemini-2.5-pro", rpm=2, input_tpm=32000, is_free_tier=True)
                    .build())

    # Example 3: Agent with world model
    agent3 = await (FlowAgentBuilder()
                    .with_name("WorldAware")
                    .with_world_model({
                        "project_name": "MyProject",
                        "environment": "production",
                        "team_members": ["Alice", "Bob"],
                        "deadlines": {"phase1": "2025-02-01"}
                    })
                    .build())

    # Example 4: Agent with MCP tools
    mcp_config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@anthropic/mcp-server-filesystem", "/workspace"]
            }
        }
    }

    agent4 = await (FlowAgentBuilder()
                    .with_name("MCPAgent")
                    .load_mcp_tools_from_config(mcp_config)
                    .enable_mcp_server(port=8000)
                    .build())

    # Example 5: Using factory methods
    dev_agent = await FlowAgentBuilder.create_developer_agent("QuickDev").build()

    print("All agents created successfully!")


if __name__ == "__main__":
    asyncio.run(example_usage())
