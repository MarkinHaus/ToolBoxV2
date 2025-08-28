import platform
import shutil

import asyncio
import json
import yaml
import logging
import os
import sys
import inspect
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Type
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from dataclasses import asdict
from datetime import datetime
import uuid

# Import agent components
from .agent import (
    FlowAgent,
    AgentModelData,
    PersonaConfig,
    FormatConfig,
    ResponseFormat,
    TextLength,
    VariableManager,
    LITELLM_AVAILABLE,
    A2A_AVAILABLE,
    MCP_AVAILABLE,
    OTEL_AVAILABLE
)

# Framework imports
if LITELLM_AVAILABLE:
    from litellm import BudgetManager
    import litellm

if OTEL_AVAILABLE:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

if MCP_AVAILABLE:
    from mcp.server.fastmcp import FastMCP

if A2A_AVAILABLE:
    from python_a2a import A2AServer, AgentCard

from toolboxv2 import get_logger

logger = get_logger()

def detect_shell() -> tuple[str, str]:
    """
    Detects the best available shell and the argument to execute a command.
    Returns:
        A tuple of (shell_executable, command_argument).
        e.g., ('/bin/bash', '-c') or ('powershell.exe', '-Command')
    """
    if platform.system() == "Windows":
        if shell_path := shutil.which("pwsh"):
            return shell_path, "-Command"
        if shell_path := shutil.which("powershell"):
            return shell_path, "-Command"
        return "cmd.exe", "/c"

    shell_env = os.environ.get("SHELL")
    if shell_env and shutil.which(shell_env):
        return shell_env, "-c"

    for shell in ["bash", "zsh", "sh"]:
        if shell_path := shutil.which(shell):
            return shell_path, "-c"

    return "/bin/sh", "-c"

# ===== PRODUCTION CONFIGURATION MODELS =====

class MCPConfig(BaseModel):
    """MCP server and tools configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = False
    config_path: Optional[str] = None  # Path to MCP tools config file
    server_name: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8000
    auto_expose_tools: bool = True
    tools_from_config: List[Dict[str, Any]] = Field(default_factory=list)


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


class TelemetryConfig(BaseModel):
    """OpenTelemetry configuration"""
    enabled: bool = False
    service_name: Optional[str] = None
    endpoint: Optional[str] = None  # OTLP endpoint
    console_export: bool = True
    batch_export: bool = True
    sample_rate: float = 1.0


class CheckpointConfig(BaseModel):
    """Checkpoint configuration"""
    enabled: bool = True
    interval_seconds: int = 300  # 5 minutes
    max_checkpoints: int = 10
    checkpoint_dir: str = "./checkpoints"
    auto_save_on_exit: bool = True


class AgentConfig(BaseModel):
    """Complete agent configuration for loading/saving"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Basic settings
    name: str = "ProductionAgent"
    description: str = "Production-ready PocketFlow agent"
    version: str = "2.0.0"

    # LLM settings
    fast_llm_model: str = "openrouter/anthropic/claude-3-haiku"
    complex_llm_model: str = "openrouter/openai/gpt-4o"
    system_message: str = """You are a production-ready autonomous agent with advanced capabilities including:
- Native MCP tool integration for extensible functionality
- A2A compatibility for agent-to-agent communication
- Dynamic task planning and execution with adaptive reflection
- Advanced context management with session awareness
- Variable system for dynamic content generation
- Checkpoint/resume capabilities for reliability

Always utilize available tools when they can help solve the user's request efficiently."""

    temperature: float = 0.7
    max_tokens_output: int = 2048
    max_tokens_input: int = 32768
    api_key_env_var: Optional[str] = "OPENROUTER_API_KEY"
    use_fast_response: bool = True

    # Features
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    # Agent behavior
    max_parallel_tasks: int = 3
    verbose_logging: bool = False

    # Persona and formatting
    active_persona: Optional[str] = None
    persona_profiles: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    default_format_config: Optional[Dict[str, Any]] = None

    # Custom variables and world model
    custom_variables: Dict[str, Any] = Field(default_factory=dict)
    initial_world_model: Dict[str, Any] = Field(default_factory=dict)


# ===== PRODUCTION FLOWAGENT BUILDER =====

class FlowAgentBuilder:
    """Production-ready FlowAgent builder focused on MCP, A2A, and robust deployment"""

    def __init__(self, config: AgentConfig = None, config_path: str = None):
        """Initialize builder with configuration"""

        if config and config_path:
            raise ValueError("Provide either config object or config_path, not both")

        if config_path:
            self.config = self.load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = AgentConfig()

        # Runtime components
        self._custom_tools: Dict[str, tuple[Callable, str]] = {}
        self._mcp_tools: Dict[str, Dict] = {}
        self._budget_manager: Optional[BudgetManager] = None
        self._tracer_provider: Optional[TracerProvider] = None
        self._mcp_server: Optional[FastMCP] = None
        self._a2a_server: Optional[Any] = None

        # Set logging level
        if self.config.verbose_logging:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info(f"FlowAgent Builder initialized: {self.config.name}")

    # ===== CONFIGURATION MANAGEMENT =====

    def load_config(self, config_path: str) -> AgentConfig:
        """Load agent configuration from file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            return AgentConfig(**data)

        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def save_config(self, config_path: str, format: str = 'yaml'):
        """Save current configuration to file"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = self.config.model_dump()

            with open(path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(data, f, indent=2)

            logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise

    @classmethod
    def from_config_file(cls, config_path: str) -> 'FlowAgentBuilder':
        """Create builder from configuration file"""
        return cls(config_path=config_path)

    # ===== FLUENT BUILDER API =====

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
        """Set system message"""
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
            logger.info(f"Budget manager enabled: ${max_cost}")
        else:
            logger.warning("LiteLLM not available, budget manager disabled")
        return self

    def verbose(self, enable: bool = True) -> 'FlowAgentBuilder':
        """Enable verbose logging"""
        self.config.verbose_logging = enable
        if enable:
            logging.getLogger().setLevel(logging.DEBUG)
        return self

    # ===== MCP INTEGRATION =====

    def enable_mcp_server(self, host: str = "0.0.0.0", port: int = 8000,
                          server_name: str = None) -> 'FlowAgentBuilder':
        """Enable MCP server"""
        if not MCP_AVAILABLE:
            logger.warning("MCP not available, cannot enable server")
            return self

        self.config.mcp.enabled = True
        self.config.mcp.host = host
        self.config.mcp.port = port
        self.config.mcp.server_name = server_name or f"{self.config.name}_MCP"

        logger.info(f"MCP server enabled: {host}:{port}")
        return self

    def _load_mcp_server_tools(self, server_name: str, server_config: Dict[str, Any]):
        """Load tools from MCP server configuration with actual command execution"""
        command = server_config.get('command')
        args = server_config.get('args', [])
        env = server_config.get('env', {})

        if not command:
            logger.warning(f"No command specified for MCP server {server_name}")
            return

        # Create a tool that can execute the MCP server command
        async def mcp_server_tool(query: str = "", **kwargs) -> str:
            """Tool backed by actual MCP server execution"""
            try:
                return await self._execute_mcp_server(command, args, env, query, **kwargs)
            except Exception as e:
                logger.error(f"MCP server tool {server_name} failed: {e}")
                return f"Error executing {server_name}: {str(e)}"

        self._mcp_tools[server_name] = {
            'function': mcp_server_tool,
            'command': command,
            'args': args,
            'env': env,
            'description': f"MCP server tool: {server_name}",
            'server_type': 'command_execution'
        }

        logger.info(f"Registered MCP server tool: {server_name} ({command})")

    async def _execute_mcp_server(self, command: str, args: List[str], env: Dict[str, str],
                                  query: str, **kwargs) -> str:
        """Execute MCP server command and handle communication"""
        try:
            # Prepare environment
            process_env = os.environ.copy()
            process_env.update(env)

            # Build full command
            shell_exe, cmd_flag = detect_shell()
            full_command = [shell_exe, cmd_flag, command] + args

            # Create the subprocess
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
                cwd=os.getcwd()
            )

            # For MCP, we need to send JSON-RPC messages
            # This is a simplified implementation - in production you'd want full MCP protocol
            mcp_request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/call",
                "params": {
                    "name": "process_query",
                    "arguments": {"query": query, **kwargs}
                }
            }

            request_json = json.dumps(mcp_request) + "\n"

            # Send request and get response
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=request_json.encode()),
                timeout=30.0  # 30 second timeout
            )

            # Parse response
            if process.returncode == 0:
                response_text = stdout.decode().strip()
                if response_text:
                    try:
                        # Try to parse as JSON-RPC response
                        response_data = json.loads(response_text)
                        if "result" in response_data:
                            return str(response_data["result"])
                        elif "error" in response_data:
                            return f"MCP Error: {response_data['error']}"
                        else:
                            return response_text
                    except json.JSONDecodeError:
                        # Return raw output if not valid JSON
                        return response_text
                else:
                    return "MCP server returned empty response"
            else:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                return f"MCP server failed (code {process.returncode}): {error_msg}"

        except asyncio.TimeoutError:
            logger.error(f"MCP server timeout for command: {command}")
            return "MCP server request timed out"
        except Exception as e:
            logger.error(f"MCP server execution error: {e}")
            return f"MCP server execution failed: {str(e)}"

    async def _execute_mcp_stdio_server(self, command: str, args: List[str], env: Dict[str, str]) -> Optional[Any]:
        """Execute MCP server using stdio transport (more robust MCP communication)"""
        if not MCP_AVAILABLE:
            logger.warning("MCP not available for stdio server execution")
            return None

        try:
            # This would use the actual MCP client to communicate with the server
            # For now, this is a placeholder for full MCP protocol implementation
            from mcp.client.stdio import stdio_client
            from mcp import ClientSession, StdioServerParameters

            # Prepare environment
            process_env = os.environ.copy()
            process_env.update(env)

            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=process_env
            )

            # This would establish proper MCP communication
            # Implementation would depend on full MCP client integration
            logger.info(f"Would establish MCP stdio connection to: {command} {' '.join(args)}")

            return None  # Placeholder - full implementation would return MCP client session

        except Exception as e:
            logger.error(f"Failed to establish MCP stdio connection: {e}")
            return None

    def load_mcp_tools_from_config(self, config_path: str) -> 'FlowAgentBuilder':
        """Enhanced MCP config loading with better error handling and validation"""
        if not MCP_AVAILABLE:
            logger.warning("MCP not available, skipping tool loading")
            return self

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"MCP config not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    mcp_config = yaml.safe_load(f)
                else:
                    mcp_config = json.load(f)

            # Parse MCP tools from official config format
            tools_loaded = 0

            # Handle standard MCP server configuration
            if 'mcpServers' in mcp_config:
                for server_name, server_config in mcp_config['mcpServers'].items():
                    # Validate server config
                    if not self._validate_mcp_server_config(server_name, server_config):
                        continue

                    self._load_mcp_server_tools(server_name, server_config)
                    tools_loaded += 1

                    logger.info(
                        f"Loaded MCP server: {server_name} - Command: {server_config.get('command')} {' '.join(server_config.get('args', []))}")

            # Handle direct tools configuration
            elif 'tools' in mcp_config:
                for tool_config in mcp_config['tools']:
                    self._load_direct_mcp_tool(tool_config)
                    tools_loaded += 1

            # Store config path for later use
            self.config.mcp.config_path = str(config_path)
            self.config.mcp.tools_from_config = mcp_config.get('tools', [])

            logger.info(f"Successfully loaded {tools_loaded} MCP configurations from {config_path}")

        except Exception as e:
            logger.error(f"Failed to load MCP config from {config_path}: {e}")
            raise

        return self

    @staticmethod
    def _validate_mcp_server_config(server_name: str, server_config: Dict[str, Any]) -> bool:
        """Validate MCP server configuration"""
        command = server_config.get('command')
        if not command:
            logger.error(f"MCP server {server_name} missing 'command' field")
            return False

        # Check if command exists and is executable
        if command in ['npx', 'node', 'python', 'python3', 'docker']:
            # These are common commands, assume they exist
            return True

        # For other commands, check if they exist
        import shutil
        if not shutil.which(command):
            logger.warning(f"MCP server {server_name}: command '{command}' not found in PATH")
            # Don't fail completely, just warn - the command might be available at runtime

        args = server_config.get('args', [])
        if not isinstance(args, list):
            logger.error(f"MCP server {server_name}: 'args' must be a list")
            return False

        env = server_config.get('env', {})
        if not isinstance(env, dict):
            logger.error(f"MCP server {server_name}: 'env' must be a dictionary")
            return False

        logger.debug(f"Validated MCP server config: {server_name}")
        return True

    def _load_direct_mcp_tool(self, tool_config: Dict[str, Any]):
        """Load tool from direct configuration"""
        name = tool_config.get('name')
        description = tool_config.get('description', '')
        function_code = tool_config.get('function_code')

        if not name or not function_code:
            logger.warning(f"Incomplete tool config: {tool_config}")
            return

        # Create function from code
        try:
            namespace = {"__builtins__": __builtins__}
            exec(function_code, namespace)

            # Find the function
            func = None
            for obj in namespace.values():
                if callable(obj) and not getattr(obj, '__name__', '').startswith('_'):
                    func = obj
                    break

            if func:
                self._mcp_tools[name] = {
                    'function': func,
                    'description': description,
                    'source': 'code'
                }
                logger.debug(f"Loaded MCP tool from code: {name}")

        except Exception as e:
            logger.error(f"Failed to load MCP tool {name}: {e}")

    def add_mcp_tool_from_code(self, name: str, code: str, description: str = "") -> 'FlowAgentBuilder':
        """Add MCP tool from code string"""
        tool_config = {
            'name': name,
            'description': description,
            'function_code': code
        }
        self._load_direct_mcp_tool(tool_config)
        return self

    # ===== A2A INTEGRATION =====

    def enable_a2a_server(self, host: str = "0.0.0.0", port: int = 5000,
                          agent_name: str = None, agent_description: str = None) -> 'FlowAgentBuilder':
        """Enable A2A server for agent-to-agent communication"""
        if not A2A_AVAILABLE:
            logger.warning("A2A not available, cannot enable server")
            return self

        self.config.a2a.enabled = True
        self.config.a2a.host = host
        self.config.a2a.port = port
        self.config.a2a.agent_name = agent_name or self.config.name
        self.config.a2a.agent_description = agent_description or self.config.description

        logger.info(f"A2A server enabled: {host}:{port}")
        return self

    # ===== TELEMETRY INTEGRATION =====

    def enable_telemetry(self, service_name: str = None, endpoint: str = None,
                         console_export: bool = True) -> 'FlowAgentBuilder':
        """Enable OpenTelemetry tracing"""
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, cannot enable telemetry")
            return self

        self.config.telemetry.enabled = True
        self.config.telemetry.service_name = service_name or self.config.name
        self.config.telemetry.endpoint = endpoint
        self.config.telemetry.console_export = console_export

        # Initialize tracer provider
        self._tracer_provider = TracerProvider()
        trace.set_tracer_provider(self._tracer_provider)

        # Add exporters
        if console_export:
            console_exporter = ConsoleSpanExporter()
            span_processor = BatchSpanProcessor(console_exporter)
            self._tracer_provider.add_span_processor(span_processor)

        if endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
                otlp_processor = BatchSpanProcessor(otlp_exporter)
                self._tracer_provider.add_span_processor(otlp_processor)
            except Exception as e:
                logger.warning(f"Failed to setup OTLP exporter: {e}")

        logger.info(f"Telemetry enabled for service: {service_name}")
        return self

    # ===== CHECKPOINT CONFIGURATION =====

    def with_checkpointing(self, enabled: bool = True, interval_seconds: int = 300,
                           checkpoint_dir: str = "./checkpoints", max_checkpoints: int = 10) -> 'FlowAgentBuilder':
        """Configure checkpointing"""
        self.config.checkpoint.enabled = enabled
        self.config.checkpoint.interval_seconds = interval_seconds
        self.config.checkpoint.checkpoint_dir = checkpoint_dir
        self.config.checkpoint.max_checkpoints = max_checkpoints

        if enabled:
            # Ensure checkpoint directory exists
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpointing enabled: {checkpoint_dir} (every {interval_seconds}s)")

        return self

    # ===== TOOL MANAGEMENT =====

    def add_tool(self, func: Callable, name: str = None, description: str = None) -> 'FlowAgentBuilder':
        """Add custom tool function"""
        tool_name = name or func.__name__
        self._custom_tools[tool_name] = (func, description or func.__doc__)

        logger.info(f"Tool added: {tool_name}")
        return self

    def add_tools_from_module(self, module, prefix: str = "", exclude: List[str] = None) -> 'FlowAgentBuilder':
        """Add all functions from a module as tools"""
        exclude = exclude or []

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name in exclude or name.startswith('_'):
                continue

            tool_name = f"{prefix}{name}" if prefix else name
            self.add_tool(obj, name=tool_name)

        logger.info(f"Added tools from module {module.__name__}")
        return self

    # ===== PERSONA MANAGEMENT =====

    def add_persona_profile(self, profile_name: str, name: str, style: str = "professional",
                            tone: str = "friendly", personality_traits: List[str] = None,
                            custom_instructions: str = "", response_format: str = None,
                            text_length: str = None) -> 'FlowAgentBuilder':
        """Add a persona profile with optional format configuration"""

        if personality_traits is None:
            personality_traits = ["helpful", "concise"]

        # Create persona config
        persona_data = {
            "name": name,
            "style": style,
            "tone": tone,
            "personality_traits": personality_traits,
            "custom_instructions": custom_instructions,
            "apply_method": "system_prompt",
            "integration_level": "light"
        }

        # Add format config if specified
        if response_format or text_length:
            format_config = {
                "response_format": response_format or "frei-text",
                "text_length": text_length or "chat-conversation",
                "custom_instructions": "",
                "strict_format_adherence": True,
                "quality_threshold": 0.7
            }
            persona_data["format_config"] = format_config

        self.config.persona_profiles[profile_name] = persona_data
        logger.info(f"Persona profile added: {profile_name}")
        return self

    def set_active_persona(self, profile_name: str) -> 'FlowAgentBuilder':
        """Set active persona profile"""
        if profile_name in self.config.persona_profiles:
            self.config.active_persona = profile_name
            logger.info(f"Active persona set: {profile_name}")
        else:
            logger.warning(f"Persona profile not found: {profile_name}")
        return self

    def with_developer_persona(self, name: str = "Senior Developer") -> 'FlowAgentBuilder':
        """Add and set a pre-built developer persona"""
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
        """Add and set a pre-built analyst persona"""
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
        """Add and set a pre-built general assistant persona"""
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
        """Add and set a pre-built creative persona"""
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
        """Add and set a pre-built executive persona"""
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

    # ===== VARIABLE MANAGEMENT =====

    def with_custom_variables(self, variables: Dict[str, Any]) -> 'FlowAgentBuilder':
        """Add custom variables"""
        self.config.custom_variables.update(variables)
        return self

    def with_world_model(self, world_model: Dict[str, Any]) -> 'FlowAgentBuilder':
        """Set initial world model"""
        self.config.initial_world_model.update(world_model)
        return self

    # ===== VALIDATION =====

    def validate_config(self) -> Dict[str, List[str]]:
        """Validate the current configuration"""
        issues = {"errors": [], "warnings": []}

        # Validate required settings
        if not self.config.fast_llm_model:
            issues["errors"].append("Fast LLM model not specified")
        if not self.config.complex_llm_model:
            issues["errors"].append("Complex LLM model not specified")

        # Validate MCP configuration
        if self.config.mcp.enabled and not MCP_AVAILABLE:
            issues["errors"].append("MCP enabled but MCP not available")

        # Validate A2A configuration
        if self.config.a2a.enabled and not A2A_AVAILABLE:
            issues["errors"].append("A2A enabled but A2A not available")

        # Validate telemetry
        if self.config.telemetry.enabled and not OTEL_AVAILABLE:
            issues["errors"].append("Telemetry enabled but OpenTelemetry not available")

        # Validate personas
        if self.config.active_persona and self.config.active_persona not in self.config.persona_profiles:
            issues["errors"].append(f"Active persona '{self.config.active_persona}' not found in profiles")

        # Validate checkpoint directory
        if self.config.checkpoint.enabled:
            try:
                Path(self.config.checkpoint.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues["warnings"].append(f"Cannot create checkpoint directory: {e}")

        return issues

    # ===== MAIN BUILD METHOD =====

    async def build(self) -> FlowAgent:
        """Build the production-ready FlowAgent"""

        logger.info(f"Building production FlowAgent: {self.config.name}")

        # Validate configuration
        validation_issues = self.validate_config()
        if validation_issues["errors"]:
            error_msg = f"Configuration validation failed: {', '.join(validation_issues['errors'])}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log warnings
        for warning in validation_issues["warnings"]:
            logger.warning(f"Configuration warning: {warning}")

        try:
            # 1. Setup API configuration
            api_key = None
            if self.config.api_key_env_var:
                api_key = os.getenv(self.config.api_key_env_var)
                if not api_key:
                    logger.warning(f"API key env var {self.config.api_key_env_var} not set")

            # 2. Create persona if configured
            active_persona = None
            if self.config.active_persona and self.config.active_persona in self.config.persona_profiles:
                persona_data = self.config.persona_profiles[self.config.active_persona]

                # Create FormatConfig if present
                format_config = None
                if "format_config" in persona_data:
                    fc_data = persona_data.pop("format_config")
                    format_config = FormatConfig(
                        response_format=ResponseFormat(fc_data.get("response_format", "frei-text")),
                        text_length=TextLength(fc_data.get("text_length", "chat-conversation")),
                        custom_instructions=fc_data.get("custom_instructions", ""),
                        strict_format_adherence=fc_data.get("strict_format_adherence", True),
                        quality_threshold=fc_data.get("quality_threshold", 0.7)
                    )

                active_persona = PersonaConfig(**persona_data)
                active_persona.format_config = format_config

                logger.info(f"Using persona: {active_persona.name}")

            # 3. Create AgentModelData
            amd = AgentModelData(
                name=self.config.name,
                fast_llm_model=self.config.fast_llm_model,
                complex_llm_model=self.config.complex_llm_model,
                system_message=self.config.system_message,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_output,
                max_input_tokens=self.config.max_tokens_input,
                api_key=api_key,
                budget_manager=self._budget_manager,
                persona=active_persona,
                use_fast_response=self.config.use_fast_response
            )

            # 4. Create FlowAgent
            agent = FlowAgent(
                amd=amd,
                world_model=self.config.initial_world_model.copy(),
                verbose=self.config.verbose_logging,
                enable_pause_resume=self.config.checkpoint.enabled,
                checkpoint_interval=self.config.checkpoint.interval_seconds,
                max_parallel_tasks=self.config.max_parallel_tasks
            )

            # 5. Add custom variables
            for key, value in self.config.custom_variables.items():
                agent.set_variable(key, value)

            # 6. Add custom tools
            tools_added = 0
            for tool_name, (tool_func, tool_description) in self._custom_tools.items():
                try:
                    await agent.add_tool(tool_func, tool_name, tool_description)
                    tools_added += 1
                except Exception as e:
                    logger.error(f"Failed to add tool {tool_name}: {e}")

            # 7. Add MCP tools
            for tool_name, tool_info in self._mcp_tools.items():
                try:
                    await agent.add_tool(
                        tool_info['function'],
                        tool_name,
                        tool_info['description']
                    )
                    tools_added += 1
                except Exception as e:
                    logger.error(f"Failed to add MCP tool {tool_name}: {e}")

            # 8. Setup MCP server
            if self.config.mcp.enabled and MCP_AVAILABLE:
                try:
                    agent.setup_mcp_server(
                        host=self.config.mcp.host,
                        port=self.config.mcp.port,
                        name=self.config.mcp.server_name
                    )
                    logger.info("MCP server configured")
                except Exception as e:
                    logger.error(f"Failed to setup MCP server: {e}")

            # 9. Setup A2A server
            if self.config.a2a.enabled and A2A_AVAILABLE:
                try:
                    agent.setup_a2a_server(
                        host=self.config.a2a.host,
                        port=self.config.a2a.port
                    )
                    logger.info("A2A server configured")
                except Exception as e:
                    logger.error(f"Failed to setup A2A server: {e}")

            # 10. Initialize enhanced session context
            try:
                await agent.initialize_session_context(max_history=200)
                logger.info("Enhanced session context initialized")
            except Exception as e:
                logger.warning(f"Session context initialization failed: {e}")

            # Final summary
            logger.info(f"ok FlowAgent built successfully!")
            logger.info(f"   Agent: {agent.amd.name}")
            logger.info(f"   Tools: {tools_added}")
            logger.info(f"   MCP: {'ok' if self.config.mcp.enabled else 'F'}")
            logger.info(f"   A2A: {'ok' if self.config.a2a.enabled else 'F'}")
            logger.info(f"   Telemetry: {'ok' if self.config.telemetry.enabled else 'F'}")
            logger.info(f"   Checkpoints: {'ok' if self.config.checkpoint.enabled else 'F'}")
            logger.info(f"   Persona: {active_persona.name if active_persona else 'Default'}")

            return agent

        except Exception as e:
            logger.error(f"Failed to build FlowAgent: {e}")
            raise

    # ===== FACTORY METHODS =====

    @classmethod
    def create_developer_agent(cls, name: str = "DeveloperAgent",
                               with_mcp: bool = True, with_a2a: bool = False) -> 'FlowAgentBuilder':
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
    def create_analyst_agent(cls, name: str = "AnalystAgent",
                             with_telemetry: bool = True) -> 'FlowAgentBuilder':
        """Create a pre-configured data analyst agent"""
        builder = (cls()
                   .with_name(name)
                   .with_analyst_persona()
                   .with_checkpointing(enabled=True)
                   .verbose(False))

        if with_telemetry:
            builder.enable_telemetry(console_export=True)

        return builder

    @classmethod
    def create_general_assistant(cls, name: str = "AssistantAgent",
                                 full_integration: bool = True) -> 'FlowAgentBuilder':
        """Create a general-purpose assistant with full integration"""
        builder = (cls()
                   .with_name(name)
                   .with_assistant_persona()
                   .with_checkpointing(enabled=True))

        if full_integration:
            builder.enable_mcp_server()
            builder.enable_a2a_server()
            builder.enable_telemetry()

        return builder

    @classmethod
    def create_creative_agent(cls, name: str = "CreativeAgent") -> 'FlowAgentBuilder':
        """Create a creative assistant agent"""
        return (cls()
                .with_name(name)
                .with_creative_persona()
                .with_temperature(0.8)  # More creative
                .with_checkpointing(enabled=True))

    @classmethod
    def create_executive_agent(cls, name: str = "ExecutiveAgent",
                               with_integrations: bool = True) -> 'FlowAgentBuilder':
        """Create an executive assistant agent"""
        builder = (cls()
                   .with_name(name)
                   .with_executive_persona()
                   .with_checkpointing(enabled=True))

        if with_integrations:
            builder.enable_a2a_server()  # Executives need A2A for delegation
            builder.enable_telemetry()  # Need metrics

        return builder


# ===== EXAMPLE USAGE =====

async def example_production_usage():
    """Production usage example with full features"""

    logger.info("=== Production FlowAgent Builder Example ===")

    # Example 1: Developer agent with full MCP integration
    logger.info("Creating developer agent with MCP integration...")

    # Add a custom tool
    def get_system_info():
        """Get basic system information"""
        import platform
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()
        }

    developer_agent = await (FlowAgentBuilder
                             .create_developer_agent("ProductionDev", with_mcp=True, with_a2a=True)
                             .add_tool(get_system_info, "get_system_info", "Get system information")
                             .enable_telemetry(console_export=True)
                             .with_custom_variables({
        "project_name": "FlowAgent Production",
        "environment": "production"
    })
                             .build())

    # Test the developer agent
    dev_response = await developer_agent.a_run(
        "Hello! I'm working on {{ project_name }}. Can you tell me about the system and create a simple Python function?"
    )
    logger.info(f"Developer agent response: {dev_response[:200]}...")

    # Example 2: Load from configuration file
    logger.info("\nTesting configuration save/load...")

    # Save current config
    config_path = "/tmp/production_agent_config.yaml"
    builder = FlowAgentBuilder.create_analyst_agent("ConfigTestAgent")
    builder.save_config(config_path)

    # Load from config
    loaded_builder = FlowAgentBuilder.from_config_file(config_path)
    config_agent = await loaded_builder.build()

    config_response = await config_agent.a_run("Analyze this data: [1, 2, 3, 4, 5]")
    logger.info(f"Config-loaded agent response: {config_response[:150]}...")

    # Example 3: Agent with MCP tools from config
    logger.info("\nTesting MCP tools integration...")

    # Create a sample MCP config
    mcp_config = {
        "tools": [
            {
                "name": "weather_checker",
                "description": "Check weather for a location",
                "function_code": '''
async def weather_checker(location: str) -> str:
    """Mock weather checker"""
    import random
    conditions = ["sunny", "cloudy", "rainy", "snowy"]
    temp = random.randint(-10, 35)
    condition = random.choice(conditions)
    return f"Weather in {location}: {condition}, {temp}°C"
'''
            }
        ]
    }

    mcp_config_path = "/tmp/mcp_tools_config.json"
    with open(mcp_config_path, 'w') as f:
        json.dump(mcp_config, f, indent=2)

    mcp_agent = await (FlowAgentBuilder()
                       .with_name("MCPTestAgent")
                       .with_assistant_persona()
                       .enable_mcp_server(port=8002)
                       .load_mcp_tools_from_config(mcp_config_path)
                       .build())

    mcp_response = await mcp_agent.a_run("What's the weather like in Berlin?")
    logger.info(f"MCP agent response: {mcp_response[:150]}...")

    # Show agent status
    logger.info("\n=== Agent Status ===")
    status = developer_agent.status(pretty_print=False)
    logger.info(f"Developer agent tools: {len(status['capabilities']['tool_names'])}")
    logger.info(f"MCP agent tools: {len(mcp_agent.shared.get('available_tools', []))}")

    # Cleanup
    await developer_agent.close()
    await config_agent.close()
    await mcp_agent.close()

    logger.info("Production example completed successfully!")


async def example_quick_start():
    """Quick start examples for common scenarios"""

    logger.info("=== Quick Start Examples ===")

    # 1. Simple developer agent
    dev_agent = await FlowAgentBuilder.create_developer_agent("QuickDev").build()
    response1 = await dev_agent.a_run("Create a Python function to validate email addresses")
    logger.info(f"Quick dev response: {response1[:100]}...")
    await dev_agent.close()

    # 2. Analyst with custom data
    analyst_agent = await (FlowAgentBuilder
                           .create_analyst_agent("QuickAnalyst")
                           .with_custom_variables({"dataset": "sales_data_2024"})
                           .build())
    response2 = await analyst_agent.a_run("Analyze the trends in {{ dataset }}")
    logger.info(f"Quick analyst response: {response2[:100]}...")
    await analyst_agent.close()

    # 3. Creative assistant
    creative_agent = await FlowAgentBuilder.create_creative_agent("QuickCreative").build()
    response3 = await creative_agent.a_run("Write a creative story about AI agents collaborating")
    logger.info(f"Quick creative response: {response3[:100]}...")
    await creative_agent.close()

    logger.info("Quick start examples completed!")


if __name__ == "__main__":
    # Run production example
    asyncio.run(example_production_usage())

    # Run quick start examples
    asyncio.run(example_quick_start())
