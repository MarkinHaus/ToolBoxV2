import asyncio
import inspect
import json
import logging
import os
import platform
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from toolboxv2 import Spinner
from toolboxv2.mods.isaa.base.Agent.mda_accomplish import bind_accomplish_to_agent
from toolboxv2.mods.isaa.base.Agent.types import CheckpointConfig

# Import agent components
from .agent import (
    A2A_AVAILABLE,
    LITELLM_AVAILABLE,
    MCP_AVAILABLE,
    OTEL_AVAILABLE,
    AgentModelData,
    FlowAgent,
    FormatConfig,
    PersonaConfig,
    ResponseFormat,
    TextLength,
    eprint,
    iprint,
    wprint,
)

# Framework imports
if LITELLM_AVAILABLE:
    from litellm import BudgetManager
else:
    BudgetManager = object

if OTEL_AVAILABLE:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        #try:
        #    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        #except ImportError:
        OTLPSpanExporter = None
    except ImportError:
        OTEL_AVAILABLE = False
        trace = None
        TracerProvider = None
        BatchSpanProcessor = None
        ConsoleSpanExporter = None
        OTLPSpanExporter = None
else:
    print("WARN: opentelemetry-api, opentelemetry-sdk not found. Observability disabled.")
    OTEL_AVAILABLE = False
    trace = None
    TracerProvider = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None
    OTLPSpanExporter = None

if MCP_AVAILABLE:
    from mcp import ClientSession
else:
    ClientSession = object

if A2A_AVAILABLE:
    pass

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
    tools_from_config: list[dict[str, Any]] = Field(default_factory=list)


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
    api_key_env_var: str | None = "OPENROUTER_API_KEY"
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
    persona_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    default_format_config: Optional[dict[str, Any]] = None

    # Custom variables and world model
    custom_variables: dict[str, Any] = Field(default_factory=dict)
    initial_world_model: dict[str, Any] = Field(default_factory=dict)

    handler_path_or_dict: Optional[str | dict] = Field(default=r"C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\.data\main-DESKTOP-CI57V1L\Agents\rate_limiter_config.json")


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
        self._custom_tools: dict[str, tuple[Callable, str]] = {}
        self._mcp_tools: dict[str, dict] = {}
        from toolboxv2.mods.isaa.extras.mcp_session_manager import MCPSessionManager

        self._mcp_session_manager = MCPSessionManager()

        self._budget_manager: BudgetManager = None
        self._tracer_provider: TracerProvider = None
        self._a2a_server: Any = None

        # Set logging level
        if self.config.verbose_logging:
            logging.getLogger().setLevel(logging.DEBUG)

        iprint(f"FlowAgent Builder initialized: {self.config.name}")

    # ===== CONFIGURATION MANAGEMENT =====

    def load_config(self, config_path: str) -> AgentConfig:
        """Load agent configuration from file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(path, encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            return AgentConfig(**data)

        except Exception as e:
            eprint(f"Failed to load config from {config_path}: {e}")
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

            iprint(f"Configuration saved to {config_path}")

        except Exception as e:
            eprint(f"Failed to save config to {config_path}: {e}")
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

    # ===== MCP INTEGRATION =====

    def enable_mcp_server(self, host: str = "0.0.0.0", port: int = 8000,
                          server_name: str = None) -> 'FlowAgentBuilder':
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

    async def _load_mcp_server_capabilities(self, server_name: str, server_config: dict[str, Any]):
        """Load all capabilities from MCP server with persistent session"""
        try:
            # Get or create persistent session
            session = await self._mcp_session_manager.get_session(server_name, server_config)
            if not session:
                eprint(f"Failed to create session for MCP server: {server_name}")
                return

            # Extract all capabilities
            capabilities = await self._mcp_session_manager.extract_capabilities(session, server_name)

            # Create tool wrappers
            for tool_name, tool_info in capabilities['tools'].items():
                wrapper_name = f"{server_name}_{tool_name}"
                tool_wrapper = self._create_tool_wrapper(server_name, tool_name, tool_info, session)
                self._mcp_tools[wrapper_name] = {
                    'function': tool_wrapper,
                    'description': tool_info['description'],
                    'type': 'tool',
                    'server': server_name,
                    'original_name': tool_name,
                    'input_schema': tool_info.get('input_schema'),
                    'output_schema': tool_info.get('output_schema')
                }

            # Create resource wrappers
            for resource_uri, resource_info in capabilities['resources'].items():
                wrapper_name = f"{server_name}_resource_{resource_info['name'].replace('/', '_')}"
                resource_wrapper = self._create_resource_wrapper(server_name, resource_uri, resource_info, session)

                self._mcp_tools[wrapper_name] = {
                    'function': resource_wrapper,
                    'description': f"Read resource: {resource_info['description']}",
                    'type': 'resource',
                    'server': server_name,
                    'original_uri': resource_uri
                }

            # Create resource template wrappers
            for template_uri, template_info in capabilities['resource_templates'].items():
                wrapper_name = f"{server_name}_template_{template_info['name'].replace('/', '_')}"
                template_wrapper = self._create_resource_template_wrapper(server_name, template_uri, template_info,
                                                                          session)

                self._mcp_tools[wrapper_name] = {
                    'function': template_wrapper,
                    'description': f"Access resource template: {template_info['description']}",
                    'type': 'resource_template',
                    'server': server_name,
                    'original_template': template_uri
                }

            # Create prompt wrappers
            for prompt_name, prompt_info in capabilities['prompts'].items():
                wrapper_name = f"{server_name}_prompt_{prompt_name}"
                prompt_wrapper = self._create_prompt_wrapper(server_name, prompt_name, prompt_info, session)

                self._mcp_tools[wrapper_name] = {
                    'function': prompt_wrapper,
                    'description': f"Execute prompt: {prompt_info['description']}",
                    'type': 'prompt',
                    'server': server_name,
                    'original_name': prompt_name,
                    'arguments': prompt_info.get('arguments', [])
                }

            total_capabilities = (len(capabilities['tools']) +
                                  len(capabilities['resources']) +
                                  len(capabilities['resource_templates']) +
                                  len(capabilities['prompts']))

            iprint(f"Created {total_capabilities} capability wrappers for server: {server_name}")

        except Exception as e:
            eprint(f"Failed to load capabilities from MCP server {server_name}: {e}")

    def _create_tool_wrapper(self, server_name: str, tool_name: str, tool_info: dict, session: ClientSession):
        """Create wrapper function for MCP tool with dynamic signature based on schema"""
        import inspect

        # Extract parameter information from input schema
        input_schema = tool_info.get('input_schema', {})
        output_schema = tool_info.get('output_schema', {})

        # Build parameter list
        parameters = []
        required_params = set(input_schema.get('required', []))
        properties = input_schema.get('properties', {})

        # Create parameters with proper types
        for param_name, param_info in properties.items():
            param_type = param_info.get('type', 'string')
            python_type = {
                'string': str,
                'integer': int,
                'number': float,
                'boolean': bool,
                'array': list,
                'object': dict
            }.get(param_type, str)

            # Determine if parameter is required
            if param_name in required_params:
                param = inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=python_type)
            else:
                # Optional parameters get default None
                param = inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                          annotation=python_type, default=None)
            parameters.append(param)

        # Determine return type from output schema
        return_type = str  # Default
        if output_schema and 'properties' in output_schema:
            output_props = output_schema['properties']
            if len(output_props) == 1:
                # Single property, return its type directly
                prop_info = list(output_props.values())[0]
                prop_type = prop_info.get('type', 'string')
                return_type = {
                    'string': str,
                    'integer': int,
                    'number': float,
                    'boolean': bool,
                    'array': list,
                    'object': dict
                }.get(prop_type, str)
            else:
                # Multiple properties, return dict
                return_type = dict

        # Create the actual function
        async def tool_wrapper(*args, **kwargs):
            try:
                # Map arguments to schema parameters
                arguments = {}
                param_names = list(properties.keys())

                # Map positional args
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        arguments[param_names[i]] = arg

                # Add keyword arguments, filtering out None for optional params
                for key, value in kwargs.items():
                    if value is not None or key in required_params:
                        arguments[key] = value

                # P0 - KRITISCH: MCP Circuit Breaker - Check if server is healthy
                agent_instance = getattr(self, '_agent_instance', None)
                if agent_instance and hasattr(agent_instance, '_check_mcp_circuit_breaker'):
                    if not agent_instance._check_mcp_circuit_breaker(server_name):
                        raise RuntimeError(f"MCP Circuit Breaker OPEN for {server_name} - too many failures")

                # Validate required parameters
                missing_required = required_params - set(arguments.keys())
                if missing_required:
                    raise ValueError(f"Missing required parameters: {missing_required}")

                # Call the actual MCP tool
                result = await session.call_tool(tool_name, arguments)

                # P0 - KRITISCH: Record success for circuit breaker
                if agent_instance and hasattr(agent_instance, '_record_mcp_success'):
                    agent_instance._record_mcp_success(server_name)

                # Handle structured vs unstructured results
                if hasattr(result, 'structuredContent') and result.structuredContent:
                    structured_data = result.structuredContent

                    # If output schema expects single property, extract it
                    if output_schema and 'properties' in output_schema:
                        output_props = output_schema['properties']
                        if len(output_props) == 1:
                            prop_name = list(output_props.keys())[0]
                            if isinstance(structured_data, dict) and prop_name in structured_data:
                                return structured_data[prop_name]

                    return structured_data

                # Fallback to content extraction
                if result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        return content.text
                    elif hasattr(content, 'data'):
                        return content.data
                    else:
                        return str(content)

                return "No content returned"

            except Exception as e:
                # P0 - KRITISCH: Record failure for circuit breaker
                if agent_instance and hasattr(agent_instance, '_record_mcp_failure'):
                    agent_instance._record_mcp_failure(server_name)

                eprint(f"MCP tool {server_name}.{tool_name} failed: {e}")
                raise RuntimeError(f"Error executing {tool_name}: {str(e)}")

        # Set dynamic signature
        signature = inspect.Signature(parameters, return_annotation=return_type)
        tool_wrapper.__signature__ = signature
        tool_wrapper.__name__ = f"{server_name}_{tool_name}"
        tool_wrapper.__doc__ = tool_info.get('description', f"MCP tool: {tool_name}")
        tool_wrapper.__annotations__ = {'return': return_type}

        # Add parameter annotations
        for param in parameters:
            tool_wrapper.__annotations__[param.name] = param.annotation

        return tool_wrapper

    def _create_resource_wrapper(self, server_name: str, resource_uri: str, resource_info: dict,
                                 session: ClientSession):
        """Create wrapper function for MCP resource with proper signature"""
        import inspect

        # Resources typically don't take parameters, return string content
        async def resource_wrapper() -> str:
            """Read MCP resource content"""
            try:
                from pydantic import AnyUrl
                result = await session.read_resource(AnyUrl(resource_uri))

                if result.contents:
                    content = result.contents[0]
                    if hasattr(content, 'text'):
                        return content.text
                    elif hasattr(content, 'data'):
                        # Handle binary data
                        if isinstance(content.data, bytes):
                            return content.data.decode('utf-8', errors='ignore')
                        return str(content.data)
                    else:
                        return str(content)

                return ""

            except Exception as e:
                eprint(f"MCP resource {resource_uri} failed: {e}")
                raise RuntimeError(f"Error reading resource: {str(e)}")

        # Set signature and metadata
        signature = inspect.Signature([], return_annotation=str)
        resource_wrapper.__signature__ = signature
        resource_wrapper.__name__ = f"{server_name}_resource_{resource_info['name'].replace('/', '_').replace(':', '_')}"
        resource_wrapper.__doc__ = f"Read MCP resource: {resource_info.get('description', resource_uri)}"
        resource_wrapper.__annotations__ = {'return': str}

        return resource_wrapper

    def _create_resource_template_wrapper(self, server_name: str, template_uri: str, template_info: dict,
                                          session: ClientSession):
        """Create wrapper function for MCP resource template with dynamic parameters"""
        import inspect
        import re

        # Extract template variables from URI (e.g., {owner}, {repo})
        template_vars = re.findall(r'\{(\w+)\}', template_uri)

        # Create parameters for each template variable
        parameters = []
        for var_name in template_vars:
            param = inspect.Parameter(var_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)
            parameters.append(param)

        async def template_wrapper(*args, **kwargs) -> str:
            """Access MCP resource template with parameters"""
            try:
                from pydantic import AnyUrl

                # Map arguments to template variables
                template_args = {}
                for i, arg in enumerate(args):
                    if i < len(template_vars):
                        template_args[template_vars[i]] = arg

                template_args.update(kwargs)

                # Validate all required template variables are provided
                missing_vars = set(template_vars) - set(template_args.keys())
                if missing_vars:
                    raise ValueError(f"Missing required template variables: {missing_vars}")

                # Replace template variables in URI
                actual_uri = template_uri
                for var_name, value in template_args.items():
                    actual_uri = actual_uri.replace(f"{{{var_name}}}", str(value))

                result = await session.read_resource(AnyUrl(actual_uri))

                if result.contents:
                    content = result.contents[0]
                    if hasattr(content, 'text'):
                        return content.text
                    elif hasattr(content, 'data'):
                        if isinstance(content.data, bytes):
                            return content.data.decode('utf-8', errors='ignore')
                        return str(content.data)
                    else:
                        return str(content)

                return ""

            except Exception as e:
                eprint(f"MCP resource template {template_uri} failed: {e}")
                raise RuntimeError(f"Error accessing resource template: {str(e)}")

        # Set dynamic signature
        signature = inspect.Signature(parameters, return_annotation=str)
        template_wrapper.__signature__ = signature
        template_wrapper.__name__ = f"{server_name}_template_{template_info['name'].replace('/', '_').replace(':', '_')}"
        template_wrapper.__doc__ = f"Access MCP resource template: {template_info.get('description', template_uri)}\nTemplate variables: {', '.join(template_vars)}"
        template_wrapper.__annotations__ = {'return': str}

        # Add parameter annotations
        for param in parameters:
            template_wrapper.__annotations__[param.name] = str

        return template_wrapper

    def _create_prompt_wrapper(self, server_name: str, prompt_name: str, prompt_info: dict, session: ClientSession):
        """Create wrapper function for MCP prompt with dynamic parameters"""
        import inspect

        # Extract parameter information from prompt arguments
        prompt_args = prompt_info.get('arguments', [])

        # Create parameters
        parameters = []
        for arg_info in prompt_args:
            arg_name = arg_info['name']
            is_required = arg_info.get('required', False)

            if is_required:
                param = inspect.Parameter(arg_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)
            else:
                param = inspect.Parameter(arg_name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                          annotation=str, default=None)
            parameters.append(param)

        async def prompt_wrapper(*args, **kwargs) -> str:
            """Execute MCP prompt with parameters"""
            try:
                # Map arguments
                prompt_arguments = {}
                arg_names = [arg['name'] for arg in prompt_args]

                # Map positional args
                for i, arg in enumerate(args):
                    if i < len(arg_names):
                        prompt_arguments[arg_names[i]] = arg

                # Add keyword arguments, filtering None for optional params
                required_args = {arg['name'] for arg in prompt_args if arg.get('required', False)}
                for key, value in kwargs.items():
                    if value is not None or key in required_args:
                        prompt_arguments[key] = value

                # Validate required parameters
                missing_required = required_args - set(prompt_arguments.keys())
                if missing_required:
                    raise ValueError(f"Missing required prompt arguments: {missing_required}")

                result = await session.get_prompt(prompt_name, prompt_arguments)

                # Extract and combine messages
                messages = []
                for message in result.messages:
                    if hasattr(message.content, 'text'):
                        messages.append(message.content.text)
                    else:
                        messages.append(str(message.content))

                return "\n".join(messages) if messages else ""

            except Exception as e:
                eprint(f"MCP prompt {prompt_name} failed: {e}")
                raise RuntimeError(f"Error executing prompt: {str(e)}")

        # Set dynamic signature
        signature = inspect.Signature(parameters, return_annotation=str)
        prompt_wrapper.__signature__ = signature
        prompt_wrapper.__name__ = f"{server_name}_prompt_{prompt_name}"

        # Build docstring with parameter info
        param_docs = []
        for arg_info in prompt_args:
            required_str = "required" if arg_info.get('required', False) else "optional"
            param_docs.append(
                f"    {arg_info['name']} ({required_str}): {arg_info.get('description', 'No description')}")

        docstring = f"Execute MCP prompt: {prompt_info.get('description', prompt_name)}"
        if param_docs:
            docstring += "\n\nParameters:\n" + "\n".join(param_docs)

        prompt_wrapper.__doc__ = docstring
        prompt_wrapper.__annotations__ = {'return': str}

        # Add parameter annotations
        for param in parameters:
            prompt_wrapper.__annotations__[param.name] = str

        return prompt_wrapper

    def load_mcp_tools_from_config(self, config_path: str | dict) -> 'FlowAgentBuilder':
        """Enhanced MCP config loading with automatic session management and full capability extraction"""
        if not MCP_AVAILABLE:
            wprint("MCP not available, skipping tool loading")
            return self

        if isinstance(config_path, dict):
            mcp_config = config_path
            from toolboxv2 import get_app
            name = self.config.name or "inline_config"
            path = Path(get_app().appdata) / "isaa" / "MCPConfig" / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(mcp_config, indent=2))
            config_path = path
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"MCP config not found: {config_path}")

            try:
                with open(config_path, encoding='utf-8') as f:
                    if config_path.suffix.lower() in ['.yaml', '.yml']:
                        mcp_config = yaml.safe_load(f)
                    else:
                        mcp_config = json.load(f)

            except Exception as e:
                eprint(f"Failed to load MCP config from {config_path}: {e}")
                raise

        # Store config for async processing
        self._mcp_config_data = mcp_config
        self.config.mcp.config_path = str(config_path)

        # Mark for processing during build
        self._mcp_needs_loading = True

        iprint(f"MCP config loaded from {config_path}, will process during build")

        return self

    async def _process_mcp_config(self):
        """Process MCP configuration with proper task management"""
        if not hasattr(self, '_mcp_config_data') or not self._mcp_config_data:
            return

        mcp_config = self._mcp_config_data

        # Handle standard MCP server configuration with sequential processing to avoid task issues
        if 'mcpServers' in mcp_config:
            servers_to_load = []

            # Validate all servers first
            for server_name, server_config in mcp_config['mcpServers'].items():
                if self._validate_mcp_server_config(server_name, server_config):
                    servers_to_load.append((server_name, server_config))
                else:
                    wprint(f"Skipping invalid MCP server config: {server_name}")

            if servers_to_load:
                iprint(f"Processing {len(servers_to_load)} MCP servers sequentially...")

                # Process servers sequentially to avoid task boundary issues
                successful_loads = 0
                for server_name, server_config in servers_to_load:
                    try:
                        result = await asyncio.wait_for(
                            self._load_single_mcp_server(server_name, server_config),
                            timeout=5.0  # Per-server timeout
                        )

                        if result:
                            successful_loads += 1
                            iprint(f"✓ Successfully loaded MCP server: {server_name}")
                        else:
                            wprint(f"⚠ MCP server {server_name} loaded with issues")

                    except TimeoutError:
                        eprint(f"✗ MCP server {server_name} timed out after 15 seconds")
                    except Exception as e:
                        eprint(f"✗ Failed to load MCP server {server_name}: {e}")

                iprint(
                    f"MCP processing complete: {successful_loads}/{len(servers_to_load)} servers loaded successfully")

        # Handle direct tools configuration (legacy)
        elif 'tools' in mcp_config:
            for tool_config in mcp_config['tools']:
                try:
                    self._load_direct_mcp_tool(tool_config)
                except Exception as e:
                    eprint(f"Failed to load direct MCP tool: {e}")

    async def _load_single_mcp_server(self, server_name: str, server_config: dict[str, Any]) -> bool:
        """Load a single MCP server with timeout and error handling"""
        try:
            iprint(f"🔄 Processing MCP server: {server_name}")

            # Get session with timeout
            session = await self._mcp_session_manager.get_session_with_timeout(server_name, server_config)
            if not session:
                eprint(f"✗ Failed to create session for MCP server: {server_name}")
                return False

            # Extract capabilities with timeout
            capabilities = await self._mcp_session_manager.extract_capabilities_with_timeout(session, server_name)
            if not any(capabilities.values()):
                wprint(f"⚠ No capabilities found for MCP server: {server_name}")
                return False

            # Create wrappers for all capabilities
            await self._create_capability_wrappers(server_name, capabilities, session)

            total_caps = sum(len(caps) for caps in capabilities.values())
            iprint(f"✓ Created {total_caps} capability wrappers for: {server_name}")

            return True

        except Exception as e:
            eprint(f"✗ Error loading MCP server {server_name}: {e}")
            return False

    async def _create_capability_wrappers(self, server_name: str, capabilities: dict, session: ClientSession):
        """Create wrappers for all capabilities with error handling"""

        # Create tool wrappers
        for tool_name, tool_info in capabilities['tools'].items():
            try:
                wrapper_name = f"{server_name}_{tool_name}"
                tool_wrapper = self._create_tool_wrapper(server_name, tool_name, tool_info, session)

                self._mcp_tools[wrapper_name] = {
                    'function': tool_wrapper,
                    'description': tool_info['description'],
                    'type': 'tool',
                    'server': server_name,
                    'original_name': tool_name,
                    'input_schema': tool_info.get('input_schema'),
                    'output_schema': tool_info.get('output_schema')
                }
            except Exception as e:
                eprint(f"Failed to create tool wrapper {tool_name}: {e}")

        # Create resource wrappers
        for resource_uri, resource_info in capabilities['resources'].items():
            try:
                safe_name = resource_info['name'].replace('/', '_').replace(':', '_')
                wrapper_name = f"{server_name}_resource_{safe_name}"
                resource_wrapper = self._create_resource_wrapper(server_name, resource_uri, resource_info, session)

                self._mcp_tools[wrapper_name] = {
                    'function': resource_wrapper,
                    'description': f"Read resource: {resource_info['description']}",
                    'type': 'resource',
                    'server': server_name,
                    'original_uri': resource_uri
                }
            except Exception as e:
                eprint(f"Failed to create resource wrapper {resource_uri}: {e}")

        # Create resource template wrappers
        for template_uri, template_info in capabilities['resource_templates'].items():
            try:
                safe_name = template_info['name'].replace('/', '_').replace(':', '_')
                wrapper_name = f"{server_name}_template_{safe_name}"
                template_wrapper = self._create_resource_template_wrapper(server_name, template_uri, template_info,
                                                                          session)

                self._mcp_tools[wrapper_name] = {
                    'function': template_wrapper,
                    'description': f"Access resource template: {template_info['description']}",
                    'type': 'resource_template',
                    'server': server_name,
                    'original_template': template_uri
                }
            except Exception as e:
                eprint(f"Failed to create template wrapper {template_uri}: {e}")

        # Create prompt wrappers
        for prompt_name, prompt_info in capabilities['prompts'].items():
            try:
                wrapper_name = f"{server_name}_prompt_{prompt_name}"
                prompt_wrapper = self._create_prompt_wrapper(server_name, prompt_name, prompt_info, session)

                self._mcp_tools[wrapper_name] = {
                    'function': prompt_wrapper,
                    'description': f"Execute prompt: {prompt_info['description']}",
                    'type': 'prompt',
                    'server': server_name,
                    'original_name': prompt_name,
                    'arguments': prompt_info.get('arguments', [])
                }
            except Exception as e:
                eprint(f"Failed to create prompt wrapper {prompt_name}: {e}")

    @staticmethod
    def _validate_mcp_server_config(server_name: str, server_config: dict[str, Any]) -> bool:
        """Validate MCP server configuration"""
        command = server_config.get('command')
        if not command:
            eprint(f"MCP server {server_name} missing 'command' field")
            return False

        # Check if command exists and is executable
        if command in ['npx', 'node', 'python', 'python3', 'docker']:
            # These are common commands, assume they exist
            return True

        if server_config.get('transport') in ['http', 'streamable-http'] and server_config.get('url'):
            return True

        # For other commands, check if they exist
        import shutil
        if not shutil.which(command):
            wprint(f"MCP server {server_name}: command '{command}' not found in PATH")
            # Don't fail completely, just warn - the command might be available at runtime

        args = server_config.get('args', [])
        if not isinstance(args, list):
            eprint(f"MCP server {server_name}: 'args' must be a list")
            return False

        env = server_config.get('env', {})
        if not isinstance(env, dict):
            eprint(f"MCP server {server_name}: 'env' must be a dictionary")
            return False

        iprint(f"Validated MCP server config: {server_name}")
        return True

    def _load_direct_mcp_tool(self, tool_config: dict[str, Any]):
        """Load tool from direct configuration"""
        name = tool_config.get('name')
        description = tool_config.get('description', '')
        function_code = tool_config.get('function_code')

        if not name or not function_code:
            wprint(f"Incomplete tool config: {tool_config}")
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
                iprint(f"Loaded MCP tool from code: {name}")

        except Exception as e:
            eprint(f"Failed to load MCP tool {name}: {e}")

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
            wprint("A2A not available, cannot enable server")
            return self

        self.config.a2a.enabled = True
        self.config.a2a.host = host
        self.config.a2a.port = port
        self.config.a2a.agent_name = agent_name or self.config.name
        self.config.a2a.agent_description = agent_description or self.config.description

        iprint(f"A2A server enabled: {host}:{port}")
        return self

    # ===== TELEMETRY INTEGRATION =====

    def enable_telemetry(self, service_name: str = None, endpoint: str = None,
                         console_export: bool = True) -> 'FlowAgentBuilder':
        """Enable OpenTelemetry tracing"""
        if not OTEL_AVAILABLE:
            wprint("OpenTelemetry not available, cannot enable telemetry")
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
                wprint(f"Failed to setup OTLP exporter: {e}")

        iprint(f"Telemetry enabled for service: {service_name}")
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
            iprint(f"Checkpointing enabled: {checkpoint_dir} (every {interval_seconds}s)")

        return self

    # ===== TOOL MANAGEMENT =====

    def add_tool(self, func: Callable, name: str = None, description: str = None) -> 'FlowAgentBuilder':
        """Add custom tool function"""
        tool_name = name or func.__name__
        self._custom_tools[tool_name] = (func, description or func.__doc__)

        iprint(f"Tool added: {tool_name}")
        return self

    def add_tools_from_module(self, module, prefix: str = "", exclude: list[str] = None) -> 'FlowAgentBuilder':
        """Add all functions from a module as tools"""
        exclude = exclude or []

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name in exclude or name.startswith('_'):
                continue

            tool_name = f"{prefix}{name}" if prefix else name
            self.add_tool(obj, name=tool_name)

        iprint(f"Added tools from module {module.__name__}")
        return self

    # ===== PERSONA MANAGEMENT =====

    def add_persona_profile(self, profile_name: str, name: str, style: str = "professional",
                            tone: str = "friendly", personality_traits: list[str] = None,
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

    def with_custom_variables(self, variables: dict[str, Any]) -> 'FlowAgentBuilder':
        """Add custom variables"""
        self.config.custom_variables.update(variables)
        return self

    def with_world_model(self, world_model: dict[str, Any]) -> 'FlowAgentBuilder':
        """Set initial world model"""
        self.config.initial_world_model.update(world_model)
        return self

    # ===== VALIDATION =====

    def validate_config(self) -> dict[str, list[str]]:
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

    def set_max_checkpoint_age(self, max_age_hours: int) -> 'FlowAgentBuilder':
        """Set the maximum age for checkpoints in hours"""
        self.config.checkpoint.max_age_hours = max_age_hours
        return self

    def set_handler_path_or_dict(self, handler_path_or_dict: str | dict) -> 'FlowAgentBuilder':
        """Set the handler path or dict"""
        self.config.handler_path_or_dict = handler_path_or_dict
        return self

    # ===== MAIN BUILD METHOD =====

    async def build(self) -> FlowAgent:
        """Build the production-ready FlowAgent"""
        from toolboxv2 import get_app
        info_print = get_app().get_mod("isaa").print

        with Spinner(message=f"Building Agent {self.config.name}", symbols='c'):
            iprint(f"Building production FlowAgent: {self.config.name}")

            # Validate configuration
            validation_issues = self.validate_config()
            if validation_issues["errors"]:
                error_msg = f"Configuration validation failed: {', '.join(validation_issues['errors'])}"
                eprint(error_msg)
                raise ValueError(error_msg)

            # Log warnings
            for warning in validation_issues["warnings"]:
                wprint(f"Configuration warning: {warning}")

            try:
                # 1. Setup API configuration
                api_key = None
                if self.config.api_key_env_var:
                    api_key = os.getenv(self.config.api_key_env_var)
                    if not api_key:
                        wprint(f"API key env var {self.config.api_key_env_var} not set")

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

                    iprint(f"Using persona: {active_persona.name}")

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

                agent.checkpoint_config = self.config.checkpoint

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
                        eprint(f"Failed to add tool {tool_name}: {e}")

                with Spinner(message="Loading MCP", symbols='w'):
                    # 6a. Process MCP configuration if needed
                    if hasattr(self, '_mcp_needs_loading') and self._mcp_needs_loading:
                        await self._process_mcp_config()

                # 7. Add MCP tools
                # P0 - KRITISCH: Set agent reference for circuit breaker
                self._agent_instance = agent

                for tool_name, tool_info in self._mcp_tools.items():
                    try:
                        await agent.add_tool(
                            tool_info['function'],
                            tool_name,
                            tool_info['description']
                        )
                        tools_added += 1
                    except Exception as e:
                        eprint(f"Failed to add MCP tool {tool_name}: {e}")

                agent._mcp_session_manager = self._mcp_session_manager

                # 8. Setup MCP server
                if self.config.mcp.enabled and MCP_AVAILABLE:
                    try:
                        agent.setup_mcp_server(
                            host=self.config.mcp.host,
                            port=self.config.mcp.port,
                            name=self.config.mcp.server_name
                        )
                        iprint("MCP server configured")
                    except Exception as e:
                        eprint(f"Failed to setup MCP server: {e}")

                # 9. Setup A2A server
                if self.config.a2a.enabled and A2A_AVAILABLE:
                    try:
                        agent.setup_a2a_server(
                            host=self.config.a2a.host,
                            port=self.config.a2a.port
                        )
                        iprint("A2A server configured")
                    except Exception as e:
                        eprint(f"Failed to setup A2A server: {e}")

                # 10. Initialize enhanced session context
                try:
                    await agent.initialize_session_context(max_history=200)
                    iprint("Enhanced session context initialized")
                except Exception as e:
                    wprint(f"Session context initialization failed: {e}")

                # 11. Reestor from checkpoint if needed
                if self.config.checkpoint.enabled:
                    info_print("loading latest checkpoint")
                    res = await agent.load_latest_checkpoint(auto_restore_history=True, max_age_hours=self.config.checkpoint.max_age_hours)
                    info_print(f"loading completed {res}")

                await agent.voting_as_tool()
                await bind_accomplish_to_agent(agent)
                # Final summary
                iprint("ok FlowAgent built successfully!")
                iprint(f"   Agent: {agent.amd.name}")
                iprint(f"   Tools: {tools_added}")
                iprint(f"   MCP: {'ok' if self.config.mcp.enabled else 'F'}")
                iprint(f"   A2A: {'ok' if self.config.a2a.enabled else 'F'}")
                iprint(f"   Telemetry: {'ok' if self.config.telemetry.enabled else 'F'}")
                iprint(f"   Checkpoints: {'ok' if self.config.checkpoint.enabled else 'F'}")
                iprint(f"   Persona: {active_persona.name if active_persona else 'Default'}")

                return agent

            except Exception as e:
                eprint(f"Failed to build FlowAgent: {e}")
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

    iprint("=== Production FlowAgent Builder Example ===")

    # Example 1: Developer agent with full MCP integration
    iprint("Creating developer agent with MCP integration...")

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
    iprint(f"Developer agent response: {dev_response[:200]}...")

    # Example 2: Load from configuration file
    iprint("\nTesting configuration save/load...")

    # Save current config
    config_path = "/tmp/production_agent_config.yaml"
    builder = FlowAgentBuilder.create_analyst_agent("ConfigTestAgent")
    builder.save_config(config_path)

    # Load from config
    loaded_builder = FlowAgentBuilder.from_config_file(config_path)
    config_agent = await loaded_builder.build()

    config_response = await config_agent.a_run("Analyze this data: [1, 2, 3, 4, 5]")
    iprint(f"Config-loaded agent response: {config_response[:150]}...")

    # Example 3: Agent with MCP tools from config
    iprint("\nTesting MCP tools integration...")

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
    iprint(f"MCP agent response: {mcp_response[:150]}...")

    # Show agent status
    iprint("\n=== Agent Status ===")
    status = developer_agent.status(pretty_print=False)
    iprint(f"Developer agent tools: {len(status['capabilities']['tool_names'])}")
    iprint(f"MCP agent tools: {len(mcp_agent.shared.get('available_tools', []))}")

    # Cleanup
    await developer_agent.close()
    await config_agent.close()
    await mcp_agent.close()

    iprint("Production example completed successfully!")


async def example_quick_start():
    """Quick start examples for common scenarios"""

    iprint("=== Quick Start Examples ===")

    # 1. Simple developer agent
    dev_agent = await FlowAgentBuilder.create_developer_agent("QuickDev").build()
    response1 = await dev_agent.a_run("Create a Python function to validate email addresses")
    iprint(f"Quick dev response: {response1[:100]}...")
    await dev_agent.close()

    # 2. Analyst with custom data
    analyst_agent = await (FlowAgentBuilder
                           .create_analyst_agent("QuickAnalyst")
                           .with_custom_variables({"dataset": "sales_data_2024"})
                           .build())
    response2 = await analyst_agent.a_run("Analyze the trends in {{ dataset }}")
    iprint(f"Quick analyst response: {response2[:100]}...")
    await analyst_agent.close()

    # 3. Creative assistant
    creative_agent = await FlowAgentBuilder.create_creative_agent("QuickCreative").build()
    response3 = await creative_agent.a_run("Write a creative story about AI agents collaborating")
    iprint(f"Quick creative response: {response3[:100]}...")
    await creative_agent.close()

    iprint("Quick start examples completed!")


if __name__ == "__main__":
    # Run production example
    asyncio.run(example_production_usage())

    # Run quick start examples
    asyncio.run(example_quick_start())
