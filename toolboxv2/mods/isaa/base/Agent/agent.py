import asyncio
import contextlib
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from importlib.metadata import version
from inspect import iscoroutinefunction
from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SkipValidation,
    TypeAdapter,
    ValidationError,
    model_validator,
)

# --- Framework Imports with Availability Checks ---

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

    # ADK Artifacts (Optional, for advanced use cases)
    # from google.adk.artifacts import ArtifactService, InMemoryArtifactService
    from google.genai.types import Content, FunctionCall, FunctionResponse, Part

    ADK_AVAILABLE = True
    print("INFO: Google ADK components found. ADK features enabled. version", version("google.adk"))
except ImportError as e:
    print(f"WARN: Google ADK components not found or import error ({e}). ADK features disabled.")
    ADK_AVAILABLE = False

    # Define dummy types for type hinting if ADK is not installed
    class BaseAgent(BaseModel): pass
    class LlmAgent(BaseAgent): pass
    class BaseTool: pass
    class FunctionTool(BaseTool): pass
    class LongRunningFunctionTool(BaseTool): pass
    class ToolContext: pass
    class CallbackContext: pass
    class BaseCodeExecutor: pass
    class CodeExecutionResult: pass
    class CodeExecutionInput: pass
    class InvocationContext: pass
    class BasePlanner: pass
    class BaseLlm: pass
    class Gemini(BaseLlm): pass
    class LiteLlm(BaseLlm): pass # Dummy
    class Runner: pass
    class InMemoryRunner(Runner): pass
    class BaseSessionService: pass
    class Session: pass
    class State(dict): pass # ADK State is dict-like
    class Event: pass
    class Content: pass
    class Part: pass
    class Example: pass
    class MCPToolset: pass # Dummy
    class StdioServerParameters: pass # Dummy
    class SseServerParams: pass # Dummy
    def adk_built_in_code_execution(): return None # Dummy
    def adk_google_search(): return None # Dummy
    class AdkVertexAiSearchTool: pass # Dummy


# core/agent.py

# ... (Imports remain the same, including A2A imports) ...
# Make sure the dummy types for A2A are updated if needed
# Specifically, ensure dummy Task includes 'id' field for cancellation example.
try:
    from python_a2a import (
        A2AClient,
        A2AServer,
        Message,
        MessageRole,
        Task,
        TaskState,
        TaskStatus,
        TextContent,
    )
    from python_a2a import agent as a2a_agent_decorator
    from python_a2a import run_server as run_a2a_server_func
    from python_a2a import skill as a2a_skill_decorator
    # Attempt to import potential cancel/error types - adjust if needed
    # from python_a2a.error import A2AError, TaskNotFoundError, TaskNotCancelableError
    # from python_a2a.jsonrpc import JSONRPCResponse, JSONRPCError

    A2A_AVAILABLE = True
    print("INFO: python-a2a components found. A2A features enabled. version", version("python_a2a"))

except ImportError as e:
    print(f"WARN: python-a2a components not found or import error ({e}). A2A features disabled.")
    A2A_AVAILABLE = False

    # Define dummy types... (ensure Task has id)
    class A2AClient: pass
    class A2AServer: pass
    def a2a_agent_decorator(**kwargs): return lambda cls: cls
    def a2a_skill_decorator(**kwargs): return lambda func: func
    def run_a2a_server_func(*args, **kwargs): pass
    class Task(BaseModel): # Pydantic dummy for type hints
        id: str = "dummy_task"
        status: Any | None = None
        message: Any | None = None
        artifacts: list[Any] | None = None
        model_config = ConfigDict(extra='allow')
    class TaskStatus(BaseModel): state: Any = None; message: Any | None=None; timestamp: str=""
    class TaskState(Enum): SUBMITTED="SUBMITTED"; WORKING="WORKING"; COMPLETED="COMPLETED"; FAILED="FAILED"; CANCELLED="CANCELLED"; UNKNOWN="UNKNOWN"; INPUT_REQUIRED="INPUT_REQUIRED"
    class Message: pass
    class TextContent: pass
    class MessageRole: pass
    # Dummy error types if needed for type hints later
    # class TaskNotFoundError(Exception): pass
    # class TaskNotCancelableError(Exception): pass


# Model Context Protocol (MCP) - Only used if building MCP servers directly
# If using ADK's MCPToolset, explicit MCP imports might not be needed here.
try:
    import mcp.server.stdio
    import mcp.types as mcp_types  # For building MCP servers
    from google.adk.tools.mcp_tool.conversion_utils import (
        adk_to_mcp_tool_type,  # If building MCP server from ADK tools
    )
    from mcp import ClientSession
    from mcp.client.sse import sse_client as mcp_sse_client
    from mcp.server.fastmcp import FastMCP
    from mcp.server.lowlevel import NotificationOptions
    from mcp.server.lowlevel import Server as MCPServerBase
    from mcp.server.models import InitializationOptions

    MCP_AVAILABLE = True
    print("INFO: MCP components found. MCP features enabled (primarily for server building).")
except ImportError as e:
    print(f"WARN: MCP components not found or import error ({e}). MCP features disabled (client usage relies on ADK's MCPToolset).")
    MCP_AVAILABLE = False
    def adk_to_mcp_tool_type(*a, **k):
        return None
    # Define dummy types
    class ClientSession: pass
    class FastMCP: pass
    class MCPServerBase: pass
    def mcp_sse_client(*args, **kwargs): pass
    def mcp_types():
        return None


    class TextContent(BaseModel):
        """Text content for a message."""

        type: Literal["text"]
        text: str
        """The text content of the message."""
        annotations: Any | None = None
        model_config = ConfigDict(extra="allow")


    mcp_types.TextContent = TextContent

# LiteLLM
try:
    import litellm
    from litellm import BudgetManager, acompletion, completion_cost, token_counter
    from litellm.exceptions import (
        APIConnectionError,
        BadRequestError,
        InternalServerError,
        RateLimitError,
    )
    from litellm.utils import get_max_tokens, trim_messages
    litellm_version = version("litellm")
    print(f"INFO: LiteLLM version {litellm_version} found.")
except ImportError:
    print("CRITICAL ERROR: LiteLLM not found. Agent cannot function without LiteLLM.")
    # Define dummy types/functions to avoid runtime errors later if LiteLLM is missing
    class BudgetManager: pass
    async def acompletion(**kwargs): raise ImportError("LiteLLM not installed")
    def completion_cost(**kwargs): return 0.0
    def token_counter(**kwargs): return 0
    class BadRequestError(Exception): pass
    class InternalServerError(Exception): pass
    class RateLimitError(Exception): pass
    class APIConnectionError(Exception): pass
    def trim_messages(messages, **kwargs): return messages
    def get_max_tokens(model): return None


# OpenTelemetry (Optional, for Observability)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (  # Example exporter
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    # Add other exporters (OTLP, Jaeger, etc.) as needed
    OTEL_AVAILABLE = True
    print("INFO: OpenTelemetry SDK found. Basic tracing enabled (requires configuration).")
except ImportError:
    print("WARN: OpenTelemetry SDK not found. Distributed tracing/observability disabled.")
    OTEL_AVAILABLE = False
    # Dummy tracer
    class DummyTracer:
        def start_as_current_span(self, *args, **kwargs):
            return contextlib.nullcontext() # No-op context manager
    trace.set_tracer_provider(trace.NoOpTracerProvider()) # Use NoOp if SDK absent
    tracer = DummyTracer()

# --- Logging Setup ---
# Configure root logger level for libraries
logging.basicConfig(level=logging.WARNING)
# Configure LiteLLM logging level specifically if needed
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
# Agent-specific logger
logger = logging.getLogger("EnhancedAgent")
logger.setLevel(logging.INFO) # Default level, builder can override via 'verbose'


# --- Helper Classes ---

@dataclass
class WorldModel:
    """Thread-safe persistent understanding of the world for the agent."""
    data: dict[str, Any] = dataclass_field(default_factory=dict)
    _lock: SkipValidation[threading.Lock] = dataclass_field(default_factory=threading.Lock, init=False, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.data.get(key, default)

    def set(self, key: str, value: Any):
        with self._lock:
            logger.debug(f"WorldModel SET: {key} = {value}")
            self.data[key] = value

    def remove(self, key: str):
        with self._lock:
            if key in self.data:
                logger.debug(f"WorldModel REMOVE: {key}")
                del self.data[key]

    def show(self) -> str:
        with self._lock:
            if not self.data:
                return "[empty]"
            try:
                items = [f"- {k}: {json.dumps(v, indent=None, ensure_ascii=False, default=str)}"
                         for k, v in self.data.items()]
                return "\n".join(items)
            except Exception:
                items = [f"- {k}: {str(v)}" for k, v in self.data.items()]
                return "\n".join(items)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return self.data.copy()

    def update_from_dict(self, data_dict: dict[str, Any]):
        with self._lock:
            self.data.update(data_dict)
            logger.debug(f"WorldModel updated from dict: {list(data_dict.keys())}")

    def sync_from_adk_state(self, adk_state: State):
        """Updates the WorldModel from an ADK Session State."""
        if not ADK_AVAILABLE or not isinstance(adk_state, State):
            return
        with self._lock:
            # Simple overwrite strategy, could be more sophisticated (merge, etc.)
            self.data = adk_state.to_dict() # ADK State is dict-like
            logger.debug(f"WorldModel synced FROM ADK state. Keys: {list(self.data.keys())}")

    def sync_to_adk_state(self, adk_state: State):
        """Updates an ADK Session State from the WorldModel."""
        if not ADK_AVAILABLE or not isinstance(adk_state, State):
            return
        with self._lock:
            # Update the ADK state dictionary directly
            adk_state.update(self.data)
            logger.debug(f"WorldModel synced TO ADK state. Keys: {list(adk_state.keys())}")


class AgentModelData(BaseModel):
    """Configuration for the LLM model and API settings via LiteLLM."""
    name: str | None = Field(default=None, description="Agent's internal name, often derived from builder.")
    model: str = Field(..., description="Primary LiteLLM model identifier (e.g., 'gemini/gemini-1.5-flash-latest', 'ollama/mistral').")
    provider: str | None = Field(default=None, description="LiteLLM provider override if needed.")
    system_message: str = Field(default="You are a helpful AI assistant.", description="Base system prompt.")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0) # Use LiteLLM defaults if None
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, description="Max tokens for LLM generation.")
    max_input_tokens: int | None = Field(default=None, ge=1, description="Max context window size (for trimming).")

    api_key: str | None = Field(default=None, description="API key (use env vars in production).")
    api_base: str | None = Field(default=None, description="API base URL (for local models/proxies).")
    api_version: str | None = Field(default=None, description="API version (e.g., Azure).")

    stop_sequence: list[str] | None = Field(default=None, alias="stop") # Alias for LiteLLM
    presence_penalty: float | None = Field(default=None)
    frequency_penalty: float | None = Field(default=None)

    user_id: str | None = Field(default=None, description="User identifier for LLM calls ('user' param).")
    budget_manager: BudgetManager | None = Field(default=None, description="LiteLLM BudgetManager instance.")
    caching: bool | None = Field(default=True, description="Enable/disable LiteLLM caching.")

    # Model config for Pydantic v2
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='ignore', # Ignore extra fields from builder/ADK init
        populate_by_name=True # Allow using 'stop' alias
    )

@dataclass
class LLMMessage:
    """Represents a message in a conversation, compatible with LiteLLM."""
    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[dict[str, Any]] # String or multimodal content (LiteLLM format)
    tool_call_id: str | None = None # For tool responses
    name: str | None = None # For tool calls/responses (function name)

    # Add tool_calls for assistant messages requesting tool use (LiteLLM format)
    tool_calls: list[dict[str, Any]] | None = None # e.g., [{"id": "call_123", "function": {"name": "...", "arguments": "{...}"}}]

    def to_dict(self) -> dict[str, Any]:
        """Converts to dict suitable for LiteLLM."""
        d = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_call_id: d["tool_call_id"] = self.tool_call_id
        if self.name: d["name"] = self.name
        if self.tool_calls: d["tool_calls"] = self.tool_calls
        return d


# Agent State Enum
class InternalAgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_TOOL = "waiting_for_tool" # If using long-running tools/A2A polling
    ERROR = "error"

# Processing Strategy Enum
class ProcessingStrategy(Enum):
    DIRECT_LLM = "direct_llm"
    ADK_RUN = "adk_run" # Includes ADK planning, tool calls, code exec etc.
    A2A_CALL = "a2a_call"
    # MCP_CALL is removed - MCP tools are handled via ADK_RUN + MCPToolset


# --- Secure Code Executor Placeholder ---
# Implement a real sandboxed executor (Docker, Firecracker, RestrictedPython) for production
if ADK_AVAILABLE:
    class SecureCodeExecutorPlaceholder(BaseCodeExecutor):
        """ Placeholder for a secure code execution environment. """
        def execute_code(self, invocation_context: InvocationContext,
                         code_execution_input: CodeExecutionInput) -> CodeExecutionResult:
            logger.critical("Attempted to execute code with SecureCodeExecutorPlaceholder. "
                           "This is NOT a secure implementation. Implement sandboxing.")
            # In a real scenario, delegate to a secure sandbox service
            # add live codeing (agent excution part)
            # result_from_sandbox = call_sandbox_service(code_execution_input.code)
            #eturn CodeExecutionResult(result=result_from_sandbox)
            return CodeExecutionResult(result="Error: Secure code execution not implemented.")

    class UnsafeSimplePythonExecutor(BaseCodeExecutor):
        """
        UNSAFE executor using Python's exec(). FOR TESTING/DEVELOPMENT ONLY.
        Requires explicit enabling via builder.
        """
        def execute_code(self, invocation_context: InvocationContext,
                         code_execution_input: CodeExecutionInput) -> CodeExecutionResult:
            logger.warning(f"Executing code with UNSAFE UnsafeSimplePythonExecutor! Code: ```\n{code_execution_input.code}\n```")
            local_vars = {}
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            result_str = ""
            try:
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    exec(code_execution_input.code, globals(), local_vars) # <<< UNSAFE!
                stdout = stdout_capture.getvalue().strip()
                stderr = stderr_capture.getvalue().strip()
                if stderr:
                    result_str = f"Execution Error:\n```\n{stderr}\n```"
                elif stdout:
                    result_str = f"Execution Output:\n```\n{stdout}\n```"
                else:
                    result_str = "Execution completed with no output."
            except Exception as e:
                result_str = f"Execution Exception: {type(e).__name__}: {e}"
                logger.error(f"Unsafe code execution failed: {e}", exc_info=True)
            finally:
                stdout_capture.close()
                stderr_capture.close()
            return CodeExecutionResult(result=result_str)
else: # Dummy executors if ADK not available
    class SecureCodeExecutorPlaceholder: pass
    class UnsafeSimplePythonExecutor: pass


# --- Main Agent Class ---

_AgentBaseClass = (LlmAgent, BaseModel) if ADK_AVAILABLE else (BaseModel, )


class EnhancedAgent(*_AgentBaseClass):
    """
    Enhanced, production-oriented Unified Agent integrating LiteLLM, ADK, A2A, and MCP (via ADK).
    """
    # --- Core Configuration ---
    amd: AgentModelData # Primary model config
    format_model: str | None = Field(default=None, description="Optional separate model for JSON formatting (a_format_class).")
    format_model_: str | None = Field(default=None, description="helper var for format_model", exclude=True)
    world_model: WorldModel = Field(default_factory=WorldModel)
    verbose: bool = Field(default=False)
    internal_state: InternalAgentState = Field(default=InternalAgentState.IDLE)

    # --- LiteLLM Specific ---
    stream: bool = Field(default=False, description="Whether LLM calls should stream chunks.")
    # Use a simple dict for history for now, can be replaced with persistent store interface
    # Keyed by session_id
    message_history: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    max_history_tokens: int | None = Field(default=None, description="Alternative to max_turns for history trimming based on token count.")
    max_history_turns: int = Field(default=20, description="Max conversation turns (user+assistant) for history.") # Used if max_history_tokens is None
    trim_strategy: Literal["litellm", "basic"] = Field(default="litellm")
    total_cost: float = Field(default=0.0, description="Accumulated cost tracked via LiteLLM.")

    # --- Framework Components (Initialized via Builder/Setup) ---
    # ADK
    adk_runner: Runner | None = Field(default=None, description="ADK Runner instance if enabled.")
    adk_session_service: BaseSessionService | None = Field(default=None, description="ADK Session Service (often from runner).")
    sync_adk_state: bool = Field(default=True, description="Sync WorldModel with ADK Session.state.")
    # Exit stack to manage lifecycles of components like MCPToolset connections
    # CRITICAL FIX: Use contextlib.AsyncExitStack type hint
    adk_exit_stack: contextlib.AsyncExitStack | None = Field(default=None, description="AsyncExitStack for managing ADK toolset lifecycles.")

    # MCP Server (Agent acts AS an MCP Server)
    mcp_server: FastMCP | None = Field(default=None, description="MCP server instance if agent exposes MCP capabilities.")
    # A2A Server (Agent acts AS an A2A Server)
    a2a_server: A2AServer | None = Field(default=None, description="A2A server instance if agent exposes A2A capabilities.")
    # A2A Client (Agent acts AS an A2A Client)
    a2a_clients: dict[str, A2AClient] = Field(default_factory=dict, description="Cached A2A clients for target agents.")
    a2a_client_lock: asyncio.Lock = Field(default_factory=asyncio.Lock, description="Lock for A2A client cache access.")
    a2a_poll_interval: float = Field(default=2.0, description="Polling interval for A2A task results (seconds).")
    a2a_poll_timeout: float = Field(default=60.0, description="Max time to wait for A2A task completion.")

    # --- Callbacks ---
    stream_callback: Callable[[str], None | Awaitable[None]] | None = Field(default=None, description="Callback for each LLM stream chunk.")
    post_run_callback: Callable[[str, str, float], None | Awaitable[None]] | None = Field(default=None, description="Callback after a_run completes (session_id, final_response, turn_cost).")
    progress_callback: Callable[[Any], None | Awaitable[None]] | None = Field(default=None, description="Callback for progress updates (e.g., tool execution, A2A polling).")
    human_in_loop_callback: Callable[[dict], str | Awaitable[str]] | None = Field(default=None, description="Callback for HIL intervention points.")

    # --- Observability ---
    tracer: Any | None = Field(default=None, description="OpenTelemetry Tracer instance.") # Type hint depends on OTel setup

    # --- Internal State ---
    last_llm_result: Any | None = Field(default=None, description="Raw result from the last LiteLLM call.")

    # Model config
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='ignore' # Critical for compatibility with ADK LlmAgent init
    )

    @model_validator(mode='after')
    def _enhanced_agent_post_init(self) -> 'EnhancedAgent':
        """
        Performs initialization steps after Pydantic has validated fields.
        """
        # --- (Existing post_init logic remains the same) ---
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        os.environ['LITELLM_LOG'] = 'DEBUG' if self.verbose else 'NONE'
        logger.debug(f"Verbose logging {'enabled' if self.verbose else 'disabled'} for agent {self.amd.name}")
        self._setup_telemetry()
        if ADK_AVAILABLE and isinstance(self, LlmAgent):
            logger.debug(f"Running post-init logic for ADK agent '{self.amd.name}'")
            self._ensure_internal_adk_tools() # Ensure tools are added *after* Pydantic init
            if self.adk_runner and hasattr(self.adk_runner, 'session_service'):
                self.adk_session_service = self.adk_runner.session_service
                logger.debug("Associated ADK session service from runner.")
        if 'default' not in self.message_history:
            self.message_history['default'] = []
        logger.info(
            f"EnhancedAgent '{self.amd.name}' initialized. Model: {self.amd.model}. "
            f"Capabilities: ADK({ADK_AVAILABLE}), A2A({A2A_AVAILABLE}), MCP({MCP_AVAILABLE})"
        )
        self.model =  LiteLlm(model=self.amd.model)
        return self

    # --- ADK Post Init (Called automatically by Pydantic if method exists in base) ---
    # This method name is expected by ADK's BaseModel integration.
    # Pydantic v2 runs validators based on MRO, so if LlmAgent has this, it runs.
    # We don't strictly need to define it here unless overriding LlmAgent's version.
    # def model_post_init(self, __context: Any) -> None:
    #     """ADK post-initialization (if inheriting from ADK BaseModel)."""
    #     # Call super() if overriding LlmAgent's method
    #     # super().model_post_init(__context) # If LlmAgent has this method
    #     logger.debug(f"ADK model_post_init for Agent '{self.amd.name}' (EnhancedAgent)")
    #     # Add post-init logic specific to ADK features here, AFTER ADK's own init
    #     self._ensure_internal_adk_tools()
    #     if self.adk_runner:
    #         self.adk_session_service = self.adk_runner.session_service


    # --- Telemetry Setup ---
    def _setup_telemetry(self):
        """Initializes the OpenTelemetry tracer."""
        if OTEL_AVAILABLE and not self.tracer:
            # Get tracer from global provider (needs to be configured elsewhere)
            # In a real app, you'd configure the TracerProvider with exporters
            # provider = TracerProvider() # Example: basic provider
            # provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter())) # Example: console output
            # trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer("enhanced_agent", "0.1.0")
            logger.info("OpenTelemetry tracer initialized.")
        elif not OTEL_AVAILABLE:
            self.tracer = DummyTracer() # Use NoOp tracer if OTel not installed
            logger.debug("OpenTelemetry not available, using NoOp tracer.")


    # --- Setup Methods (Called by Builder) ---

    def setup_mcp_server(self, host="0.0.0.0", port=8000, **mcp_kwargs):
        """Initialize and configure the MCP server capabilities *for this agent*.
           This agent will ACT AS an MCP Server.
        """
        if not MCP_AVAILABLE:
            logger.warning("MCP library not installed. Cannot setup MCP server.")
            return None
        if self.mcp_server:
            logger.warning("MCP server already initialized.")
            return self.mcp_server
        name = mcp_kwargs.get("name")
        del mcp_kwargs["name"]
        self.mcp_server = FastMCP(name=name or f"{self.amd.name}-mcp-server",
                                  description=f"MCP interface for EnhancedAgent {self.amd.name}",
                                  **mcp_kwargs)
        logger.info(f"Setting up MCP server for agent '{self.amd.name}' on {host}:{port}")

        # --- Register Agent's core functionalities as MCP services ---
        # Example: Expose World Model (Read-only for safety)
        @self.mcp_server.resource(f"agent://{self.amd.name}/world_model")
        def mcp_get_world_model_resource() -> dict[str, Any]:
            """Gets the agent's world model."""
            logger.debug(f"[MCP Resource] agent://{self.amd.name}/world_model accessed")
            return self.world_model.to_dict()

        # Example: Expose a simple query tool via MCP
        @self.mcp_server.tool(name="simple_llm_query")
        async def mcp_simple_query(prompt: str) -> str:
            """Sends a simple prompt to the agent's LLM (non-persistent run)."""
            logger.debug(f"[MCP Tool] simple_llm_query called: {prompt[:50]}...")
            # Use a minimal, non-persistent run, disable recursive calls
            response = await self.a_run(
                prompt, session_id=f"mcp_query_{uuid.uuid4()}",
                persist_history=False, strategy_override=ProcessingStrategy.DIRECT_LLM
            )
            return response

        # If ADK tools exist, potentially expose them via MCP automatically?
        if ADK_AVAILABLE and isinstance(self, LlmAgent) and self.tools:
             logger.info("Attempting to expose ADK tools via MCP server...")
             for adk_tool in self.tools:
                 if adk_tool.name in ["code_execution", "adk_tool_a2a_send_and_wait", "adk_tool_a2a_send_no_wait", "adk_tool_a2a_get_task_status", "adk_tool_a2a_cancel_task"]:
                     continue
                 if not isinstance(adk_tool, BaseTool): continue
                 try:
                     mcp_schema = adk_to_mcp_tool_type(adk_tool)

                     # Define the MCP tool handler dynamically
                     async def mcp_tool_handler(tool_name=adk_tool.name, **kwargs):
                         logger.info(f"[MCP Tool via ADK] Calling {tool_name} with {kwargs}")
                         # ADK tools expect ToolContext, which we don't have here.
                         # We might need to simulate it or adapt the tool execution.
                         # This simple version calls the tool's underlying function if possible.
                         # WARNING: This bypasses ADK's standard tool execution flow.
                         if hasattr(adk_tool, 'func') and callable(adk_tool.func):
                             # This assumes the function doesn't need ToolContext
                             result = await adk_tool.func(**kwargs)
                             # Convert result to MCP content (e.g., TextContent)
                             if isinstance(result, str):
                                 return [mcp_types.TextContent(type="text", text=result)]
                             else:
                                 try:
                                     return [mcp_types.TextContent(type="text", text=json.dumps(result))]
                                 except:
                                     return [mcp_types.TextContent(type="text", text=str(result))]
                         else:
                             logger.warning(f"Cannot directly call ADK tool {tool_name} via MCP.")
                             return [mcp_types.TextContent(type="text", text=f"Error: Cannot execute ADK tool {tool_name} directly.")]

                     # Register the dynamic handler with the MCP server
                     self.mcp_server.tool(name=mcp_schema.name)(mcp_tool_handler)
                     logger.info(f"Exposed ADK tool '{adk_tool.name}' as MCP tool '{mcp_schema.name}'.")

                 except Exception as e:
                     logger.warning(f"Failed to expose ADK tool '{adk_tool.name}' via MCP: {e}")


        logger.info(f"MCP server setup complete for agent '{self.amd.name}'. Run `agent.run_mcp_server()` to start.")
        return self.mcp_server

    def run_mcp_server(self, transport='sse', **kwargs):
        """Starts the MCP server (blocking)."""
        if not self.mcp_server:
            logger.error("MCP server not initialized. Call setup_mcp_server first.")
            return
        if not MCP_AVAILABLE:
             logger.error("MCP library not available. Cannot run MCP server.")
             return
        logger.info(f"Starting MCP server for agent '{self.amd.name}' using {transport} transport...")
        # This is blocking, run in a separate process/thread for a long-running agent
        try:
            self.mcp_server.run(transport=transport, **kwargs)
        except Exception as e:
            logger.error(f"MCP server failed to run: {e}", exc_info=True)

    # MCP Client Setup is now handled by ADK's MCPToolset via the Builder


    def setup_a2a_server(self, host="0.0.0.0", port=5000, **a2a_server_options):
        """
        Initialize and configure the A2A server capabilities using python-a2a.
        This dynamically creates a server class with the agent's capabilities.
        """
        if not A2A_AVAILABLE:
            logger.warning("python-a2a library not installed. Cannot setup A2A server.")
            return None
        if self.a2a_server:
            logger.warning("A2A server already initialized.")
            return self.a2a_server

        logger.info(f"Setting up A2A server for agent '{self.amd.name}' on {host}:{port}")

        agent_instance = self # Reference to the current EnhancedAgent instance

        # Define the A2A Server class dynamically using the decorator
        @a2a_agent_decorator(
            name=self.amd.name or "EnhancedAgent",
            description=f"Enhanced Agent '{self.amd.name}' - Capabilities: ADK({ADK_AVAILABLE}), MCP({MCP_AVAILABLE}), A2A({A2A_AVAILABLE})",
            version="1.0.0",
            # Other AgentCard fields...
        )
        class DynamicA2AServer(A2AServer):
            bound_agent: EnhancedAgent = agent_instance

            def handle_task(self, task: Task) -> Task:
                """ Handles incoming A2A tasks by calling the EnhancedAgent's async logic. """
                # --- (handle_task implementation remains the same as before) ---
                logger.info(f"[A2A Server {self.bound_agent.amd.name}] Received task: {task.id}")
                async def run_agent_async():
                    # ... (logic to extract prompt, call a_run, update task) ...
                    try:
                        user_prompt = ""
                        # ... (extract user_prompt from task.message) ...
                        if task.message and task.message.get("content"):
                            content = task.message["content"]
                            if isinstance(content, dict) and content.get("type") == "text":
                                user_prompt = content.get("text", "").strip()
                            elif isinstance(content, str):
                                user_prompt = content.strip()

                        if not user_prompt:
                            raise ValueError("Task message has no text content.")

                        session_id = task.message.get("session_id", task.id)
                        agent_response = await self.bound_agent.a_run(
                            user_prompt,
                            session_id=session_id,
                            persist_history=False,
                            a2a_task_id=task.id
                        )
                        task.artifacts = [{"parts": [{"type": "text", "text": str(agent_response)}]}]
                        task.status = TaskStatus(state=TaskState.COMPLETED)
                    except Exception as e:
                        # ... (error handling) ...
                        logger.error(f"[A2A Task {task.id}] Error during processing: {e}", exc_info=True)
                        error_msg = f"Internal agent error: {str(e)}"
                        task.artifacts = [{"parts": [{"type": "text", "text": error_msg}]}]
                        task.status = TaskStatus(state=TaskState.FAILED, message={"role": "agent", "content": {"type": "text", "text": error_msg}})
                    return task
                try:
                    updated_task = asyncio.run(run_agent_async())
                    return updated_task
                except RuntimeError as e:
                    # ... (handle RuntimeError) ...
                    logger.error(f"RuntimeError calling asyncio.run in handle_task: {e}.")
                    task.status = TaskStatus(state=TaskState.FAILED, message={"role": "agent", "content": {"type": "text", "text": "Internal Server Error processing task asynchronously."}})
                    return task
                # --- (end of handle_task logic) ---


            # --- Expose Skills ---
            @a2a_skill_decorator(
                name="General Query",
                description="Process general natural language queries using the agent's primary LLM.",
                examples=["What is the capital of France?", "Summarize the plot of Hamlet."]
            )
            def general_query_skill(self, query: str) -> str:
                """Handles general queries via the skill mechanism by calling a_run."""
                logger.info(f"[A2A Skill] Received general_query: {query[:50]}...")
                async def run_skill_async():
                    # Call a_run, forcing direct LLM strategy for simple queries
                    response = await self.bound_agent.a_run(
                        query,
                        a2a_task_id=f"skill_query_{uuid.uuid4()}",
                        strategy_override=ProcessingStrategy.DIRECT_LLM,
                        persist_history=False
                        )
                    return response
                try:
                    # Bridge sync skill call to async agent logic
                    return asyncio.run(run_skill_async())
                except RuntimeError:
                     logger.error("RuntimeError calling asyncio.run in general_query_skill.")
                     return "Error: Could not process skill asynchronously."

            # --- FIXED: Generic Skill for ADK Tools ---
            if ADK_AVAILABLE and isinstance(agent_instance, LlmAgent) and agent_instance.tools:
                # Check if there are any ADK tools to expose
                adk_tool_list = [t for t in agent_instance.tools if isinstance(t, BaseTool)]
                if adk_tool_list:
                    logger.info(f"Exposing {len(adk_tool_list)} ADK tools via 'execute_adk_tool' A2A skill.")

                    @a2a_skill_decorator(
                        name="execute_adk_tool",
                        description=f"Executes a registered ADK tool. Available tools: {', '.join([t.name for t in adk_tool_list])}",
                        examples=["Execute tool 'some_tool_name' with argument 'arg1'='value1'"] # Generic example
                    )
                    def execute_adk_tool_skill(self, tool_name: str, arguments: dict[str, Any]) -> str:
                        """Generic skill to execute an ADK tool by name with arguments."""
                        logger.info(f"[A2A Skill] Request to execute ADK tool: {tool_name} with args: {arguments}")

                        # Find the ADK tool instance on the bound agent
                        tool_to_call: BaseTool | None = None
                        for tool in self.bound_agent.tools:
                            if isinstance(tool, BaseTool) and tool.name == tool_name:
                                tool_to_call = tool
                                break

                        if not tool_to_call:
                            logger.warning(f"[A2A Skill] ADK tool '{tool_name}' not found.")
                            return f"Error: ADK tool '{tool_name}' not found on this agent."

                        # --- Bridge sync skill call to async ADK tool execution ---
                        async def run_adk_tool_async():
                            try:
                                # ADK tools require ToolContext. We can provide a minimal one or None.
                                # Providing None might limit tool functionality.
                                # Let's try providing None for simplicity first.
                                adk_tool_context = None

                                # Check if the tool has an async run method (most ADK tools should)
                                if hasattr(tool_to_call, 'run_async') and iscoroutinefunction(tool_to_call.run_async):
                                    # Pass arguments directly to run_async
                                    result = await tool_to_call.run_async(args=arguments, tool_context=adk_tool_context)
                                    # Convert result to string for A2A response
                                    if isinstance(result, str): return result
                                    try: return json.dumps(result)
                                    except: return str(result)
                                elif hasattr(tool_to_call, 'run') and callable(tool_to_call.run):
                                    # Fallback to synchronous run in thread pool
                                    logger.warning(f"ADK tool '{tool_name}' has no run_async, using synchronous run in thread.")
                                    result = await asyncio.to_thread(tool_to_call.run, args=arguments, tool_context=adk_tool_context)
                                    if isinstance(result, str): return result
                                    try: return json.dumps(result)
                                    except: return str(result)
                                else:
                                     return f"Error: ADK tool '{tool_name}' has no callable run or run_async method."

                            except Exception as e:
                                logger.error(f"[A2A Skill] Error executing ADK tool '{tool_name}': {e}", exc_info=True)
                                return f"Error executing ADK tool {tool_name}: {e}"

                        # Execute the async tool runner
                        try:
                            return asyncio.run(run_adk_tool_async())
                        except RuntimeError:
                            logger.error(f"RuntimeError calling asyncio.run in execute_adk_tool_skill for tool {tool_name}.")
                            return "Error: Could not execute ADK tool asynchronously."

            # --- End of Skill Definitions ---

        # Instantiate the dynamic server class
        try:
             self.a2a_server = DynamicA2AServer(**a2a_server_options)
             logger.info(f"A2A server instance created for agent '{self.amd.name}'.")
             return self.a2a_server
        except Exception as e:
             logger.error(f"Failed to instantiate dynamic A2A Server: {e}", exc_info=True)
             return None


    def run_a2a_server(self, host="0.0.0.0", port=5000, **kwargs):
        """Starts the A2A server (blocking) using the python-a2a run_server function."""
        if not self.a2a_server:
            logger.error("A2A server not initialized. Call setup_a2a_server first.")
            return
        if not A2A_AVAILABLE:
            logger.error("python-a2a library not available. Cannot run A2A server.")
            return

        # Get effective host/port from server instance if set, otherwise use args
        effective_host = getattr(self.a2a_server, 'host', host)
        effective_port = getattr(self.a2a_server, 'port', port)

        logger.info(f"Starting A2A server for agent '{self.amd.name}' via run_server_func on {effective_host}:{effective_port}...")
        try:
            # Call the imported run_server function, passing the agent instance
            run_a2a_server_func(self.a2a_server, host=effective_host, port=effective_port, **kwargs) # This blocks
        except Exception as e:
            logger.error(f"A2A server failed to run: {e}", exc_info=True)

    async def setup_a2a_client(self, target_agent_url: str) -> A2AClient | None:
        """Gets or creates an A2A client for a specific target agent URL using python-a2a."""
        if not A2A_AVAILABLE:
            logger.warning("python-a2a library not installed. Cannot setup A2A client.")
            return None

        async with self.a2a_client_lock:
            if target_agent_url in self.a2a_clients:
                logger.debug(f"Reusing cached A2A client for {target_agent_url}")
                return self.a2a_clients[target_agent_url]

            logger.info(f"Setting up A2A client for target: {target_agent_url}")
            try:
                # python-a2a client likely fetches card on init or first call
                client = A2AClient(base_url=target_agent_url) # Pass the URL directly
                # Verify connection implicitly by getting card (optional, client might do lazy loading)
                # agent_card = await client.get_agent_card() # If method exists
                # logger.info(f"Successfully connected A2A client to agent: {agent_card.name}")
                self.a2a_clients[target_agent_url] = client
                logger.info(f"A2A client created for target: {target_agent_url}")
                return client
            except Exception as e:
                logger.error(f"Failed to setup A2A client for {target_agent_url}: {e}", exc_info=True)
                return None

    async def close_a2a_clients(self):
        """Closes all cached A2A client connections."""
        async with self.a2a_client_lock:
            logger.info(f"Closing {len(self.a2a_clients)} A2A clients.")
            # A2AClient may manage underlying httpx clients automatically.
            # If explicit close needed in future versions, add here.
            # for client in self.a2a_clients.values():
            #     await client.close() # If available
            self.a2a_clients.clear()

    def setup_adk_runner(self, runner_options: dict[str, Any] | None = None):
        """Initializes an ADK runner for this agent (if ADK enabled)."""
        if not ADK_AVAILABLE:
            logger.warning("ADK not available. Cannot setup ADK runner.")
            return None
        if not isinstance(self, LlmAgent):
            logger.error("Agent must inherit from LlmAgent to use ADK runner directly.")
            return None
        if self.adk_runner:
            logger.warning("ADK runner already initialized.")
            return self.adk_runner

        runner_opts = runner_options or {}
        runner_class = runner_opts.pop("runner_class", InMemoryRunner) # Default to InMemory
        app_name = runner_opts.pop("app_name", f"{self.amd.name}_ADKApp")

        if runner_class == InMemoryRunner:
            runner_opts = {}

        logger.info(f"Setting up ADK Runner ({runner_class.__name__}) for app '{app_name}'...")

        try:
             # Pass the agent instance and other options to the runner constructor
            self.adk_runner = runner_class(agent=self, app_name=app_name, **runner_opts)
            self.adk_session_service = self.adk_runner.session_service # Store session service
            logger.info(f"ADK {runner_class.__name__} setup complete for agent '{self.amd.name}'.")
            return self.adk_runner
        except Exception as e:
            logger.error(f"Failed to setup ADK runner: {e}", exc_info=True)
            self.adk_runner = None
            self.adk_session_service = None
            return None


    # --- Core Agent Logic (`a_run`) ---

    async def a_run(self,
                    user_input: str,
                    session_id: str | None = None,
                    persist_history: bool = True,
                    strategy_override: ProcessingStrategy | None = None,
                    kwargs_override: dict[str, Any] | None = None, # For fine-grained control
                    a2a_task_id: str | None = None # Context if called from A2A task
                    ) -> str:
        """
        Main asynchronous execution logic for the agent turn.

        Orchestrates world model updates, state sync, strategy selection,
        execution, cost tracking, and callbacks.
        """
        self.internal_state = InternalAgentState.PROCESSING
        start_time = time.monotonic()
        session_id = session_id or "default" # Use 'default' if none provided
        response = "Error: Processing failed." # Default error
        turn_cost = 0.0
        span = None # OTel span

        if not self.tracer: self._setup_telemetry() # Ensure tracer exists

        try:
            with self.tracer.start_as_current_span(f"Agent Run: {self.amd.name}", attributes={"session_id": session_id}) as span:

                # Ensure session history list exists
                if session_id not in self.message_history:
                    logger.debug(f"Initializing history for session: {session_id}")
                    self.message_history[session_id] = []

                logger.info(f"--- Agent Run Start (Session: {session_id}) ---")
                span.add_event("Agent run started")
                logger.info(f"User Input: {user_input[:100]}...")
                span.set_attribute("user_input", user_input[:500]) # Log truncated input

                # 0. Get ADK Session State (if ADK enabled and syncing)
                adk_session_state = None
                if self.sync_adk_state and self.adk_session_service:
                    try:
                        # ADK SessionService methods are typically synchronous
                        # Run in threadpool to avoid blocking
                        adk_session = await asyncio.to_thread(
                             self.adk_session_service.get_session,
                             app_name=self.adk_runner.app_name, # Assuming runner is set if syncing
                             user_id=self.amd.user_id or "adk_user", # Needs consistent user ID
                             session_id=session_id
                        )
                        if adk_session:
                            adk_session_state = adk_session.state
                        else:
                            logger.warning(f"ADK Session '{session_id}' not found for state sync.")
                            # Optionally create session here? Be careful about race conditions.
                    except Exception as sync_e:
                        logger.error(f"Error getting ADK session state for sync: {sync_e}")

                # 1. Update World Model & Sync State (Run *before* strategy selection)
                # flow_world_model is now responsible for syncing *from* ADK state initially
                await self.flow_world_model(user_input, session_id, adk_session_state)
                span.add_event("World model updated")

                # 2. Prepare message history for this turn
                current_turn_messages = self._prepare_llm_messages(user_input, session_id)
                span.set_attribute("history_length", len(current_turn_messages) -1) # Exclude current input

                # 3. Determine Processing Strategy
                if strategy_override:
                    strategy = strategy_override
                    strategy_reasoning = "Strategy overridden by caller."
                    logger.info(f"Strategy forced by override: {strategy.value}")
                else:
                    strategy, strategy_reasoning = self._determine_strategy_heuristic(user_input, current_turn_messages)
                    logger.info(f"Strategy Selected: {strategy.value} (Reason: {strategy_reasoning})")
                span.set_attribute("selected_strategy", strategy.value)
                span.set_attribute("strategy_reasoning", strategy_reasoning)


                # --- Prepare kwargs for execution based on strategy ---
                exec_kwargs = kwargs_override or {}
                exec_kwargs['session_id'] = session_id
                exec_kwargs['user_input'] = user_input
                exec_kwargs['current_turn_messages'] = current_turn_messages
                exec_kwargs['adk_session_state'] = adk_session_state # Pass state for potential use/update


                # 4. Execute Selected Strategy
                logger.info(f"Executing strategy: {strategy.value}")
                if strategy == ProcessingStrategy.ADK_RUN:
                    if ADK_AVAILABLE and self.adk_runner:
                        response = await self._execute_adk_run(**exec_kwargs)
                    else:
                        logger.warning("ADK_RUN strategy selected, but ADK runner not available/configured. Falling back.")
                        # Fallback strategy? Maybe DIRECT_LLM?
                        strategy = ProcessingStrategy.DIRECT_LLM
                        response = await self._execute_direct_llm(**exec_kwargs)

                elif strategy == ProcessingStrategy.A2A_CALL:
                    if A2A_AVAILABLE:
                        response = await self._execute_a2a_call(**exec_kwargs)
                    else:
                        logger.warning("A2A_CALL strategy selected, but A2A not available. Falling back.")
                        strategy = ProcessingStrategy.DIRECT_LLM
                        response = await self._execute_direct_llm(**exec_kwargs)

                else: # Default: DIRECT_LLM
                    response = await self._execute_direct_llm(**exec_kwargs)

                span.set_attribute("raw_response_length", len(response))
                span.add_event("Strategy execution complete")

                # 5. Persist History (if successful and enabled)
                # Add assistant response to history
                if persist_history and not response.startswith("Error:"):
                     self._add_to_history(session_id, LLMMessage(role="assistant", content=response).to_dict())

                # 6. Sync World Model *back* to ADK State (if changed and enabled)
                if self.sync_adk_state and adk_session_state is not None:
                    try:
                        self.world_model.sync_to_adk_state(adk_session_state)
                        span.add_event("ADK state synchronized and updated")
                    except Exception as sync_e:
                         logger.error(f"Error syncing/updating ADK session state: {sync_e}")
                         span.record_exception(sync_e)

                # 7. Track Cost (using last_llm_result if available)
                if self.last_llm_result:
                    try:
                        cost = completion_cost(completion_response=self.last_llm_result, model=self.amd.model)
                        if cost:
                            turn_cost = cost
                            self.total_cost += turn_cost
                            logger.info(f"Turn Cost: ${turn_cost:.6f}, Total Cost: ${self.total_cost:.6f}")
                            span.set_attribute("llm_cost", turn_cost)
                            span.set_attribute("total_agent_cost", self.total_cost)
                        self.last_llm_result = None # Clear after use
                    except Exception as cost_e:
                        logger.warning(f"Failed to calculate cost: {cost_e}")
                        span.add_event("Cost calculation failed", attributes={"error": str(cost_e)})


                # 8. Run Post Callback
                if self.post_run_callback and not response.startswith("Error:"):
                    try:
                        if iscoroutinefunction(self.post_run_callback):
                            await self.post_run_callback(session_id, response, turn_cost)
                        else:
                            self.post_run_callback(session_id, response, turn_cost)
                        span.add_event("Post-run callback executed")
                    except Exception as cb_e:
                        logger.error(f"Post-run callback failed: {cb_e}", exc_info=True)
                        span.record_exception(cb_e)


                logger.info(f"Agent Run finished in {time.monotonic() - start_time:.2f}s. Response: {response[:100]}...")

        except Exception as e:
            logger.error(f"Error during agent run (Session: {session_id}): {e}", exc_info=True)
            self.internal_state = InternalAgentState.ERROR
            response = f"Error: An internal error occurred during processing: {str(e)}"
            if span:
                 span.set_status(trace.Status(trace.StatusCode.ERROR, f"Agent run failed: {e}"))
                 span.record_exception(e)
        finally:
            self.internal_state = InternalAgentState.IDLE
            if span: span.end() # Ensure span is closed
            logger.info(f"--- Agent Run End (Session: {session_id}) ---")

        return str(response) # Ensure string output

    def run(self, user_input: str, session_id: str | None = None, **kwargs) -> str:
        """Synchronous wrapper for a_run."""
        try:
            # get_event_loop() is deprecated in 3.10+, use get_running_loop() or new_event_loop()
            try:
                asyncio.get_running_loop()
                # If loop is running, cannot use asyncio.run. Need to schedule and wait.
                # This is complex to get right universally (e.g., in notebooks vs servers).
                # Simplest approach for sync call from sync context is asyncio.run()
                # If called from async context, user should await a_run() directly.
                logger.warning("Synchronous 'run' called from a running event loop. "
                               "This might block the loop. Consider using 'await a_run'.")
                # Fallback to basic run, may error if loop is running
                return asyncio.run(self.a_run(user_input, session_id=session_id, **kwargs))
            except RuntimeError: # No running event loop
                 return asyncio.run(self.a_run(user_input, session_id=session_id, **kwargs))
        except Exception as e:
            logger.error(f"Error in synchronous run wrapper: {e}", exc_info=True)
            return f"Error: Failed to execute synchronous run: {e}"

    # --- Strategy Determination ---

    def _determine_strategy_heuristic(self, user_input: str, messages: list[dict]) -> tuple[ProcessingStrategy, str]:
        """Determines the processing strategy using heuristics (faster than LLM)."""
        # 1. Check for keywords indicating specific needs
        input_lower = user_input.lower()
        # Example Keywords:
        code_keywords = {"execute", "run code", "python", "calculate", "script"}
        search_keywords = {"search", "google", "find information", "what is", "who is"}
        agent_keywords = {"ask agent", "tell agent", "delegate to"} # Keywords for A2A/MCP delegation
        tool_keywords = {"use tool", "run tool"} # Keywords for specific tool use

        # 2. Check Agent Capabilities (Tools, Servers, Clients)
        has_adk_tools = ADK_AVAILABLE and isinstance(self, LlmAgent) and bool(self.tools)
        has_adk_code_executor = ADK_AVAILABLE and isinstance(self, LlmAgent) and self.code_executor is not None
        can_do_adk_search = any(isinstance(t, type(adk_google_search) | AdkVertexAiSearchTool) for t in getattr(self, 'tools', []))
        can_do_a2a = A2A_AVAILABLE and bool(self.a2a_clients) # Check if clients configured
        # MCP check relies on tools being added via MCPToolset in ADK
        has_adk_tools and any(isinstance(t, BaseTool) and getattr(t, '_is_mcp_tool', False) for t in self.tools) # Heuristic


        # --- Strategy Logic ---
        # Priority: ADK (if tools/code/search needed) > A2A (if delegation requested) > Direct LLM

        # ADK: If code execution or search is explicitly requested or implied, or specific ADK tools mentioned
        if ADK_AVAILABLE and self.adk_runner:
            if has_adk_code_executor and any(kw in input_lower for kw in code_keywords):
                return ProcessingStrategy.ADK_RUN, "Input suggests code execution, using ADK."
            if can_do_adk_search and any(kw in input_lower for kw in search_keywords):
                 return ProcessingStrategy.ADK_RUN, "Input suggests web/data search, using ADK."
            # Check if input mentions names of specific ADK tools
            if has_adk_tools:
                tool_names = {t.name.lower() for t in self.tools}
                if any(f" {name} " in input_lower for name in tool_names) or any(kw in input_lower for kw in tool_keywords):
                     return ProcessingStrategy.ADK_RUN, "Input mentions specific ADK tool or requests tool use."
            # General ADK case: If ADK is primary mode and input isn't trivial
            if len(user_input.split()) > 5: # Simple heuristic for non-trivial input
                # If ADK tools exist, assume ADK might be needed for planning
                if has_adk_tools:
                    return ProcessingStrategy.ADK_RUN, "Complex input and ADK tools available, using ADK planning."
                # If only basic LLM agent, still might use ADK runner for session mgmt? Check config.
                # Defaulting to DIRECT_LLM if no specific ADK features seem required.

        # A2A: If delegation is requested and A2A clients are available
        if can_do_a2a and any(kw in input_lower for kw in agent_keywords):
             # : Could use LLM here to extract target agent if multiple clients exist
            return ProcessingStrategy.A2A_CALL, "Input suggests delegating to another agent."

        # Fallback: Direct LLM
        return ProcessingStrategy.DIRECT_LLM, "Input seems suitable for direct LLM processing."


    # --- Strategy Execution Helpers ---

    def _prepare_llm_messages(self, user_input: str, session_id: str) -> list[dict]:
        """Prepares the list of messages for the LLM call, including history and system prompts."""
        session_history = self.message_history.get(session_id, [])

        # Construct message list
        messages: list[dict] = []
        messages.extend(self.construct_initial_prompts()) # System/world model/tool prompts
        # Add history (ensure alternating roles if possible, handle potential issues)
        messages.extend(session_history)
        # Add current user input
        messages.append(LLMMessage(role="user", content=user_input).to_dict())

        # Trim messages based on token count or turn limit
        trimmed_messages = self._trim_messages(messages)

        # Add user input to persistent history *before* LLM call
        # Note: assistant response added *after* successful call in a_run
        self._add_to_history(session_id, LLMMessage(role="user", content=user_input).to_dict())

        return trimmed_messages

    async def _execute_direct_llm(self, current_turn_messages: list[dict], session_id: str, **kwargs) -> str:
        """Executes a direct call to the LLM using LiteLLM."""
        logger.debug("Executing direct LLM call...")
        if not current_turn_messages: return "Error: No messages prepared for LLM."
        try:
            response_content = await self.a_run_llm_completion(current_turn_messages)
            return response_content
        except Exception as e:
            logger.error(f"Direct LLM execution failed: {e}", exc_info=True)
            return f"Error during LLM generation: {e}"

    async def _execute_adk_run(self, user_input: str, session_id: str, adk_session_state: State | None, **kwargs) -> str:
        """Executes the agent's logic using the configured ADK runner."""
        if not self.adk_runner or not self.adk_session_service:
            return "Error: ADK Runner or Session Service is not configured for this agent."

        logger.debug(f"Executing ADK run for session {session_id}...")
        final_response_text = "Error: ADK processing did not yield a final textual response."
        # Use user_id from AMD if available, default otherwise
        user_id = self.amd.user_id or "adk_user"
        app_name = self.adk_runner.app_name

        try:
            # 1. Ensure ADK session exists
            try:
                # Check and potentially create session (synchronous, run in thread)
                session_exists = await asyncio.to_thread(
                    self.adk_session_service.get_session, app_name=app_name, user_id=user_id, session_id=session_id
                )
                if not session_exists:
                     logger.info(f"Creating ADK session {session_id} for user {user_id} in app {app_name}")
                     # Pass initial state from World Model if syncing
                     initial_state = self.world_model.to_dict() if self.sync_adk_state else {}
                     await asyncio.to_thread(
                         self.adk_session_service.create_session,
                         app_name=app_name, user_id=user_id, session_id=session_id,
                         state=initial_state
                     )
                elif adk_session_state is None and self.sync_adk_state:
                    # If session existed but we couldn't get state earlier, try again
                     session = await asyncio.to_thread(self.adk_session_service.get_session, app_name=app_name, user_id=user_id, session_id=session_id)
                     if session: adk_session_state = session.state

            except Exception as session_e:
                logger.error(f"Failed to ensure ADK session {session_id}: {session_e}", exc_info=True)
                return f"Error setting up ADK session: {session_e}"

            # 2. Prepare ADK input (handle multi-modal later)
            # Assuming user_input is text for now
            adk_input_content = Content(role='user', parts=[Part(text=user_input)])

            # 3. Execute ADK run_async
            all_events_str = [] # For logging/debugging
            async for event in self.adk_runner.run_async(
                user_id=user_id, session_id=session_id, new_message=adk_input_content):

                # Log event details (optional, can be verbose)
                try:
                    event_dict = event.model_dump(exclude_none=True)
                    all_events_str.append(json.dumps(event_dict, default=str)) # Serialize complex types
                    logger.debug(f"ADK Event ({event.author}): {all_events_str[-1]}")
                except Exception as log_e:
                    logger.debug(f"ADK Event ({event.author}): [Error logging event details: {log_e}]")

                # Call progress callback
                if self.progress_callback:
                     try:
                         progress_data = {"type": "adk_event", "event": event.model_dump(exclude_none=True)}
                         if iscoroutinefunction(self.progress_callback): await self.progress_callback(progress_data)
                         else: self.progress_callback(progress_data)
                     except Exception as cb_e: logger.warning(f"Progress callback failed for ADK event: {cb_e}")

                # Check for Human-in-Loop triggers (example)
                #if event.actions and event.actions.request_human_input:
                #     if self.human_in_loop_callback:
                #         logger.info(f"ADK requesting human input: {event.actions.request_human_input.reason}")
                         # This needs a mechanism to pause and resume the run_async loop
                         # HIL is complex with async generators. Placeholder for now.
                         # human_response = await self.human_in_loop_callback(...)
                         # Need to inject response back into ADK runner - not straightforward
               #          logger.warning("Human-in-Loop requested by ADK, but interaction is not implemented.")
                         # Could potentially send an error response back?
               #      else:
               #         logger.warning("ADK requested human input, but no HIL callback is configured.")


                # Extract final textual response
                if event.is_final_response():
                    # Prioritize text part
                    if event.content and event.content.parts:
                        text_parts = [p.text for p in event.content.parts if hasattr(p, 'text')]
                        if text_parts:
                            final_response_text = "\n".join(text_parts).strip()
                        else: # Handle other content types if needed (e.g., function call results as final)
                            # For now, just serialize the first part if no text found
                            final_response_text = str(event.content.parts[0]) if event.content.parts else "ADK finished with non-text content."
                    elif event.actions and event.actions.escalate:
                        final_response_text = f"Error: Agent escalated: {event.error_message or 'No specific message.'}"
                    elif event.error_message:
                         final_response_text = f"Error: ADK processing failed: {event.error_message}"
                    else:
                         final_response_text = "ADK processing finished without a clear textual response."
                    break # Stop processing events

            # 4. Update World Model from final ADK state (if syncing)
            # This happens *after* the run completes, the sync in a_run updates the persisted state.
            if self.sync_adk_state and adk_session_state is not None:
                 # Fetch potentially updated state after run completion
                 try:
                     final_session = await asyncio.to_thread(self.adk_session_service.get_session, app_name=app_name, user_id=user_id, session_id=session_id)
                     if final_session:
                         self.world_model.sync_from_adk_state(final_session.state)
                     else:
                         logger.warning(f"Could not fetch final ADK state for session {session_id} after run.")
                 except Exception as sync_e:
                     logger.error(f"Error fetching final ADK state: {sync_e}")


            logger.debug("ADK run finished.")
            return final_response_text

        except Exception as e:
            logger.error(f"ADK execution failed: {e}", exc_info=True)
            # Return partial events log on error for debugging
            events_preview = "\n".join(all_events_str[:5])
            return f"Error during ADK processing: {e}\nFirst Events:\n{events_preview}"

    async def _execute_a2a_call(self, user_input: str, session_id: str, **kwargs) -> str:
        """Executes a call to another agent via A2A using python-a2a and waits for the result."""

        client = None
        task_id = None

        if not A2A_AVAILABLE: return "Error: python-a2a library not available."

        logger.debug("Executing A2A call...")

        target_agent_url = kwargs.get('target_a2a_agent_url')
        task_prompt = kwargs.get('a2a_task_prompt', user_input)

        if not target_agent_url:
            if len(self.a2a_clients) == 1:
                target_agent_url = list(self.a2a_clients.keys())[0]
                logger.info(f"Using only available A2A client target: {target_agent_url}")
            else:
                 return "Error: Target A2A agent URL not specified and multiple clients configured."
        try:
            client = await self.setup_a2a_client(target_agent_url)
            if not client:
                return f"Error: Could not connect to A2A agent at {target_agent_url}"

            task_id = str(uuid.uuid4())
            a2a_session_id = f"a2a_{session_id}_{task_id[:8]}"

            logger.info(f"Sending A2A task '{task_id}' to {target_agent_url}...")

            # --- Call python-a2a client's task sending method ---
            # The library might have a high-level `create_task` or similar.
            # Let's assume a `send_task` method exists that takes message content.
            # We construct the message payload expected by the library.
            # This structure might need adjustment based on python-a2a's specifics.
            message_payload = {
                "role": "user", # Assuming MessageRole.USER maps to "user"
                "content": {
                    "type": "text", # Assuming TextContent maps to this
                    "text": task_prompt
                 }
            }
            # The client method might take id/sessionId separately or as part of a task object
            # Assuming a method signature like: send_task(message: Dict, task_id: str, session_id: str)
            # This is an *assumption* based on typical A2A needs.
            if hasattr(client, 'send_task'):
                initial_task_info = await client.send_task(
                    message=message_payload,
                    task_id=task_id,
                    session_id=a2a_session_id
                ) # Adjust call based on actual method signature
            elif hasattr(client, 'create_task'): # Alternative common pattern
                 initial_task_info = await client.create_task(
                     message=message_payload,
                     task_id=task_id,
                     session_id=a2a_session_id
                 )
            else:
                 # Fallback to 'ask' if specific task methods are unavailable (less control)
                 logger.warning("A2A client lacks specific send_task/create_task method, using high-level 'ask'. Polling might not work.")
                 # 'ask' likely blocks and returns the final result directly
                 response_text = await client.ask(task_prompt, session_id=a2a_session_id)
                 return response_text


            # --- Process initial response and Poll ---
            # Check the structure of initial_task_info (might be a Task object, dict, etc.)
            # Extract initial state if possible
            initial_state = TaskState.SUBMITTED # Default if state not returned immediately
            if isinstance(initial_task_info, dict) and initial_task_info.get('status'):
                initial_state_val = initial_task_info['status'].get('state')
                if initial_state_val: initial_state = TaskState(initial_state_val) # Convert string to Enum
            elif hasattr(initial_task_info, 'status') and hasattr(initial_task_info.status, 'state'):
                 initial_state = initial_task_info.status.state

            logger.info(f"A2A task submitted (ID: {task_id}). Initial State: {initial_state}")

            # Don't poll if initial state is already final (unlikely but possible)
            if initial_state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
                 logger.warning(f"A2A task {task_id} already in final state {initial_state} after submission.")
                 # Need to extract result from initial_task_info here
                 # ... logic to extract result based on initial_task_info structure ...
                 return f"Task finished immediately with state {initial_state}." # Placeholder

            self.internal_state = InternalAgentState.WAITING_FOR_TOOL
            final_result = await self._poll_a2a_task(client, task_id, target_agent_url)
            self.internal_state = InternalAgentState.PROCESSING
            return final_result

        except TimeoutError:
             logger.error(f"A2A task {task_id} timed out after {self.a2a_poll_timeout}s.")
             # Attempt cancellation?
             cancel_response = "No clinet"
             if client:
                cancel_response = await client.cancel_task(task_id=task_id)
             return f"Error: A2A task timed out waiting for result from {target_agent_url} {cancel_response}."
        except Exception as e:
            logger.error(f"A2A execution failed: {e}", exc_info=True)
            return f"Error during A2A call: {e}"

    async def _poll_a2a_task(self, client: A2AClient, task_id: str, target_url: str) -> str:
        """Polls the GetTask endpoint using python-a2a client until a final state."""
        if not hasattr(client, 'get_task'):
             raise NotImplementedError(f"A2A client for {target_url} does not support 'get_task' for polling.")

        logger.debug(f"Polling A2A task {task_id} on {target_url}...")
        start_time = time.monotonic()

        while time.monotonic() - start_time < self.a2a_poll_timeout:
            try:
                # Assume get_task takes task_id (and potentially historyLength)
                task_details = await client.get_task(task_id=task_id, history_length=1)

                # --- Parse the response (structure depends on python-a2a implementation) ---
                current_state = TaskState.UNKNOWN
                final_text = f"A2A Task {task_id} finished."
                error_message = None

                # Example parsing assuming task_details is dict-like or object-like
                status_info = None
                if isinstance(task_details, dict):
                    status_info = task_details.get('status')
                elif hasattr(task_details, 'status'):
                    status_info = task_details.status

                if status_info:
                    state_val = status_info.get('state') if isinstance(status_info, dict) else getattr(status_info, 'state', None)
                    if state_val:
                        try:
                            current_state = TaskState(state_val) # Convert string to Enum
                        except ValueError:
                             logger.warning(f"Received unknown task state '{state_val}' for task {task_id}")

                    logger.debug(f"A2A task {task_id} current state: {current_state}")

                    # Call progress callback
                    if self.progress_callback:
                         # ... (progress callback logic remains the same) ...
                        pass

                    # Check for final state
                    if current_state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
                        logger.info(f"A2A task {task_id} reached final state: {current_state}")

                        # Extract final result from artifacts
                        artifacts = task_details.get('artifacts') if isinstance(task_details, dict) else getattr(task_details, 'artifacts', None)
                        if artifacts and isinstance(artifacts, list) and artifacts:
                            # Simple extraction: assume first artifact, first part is text
                            try:
                                parts = artifacts[0].get('parts') if isinstance(artifacts[0], dict) else getattr(artifacts[0], 'parts', [])
                                if parts and isinstance(parts, list) and parts:
                                    text_part = parts[0].get('text') if isinstance(parts[0], dict) else getattr(parts[0], 'text', None)
                                    if text_part:
                                        final_text = str(text_part).strip()
                            except Exception as parse_e:
                                logger.warning(f"Could not parse artifacts for task {task_id}: {parse_e}")
                                final_text = "[Could not parse final artifact]"

                        # Handle failed/cancelled states
                        if current_state == TaskState.FAILED:
                            # Try to extract error message from status
                            status_message_info = status_info.get('message') if isinstance(status_info, dict) else getattr(status_info, 'message', None)
                            if status_message_info:
                                # Assuming message content is similar structure to artifacts
                                try:
                                     err_content = status_message_info.get('content') if isinstance(status_message_info, dict) else getattr(status_message_info, 'content', None)
                                     if err_content:
                                         error_message = err_content.get('text') if isinstance(err_content, dict) else getattr(err_content, 'text', 'Unknown error')
                                except: pass # Ignore parsing errors here
                            return f"Error: A2A task failed on {target_url}: {error_message or final_text}"
                        elif current_state == TaskState.CANCELLED:
                            return f"Info: A2A task was cancelled on {target_url}."
                        else: # Completed
                            return final_text

                else:
                    logger.warning(f"A2A get_task for {task_id} returned no status info: {task_details}")

            except APIConnectionError as conn_e:
                 logger.warning(f"Connection error polling A2A task {task_id}: {conn_e}. Retrying...")
            except Exception as e:
                logger.error(f"Error polling A2A task {task_id}: {e}", exc_info=True)
                return f"Error polling A2A task status: {e}"

            await asyncio.sleep(self.a2a_poll_interval)

        raise TimeoutError(f"Polling A2A task {task_id} timed out.")

    # --- Internal Helper Methods ---

    def construct_initial_prompts(self) -> list[dict]:
        """Constructs the initial system/context messages for the LLM prompt."""
        messages = []
        # Base System Prompt
        if self.amd.system_message:
            messages.append(LLMMessage("system", self.amd.system_message).to_dict())

        # World Model Context
        wm_repr = self.world_model.show()
        if wm_repr != "[empty]":
            messages.append(LLMMessage("system", f"Current World State:\n{wm_repr}").to_dict())

        # Capabilities Overview (ADK specific parts depend on LlmAgent inheritance)
        caps = ["LiteLLM (Core LLM access)"]
        if ADK_AVAILABLE and isinstance(self, LlmAgent):
            if self.tools: caps.append("ADK Tools (including potential MCP/A2A wrappers)")
            if self.code_executor: caps.append("ADK Code Execution")
            if any(isinstance(t, type(adk_google_search) | AdkVertexAiSearchTool) for t in getattr(self, 'tools', [])):
                 caps.append("ADK Search")
        if A2A_AVAILABLE and self.a2a_clients: caps.append("A2A Client (delegate to other agents)")
        if self.mcp_server: caps.append("MCP Server (exposes capabilities)")
        if self.a2a_server: caps.append("A2A Server (receives tasks)")

        messages.append(LLMMessage("system", f"Your Capabilities: {', '.join(caps)}.").to_dict())

        # ADK Tool Instructions (if ADK enabled and tools exist)
        if ADK_AVAILABLE and isinstance(self, LlmAgent) and self.tools:
            try:
                # Use ADK's internal method to get schema if possible, otherwise basic list
                tool_schemas = getattr(self, 'tool_schemas', None) # ADK might populate this
                if tool_schemas:
                     tool_list_str = json.dumps(tool_schemas, indent=2)
                     messages.append(LLMMessage("system", f"You have access to the following tools (use FunctionCall format):\n{tool_list_str}").to_dict())
                else: # Fallback to basic list
                    tool_list = "\n".join([f"- {tool.name}: {tool.description or 'No description'}" for tool in self.tools])
                    messages.append(LLMMessage("system", f"You can use the following tools:\n{tool_list}\nRespond with a FunctionCall to use a tool.").to_dict())
            except Exception as e:
                 logger.warning(f"Could not generate detailed ADK tool instructions: {e}")


        # Add specific instructions for A2A delegation if needed
        if A2A_AVAILABLE and self.a2a_clients:
             client_names = list(self.a2a_clients.keys()) # Target URLs act as names here
             messages.append(LLMMessage("system", f"You can delegate tasks to other agents via A2A using their URLs (e.g., {client_names[0]} if available). Indicate clearly if you want to delegate.").to_dict())

        return messages

    def _add_to_history(self, session_id: str, message: dict[str, Any]):
         """Adds a message to the session history, respecting limits."""
         if session_id not in self.message_history:
              self.message_history[session_id] = []
         self.message_history[session_id].append(message)

         # Apply trimming immediately after adding (simpler than doing it before call)
         self.message_history[session_id] = self._trim_messages(self.message_history[session_id])


    def _trim_messages(self, messages: list[dict]) -> list[dict]:
        """Trims message list based on configured strategy (tokens or turns)."""
        if self.max_history_tokens and self.amd.model:
            # Token-based trimming
            max_tokens = self.max_history_tokens
            if self.trim_strategy == "litellm":
                try:
                    trimmed = trim_messages(messages, model=self.amd.model, max_tokens=max_tokens)
                    if len(trimmed) < len(messages):
                        logger.debug(f"Trimmed history from {len(messages)} to {len(trimmed)} messages using LiteLLM token strategy ({max_tokens} tokens).")
                    return trimmed
                except Exception as e:
                    logger.warning(f"LiteLLM trimming failed ({e}), falling back to basic token trim.")
                    # Fallthrough to basic token trim
            # Basic token trim (keep system, remove oldest convo pairs)
            system_msgs = [m for m in messages if m.get('role') == 'system']
            convo_msgs = [m for m in messages if m.get('role') != 'system']
            current_tokens = token_counter(messages=messages, model=self.amd.model)
            while current_tokens > max_tokens and len(convo_msgs) >= 2:
                 convo_msgs = convo_msgs[2:] # Remove oldest pair
                 current_tokens = token_counter(messages=system_msgs + convo_msgs, model=self.amd.model)
            final_messages = system_msgs + convo_msgs
            if len(final_messages) < len(messages):
                 logger.debug(f"Trimmed history from {len(messages)} to {len(final_messages)} messages using basic token strategy ({max_tokens} tokens).")
            return final_messages

        elif self.max_history_turns > 0:
            # Turn-based trimming
            system_msgs = [m for m in messages if m.get('role') == 'system']
            convo_msgs = [m for m in messages if m.get('role') != 'system']
            # Keep last N turns (each turn = user + assistant = 2 messages)
            max_convo_messages = self.max_history_turns * 2
            if len(convo_msgs) > max_convo_messages:
                trimmed_convo = convo_msgs[-max_convo_messages:]
                logger.debug(f"Trimmed history from {len(convo_msgs)//2} to {len(trimmed_convo)//2} turns.")
                return system_msgs + trimmed_convo
            else:
                return messages # No trimming needed
        else:
            # No trimming configured or possible
            logger.warning("History trimming not configured or possible (missing max_tokens/model or max_turns).")
            return messages


    async def a_run_llm_completion(self, llm_messages: list[dict], **kwargs) -> str:
        """Core wrapper around LiteLLM acompletion with error handling, streaming, and cost tracking."""
        if not llm_messages:
            logger.warning("a_run_llm_completion called with empty message list.")
            return "Error: No message provided to the model."

        self.print_verbose(f"Running model '{self.amd.model}' with {len(llm_messages)} messages.")
        # self.print_verbose("Messages:", json.dumps(llm_messages, indent=2)) # Very verbose

        # Prepare LiteLLM parameters from AgentModelData and kwargs overrides
        params = {
            'model': self.format_model or self.amd.model,
            'messages': llm_messages,
            'temperature': self.amd.temperature,
            'top_p': self.amd.top_p,
            'top_k': self.amd.top_k,
            'max_tokens': self.amd.max_tokens,
            'stream': self.stream,
            'stop': self.amd.stop_sequence,
            'user': self.amd.user_id,
            'api_base': self.amd.api_base,
            'api_version': self.amd.api_version,
            'api_key': self.amd.api_key,
            'presence_penalty': self.amd.presence_penalty,
            'frequency_penalty': self.amd.frequency_penalty,
            'caching': self.amd.caching,
            'response_format': kwargs.get('response_format'), # For a_format_class
            'tools': kwargs.get('tools'), # For LiteLLM function calling (less common now with ADK)
        }
        # Filter out None values as LiteLLM prefers absence over None for some params
        params = {k: v for k, v in params.items() if v is not None}

        # Add budget manager if present
        if self.amd.budget_manager: params['budget_manager'] = self.amd.budget_manager

        full_response_content = ""
        tool_calls_requested = None # Store tool calls if generated

        try:
            response_object = await acompletion(**params)

            if self.stream:
                collected_chunks = []
                async for chunk in response_object:
                    # Store raw chunk for potential analysis or replay
                    collected_chunks.append(chunk)
                    # Extract text delta
                    chunk_delta = chunk.choices[0].delta.content or ""
                    if chunk_delta:
                        full_response_content += chunk_delta
                        if self.stream_callback:
                             try:
                                 # Provide only the new text chunk
                                 if iscoroutinefunction(self.stream_callback): await self.stream_callback(chunk_delta)
                                 else: self.stream_callback(chunk_delta)
                             except Exception as cb_e:
                                 logger.warning(f"Stream callback failed: {cb_e}")
                    # Check for tool call deltas (less common in streaming)
                    tool_deltas = chunk.choices[0].delta.tool_calls
                    if tool_deltas:
                         logger.warning("Received tool call delta during streaming - handling may be incomplete.")
                         # : Implement robust handling of streaming tool calls if needed

                # After stream, construct a final response object mimicking non-streaming one for cost tracking
                # This is an approximation, LiteLLM might offer better ways.
                final_choice = {"message": {"role": "assistant", "content": full_response_content}}
                # If tool calls were detected during streaming, add them (complex to reconstruct accurately)
                # if reconstructed_tool_calls: final_choice["message"]["tool_calls"] = reconstructed_tool_calls
                self.last_llm_result = {
                    "choices": [{"message": final_choice["message"]}],
                    "model": self.amd.model, # Needed for cost tracking
                    # Usage stats are often missing or zero in streaming chunks, need final value if available
                    "usage": getattr(collected_chunks[-1], 'usage', {"prompt_tokens": 0, "completion_tokens": 0})
                }

            else: # Non-streaming
                self.last_llm_result = response_object # Store the full response
                # Extract content and potential tool calls
                message = response_object.choices[0].message
                full_response_content = message.content or ""
                tool_calls_requested = message.tool_calls # List of ToolCall objects

                # Check if LiteLLM did function/tool calling (different from ADK tools)
                # This path is less likely if using ADK, but supported by LiteLLM
                if tool_calls_requested:
                    logger.info(f"LiteLLM requested {len(tool_calls_requested)} tool calls.")
                    # This requires a separate mechanism to execute these LiteLLM-requested tools
                    # and send back 'tool' role messages in the next turn.
                    # Not implemented here as focus is on ADK/A2A tools.
                    # For now, return a message indicating tool call request.
                    calls_repr = ", ".join([f"{tc.function.name}" for tc in tool_calls_requested])
                    return f"Info: LLM requested tool calls ({calls_repr}). Direct execution not implemented."


            self.print_verbose(f"Model Response: {full_response_content[:100]}...")
            return full_response_content

        except RateLimitError as e:
            logger.error(f"Rate limit error from {self.amd.model}: {e}")
            # Implement backoff/retry? For now, re-raise.
            raise
        except (BadRequestError, APIConnectionError, InternalServerError) as e:
            logger.error(f"API/Server error during LiteLLM call for {self.amd.model}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LiteLLM completion: {e}", exc_info=True)
            raise

    async def a_format_class(self,
                             pydantic_model: type[BaseModel],
                             prompt: str,
                             message_context: list[dict] | None = None,
                             max_retries: int = 2) -> dict[str, Any]:
        """Uses LiteLLM's response_format feature to get structured JSON output, with retries."""
        logger.debug(f"Formatting prompt for Pydantic model: {pydantic_model.__name__}")
        model_schema = pydantic_model.model_json_schema()

        messages = message_context or []
        # System prompt explaining the task and schema
        messages.append({
            "role": "system",
            "content": f"Your task is to analyze the user's request and extract information into a JSON object.\n"
                       f"Strictly adhere to the following Pydantic schema:\n"
                       f"```json\n{json.dumps(model_schema, indent=2)}\n```\n"
                       f"Guidelines:\n"
                       f"- Analyze the request carefully.\n"
                       f"- Output *only* the JSON object, nothing else (no explanations, apologies, or markdown).\n"
                       f"- Ensure the JSON is valid and conforms exactly to the schema.\n"
                       f"- Omit optional fields if the information is not present in the request."
        })
        messages.append({"role": "user", "content": prompt})

        # Use LiteLLM's JSON mode (requires compatible model/provider)
        response_format_config = {"type": "json_object"}
        # Some providers might need the schema explicitly even in json_object mode
        # response_format_config = {"type": "json_object", "schema": model_schema}

        original_stream_state = self.stream
        self.stream = False # Ensure streaming is off for structured output
        try:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    logger.debug(f"Attempt {attempt + 1}/{max_retries + 1} to get structured JSON.")
                    # Use a potentially faster/cheaper model optimized for JSON tasks if configured?
                    self.format_model = self.format_model_
                    response_text = await self.a_run_llm_completion(messages, response_format=response_format_config)
                    self.format_model = None
                    # Clean and parse the JSON response
                    try:
                         # Basic cleaning: remove potential markdown fences
                        cleaned_response = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)

                         # Try parsing using Pydantic's TypeAdapter for direct validation
                        adapter = TypeAdapter(pydantic_model)
                        validated_obj = adapter.validate_json(cleaned_response)
                        result_dict = validated_obj.model_dump(mode='json') # Get dict representation

                        logger.debug(f"Successfully formatted and validated JSON: {result_dict}")
                        return result_dict

                    except (json.JSONDecodeError, ValidationError) as e:
                        logger.warning(f"Attempt {attempt + 1} failed: Invalid JSON or schema mismatch. Error: {e}. Response: {response_text[:500]}")
                        last_exception = ValueError(f"LLM response did not match schema after cleaning. Error: {e}. Response: '{response_text[:200]}...'")
                        # Add feedback to the model for retry
                        messages.append({"role": "assistant", "content": response_text}) # Show previous attempt
                        messages.append({"role": "system", "content": f"Your previous response was invalid ({e}). Please try again, ensuring you output *only* valid JSON matching the schema."})

                except Exception as e:
                    logger.error(f"Error during a_format_class (attempt {attempt + 1}): {e}", exc_info=True)
                    last_exception = e
                    # Don't retry on non-parsing errors immediately, could be API issue
                    break

                # Wait before retrying
                if attempt < max_retries:
                     await asyncio.sleep(1.5 ** attempt) # Exponential backoff

            # If all retries fail
            logger.error(f"Failed to get valid structured JSON after {max_retries + 1} attempts.")
            raise last_exception or ValueError("Failed to get structured JSON response from LLM.")

        finally:
            self.stream = original_stream_state # Restore stream setting


    async def flow_world_model(self, text_input: str, session_id: str, adk_session_state: State | None):
        """
        Analyzes input, updates internal WorldModel, and syncs with ADK state if enabled.
        Sync Priority: If ADK state exists, sync *from* it first. Then update based on text.
                     The sync *to* ADK happens after the agent run completes.
        """
        logger.debug(f"Flowing world model based on text: {text_input[:100]}...")

        # 1. Sync FROM ADK State (if enabled and state available)
        if self.sync_adk_state and adk_session_state is not None:
             logger.debug("Syncing World Model FROM ADK session state...")
             self.world_model.sync_from_adk_state(adk_session_state)

        # 2. Update World Model based on Text Input (using LLM)
        # This adds/modifies based on the current turn's input
        # Define Pydantic model for structured update extraction
        current_keys = list(self.world_model.to_dict().keys())
        class WorldModelAdaption(BaseModel):
            action: Literal['add', 'update', 'remove', 'none'] = Field(..., description="Action on the world model.")
            key: str | None = Field(None, description=f"Key to modify/add/remove (e.g., 'user_location', 'task_status'). Existing keys: {current_keys}")
            value: Any | None = Field(None, description="New value (for 'add'/'update'). Should be JSON serializable.")
            reasoning: str = Field(..., description="Why this change (or no change) is needed based on the input.")

        prompt = (f"Analyze the following text and current world state to determine if the agent's world model needs changes.\n"
                  f"Current World State Keys: {current_keys}\n"
                  f"Text Input: ```\n{text_input}\n```\n"
                  f"Decide action, key, value, and reasoning. Focus on factual updates derived *from the text*. Do not hallucinate.")

        try:
            # Use a potentially faster/cheaper model for this classification task
            # Could eventually use a separate AMD config for this call
            adaption_dict = await self.a_format_class(WorldModelAdaption, prompt)
            adaption = WorldModelAdaption(**adaption_dict)

            logger.info(f"World Model Adaption proposed: {adaption.action} on key '{adaption.key}'. Reason: {adaption.reasoning}")

            if adaption.action == 'add' or adaption.action == 'update':
                if adaption.key and adaption.value is not None:
                    self.world_model.set(adaption.key, adaption.value)
                else:
                    logger.warning("World model 'add'/'update' ignored: missing key or value.")
            elif adaption.action == 'remove':
                if adaption.key:
                    self.world_model.remove(adaption.key)
                else:
                    logger.warning("World model 'remove' ignored: missing key.")
            # Else ('none'): do nothing

        except (ValidationError, Exception) as e:
            logger.warning(f"Failed to determine world model adaption via LLM: {e}. World model may be based only on ADK sync or previous state.")

        # NOTE: Sync TO ADK happens *after* the full agent run in a_run()


    # --- ADK Tool Implementations (Internal Wrappers) ---
    def _ensure_internal_adk_tools(self):
        """Adds essential internal ADK tools if not already present."""
        if not ADK_AVAILABLE or not isinstance(self, LlmAgent):
            return
        if self.tools is None: self.tools = []

        existing_tool_names = {tool.name for tool in self.tools if isinstance(tool, BaseTool)}

        internal_adk_tools = {
            "get_world_model_key": self.adk_tool_world_model_get,
            "show_world_model": self.adk_tool_world_model_show,
        }
        if A2A_AVAILABLE:
            internal_adk_tools["a2a_send_and_wait"] = self.adk_tool_a2a_send_and_wait
            # Add NEW tools
            internal_adk_tools["a2a_send_no_wait"] = self.adk_tool_a2a_send_no_wait
            internal_adk_tools["a2a_get_task_status"] = self.adk_tool_a2a_get_task_status
            internal_adk_tools["a2a_cancel_task"] = self.adk_tool_a2a_cancel_task

        for name, func in internal_adk_tools.items():
            if name not in existing_tool_names:
                try:
                    tool_instance = FunctionTool(func=func) # ADK infers from func signature/docstring
                    self.tools.append(tool_instance)
                    logger.debug(f"Registered internal ADK tool: {name}")
                except Exception as e:
                    logger.warning(f"Failed to register internal ADK tool '{name}': {e}.")

    # --- Existing ADK Tools ---
    async def adk_tool_world_model_get(self, tool_context: ToolContext | None, key: str) -> Any | None:
        """ADK Tool: Retrieves a specific value from the agent's world model."""
        # ... (implementation remains the same) ...
        logger.info(f"[ADK Tool] get_world_model_key called for key: {key}")
        return self.world_model.get(key)

    async def adk_tool_world_model_show(self, tool_context: ToolContext | None) -> str:
        """ADK Tool: Returns a string representation of the agent's entire world model."""
        # ... (implementation remains the same) ...
        logger.info("[ADK Tool] show_world_model called")
        return self.world_model.show()

    async def adk_tool_a2a_send_and_wait(self,
                                         tool_context: ToolContext | None,
                                         target_agent_url: str,
                                         task_prompt: str,
                                         session_id: str | None = None
                                         ) -> str:
        """ADK Tool: Sends a task to another agent via A2A and waits for the final text result."""
        # ... (implementation remains the same, calls _execute_a2a_call) ...
        if not A2A_AVAILABLE: return "Error: python-a2a library not available."
        logger.info(f"[ADK Tool] a2a_send_and_wait called for target: {target_agent_url}")
        tool_session_id = session_id or f"adk_tool_a2a_{uuid.uuid4()}"
        try:
            return await self._execute_a2a_call(
                 user_input=task_prompt,
                 session_id=tool_session_id,
                 target_a2a_agent_url=target_agent_url,
                 a2a_task_prompt=task_prompt
            )
        except Exception as e:
             logger.error(f"[ADK Tool] a2a_send_and_wait failed: {e}", exc_info=True)
             return f"Error executing A2A task via ADK tool: {e}"

        # --- NEW ADK Tools for A2A ---

    async def adk_tool_a2a_send_no_wait(self,
                                        tool_context: ToolContext | None,
                                        target_agent_url: str,
                                        task_prompt: str,
                                        session_id: str | None = None
                                        ) -> str:
        """ADK Tool: Sends a task to another agent via A2A and returns the task ID immediately.

        Args:
            target_agent_url: The full URL of the target A2A agent.
            task_prompt: The natural language prompt or task for the target agent.
            session_id: Optional session ID to use for the A2A interaction.

        Returns:
            The unique ID of the submitted A2A task, or an error message.
        """
        if not A2A_AVAILABLE: return "Error: python-a2a library not available."
        logger.info(f"[ADK Tool] a2a_send_no_wait called for target: {target_agent_url}")

        try:
            client = await self.setup_a2a_client(target_agent_url)
            if not client:
                return f"Error: Could not connect to A2A agent at {target_agent_url}"

            task_id = str(uuid.uuid4())
            a2a_session_id = session_id or f"a2a_tool_nowait_{task_id[:8]}"

            message_payload = {"role": "user", "content": {"type": "text", "text": task_prompt}}

            initial_task_info = None
            if hasattr(client, 'send_task'):
                initial_task_info = await client.send_task(message=message_payload, task_id=task_id,
                                                           session_id=a2a_session_id)
            elif hasattr(client, 'create_task'):
                initial_task_info = await client.create_task(message=message_payload, task_id=task_id,
                                                             session_id=a2a_session_id)
            else:
                return "Error: A2A client does not support send_task or create_task."

            # Check for immediate errors from the submission call
            # Structure depends on python-a2a's return value
            error_info = None
            if isinstance(initial_task_info, dict):
                error_info = initial_task_info.get('error')
            elif hasattr(initial_task_info, 'error'):
                error_info = initial_task_info.error

            if error_info:
                err_msg = error_info.get('message', str(error_info)) if isinstance(error_info, dict) else str(
                    error_info)
                logger.error(f"A2A send_task (no wait) failed immediately: {err_msg}")
                return f"Error submitting A2A task: {err_msg}"
            else:
                logger.info(f"A2A task '{task_id}' submitted successfully (no wait) to {target_agent_url}.")
                return task_id  # Return the ID for later polling/checking

        except Exception as e:
            logger.error(f"[ADK Tool] a2a_send_no_wait failed: {e}", exc_info=True)
            return f"Error sending A2A task (no wait): {e}"

    async def adk_tool_a2a_get_task_status(self,
                                           tool_context: ToolContext | None,
                                           target_agent_url: str,
                                           task_id: str
                                           ) -> dict[str, Any]:
        """ADK Tool: Gets the current status and details of an A2A task.

        Args:
            target_agent_url: The URL of the agent hosting the task.
            task_id: The ID of the task to check.

        Returns:
            A dictionary containing task status details (state, message, artifacts) or an error.
        """
        if not A2A_AVAILABLE: return {"error": "python-a2a library not available."}
        logger.info(f"[ADK Tool] a2a_get_task_status called for task {task_id} on {target_agent_url}")

        try:
            client = await self.setup_a2a_client(target_agent_url)
            if not client:
                return {"error": f"Could not connect to A2A agent at {target_agent_url}"}

            if not hasattr(client, 'get_task'):
                return {"error": f"A2A client for {target_agent_url} does not support 'get_task'."}

            # Get task details from the client
            task_details = await client.get_task(task_id=task_id, history_length=1)  # History=1 gets latest status

            # Parse and return relevant info
            if isinstance(task_details, dict):
                # Basic parsing, adjust based on actual python-a2a structure
                status_info = task_details.get('status', {})
                artifacts = task_details.get('artifacts')
                return {
                    "task_id": task_id,
                    "state": status_info.get('state', 'UNKNOWN'),
                    "status_message": status_info.get('message'),  # Might be complex object
                    "artifacts": artifacts,  # Might be complex list
                    "raw_response": task_details  # Include raw for debugging
                }
            elif hasattr(task_details, 'status'):  # Object-like response
                status_obj = task_details.status
                artifacts_obj = getattr(task_details, 'artifacts', None)
                return {
                    "task_id": task_id,
                    "state": getattr(status_obj, 'state', TaskState.UNKNOWN).value,  # Get enum value
                    "status_message": getattr(status_obj, 'message', None),
                    "artifacts": artifacts_obj,
                    "raw_response": vars(task_details)  # Example conversion
                }
            else:
                return {"error": "Received unexpected response structure from get_task.", "raw_response": task_details}

        except Exception as e:
            # Catch specific errors from python-a2a if they exist (e.g., TaskNotFoundError)
            # if isinstance(e, TaskNotFoundError):
            #    logger.warning(f"[ADK Tool] A2A Task {task_id} not found on {target_agent_url}.")
            #    return {"error": f"Task {task_id} not found."}
            logger.error(f"[ADK Tool] a2a_get_task_status failed: {e}", exc_info=True)
            return {"error": f"Error getting A2A task status: {e}"}

    async def adk_tool_a2a_cancel_task(self,
                                       tool_context: ToolContext | None,
                                       target_agent_url: str,
                                       task_id: str
                                       ) -> dict[str, Any]:
        """ADK Tool: Attempts to cancel an ongoing A2A task.

        Args:
            target_agent_url: The URL of the agent hosting the task.
            task_id: The ID of the task to cancel.

        Returns:
            A dictionary indicating success or failure, possibly with the task's state after cancellation attempt.
        """
        if not A2A_AVAILABLE: return {"error": "python-a2a library not available."}
        logger.info(f"[ADK Tool] a2a_cancel_task called for task {task_id} on {target_agent_url}")

        try:
            client = await self.setup_a2a_client(target_agent_url)
            if not client:
                return {"error": f"Could not connect to A2A agent at {target_agent_url}"}

            if not hasattr(client, 'cancel_task'):
                return {"error": f"A2A client for {target_agent_url} does not support 'cancel_task'."}

            # Call the client's cancel method
            # The response structure depends heavily on the library implementation
            cancel_response = await client.cancel_task(task_id=task_id)

            # Parse response - could be simple success/fail, or updated task state
            if isinstance(cancel_response, dict):
                if 'error' in cancel_response:
                    error_info = cancel_response['error']
                    err_msg = error_info.get('message', str(error_info)) if isinstance(error_info, dict) else str(
                        error_info)
                    logger.warning(f"A2A cancel_task failed for {task_id}: {err_msg}")
                    return {"success": False, "error": err_msg, "raw_response": cancel_response}
                else:
                    # Assume success, response might contain updated task state
                    logger.info(f"A2A task {task_id} cancellation requested successfully.")
                    # Try to extract state if returned
                    state = cancel_response.get('result', {}).get('status', {}).get('state', 'UNKNOWN')
                    return {"success": True, "state_after_request": state, "raw_response": cancel_response}
            elif cancel_response is True:  # Simple boolean success
                return {"success": True, "state_after_request": "UNKNOWN"}
            else:  # Assume object-like or other structure
                # Add parsing based on observed python-a2a behavior
                logger.info(f"A2A task {task_id} cancellation request sent, parsing result.")
                # Example: Check for specific attributes if object is returned
                state = getattr(getattr(getattr(cancel_response, 'result', None), 'status', None), 'state',
                                TaskState.UNKNOWN).value
                return {"success": True, "state_after_request": state,
                        "raw_response": vars(cancel_response) if hasattr(cancel_response, '__dict__') else str(
                            cancel_response)}


        except Exception as e:
            # Catch specific errors like TaskNotFound, TaskNotCancelable if defined by python-a2a
            # if isinstance(e, TaskNotFoundError):
            #    return {"success": False, "error": f"Task {task_id} not found."}
            # if isinstance(e, TaskNotCancelableError):
            #    return {"success": False, "error": f"Task {task_id} is not in a cancelable state."}
            logger.error(f"[ADK Tool] a2a_cancel_task failed: {e}", exc_info=True)
            return {"success": False, "error": f"Error cancelling A2A task: {e}"}

    # async def adk_tool_a2a_get_task(self, tool_context: Optional[ToolContext], target_agent_url: str, task_id: str) -> Dict:
    #     """ADK Tool: Gets the current status and details of an A2A task."""
    #     # Implementation would be similar to _poll_a2a_task but return the status dict directly
    #     pass


    # --- Cost Tracking ---
    def _track_cost(self, response_obj: Any):
        """Updates cost using LiteLLM."""
        if not response_obj: return
        try:
            cost = completion_cost(completion_response=response_obj, model=self.amd.model)
            if cost is not None:
                self.total_cost += cost
                logger.info(f"Turn Cost: ${cost:.6f}, Total Accumulated Cost: ${self.total_cost:.6f}")
            else:
                 logger.debug("Cost calculation returned None (possibly streaming or non-standard response).")
        except Exception as e:
            logger.warning(f"Failed to calculate/track cost: {e}")


    # --- Cleanup ---
    async def close(self):
        """Gracefully close connections and resources."""
        logger.info(f"Closing resources for agent '{self.amd.name}'...")
        # Close A2A resources
        if self.a2a_server and hasattr(self.a2a_server, 'stop'): # Check if server has stop method
             logger.info("Stopping A2A server...")
             try:
                 await self.a2a_server.stop() # Assuming stop is async
             except Exception as e: logger.warning(f"Error stopping A2A server: {e}")
        if hasattr(self, '_a2a_task_manager_instance') and hasattr(self._a2a_task_manager_instance, 'close'):
             logger.info("Closing A2A task manager...")
             await self._a2a_task_manager_instance.close()
        await self.close_a2a_clients()

        # Close MCP server if running
        if self.mcp_server and hasattr(self.mcp_server, 'stop'): # Check for stop method
             logger.info("Stopping MCP server...")
             try:
                 # MCP server run is blocking, stop might need separate mechanism
                 # or be handled by process termination. If stop method exists:
                 # await self.mcp_server.stop() # Assuming async stop
                 logger.warning("MCP server 'stop' might need manual implementation or process signal.")
             except Exception as e: logger.warning(f"Error stopping MCP server: {e}")


        # Close ADK resources (MCPToolset connections managed by exit stack)
        if self.adk_exit_stack:
            logger.info("Closing ADK AsyncExitStack (manages MCPToolset connections)...")
            try:
                await self.adk_exit_stack.aclose()
            except Exception as e:
                logger.warning(f"Error closing ADK exit stack: {e}")

        # Close ADK runner if it has a close method
        if self.adk_runner and hasattr(self.adk_runner, 'close'):
             logger.info("Closing ADK runner...")
             try:
                  # Check if close is async
                 if iscoroutinefunction(self.adk_runner.close):
                     await self.adk_runner.close()
                 else:
                     self.adk_runner.close()
             except Exception as e: logger.warning(f"Error closing ADK runner: {e}")


        logger.info(f"Agent '{self.amd.name}' resource cleanup finished.")

    def print_verbose(self, *args):
        """Conditional logging helper."""
        if self.verbose:
            logger.debug(' '.join(map(str, args)))

# --- End of EnhancedAgent Class ---


# --- Builder Class ---

# **To make this code runnable and truly production-ready, you would still need to:**
#
# 1.  **Install Dependencies:** `pip install litellm google-cloud-aiplatform google-generativeai python-a2a mcp opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp google-adk` (adjust based on specific ADK installation instructions and desired OTel exporters).
# 2.  **Configure API Keys:** Set environment variables (e.g., `GOOGLE_API_KEY`, `OPENAI_API_KEY`) or use secure credential management.
# 3.  **Implement Secure Code Execution:** Replace `SecureCodeExecutorPlaceholder` with a real sandboxed solution if code execution is needed and ADK's built-in isn't sufficient/available.
# 4.  **Configure OpenTelemetry:** Set up the `TracerProvider` with appropriate exporters (OTLP, Jaeger, Prometheus, etc.) and resource attributes in your main application entry point.
# 5.  **Persistent Storage:** If needed, replace `InMemoryRunner` with a persistent ADK runner (e.g., database-backed) and implement persistent storage for `message_history` and `WorldModel`.
# 6.  **Configuration Management:** Load agent configurations from files (YAML, JSON) instead of hardcoding in the builder calls.
# 7.  **Error Handling & Retries:** Enhance error handling for external calls (LLM, A2A, MCP) with more specific exceptions and robust retry logic (e.g., using libraries like `tenacity`).
# 8.  **Server Deployment:** Run A2A/MCP servers in separate, managed processes or containers (e.g., using `uvicorn` for FastAPI-based servers like A2A/FastMCP, or appropriate process managers). The `run_a2a_server`/`run_mcp_server` methods are blocking.
# 9.  **Testing:** Implement comprehensive unit, integration, and end-to-end tests.
# 10. **Human-in-the-Loop:** Design and implement the actual HIL interaction points and UI/callback mechanisms.
