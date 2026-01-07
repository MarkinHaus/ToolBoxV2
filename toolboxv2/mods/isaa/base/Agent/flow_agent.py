"""
FlowAgent V2 - Production-ready Agent System

Refactored architecture:
- SessionManager: Session lifecycle with ChatSession integration
- ToolManager: Unified tool registry (local, MCP, A2A)
- CheckpointManager: Full state persistence
- BindManager: Agent-to-agent binding
- ExecutionEngine: MAKER/RLM inspired orchestration with Pause/Continue

Author: FlowAgent V2
"""

import asyncio
import json
import os
import time
import uuid
import yaml
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, AsyncGenerator, Callable

from pydantic import BaseModel, ValidationError

from toolboxv2 import get_logger
from toolboxv2.mods.isaa.base.Agent.chain import Chain, ConditionalChain
from toolboxv2.mods.isaa.base.Agent.types import (
    AgentModelData,
    PersonaConfig,
    ProgressEvent,
    ProgressTracker,
    NodeStatus,
)

# Framework imports
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

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


logger = get_logger()
AGENT_VERBOSE = os.environ.get("AGENT_VERBOSE", "false").lower() == "true"


class FlowAgent:
    """Production-ready autonomous agent with session isolation."""

    def __init__(
        self,
        amd: AgentModelData,
        verbose: bool = False,
        max_parallel_tasks: int = 3,
        auto_load_checkpoint: bool = True,
        rule_config_path: str | None = None,
        progress_callback: Callable | None = None,
        stream: bool = True,
        **kwargs
    ):
        self.amd = amd
        self.verbose = verbose
        self.stream = stream
        self._rule_config_path = rule_config_path

        self.is_running = False
        self.active_session: str | None = None

        # Statistics
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_accumulated = 0.0
        self.total_llm_calls = 0

        # Progress tracking
        self.progress_tracker = ProgressTracker(
            progress_callback=progress_callback,
            agent_name=amd.name
        )

        self.executor = ThreadPoolExecutor(max_workers=max_parallel_tasks)

        # Servers
        self.a2a_server: A2AServer | None = None
        self.mcp_server: FastMCP | None = None

        # Execution engine instance (lazy loaded)
        self._execution_engine = None

        self._init_managers(auto_load_checkpoint)
        self._init_rate_limiter()

        logger.info(f"FlowAgent '{amd.name}' initialized")

    def _init_managers(self, auto_load_checkpoint: bool):
        from toolboxv2.mods.isaa.base.Agent.session_manager import SessionManager
        from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolManager
        from toolboxv2.mods.isaa.base.Agent.checkpoint_manager import CheckpointManager
        from toolboxv2.mods.isaa.base.Agent.bind_manager import BindManager

        self.session_manager = SessionManager(
            agent_name=self.amd.name,
            default_max_history=100,
            vfs_max_window_lines=self.amd.vfs_max_window_lines,
            rule_config_path=self._rule_config_path,
            summarizer=self._create_summarizer()
        )

        self.tool_manager = ToolManager()

        self.checkpoint_manager = CheckpointManager(
            agent=self,
            auto_load=auto_load_checkpoint
        )

        self.bind_manager = BindManager(agent=self)

    def _init_rate_limiter(self):
        from toolboxv2.mods.isaa.base.IntelligentRateLimiter.intelligent_rate_limiter import (
            LiteLLMRateLimitHandler,
            load_handler_from_file,
            create_handler_from_config,
        )

        if isinstance(self.amd.handler_path_or_dict, dict):
            self.llm_handler = create_handler_from_config(self.amd.handler_path_or_dict)
        elif isinstance(self.amd.handler_path_or_dict, str) and os.path.exists(self.amd.handler_path_or_dict):
            self.llm_handler = load_handler_from_file(self.amd.handler_path_or_dict)
        else:
            self.llm_handler = LiteLLMRateLimitHandler(max_retries=3)

    def _create_summarizer(self) -> Callable:
        async def summarize(content: str) -> str:
            try:
                result = await self.a_run_llm_completion(
                    messages=[{"role": "user", "content": f"Summarize in 1-2 sentences:\n\n{content[:2000]}"}],
                    max_tokens=100,
                    temperature=0.3,
                    with_context=False,
                    model_preference="fast",
                    task_id="vfs_summarize"
                )
                return result.strip()
            except Exception:
                return f"[{len(content)} chars]"
        return summarize

    def _get_execution_engine(self, **kwargs):
        """Get or create execution engine"""
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

        return ExecutionEngine(
            agent=self,
            use_native_tools=kwargs.get('use_native_tools', True),
            human_online=kwargs.get('human_online', False),
            intermediate_callback=kwargs.get('intermediate_callback')
        )

    # =========================================================================
    # CORE: a_run_llm_completion
    # =========================================================================

    async def a_run_llm_completion(
        self,
        messages: list[dict],
        model_preference: str = "fast",
        with_context: bool = True,
        stream: bool | None = None,
        get_response_message: bool = False,
        task_id: str = "unknown",
        session_id: str | None = None,
        **kwargs
    ) -> str | Any:
        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM required")

        model = kwargs.pop('model', None) or (
            self.amd.fast_llm_model if model_preference == "fast" else self.amd.complex_llm_model
        )
        use_stream = stream if stream is not None else self.stream

        llm_kwargs = {'model': model, 'messages': messages.copy(), 'stream': use_stream, **kwargs}
        session_id = session_id or self.active_session
        system_msg = self.amd.get_system_message()
        history = []
        if session_id:
            session = self.session_manager.get(session_id)
            if session:
                await session.initialize()
                system_msg += "\n\n"+  session.build_vfs_context()
                history = session.get_history(5)
                if len(history) >= 1:
                    history = history[:-1]
        if with_context:
            llm_kwargs['messages'] = [{"role": "system", "content": f"{system_msg}"}] + history + llm_kwargs['messages']

        if 'api_key' not in llm_kwargs:
            llm_kwargs['api_key'] = self._get_api_key_for_model(model)

        try:
            if use_stream:
                llm_kwargs["stream_options"] = {"include_usage": True}

            response = await self.llm_handler.completion_with_rate_limiting(litellm, **llm_kwargs)

            if use_stream:
                result, usage = await self._process_streaming_response(response, task_id, model, get_response_message)
            else:
                result = response.choices[0].message.content
                usage = response.usage
                if get_response_message:
                    result = response.choices[0].message

            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            cost = self.progress_tracker.calculate_llm_cost(model, input_tokens, output_tokens, response)

            self.total_tokens_in += input_tokens
            self.total_tokens_out += output_tokens
            self.total_cost_accumulated += cost
            self.total_llm_calls += 1

            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def _process_streaming_response(self, response, task_id, model, get_response_message):
        from litellm.types.utils import Message, ChatCompletionMessageToolCall, Function

        result = ""
        tool_calls_acc = {}
        final_chunk = None

        async for chunk in response:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            result += content

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = ChatCompletionMessageToolCall(id=tc.id, type="function", function=Function(name="", arguments=""))
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx].function.name = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx].function.arguments += tc.function.arguments
            final_chunk = chunk

        usage = final_chunk.usage if hasattr(final_chunk, "usage") else None

        if get_response_message:
            result = Message(role="assistant", content=result or None, tool_calls=list(tool_calls_acc.values()) if tool_calls_acc else [])

        return result, usage

    def _get_api_key_for_model(self, model: str) -> str | None:
        prefix = model.split("/")[0]
        return {"openrouter": os.getenv("OPENROUTER_API_KEY"), "openai": os.getenv("OPENAI_API_KEY"),
                "anthropic": os.getenv("ANTHROPIC_API_KEY"), "google": os.getenv("GOOGLE_API_KEY"),
                "groq": os.getenv("GROQ_API_KEY")}.get(prefix)

    # =========================================================================
    # CORE: arun_function
    # =========================================================================

    async def arun_function(self, function_name: str, **kwargs) -> Any:
        if self.active_session:
            session = self.session_manager.get(self.active_session)
            if session and not session.is_tool_allowed(function_name):
                raise PermissionError(f"Tool '{function_name}' restricted in session '{self.active_session}'")

        start_time = time.perf_counter()
        result = await self.tool_manager.execute(function_name, **kwargs)

        if self.progress_tracker:
            await self.progress_tracker.emit_event(ProgressEvent(
                event_type="tool_call", node_name="FlowAgent", status=NodeStatus.COMPLETED, success=True,
                duration=time.perf_counter() - start_time, tool_name=function_name, tool_args=kwargs, tool_result=result,
            ))
        return result

    # =========================================================================
    # CORE: a_format_class
    # =========================================================================

    async def a_format_class(
        self,
        pydantic_model: type[BaseModel],
        prompt: str,
        message_context: list[dict] | None = None,
        max_retries: int = 1,
        model_preference: str = "fast",
        auto_context: bool = False,
        **kwargs
    ) -> dict[str, Any]:
        schema = pydantic_model.model_json_schema()
        model_name = pydantic_model.__name__

        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        fields_desc = [f"  {name}{'*' if name in required else ''}: {info.get('type', 'string')}" for name, info in props.items()]

        enhanced_prompt = f"{prompt}"

        messages = (message_context or []) + [{"role": "user", "content": enhanced_prompt} , {"role": "system", "content": "Return YAML with fields:\n" + "\n".join(fields_desc)}]

        for attempt in range(max_retries + 1):
            try:
                response = await self.a_run_llm_completion(
                    messages=messages, model_preference=model_preference, stream=False,
                    with_context=auto_context, temperature=0.1 + (attempt * 0.1),
                    max_tokens=500, task_id=f"format_{model_name.lower()}_{attempt}"
                )

                if not response or not response.strip():
                    raise ValueError("Empty response")

                yaml_content = self._extract_yaml_content(response)
                if not yaml_content:
                    raise ValueError("No YAML found")

                parsed_data = yaml.safe_load(yaml_content)
                if not isinstance(parsed_data, dict):
                    raise ValueError(f"Expected dict, got {type(parsed_data)}")

                validated = pydantic_model.model_validate(parsed_data)
                return validated.model_dump()

            except Exception as e:
                if attempt < max_retries:
                    messages[-1]["content"] = enhanced_prompt + f"\n\nFix error: {str(e)}"
                else:
                    raise RuntimeError(f"Failed after {max_retries + 1} attempts: {e}")

    def _extract_yaml_content(self, response: str) -> str:
        if "```yaml" in response:
            try:
                return response.split("```yaml")[1].split("```")[0].strip()
            except IndexError:
                pass
        if "```" in response:
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    lines = part.strip().split('\n')
                    if len(lines) > 1:
                        return '\n'.join(lines[1:]).strip() if lines[0].strip().isalpha() else part.strip()
        if ':' in response and not response.strip().startswith('<'):
            return response.strip()
        return ""

    # =========================================================================
    # CORE: a_run - ExecutionEngine based with Pause/Continue
    # =========================================================================

    async def a_run(
        self,
        query: str,
        session_id: str = "default",
        remember: bool = True,
        execution_id: str | None = None,
        use_native_tools: bool = True,
        human_online: bool = False,
        intermediate_callback: Callable[[str], None] | None = None,
        human_response: str | None = None,
        max_iterations: int = 15,
        token_budget: int = 10000,
        **kwargs
    ) -> str:
        """
        Main entry point for agent execution.

        Architecture: MAKER (parallel decomposition) + RLM (VFS-based context)

        Features:
        - Auto Intent Detection → Immediate/Tools/Decomposition
        - Category-based tool selection (max 5 tools)
        - RLM-VFS style ReAct loop
        - Parallel microagent execution for complex tasks
        - Pause/Continue support
        - Human-in-the-loop
        - Transaction-based rollback
        - Non-blocking learning

        Args:
            query: User query
            session_id: Session identifier
            remember: Save to history
            execution_id: For continuing paused execution
            use_native_tools: LiteLLM native tool calling vs a_format_class
            human_online: Allow human-in-the-loop
            intermediate_callback: User-facing status messages
            human_response: Response from human (for continuation)
            max_iterations: Max ReAct iterations (default 15)
            token_budget: Token budget per iteration (default 10000)
            **kwargs: Additional options

        Returns:
            Response string or special response for paused states:
            - "__PAUSED__:{execution_id}" - Execution paused
            - "__NEEDS_HUMAN__:{execution_id}:{question}" - Waiting for human
        """
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

        print(f"ACTIVE SESSION IN A_RUN {self.active_session=} {session_id=}")
        self.active_session = session_id
        self.is_running = True

        try:
            # Create execution engine
            engine = self._get_execution_engine(
                use_native_tools=use_native_tools,
                human_online=human_online,
                intermediate_callback=intermediate_callback
            )

            # Execute
            result = await engine.execute(
                query=query,
                session_id=session_id,
                execution_id=execution_id,
                remember=remember,
                with_context=kwargs.get('with_context', True),
                human_response=human_response,
                max_iterations=max_iterations,
                token_budget=token_budget,
                **kwargs
            )

            # Handle special states
            if result.paused:
                if result.needs_human:
                    return f"__NEEDS_HUMAN__:{result.execution_id}:{result.human_query}"
                return f"__PAUSED__:{result.execution_id}"

            return result.response

        except Exception as e:
            logger.error(f"a_run failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
        finally:
            self.is_running = False
            # self.active_session = None

    async def continue_execution(
        self,
        execution_id: str,
        human_response: str | None = None,
        **kwargs
    ) -> str:
        """
        Continue a paused execution.

        Args:
            execution_id: ID of paused execution
            human_response: Response from human (if was waiting)

        Returns:
            Response string
        """
        return await self.a_run(
            query="",  # Ignored for continuation
            execution_id=execution_id,
            human_response=human_response,
            **kwargs
        )

    async def pause_execution(self, execution_id: str) -> dict | None:
        """
        Pause a running execution.

        Returns:
            Execution state dict or None if not found
        """
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

        engine = self._get_execution_engine()
        state = await engine.pause(execution_id)
        return state.to_checkpoint() if state else None

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an execution and rollback changes.

        Returns:
            True if cancelled
        """
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

        engine = self._get_execution_engine()
        return await engine.cancel(execution_id)

    def list_executions(self) -> list[dict]:
        """List all active/paused executions."""
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

        engine = self._get_execution_engine()
        return engine.list_executions()

    # =========================================================================
    # CORE: a_stream - Voice-First Intelligent Streaming
    # =========================================================================

    async def a_stream(
        self,
        query: str,
        session_id: str = "default",
        remember: bool = True,
        language: str = "en",
        wait_for_hard: bool = False,
        force_mode: str | None = None,
        callback_on_complete: Callable[[str, str], None] | None = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Voice-first intelligent streaming.

        Auto-detects query complexity:
        - INSTANT: Simple questions → Stream directly
        - QUICK_TOOL: Single tool needed → Inline execution
        - HARD_TASK: Complex task → Background a_run

        Natural voice flow:
        - User: "What's the weather?"
        - Agent: "Let me check... It's 22°C and sunny."

        - User: "Create a Discord bot for moderation"
        - Agent: "Ok, I'll take care of that... [works] ... Done! Here's the result..."

        Args:
            query: User query
            session_id: Session identifier
            remember: Save to history
            language: Response language ("en", "de")
            wait_for_hard: Wait for hard tasks or return task_id immediately
            force_mode: Force complexity ("instant", "quick_tool", "hard_task", None=auto)
            callback_on_complete: Callback when background task completes
            **kwargs: Additional options

        Yields:
            str: Response chunks

        Special yields:
            "__TASK_STARTED__:{task_id}" - Background task started
            "__TASK_DONE__:{task_id}:{result}" - Task completed (via callback)

        Example:
            async for chunk in agent.a_stream("Analyze my data", language="de"):
                print(chunk, end="", flush=True)
        """
        from toolboxv2.mods.isaa.base.Agent.voice_stream import VoiceStreamEngine

        engine = VoiceStreamEngine(
            agent=self,
            language=language,
            callback_on_complete=callback_on_complete
        )

        async for chunk in engine.stream(
            query=query,
            session_id=session_id,
            remember=remember,
            force_mode=force_mode,
            wait_for_hard=wait_for_hard,
            **kwargs
        ):
            yield chunk

    async def a_stream_simple(
        self,
        query: str,
        session_id: str = "default",
        remember: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Simple streaming without capability detection (legacy mode)."""
        self.active_session = session_id
        self.is_running = True

        try:
            session = await self.session_manager.get_or_create(session_id)

            if remember:
                await session.add_message({"role": "user", "content": query})

            messages = session.get_history_for_llm(last_n=10)
            if asyncio.iscoroutine(messages):
                messages = await messages
            messages.append({"role": "user", "content": query})

            model = self.amd.fast_llm_model
            llm_kwargs = {
                'model': model,
                'messages': messages,
                'stream': True,
                'stream_options': {"include_usage": True}
            }

            if 'api_key' not in llm_kwargs:
                llm_kwargs['api_key'] = self._get_api_key_for_model(model)

            response = await self.llm_handler.completion_with_rate_limiting(litellm, **llm_kwargs)

            full_response = ""
            async for chunk in response:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                if content:
                    yield content

            if remember:
                await session.add_message({"role": "assistant", "content": full_response})

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"Error: {str(e)}"
        finally:
            self.is_running = False
            # self.active_session = None

    # =========================================================================
    # TOOL MANAGEMENT
    # =========================================================================

    async def add_tool(
        self,
        tool_func: Callable,
        name: str | None = None,
        description: str | None = None,
        category: list[str] | str | None = None,
        flags: dict[str, bool] | None = None
    ):
        """Register a tool."""
        self.tool_manager.register(
            func=tool_func,
            name=name,
            description=description,
            category=category,
            flags=flags
        )

    def get_tool(self, name: str) -> Callable | None:
        """Get tool function by name."""
        return self.tool_manager.get_function(name)

    # =========================================================================
    # CHECKPOINT
    # =========================================================================

    async def save(self) -> str:
        """Save checkpoint."""
        return await self.checkpoint_manager.save_current()

    async def restore(self, function_registry: dict[str, Callable] | None = None) -> dict:
        """Restore from checkpoint."""
        return await self.checkpoint_manager.auto_restore(function_registry)

    # =========================================================================
    # BINDING
    # =========================================================================

    async def bind(self, partner: 'FlowAgent', mode: str = 'public', session_id: str = 'default'):
        """Bind to another agent."""
        return await self.bind_manager.bind(partner, mode, session_id)

    def unbind(self, partner_name: str) -> bool:
        """Unbind from partner."""
        return self.bind_manager.unbind(partner_name)

    # =========================================================================
    # SERVERS
    # =========================================================================

    def setup_mcp_server(self, name: str | None = None):
        if not MCP_AVAILABLE:
            logger.warning("MCP not available")
            return

        server_name = name or f"{self.amd.name}_MCP"
        self.mcp_server = FastMCP(server_name)

        @self.mcp_server.tool()
        async def agent_run(query: str, session_id: str = "mcp_session") -> str:
            return await self.a_run(query, session_id=session_id)

    def setup_a2a_server(self, host: str = "0.0.0.0", port: int = 5000):
        if not A2A_AVAILABLE:
            logger.warning("A2A not available")
            return

        self.a2a_server = A2AServer(
            host=host, port=port,
            agent_card=AgentCard(name=self.amd.name, description="FlowAgent", version="2.0")
        )

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def close(self):
        """Clean shutdown."""
        self.is_running = False
        print("Saving checkpoint...")
        await self.save()
        await self.session_manager.close_all()
        self.executor.shutdown(wait=True)

        if self.a2a_server:
            await self.a2a_server.close()
        if self.mcp_server:
            await self.mcp_server.close()
        print("Checkpoint saved")
        logger.info(f"FlowAgent '{self.amd.name}' closed")

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def total_cost(self) -> float:
        return self.total_cost_accumulated

    def get_stats(self) -> dict:
        return {
            'agent_name': self.amd.name,
            'total_tokens_in': self.total_tokens_in,
            'total_tokens_out': self.total_tokens_out,
            'total_cost': self.total_cost_accumulated,
            'total_llm_calls': self.total_llm_calls,
            'sessions': self.session_manager.get_stats(),
            'tools': self.tool_manager.get_stats(),
            'bindings': self.bind_manager.get_stats(),
        }

    def __repr__(self) -> str:
        return f"<FlowAgent '{self.amd.name}' [{len(self.session_manager.sessions)} sessions]>"


    def __rshift__(self, other):
        return Chain(self) >> other

    def __add__(self, other):
        return Chain(self) + other

    def __and__(self, other):
        return Chain(self) & other

    def __mod__(self, other):
        """Implements % operator for conditional branching"""
        return ConditionalChain(self, other)
