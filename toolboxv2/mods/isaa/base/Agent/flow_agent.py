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
from pathlib import Path

import yaml
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Generator, Coroutine, Union

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
        self.last_result = None
        self.amd = amd
        self.verbose = verbose
        self.stream = stream
        self._rule_config_path = rule_config_path

        self.is_running = False
        self.active_session: str | None = None
        self.active_execution_id: str | None = None

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
        do_tool_execution: bool = False,
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
        session = None
        if session_id:
            session = self.session_manager.get(session_id)
            if session:
                await session.initialize()
                system_msg += "\n\n"+  session.build_vfs_context()
        if with_context:
            if session:
                sysmsg = [{"role": "system", "content": f"{system_msg}"}]
                full_history = session.get_history(kwargs.get("history_size", 6))
                current_msg = llm_kwargs['messages']
                for msg in full_history:

                    if not current_msg:
                        break

                    if msg['role'] != 'user':
                        continue

                    content = msg['content']

                    if current_msg[0]['role'] == 'user' and current_msg[0]['content'] == content:
                        current_msg = current_msg[1:]
                        break

                    if len(current_msg) > 1 and current_msg[-1]['role'] == 'user' and current_msg[-1]['content'] == content:
                        current_msg = current_msg[:-1]
                        break

                llm_kwargs['messages'] = sysmsg + full_history + current_msg
            else:
                llm_kwargs['messages'] = [{"role": "system", "content": f"{system_msg}"}] + llm_kwargs['messages']

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

            if do_tool_execution and 'tools' in llm_kwargs:
                tool_response = await self.run_tool_response(result if get_response_message else response.choices[0].message, session_id)
                llm_kwargs['messages'] += [{"role": "assistant", "content":result.content if get_response_message else result}]+tool_response
                del kwargs['tools']
                return await self.a_run_llm_completion(llm_kwargs['messages'], model_preference, with_context, stream, get_response_message, task_id, session_id, **kwargs)

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

    async def run_tool_response(self, response, session_id):

        tool_calls = response.tool_calls
        session = None
        if session_id:
            session = self.session_manager.get(session_id)
        all_results = []
        for tc in tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments or "{}")
            try:
                result = await self.arun_function(tool_name, **tool_args)
            except Exception as e:
                result = f"Error: {str(e)}"
            tool_response = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            }
            all_results.append(tool_response)
            if session:
                await session.add_message(tool_response)
        return all_results

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
        max_tokens: int | None = None,
        **kwargs
    ) -> dict[str, Any]:
        schema = pydantic_model.model_json_schema()
        model_name = pydantic_model.__name__

        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        fields_desc = [f"  {name}{'*' if name in required else ''}: {info.get('type', 'string')}" for name, info in props.items()]

        enhanced_prompt = f"{prompt}"

        try:
            from litellm import supports_response_schema

            for mp in [model_preference, "complex" if model_preference == "fast" else "fast"]:
                data = await self.a_run_llm_completion(
                    messages=[{"role": "user", "content": enhanced_prompt}], model_preference=mp, stream=False,
                    with_context=auto_context,
                    max_tokens=max_tokens, task_id=f"format_{model_name.lower()}", response_format=pydantic_model
                )
                if isinstance(data, str):
                    data = json.loads(data)
                validated = pydantic_model.model_validate(data)
                return validated.model_dump()


        except ImportError as e:
            logger.error(f"LLM call failed: {e}")
            print("LLM call failed:", e, "falling back to YAML")


        messages = (message_context or []) + [{"role": "system", "content": "You are a YAML formatter. format the input to valid YAML."}, {"role": "user", "content": enhanced_prompt} , {"role": "system", "content": "Return YAML with fields:\n" + "\n".join(fields_desc)}]

        for attempt in range(max_retries + 1):
            try:
                response = await self.a_run_llm_completion(
                    messages=messages, model_preference=model_preference, stream=False,
                    with_context=auto_context, temperature=0.1 + (attempt * 0.1),
                    max_tokens=max_tokens, task_id=f"format_{model_name.lower()}_{attempt}"
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

        if not session_id:
            session_id = "default"
        if session_id == "default" and self.active_session is not None:
            session_id = self.active_session

        self.active_session = session_id
        self.is_running = True
        if execution_id is None and self.active_execution_id is not None:
            execution_id = self.active_execution_id
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
                human_response=human_response,
                max_iterations=max_iterations,
                token_budget=token_budget,
                **kwargs
            )

            # Handle special states
            if result.paused:
                self.active_execution_id = result.execution_id
                if result.needs_human:
                    return f"__NEEDS_HUMAN__:{result.human_query}"
                return f"__PAUSED__"
            self.active_execution_id = None

            response = result.response
            # Ensure response is a string (a_run can return various types)
            if response is None:
                response = ""
            elif not isinstance(response, str):
                # Handle Message objects, dicts, or other types
                if hasattr(response, 'content'):
                    response = str(response.content)
                elif hasattr(response, 'text'):
                    response = str(response.text)
                else:
                    response = str(response)
            self.last_result = result
            return response

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
        Main entry point for streaming agent execution.

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

        if not session_id:
            session_id = "default"
        if session_id == "default" and self.active_session is not None:
            session_id = self.active_session

        self.active_session = session_id
        self.is_running = True
        if execution_id is None and self.active_execution_id is not None:
            execution_id = self.active_execution_id

        try:
            # Create execution engine
            engine = self._get_execution_engine(
                use_native_tools=use_native_tools,
                human_online=human_online,
                intermediate_callback=intermediate_callback
            )

            # Execute
            stream_func, state = await engine.execute(
                query=query,
                session_id=session_id,
                execution_id=execution_id,
                human_response=human_response,
                max_iterations=max_iterations,
                token_budget=token_budget,
                do_stream=True,
                **kwargs
            )

            async for result in stream_func(state):
                if hasattr(result, 'paused'):
                    if result.paused:
                        self.active_execution_id = result.execution_id
                        yield result.human_query if result.needs_human else "I am Paused"
                        break
                    elif result.success:
                        self.active_execution_id = None
                        yield result.response
                        break
                    elif not result.success:
                        self.active_execution_id = None
                        yield result.response
                        break
                else:
                    yield result

        except Exception as e:
            logger.error(f"a_run failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"Error: {str(e)}"
        finally:
            self.is_running = False
            # self.active_session = None


    # =========================================================================
    # audio processing
    # =========================================================================

    async def a_stream_audio(
        self,
        audio_chunks: Generator[bytes, None, None],
        session_id: str = "default",
        language: str = "en",
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Process a stream of audio chunks through the agent.

        Use this for real-time audio processing where you want
        to yield audio output as soon as possible.

        Args:
            audio_chunks: Generator yielding audio byte chunks
            session_id: Session identifier
            language: Response language ("en", "de")
            **kwargs: Additional options

        Yields:
            Audio bytes chunks for immediate playback
        """
        from toolboxv2.mods.isaa.base.audio_io.audioIo import process_audio_stream

        self.active_session = session_id
        async for chunk in process_audio_stream(
            audio_chunks, self.a_stream, language=language, **kwargs
        ):
            yield chunk

    async def a_audio(
        self,
        audio: Union[bytes, Path, str],
        session_id: str = "default",
        language: str = "en",
        **kwargs
    ) -> tuple[bytes | None, str, list, dict]:
        """
        Process a complete audio file/buffer through the agent.

        This function handles the full pipeline:
        1. Audio input (file, bytes, or path)
        2. Understanding (STT or native audio model)
        3. Processing (your agent logic via processor callback)
        4. Response generation (TTS or native audio model)

        Args:
            audio: Audio input (bytes, file path, or Path object)
            session_id: Session identifier
            language: Response language ("en", "de")
            **kwargs: Additional options

        Returns:
            Audio bytes for playback
        """
        from toolboxv2.mods.isaa.base.audio_io.audioIo import process_audio_raw
        self.active_session = session_id
        result = await process_audio_raw(audio, self.a_run, language=language, **kwargs)
        # text_input = result.text_input
        text_output = result.text_output
        audio_output = result.audio_output
        tool_calls = result.tool_calls
        metadata = result.metadata

        return audio_output, text_output, tool_calls, metadata

    @staticmethod
    async def tts(text: str, language: str = "en", **kwargs) -> 'TTSResult':
        from toolboxv2.mods.isaa.base.audio_io.Tts import synthesize, TTSResult
        return synthesize(text, language=language, **kwargs)

    @staticmethod
    async def stt(audio: Union[bytes, Path, str], language: str = "en", **kwargs) -> 'STTResult':
        from toolboxv2.mods.isaa.base.audio_io.Stt import transcribe, STTResult
        return transcribe(audio, language=language, **kwargs)


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
    # SESSION TOOLS INITIALIZATION
    # =========================================================================

    def clear_session_history(self, session_id: str = None):
        session_id = session_id or self.active_session
        _session = self.session_manager.get(session_id)
        if _session:
            _session.clear_history()

    async def init_session_tools(self, session_id: str = None):
        """
        Initializes and registers standard session-aware tools (VFS, Memory, Situation).
        These tools allow the LLM to interact with the active AgentSession.
        """
        session_id = session_id or self.active_session
        _session = await self.session_manager.get_or_create(session_id) if isinstance(session_id, str) else session_id
        session_id = _session.session_id
        if not self.session_manager.exists(session_id):
            raise ValueError(f"Session '{session_id}' not found. create it using the session_manager get_or_create")
        if _session.tools_initialized:
            return
        # --- Helper to safely get session ---
        def get_session():
            session = self.session_manager.get(session_id)
            if not session:
                raise ValueError(f"Session '{self.active_session}' not found.")
            return session

        # ==========================
        # 1. Virtual File System (VFS)
        # ==========================

        async def vfs_list_files():
            """List all files in the virtual file system with their status and size."""
            return get_session().vfs_list()

        async def vfs_read_file(filename: str):
            """Read the full content of a file from the VFS."""
            return get_session().vfs_read(filename)

        async def vfs_create_file(filename: str, content: str):
            """Create a new file in the VFS with the given content."""
            return get_session().vfs_create(filename, content)

        async def vfs_write_file(filename: str, content: str):
            """Overwrite an existing file completely with new content."""
            return get_session().vfs_write(filename, content)

        async def vfs_edit_file(filename: str, line_start: int, line_end: int, new_content: str):
            """
            Edit specific lines in a file. Replaces lines from line_start to line_end with new_content.
            Use this for precise edits to large files to save tokens.
            """
            return get_session().vfs.edit(filename, line_start, line_end, new_content)

        async def vfs_append_file(filename: str, content: str):
            """Append text to the end of a file."""
            return get_session().vfs.append(filename, content)

        async def vfs_open_file(filename: str, line_start: int = 1, line_end: int = -1):
            """
            'Open' a file to make it visible in your context window.
            Optionally specify line range.
            Only open files are "seen" by the agent automatically.
            """
            return get_session().vfs_open(filename, line_start, line_end)

        async def vfs_close_file(filename: str):
            """
            'Close' a file to remove it from context window and save tokens.
            Automatically generates a summary of the closed file.
            """
            return await get_session().vfs_close(filename)

        # Register VFS Tools
        vfs_category = ["system", "vfs"]
        vfs_flags = {"virtual": True, "context_modifying": True}

        await self.add_tool(vfs_list_files, name="vfs_list", description="List all VFS files",
                            category=vfs_category, flags=vfs_flags)
        await self.add_tool(vfs_read_file, name="vfs_read", description="Read file content",
                            category=vfs_category, flags=vfs_flags)
        await self.add_tool(vfs_create_file, name="vfs_create", description="Create new file",
                            category=vfs_category, flags=vfs_flags)
        await self.add_tool(vfs_write_file, name="vfs_write", description="Overwrite file", category=vfs_category,
                            flags=vfs_flags)
        await self.add_tool(vfs_edit_file, name="vfs_edit", description="Edit specific file lines",
                            category=vfs_category, flags=vfs_flags)
        await self.add_tool(vfs_append_file, name="vfs_append", description="Append to file",
                            category=vfs_category, flags=vfs_flags)
        await self.add_tool(vfs_open_file, name="vfs_open", description="Add file to context",
                            category=vfs_category, flags=vfs_flags)
        await self.add_tool(vfs_close_file, name="vfs_close", description="Remove file from context",
                            category=vfs_category, flags=vfs_flags)

        # ==========================
        # 2. Memory & RAG
        # ==========================

        async def memory_recall(query: str, concepts: bool = False):
            """
            Search long-term memory/knowledge base for relevant information.
            Use this when you need context not present in the current conversation.
            """
            return await get_session().get_reference(query, concepts=concepts)

        async def memory_history(last_n: int = 10):
            """Retrieve the last N messages of the conversation history."""
            return get_session().get_history(last_n)

        mem_category = ["system", "memory"]

        await self.add_tool(memory_recall, name="memory_recall",
                            description="Search knowledge base/long-term memory", category=mem_category,
                            flags={"readonly": True})
        await self.add_tool(memory_history, name="memory_history", description="Get recent chat history",
                            category=mem_category, flags={"readonly": True})

        # ==========================
        # 3. Situation & Behavior
        # ==========================

        async def set_agent_situation(situation: str, intent: str):
            """
            Update the current situation and intent.
            This changes which rules apply to the agent's behavior.
            """
            get_session().set_situation(situation, intent)
            return f"Situation set to: {situation} | Intent: {intent}"

        async def check_permissions(action: str):
            """Check if a specific action is allowed under current rules."""
            result = get_session().rule_on_action(action)
            return f"Action '{action}' allowed: {result.allowed}. Reason: {result.reason}"

        meta_category = ["system", "meta"]

        await self.add_tool(set_agent_situation, name="situation_set",
                            description="Update current agent situation/context", category=meta_category,
                            flags={"affects_rules": True})
        await self.add_tool(check_permissions, name="check_rules", description="Check if action is allowed",
                            category=meta_category, flags={"readonly": True})

        logger.info(f"Initialized standard session tools for agent '{self.amd.name}'")

        _session.tools_initialized = True

    # =========================================================================
    # CONTEXT AWARENESS & ANALYTICS
    # =========================================================================

    async def context_overview(self, session_id: str | None = None, print_visual: bool = True) -> dict:
        """
        Analysiert den aktuellen Token-Verbrauch des Kontexts und gibt eine Übersicht zurück.

        Args:
            session_id: Die zu analysierende Session (oder None für generische Analyse)
            print_visual: Ob eine grafische CLI-Anzeige ausgegeben werden soll

        Returns:
            Ein Dictionary mit den detaillierten Token-Metriken.
        """
        if not LITELLM_AVAILABLE:
            logger.warning("LiteLLM not available, cannot count tokens.")
            return {}

        # 1. Setup & Defaults
        target_session = session_id or self.active_session or "default"
        model = self.amd.fast_llm_model.split("/")[-1]  # Wir nutzen das schnelle Modell für die Tokenizer-Logik

        # Holen der Context Window Size (Fallback auf 128k wenn unbekannt)
        try:
            model_info = litellm.get_model_info(model)
            context_limit = model_info.get("max_input_tokens") or model_info.get("max_tokens") or 128000
        except Exception:
            context_limit = 128000

        metrics = {
            "system_prompt": 0,
            "tool_definitions": 0,
            "vfs_context": 0,
            "history": 0,
            "overhead": 0,
            "total": 0,
            "limit": context_limit,
            "session_id": target_session if session_id else "NONE (Base Config)"
        }

        # 2. System Prompt Berechnung
        # Wir simulieren den Prompt, den die Engine bauen würde
        base_system_msg = self.amd.get_system_message()
        # Hinweis: ExecutionEngine fügt oft noch spezifische Prompts hinzu (Immediate/React)
        # Wir nehmen hier eine repräsentative Größe an.
        from toolboxv2.mods.isaa.base.Agent.execution_engine import IMMEDIATE_RESPONSE_SYSTEM_PROMPT
        full_sys_msg = f"{base_system_msg}\n\n{IMMEDIATE_RESPONSE_SYSTEM_PROMPT}"
        metrics["system_prompt"] = litellm.token_counter(model=model, text=full_sys_msg)

        # 3. Tools Definitions Berechnung
        # Wir sammeln alle Tools + Standard VFS Tools um die Definition-Größe zu berechnen
        from toolboxv2.mods.isaa.base.Agent.execution_engine import VFS_TOOLS_LITELLM, CONTROL_TOOLS_LITELLM

        user_tools = self.tool_manager.get_all_litellm()
        # System Tools die immer injected werden
        all_tools = user_tools + VFS_TOOLS_LITELLM + CONTROL_TOOLS_LITELLM

        # LiteLLM Token Counter kann Tools nicht direkt, wir dumpen das JSON als Näherungswert
        # (Dies ist oft genauer als man denkt, da Definitionen als Text/JSON injected werden)
        tools_json = json.dumps(all_tools)
        metrics["tool_definitions"] = litellm.token_counter(model=model, text=tools_json)

        # 4. Session Specific Data (VFS & History)
        if session_id:
            session = await self.session_manager.get_or_create(target_session)

            # VFS Context
            # Wir rufen build_context_string auf, um genau zu sehen, was das LLM sieht
            vfs_str = session.build_vfs_context()
            # Plus Auto-Focus (Letzte Änderungen)
            if self._execution_engine:  # Falls Engine instanziiert, holen wir AutoFocus
                # Wir müssen hier tricksen, da AutoFocus in der Engine Instanz liegt
                # und private ist. Wir nehmen an, dass es leer ist oder klein,
                # oder wir instanziieren eine temporäre Engine.
                # Für Performance nehmen wir hier nur den VFS String.
                pass

            metrics["vfs_context"] = litellm.token_counter(model=model, text=vfs_str)

            # Chat History
            # Wir nehmen an, dass standardmäßig ca. 10-15 Nachrichten gesendet werden
            history = session.get_history_for_llm(last_n=15)
            metrics["history"] = litellm.token_counter(model=model, messages=history)

        # 5. Summe
        # Puffer für Protokoll-Overhead (Role-Tags, JSON-Formatierung) ~50 Tokens
        metrics["overhead"] = 50
        metrics["total"] = sum(
            [v for k, v in metrics.items() if isinstance(v, (int, float)) and k not in ["limit", "total"]])

        # 6. Visualisierung
        if print_visual:
            self._print_context_visual(metrics, model)

        return metrics

    def _print_context_visual(self, metrics: dict, model_name: str):
        """Helper für die CLI Visualisierung"""
        total = metrics["total"]
        limit = metrics["limit"]
        percent = min(100, (total / limit) * 100)

        # Farben (ANSI)
        C_RESET = "\033[0m"
        C_BOLD = "\033[1m"
        C_GREEN = "\033[32m"
        C_YELLOW = "\033[33m"
        C_RED = "\033[31m"
        C_BLUE = "\033[34m"
        C_GRAY = "\033[90m"

        # Farbe basierend auf Auslastung
        bar_color = C_GREEN
        if percent > 70: bar_color = C_YELLOW
        if percent > 90: bar_color = C_RED

        # Progress Bar bauen (Breite 30 Zeichen)
        bar_width = 30
        filled = int((percent / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        print(f"\n{C_BOLD}CONTEXT OVERVIEW{C_RESET} | Session: {C_BLUE}{metrics['session_id']}{C_RESET}")
        print(f"{C_GRAY}Model: {model_name} | Limit: {limit:,} tokens{C_RESET}\n")

        print(f"Usage:")
        print(f"{bar_color}[{bar}]{C_RESET} {C_BOLD}{percent:.1f}%{C_RESET} ({total:,} / {limit:,})")

        print(f"\n{C_BOLD}Breakdown:{C_RESET}")

        def print_row(label, value, color=C_RESET):
            pct = (value / total * 100) if total > 0 else 0
            print(f" • {label:<18} {color}{value:>6,}{C_RESET} tokens {C_GRAY}({pct:>4.1f}%){C_RESET}")

        print_row("System Prompts", metrics["system_prompt"], C_YELLOW)
        print_row("Tools (Defs)", metrics["tool_definitions"], C_BLUE)
        if metrics["vfs_context"] > 0:
            print_row("VFS / Files", metrics["vfs_context"], C_GREEN)
        if metrics["history"] > 0:
            print_row("Chat History", metrics["history"], C_BLUE)

        # Leerer Platz Berechnung
        remaining = limit - total
        print("-" * 40)
        print(f" {C_BOLD}{'TOTAL':<18} {total:>6,}{C_RESET}")
        print(f" {C_GRAY}{'Remaining':<18} {remaining:>6,}{C_RESET}")
        print("")

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
