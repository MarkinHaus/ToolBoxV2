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
        self.active_execution_id: str | None = None

        # Statistics
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_accumulated = 0.0
        self.total_llm_calls = 0

        self.executor = ThreadPoolExecutor(max_workers=max_parallel_tasks)

        # Servers
        self.a2a_server: A2AServer | None = None
        self.mcp_server: FastMCP | None = None

        # Execution engine instance (lazy loaded)
        self._execution_engine_cache = None

        self._init_managers(auto_load_checkpoint)
        self._init_rate_limiter()

        logger.info(f"FlowAgent '{amd.name}' initialized")

    def _init_managers(self, auto_load_checkpoint: bool):
        from toolboxv2.mods.isaa.base.Agent.session_manager import SessionManager
        from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolManager
        from toolboxv2.mods.isaa.base.Agent.checkpoint_manager import CheckpointManager
        from toolboxv2.mods.isaa.base.Agent.bind_manager import BindManager
        from toolboxv2.mods.isaa.base.Agent.docker_vfs import DockerConfig

        self.session_manager = SessionManager(
            agent_name=self.amd.name,
            default_max_history=100,
            vfs_max_window_lines=self.amd.vfs_max_window_lines,
            rule_config_path=self._rule_config_path,
            summarizer=self._create_summarizer(),

            enable_lsp = self.amd.enable_lsp,
            enable_docker = self.amd.enable_docker,
            docker_config = self.amd.docker_config or DockerConfig(
                memory_limit="4g",
                timeout_seconds=600
            ),
            toolboxv2_wheel_path = os.getenv("TOOLBV2_WHEEL_PATH", "C:/Users/Markin/Workspace/ToolBoxV2/dist/toolboxv2-0.1.24-py2.py3-none-any.whl")
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
            human_online=kwargs.get('human_online', False),
            callback=kwargs.get('intermediate_callback')
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
                if 'true_stream' in llm_kwargs and llm_kwargs['true_stream']:
                    return response
                result, usage = await self._process_streaming_response(response, task_id, model, get_response_message)
            else:
                result = response.choices[0].message.content
                usage = response.usage
                if get_response_message:
                    result = response.choices[0].message

            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            cost = self.calculate_llm_cost(model, input_tokens, output_tokens, response)

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


    def calculate_llm_cost(self, model: str, input_tokens: int, output_tokens: int,completion_response:Any=None) -> float:
        """Calculate approximate LLM cost"""
        cost = (input_tokens / 1000) * 0.002 + (output_tokens / 1000) * 0.01
        if hasattr(completion_response, "_hidden_params"):
            cost = completion_response._hidden_params.get("response_cost", 0)
        try:
            import litellm
            cost = litellm.completion_cost(model=model, completion_response=completion_response)
        except ImportError:
            pass
        except Exception as e:
            try:
                import litellm
                cost = litellm.completion_cost(model=model.split('/')[-1], completion_response=completion_response)
            except Exception:
                pass
        return cost or (input_tokens / 1000) * 0.002 + (output_tokens / 1000) * 0.01


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
    def _get_execution_engine(
        self,
        use_native_tools: bool = True,
        human_online: bool = False,
        intermediate_callback: Callable[[str], None] | None = None
    ) -> 'ExecutionEngine':
        """
        Get or create ExecutionEngine instance.

        Caches the engine for reuse within the same agent instance.
        Creates a new engine only if parameters change significantly.

        Args:
            use_native_tools: Use LiteLLM native tool calling
            human_online: Allow human-in-the-loop
            intermediate_callback: Callback for status messages

        Returns:
            ExecutionEngine instance
        """
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

        # Check if we can reuse cached engine
        if self._execution_engine_cache is not None:
            # Update dynamic parameters
            self._execution_engine_cache.human_online = human_online
            self._execution_engine_cache.callback = intermediate_callback
            return self._execution_engine_cache

        # Create new engine
        engine = ExecutionEngine(
            agent=self,
            human_online=human_online,
            callback=intermediate_callback,
            is_sub_agent=False
        )

        self._execution_engine_cache = engine
        return engine

    # =========================================================================
    # MAIN ENTRY POINT (your existing a_run should work with this)
    # =========================================================================

    async def a_run(
        self,
        query: str,
        session_id: str = "default",
        execution_id: str | None = None,
        use_native_tools: bool = True,
        human_online: bool = False,
        intermediate_callback: Callable[[str], None] | None = None,
        max_iterations: int = 15,
        get_ctx: bool = False,
        **kwargs
    ) -> str | tuple[str, Any]:
        """
        Main entry point for agent execution.

        Args:
            query: User query
            session_id: Session identifier
            execution_id: For continuing paused execution
            use_native_tools: LiteLLM native tool calling
            human_online: Allow human-in-the-loop
            intermediate_callback: User-facing status messages
            human_response: Response from human (for continuation)
            max_iterations: Max ReAct iterations (default 15)
            token_budget: Token budget per iteration
            get_ctx: Return (result, ExecutionContext) tuple
            **kwargs: Additional options

        Returns:
            Response string, or (response, ctx) if get_ctx=True
        """
        # Handle defaults
        if not session_id:
            session_id = "default"
        if session_id == "default" and getattr(self, 'active_session', None):
            session_id = self.active_session

        self.active_session = session_id
        self.is_running = True

        # Check for resume
        ctx = None
        if execution_id:
            engine = self._get_execution_engine(
                use_native_tools=use_native_tools,
                human_online=human_online,
                intermediate_callback=intermediate_callback
            )
            ctx = engine.get_execution(execution_id)

        try:
            # Get or create execution engine
            engine = self._get_execution_engine(
                use_native_tools=use_native_tools,
                human_online=human_online,
                intermediate_callback=intermediate_callback
            )

            # Execute
            result = await engine.execute(
                query=query,
                session_id=session_id,
                max_iterations=max_iterations,
                ctx=ctx,
                get_ctx=get_ctx
            )

            return result

        except Exception as e:
            import logging
            import traceback
            logging.error(f"a_run failed: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"
        finally:
            self.is_running = False

    # =========================================================================
    # PAUSE / CANCEL / LIST (these should work as-is)
    # =========================================================================

    async def pause_execution(self, execution_id: str) -> dict | None:
        """
        Pause a running execution.

        Args:
            execution_id: The run_id to pause

        Returns:
            Checkpoint dict or None
        """
        engine = self._get_execution_engine()
        ctx = await engine.pause(execution_id)
        return ctx.to_checkpoint() if ctx else None

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an execution.

        Args:
            execution_id: The run_id to cancel

        Returns:
            True if cancelled
        """
        engine = self._get_execution_engine()
        return await engine.cancel(execution_id)

    def list_executions(self) -> list[dict]:
        """
        List all active/paused executions.

        Returns:
            List of execution summaries
        """
        engine = self._get_execution_engine()
        return engine.list_executions()

    async def resume_execution(
        self,
        execution_id: str,
        max_iterations: int = 15
    ) -> str:
        """
        Resume a paused execution.

        Args:
            execution_id: The run_id to resume
            max_iterations: Max additional iterations

        Returns:
            Final response
        """
        engine = self._get_execution_engine()
        return await engine.resume(execution_id, max_iterations)

    def get_execution_state(self, execution_id: str) -> dict | None:
        """
        Get current state of an execution.

        Args:
            execution_id: The run_id

        Returns:
            Checkpoint dict or None
        """
        engine = self._get_execution_engine()
        ctx = engine.get_execution(execution_id)
        return ctx.to_checkpoint() if ctx else None

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
        max_iterations: int = 15,
        **kwargs
    ):
        """
        Streaming execution - yields chunks during execution.

        Yields:
            dict with 'type' key:
            - {"type": "content", "chunk": "..."}
            - {"type": "tool_start", "name": "..."}
            - {"type": "tool_result", "name": "...", "result": "..."}
            - {"type": "final_answer", "answer": "..."}
            - {"type": "paused", "run_id": "..."}
            - {"type": "done", "success": bool, "final_answer": "..."}
        """
        if not session_id:
            session_id = "default"
        if session_id == "default" and self.active_session is not None:
            session_id = self.active_session

        self.active_session = session_id
        self.is_running = True

        # Check for resume
        ctx = None
        if execution_id:
            engine = self._get_execution_engine(
                use_native_tools=use_native_tools,
                human_online=human_online,
                intermediate_callback=intermediate_callback
            )
            ctx = engine.get_execution(execution_id)

        try:
            engine = self._get_execution_engine(
                use_native_tools=use_native_tools,
                human_online=human_online,
                intermediate_callback=intermediate_callback
            )

            # Get stream generator
            stream_func, ctx = await engine.execute_stream(
                query=query,
                session_id=session_id,
                max_iterations=max_iterations,
                ctx=ctx
            )

            # Yield all chunks
            async for chunk in stream_func(ctx):
                yield chunk

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"type": "error", "error": str(e)}
        finally:
            self.is_running = False

    async def a_stream_verbose(
        self,
        query: str,
        session_id: str = "default",
        execution_id: str | None = None,
        use_native_tools: bool = True,
        human_online: bool = False,
        intermediate_callback: Callable[[str], None] | None = None,
        max_iterations: int = 15,
        **kwargs
    ):
        """
        Vereinfachtes Streaming - gibt nur Strings zurück.

        Yields:
            str: Lesbarer Text für jeden Schritt
        """
        async for chunk in self.a_stream(
            query=query,
            session_id=session_id,
            execution_id=execution_id,
            use_native_tools=use_native_tools,
            human_online=human_online,
            intermediate_callback=intermediate_callback,
            max_iterations=max_iterations,
            **kwargs
        ):
            chunk_type = chunk.get("type", "")

            if chunk_type == "content":
                yield chunk["chunk"]

            elif chunk_type == "tool_start":
                yield f"\n🔧 Nutze Tool: {chunk['name']}\n"

            elif chunk_type == "tool_result":
                result = chunk.get("result", "")
                if len(result) > 200:
                    result = result[:200] + "..."
                yield f"   ✓ Ergebnis: {result}\n"

            elif chunk_type == "final_answer":
                yield f"\n\n📝 {chunk['answer']}"

            elif chunk_type == "paused":
                yield f"\n⏸️ Pausiert (ID: {chunk['run_id']})\n"

            elif chunk_type == "max_iterations":
                yield f"\n⚠️ Max Iterationen erreicht\n{chunk['answer']}"

            elif chunk_type == "error":
                yield f"\n❌ Fehler: {chunk['error']}\n"

            elif chunk_type == "done":
                status = "✅" if chunk.get("success") else "⚠️"
                yield f"\n{status} Fertig\n"


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
            audio_chunks, self.a_stream_verbose, language=language, **kwargs
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

    def add_tool(
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

    def init_session_tools(self, session: 'AgentSession'):
        """
        Initialize session-specific tools for VFS V2, Docker, and filesystem operations.

        Tools are categorized:
        - vfs: Virtual File System operations
        - docker: Container execution (flag: requires_docker)
        - filesystem: Real filesystem copy operations (flag: filesystem_access)
        - memory: RAG and history
        - situation: Behavior control
        """

        # =========================================================================
        # VFS TOOLS (V2)
        # =========================================================================

        # --- File Operations ---

        def vfs_list(path: str = "/", recursive: bool = False) -> dict:
            """
            List directory contents in VFS.

            Args:
                path: Directory path to list (default: root)
                recursive: If True, list recursively

            Returns:
                Dict with contents list including files and directories
            """
            return session.vfs_ls(path, recursive)

        def vfs_read(path: str) -> dict:
            """
            Read file content from VFS.

            Args:
                path: Path to file (e.g., "/src/main.py")

            Returns:
                Dict with file content
            """
            return session.vfs_read(path)

        def vfs_create(path: str, content: str = "") -> dict:
            """
            Create a new file in VFS.

            Args:
                path: Path for new file (e.g., "/src/utils.py")
                content: Initial file content

            Returns:
                Dict with success status and file type info
            """
            return session.vfs_create(path, content)

        def vfs_write(path: str, content: str) -> dict:
            """
            Write/overwrite file content in VFS.

            Args:
                path: Path to file
                content: New content

            Returns:
                Dict with success status
            """
            return session.vfs_write(path, content)

        def vfs_edit(path: str, line_start: int, line_end: int, new_content: str) -> dict:
            """
            Edit file by replacing lines (1-indexed).

            Args:
                path: Path to file
                line_start: First line to replace (1-indexed)
                line_end: Last line to replace (inclusive)
                new_content: New content for those lines

            Returns:
                Dict with success status
            """
            return session.vfs.edit(path, line_start, line_end, new_content)

        def vfs_append(path: str, content: str) -> dict:
            """
            Append content to a file.

            Args:
                path: Path to file
                content: Content to append

            Returns:
                Dict with success status
            """
            return session.vfs.append(path, content)

        def vfs_delete(path: str) -> dict:
            """
            Delete a file from VFS.

            Args:
                path: Path to file

            Returns:
                Dict with success status
            """
            return session.vfs.delete(path)

        # --- Directory Operations ---

        def vfs_mkdir(path: str, parents: bool = True) -> dict:
            """
            Create a directory in VFS.

            Args:
                path: Directory path (e.g., "/src/components")
                parents: If True, create parent directories as needed

            Returns:
                Dict with success status
            """
            return session.vfs_mkdir(path, parents)

        def vfs_rmdir(path: str, force: bool = False) -> dict:
            """
            Remove a directory from VFS.

            Args:
                path: Directory path
                force: If True, remove non-empty directories recursively

            Returns:
                Dict with success status
            """
            return session.vfs_rmdir(path, force)

        def vfs_mv(source: str, destination: str) -> dict:
            """
            Move/rename a file or directory.

            Args:
                source: Source path
                destination: Destination path

            Returns:
                Dict with success status
            """
            return session.vfs_mv(source, destination)

        # --- Open/Close Operations ---

        def vfs_open(path: str, line_start: int = 1, line_end: int = -1) -> dict:
            """
            Open a file (make visible in LLM context).

            Args:
                path: Path to file
                line_start: First line to show (1-indexed)
                line_end: Last line to show (-1 = all)

            Returns:
                Dict with preview of content
            """
            return session.vfs_open(path, line_start, line_end)

        async def vfs_close(path: str) -> dict:
            """
            Close a file (remove from context, generate summary).

            Args:
                path: Path to file

            Returns:
                Dict with generated summary
            """
            return await session.vfs_close(path)

        def vfs_view(path: str, line_start: int = 1, line_end: int = -1) -> dict:
            """
            View/adjust visible window of an open file.

            Args:
                path: Path to file
                line_start: First line to show
                line_end: Last line to show

            Returns:
                Dict with visible content
            """
            return session.vfs.view(path, line_start, line_end)

        # --- Info & Diagnostics ---

        def vfs_info(path: str) -> dict:
            """
            Get detailed info about a file or directory.

            Args:
                path: Path to file or directory

            Returns:
                Dict with metadata (type, size, lines, file_type, lsp_enabled, etc.)
            """
            return session.vfs.get_file_info(path)

        async def vfs_diagnostics(path: str) -> dict:
            """
            Get LSP diagnostics (errors, warnings, hints) for a code file.

            Args:
                path: Path to code file

            Returns:
                Dict with diagnostics list, error/warning/hint counts
            """
            return await session.vfs_diagnostics(path)

        def vfs_executables() -> list[dict]:
            """
            Get list of all executable files in VFS.

            Returns:
                List of executable files with path, language, size
            """
            return session.vfs.get_executable_files()

        # =========================================================================
        # FILESYSTEM COPY TOOLS (Flag: filesystem_access)
        # =========================================================================

        def fs_copy_to_vfs(
            local_path: str,
            vfs_path: str | None = None,
            allowed_dirs: list[str] | None = None,
            max_size_bytes: int = 1024 * 1024
        ) -> dict:
            """
            Copy a file from real filesystem into VFS.

            Args:
                local_path: Path on real filesystem
                vfs_path: Destination path in VFS (default: /<filename>)
                allowed_dirs: List of allowed directories for security
                max_size_bytes: Maximum file size (default: 1MB)

            Returns:
                Dict with success status, vfs_path, size, lines, file_type

            Security:
                Requires filesystem_access flag.
                Only reads from allowed_dirs if specified.
            """
            return session.vfs.load_from_local(
                local_path=local_path,
                vfs_path=vfs_path,
                allowed_dirs=allowed_dirs,
                max_size_bytes=max_size_bytes
            )

        def fs_copy_from_vfs(
            vfs_path: str,
            local_path: str,
            allowed_dirs: list[str] | None = None,
            overwrite: bool = False,
            create_dirs: bool = True
        ) -> dict:
            """
            Copy a file from VFS to real filesystem.

            Args:
                vfs_path: Path in VFS
                local_path: Destination path on real filesystem
                allowed_dirs: List of allowed directories for security
                overwrite: Allow overwriting existing files
                create_dirs: Create parent directories if needed

            Returns:
                Dict with success status, saved_path, size, lines

            Security:
                Requires filesystem_access flag.
                Only writes to allowed_dirs if specified.
            """
            return session.vfs.save_to_local(
                vfs_path=vfs_path,
                local_path=local_path,
                allowed_dirs=allowed_dirs,
                overwrite=overwrite,
                create_dirs=create_dirs
            )

        def fs_copy_folder_to_vfs(
            local_path: str,
            vfs_path: str = "/",
            allowed_dirs: list[str] | None = None,
            max_size_bytes: int = 1024 * 1024,
            max_files: int = 100,
            include_patterns: list[str] | None = None,
            exclude_patterns: list[str] | None = None
        ) -> dict:
            """
            Copy a folder from real filesystem into VFS recursively.

            Args:
                local_path: Path to folder on real filesystem
                vfs_path: Destination path in VFS (default: root)
                allowed_dirs: List of allowed directories for security
                max_size_bytes: Maximum size per file (default: 1MB)
                max_files: Maximum number of files to copy (default: 100)
                include_patterns: Only include files matching these patterns (e.g., ["*.py", "*.js"])
                exclude_patterns: Exclude files matching these patterns (e.g., ["__pycache__", "*.pyc", ".git"])

            Returns:
                Dict with success status, copied files count, skipped files, errors

            Security:
                Requires filesystem_access flag.
                Only reads from allowed_dirs if specified.
            """
            import os
            import fnmatch

            results = {
                "success": True,
                "copied_files": [],
                "copied_dirs": [],
                "skipped": [],
                "errors": [],
                "total_size": 0
            }

            # Default exclude patterns
            if exclude_patterns is None:
                exclude_patterns = [
                    "__pycache__", "*.pyc", "*.pyo", ".git", ".svn",
                    "node_modules", ".venv", "venv", "*.egg-info",
                    ".DS_Store", "Thumbs.db", "*.log"
                ]

            try:
                resolved_path = os.path.abspath(os.path.expanduser(local_path))
            except Exception as e:
                return {"success": False, "error": f"Invalid path: {e}"}

            # Security check
            if allowed_dirs:
                allowed = any(
                    resolved_path.startswith(os.path.abspath(os.path.expanduser(d)))
                    for d in allowed_dirs
                )
                if not allowed:
                    return {"success": False, "error": f"Path not in allowed directories: {resolved_path}"}

            if not os.path.exists(resolved_path):
                return {"success": False, "error": f"Folder not found: {resolved_path}"}

            if not os.path.isdir(resolved_path):
                return {"success": False, "error": f"Not a directory: {resolved_path}"}

            def should_include(filename: str) -> bool:
                """Check if file should be included based on patterns"""
                # Check exclude patterns first
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(os.path.basename(filename), pattern):
                        return False

                # If include patterns specified, file must match at least one
                if include_patterns:
                    return any(
                        fnmatch.fnmatch(filename, p) or fnmatch.fnmatch(os.path.basename(filename), p)
                        for p in include_patterns
                    )

                return True

            def should_include_dir(dirname: str) -> bool:
                """Check if directory should be traversed"""
                basename = os.path.basename(dirname)
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(basename, pattern):
                        return False
                return True

            # Normalize vfs_path
            vfs_base = vfs_path.rstrip("/")
            if not vfs_base:
                vfs_base = ""

            file_count = 0

            # Walk the directory
            for root, dirs, files in os.walk(resolved_path):
                # Filter directories in-place to prevent traversal
                dirs[:] = [d for d in dirs if should_include_dir(os.path.join(root, d))]

                # Calculate relative path
                rel_root = os.path.relpath(root, resolved_path)
                if rel_root == ".":
                    vfs_dir = vfs_base if vfs_base else "/"
                else:
                    vfs_dir = f"{vfs_base}/{rel_root.replace(os.sep, '/')}"

                # Create directory in VFS
                if vfs_dir and vfs_dir != "/":
                    dir_result = session.vfs_mkdir(vfs_dir, parents=True)
                    if dir_result.get("success"):
                        results["copied_dirs"].append(vfs_dir)

                # Copy files
                for filename in files:
                    if file_count >= max_files:
                        results["skipped"].append(f"{root}/{filename} (max files reached)")
                        continue

                    local_file = os.path.join(root, filename)

                    if not should_include(local_file):
                        results["skipped"].append(f"{local_file} (excluded by pattern)")
                        continue

                    # Check file size
                    try:
                        file_size = os.path.getsize(local_file)
                        if file_size > max_size_bytes:
                            results["skipped"].append(f"{local_file} (too large: {file_size} bytes)")
                            continue
                    except Exception as e:
                        results["errors"].append(f"{local_file}: {e}")
                        continue

                    # Build VFS path
                    vfs_file_path = f"{vfs_dir}/{filename}" if vfs_dir != "/" else f"/{filename}"

                    # Copy file
                    copy_result = session.vfs.load_from_local(
                        local_path=local_file,
                        vfs_path=vfs_file_path,
                        allowed_dirs=allowed_dirs,
                        max_size_bytes=max_size_bytes
                    )

                    if copy_result.get("success"):
                        results["copied_files"].append({
                            "local": local_file,
                            "vfs": vfs_file_path,
                            "size": copy_result.get("size_bytes", 0),
                            "type": copy_result.get("file_type", "Unknown")
                        })
                        results["total_size"] += copy_result.get("size_bytes", 0)
                        file_count += 1
                    else:
                        results["errors"].append(f"{local_file}: {copy_result.get('error')}")

            results["files_copied"] = len(results["copied_files"])
            results["dirs_created"] = len(results["copied_dirs"])

            if results["errors"]:
                results["success"] = len(results["copied_files"]) > 0  # Partial success

            return results

        def fs_copy_folder_from_vfs(
            vfs_path: str,
            local_path: str,
            allowed_dirs: list[str] | None = None,
            overwrite: bool = False,
            create_dirs: bool = True,
            include_patterns: list[str] | None = None,
            exclude_patterns: list[str] | None = None
        ) -> dict:
            """
            Copy a folder from VFS to real filesystem recursively.

            Args:
                vfs_path: Path to folder in VFS
                local_path: Destination path on real filesystem
                allowed_dirs: List of allowed directories for security
                overwrite: Allow overwriting existing files
                create_dirs: Create parent directories if needed
                include_patterns: Only include files matching these patterns
                exclude_patterns: Exclude files matching these patterns

            Returns:
                Dict with success status, copied files count, skipped files, errors

            Security:
                Requires filesystem_access flag.
                Only writes to allowed_dirs if specified.
            """
            import os
            import fnmatch

            results = {
                "success": True,
                "copied_files": [],
                "created_dirs": [],
                "skipped": [],
                "errors": [],
                "total_size": 0
            }

            # Default exclude patterns
            if exclude_patterns is None:
                exclude_patterns = []

            # Normalize VFS path
            vfs_path = vfs_path.rstrip("/")
            if not vfs_path:
                vfs_path = "/"

            # Check if VFS path exists and is a directory
            if not session.vfs._is_directory(vfs_path) and vfs_path != "/":
                # Maybe it's root or doesn't exist
                if vfs_path != "/" and not session.vfs._path_exists(vfs_path):
                    return {"success": False, "error": f"VFS path not found: {vfs_path}"}

            try:
                resolved_local = os.path.abspath(os.path.expanduser(local_path))
            except Exception as e:
                return {"success": False, "error": f"Invalid local path: {e}"}

            # Security check
            if allowed_dirs:
                allowed = any(
                    resolved_local.startswith(os.path.abspath(os.path.expanduser(d)))
                    for d in allowed_dirs
                )
                if not allowed:
                    return {"success": False, "error": f"Path not in allowed directories: {resolved_local}"}

            def should_include(filename: str) -> bool:
                """Check if file should be included based on patterns"""
                basename = os.path.basename(filename)

                # Check exclude patterns
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(basename, pattern) or fnmatch.fnmatch(filename, pattern):
                        return False

                # If include patterns specified, must match at least one
                if include_patterns:
                    return any(
                        fnmatch.fnmatch(basename, p) or fnmatch.fnmatch(filename, p)
                        for p in include_patterns
                    )

                return True

            def copy_vfs_directory(vfs_dir: str, local_dir: str):
                """Recursively copy VFS directory to local"""
                # Create local directory
                if not os.path.exists(local_dir):
                    if create_dirs:
                        try:
                            os.makedirs(local_dir, exist_ok=True)
                            results["created_dirs"].append(local_dir)
                        except Exception as e:
                            results["errors"].append(f"Cannot create {local_dir}: {e}")
                            return
                    else:
                        results["errors"].append(f"Directory does not exist: {local_dir}")
                        return

                # List VFS directory contents
                ls_result = session.vfs.ls(vfs_dir, recursive=False)
                if not ls_result.get("success"):
                    results["errors"].append(f"Cannot list {vfs_dir}: {ls_result.get('error')}")
                    return

                for item in ls_result.get("contents", []):
                    item_name = item["name"]
                    item_vfs_path = item["path"]
                    item_local_path = os.path.join(local_dir, item_name)

                    if item["type"] == "directory":
                        # Check exclude patterns for directories
                        skip = False
                        for pattern in exclude_patterns:
                            if fnmatch.fnmatch(item_name, pattern):
                                results["skipped"].append(f"{item_vfs_path} (excluded directory)")
                                skip = True
                                break

                        if not skip:
                            copy_vfs_directory(item_vfs_path, item_local_path)

                    else:  # file
                        if not should_include(item_name):
                            results["skipped"].append(f"{item_vfs_path} (excluded by pattern)")
                            continue

                        # Skip readonly/system files
                        vfs_file = session.vfs.files.get(item_vfs_path)
                        if vfs_file and vfs_file.readonly:
                            results["skipped"].append(f"{item_vfs_path} (system file)")
                            continue

                        # Check if local file exists
                        if os.path.exists(item_local_path) and not overwrite:
                            results["skipped"].append(f"{item_vfs_path} (file exists, overwrite=False)")
                            continue

                        # Copy file
                        save_result = session.vfs.save_to_local(
                            vfs_path=item_vfs_path,
                            local_path=item_local_path,
                            allowed_dirs=allowed_dirs,
                            overwrite=overwrite,
                            create_dirs=create_dirs
                        )

                        if save_result.get("success"):
                            results["copied_files"].append({
                                "vfs": item_vfs_path,
                                "local": item_local_path,
                                "size": save_result.get("size_bytes", 0)
                            })
                            results["total_size"] += save_result.get("size_bytes", 0)
                        else:
                            results["errors"].append(f"{item_vfs_path}: {save_result.get('error')}")

            # Start recursive copy
            copy_vfs_directory(vfs_path, resolved_local)

            results["files_copied"] = len(results["copied_files"])
            results["dirs_created"] = len(results["created_dirs"])

            if results["errors"]:
                results["success"] = len(results["copied_files"]) > 0  # Partial success

            return results

        # =========================================================================
        # DOCKER TOOLS (Flag: requires_docker)
        # =========================================================================

        async def docker_run(
            command: str,
            timeout: int = 300,
            sync_before: bool = True,
            sync_after: bool = True
        ) -> dict:
            """
            Execute a command in the Docker container.

            The container has VFS files synced to /workspace.
            Changes made in the container are synced back to VFS.

            Args:
                command: Shell command to execute
                timeout: Timeout in seconds (default: 300)
                sync_before: Sync VFS to container before execution
                sync_after: Sync container to VFS after execution

            Returns:
                Dict with stdout, stderr, exit_code, duration, success
            """
            return await session.docker_run_command(command, timeout, sync_before, sync_after)

        async def docker_start_app(
            entrypoint: str,
            port: int = 8080,
            env: dict[str, str] | None = None
        ) -> dict:
            """
            Start a web application in the Docker container.

            Args:
                entrypoint: Command to start the app (e.g., "python app.py")
                port: Port the app listens on (default: 8080)
                env: Environment variables

            Returns:
                Dict with url, host_port, status
            """
            return await session.docker_start_web_app(entrypoint, port, env)

        async def docker_stop_app() -> dict:
            """
            Stop the running web application.

            Returns:
                Dict with success status
            """
            return await session.docker_stop_web_app()

        async def docker_logs(lines: int = 100) -> dict:
            """
            Get logs from the web application.

            Args:
                lines: Number of log lines to retrieve

            Returns:
                Dict with logs content
            """
            return await session.docker_get_logs(lines)

        def docker_status() -> dict:
            """
            Get Docker container status.

            Returns:
                Dict with is_running, container_id, exposed_ports, etc.
            """
            return session.docker_status()

        # =========================================================================
        # MEMORY/RAG TOOLS
        # =========================================================================

        async def recall(query: str, max_entries: int = 5) -> str:
            """
            Query RAG memory for relevant context.

            Args:
                query: Search query
                max_entries: Maximum results to return

            Returns:
                Formatted context string from memory
            """
            return await session.get_reference(query, max_entries=max_entries)

        def history(last_n: int = 10) -> list[dict]:
            """
            Get recent conversation history.

            Args:
                last_n: Number of recent messages

            Returns:
                List of message dicts with role and content
            """
            return session.get_history_for_llm(last_n)

        # =========================================================================
        # SITUATION/BEHAVIOR TOOLS
        # =========================================================================

        def set_agent_situation(situation: str, intent: str) -> dict:
            """
            Set the current situation and intent for rule-based behavior.

            Args:
                situation: Current situation description
                intent: Current intent/goal

            Returns:
                Confirmation dict
            """
            session.set_situation(situation, intent)
            return {"success": True, "situation": situation, "intent": intent}

        def check_permissions(action: str, context: dict | None = None) -> dict:
            """
            Check if an action is allowed under current rules.

            Args:
                action: Action to check
                context: Optional context for rule evaluation

            Returns:
                Dict with allowed status and reason
            """
            result = session.rule_on_action(action, context)
            return {
                "allowed": result.allowed,
                "reason": result.reason,
                "rule": result.rule_name
            }

        # =========================================================================
        # REGISTER ALL TOOLS
        # =========================================================================

        tools = [
            # VFS File Operations
            {"function": vfs_list, "name": "vfs_list", "category": ["vfs", "read"]},
            {"function": vfs_read, "name": "vfs_read", "category": ["vfs", "read"]},
            {"function": vfs_create, "name": "vfs_create", "category": ["vfs", "write"]},
            {"function": vfs_write, "name": "vfs_write", "category": ["vfs", "write"]},
            {"function": vfs_edit, "name": "vfs_edit", "category": ["vfs", "write"]},
            {"function": vfs_append, "name": "vfs_append", "category": ["vfs", "write"]},
            {"function": vfs_delete, "name": "vfs_delete", "category": ["vfs", "write"]},

            # VFS Directory Operations
            {"function": vfs_mkdir, "name": "vfs_mkdir", "category": ["vfs", "write"]},
            {"function": vfs_rmdir, "name": "vfs_rmdir", "category": ["vfs", "write"]},
            {"function": vfs_mv, "name": "vfs_mv", "category": ["vfs", "write"]},

            # VFS Open/Close
            {"function": vfs_open, "name": "vfs_open", "category": ["vfs", "context"]},
            {"function": vfs_close, "name": "vfs_close", "category": ["vfs", "context"], "is_async": True},
            {"function": vfs_view, "name": "vfs_view", "category": ["vfs", "context"]},

            # VFS Info & Diagnostics
            {"function": vfs_info, "name": "vfs_info", "category": ["vfs", "read"]},
            {"function": vfs_diagnostics, "name": "vfs_diagnostics", "category": ["vfs", "lsp"], "is_async": True},
            {"function": vfs_executables, "name": "vfs_executables", "category": ["vfs", "read"]},

            # Filesystem Copy (Flag-based)
            {
                "function": fs_copy_to_vfs,
                "name": "fs_copy_to_vfs",
                "category": ["filesystem", "vfs"],
                "flags": {"filesystem_access": True},
                "description": "Copy file from real filesystem to VFS"
            },
            {
                "function": fs_copy_from_vfs,
                "name": "fs_copy_from_vfs",
                "category": ["filesystem", "vfs"],
                "flags": {"filesystem_access": True},
                "description": "Copy file from VFS to real filesystem"
            },
            {
                "function": fs_copy_folder_to_vfs,
                "name": "fs_copy_folder_to_vfs",
                "category": ["filesystem", "vfs"],
                "flags": {"filesystem_access": True},
                "description": "Copy folder from real filesystem to VFS recursively"
            },
            {
                "function": fs_copy_folder_from_vfs,
                "name": "fs_copy_folder_from_vfs",
                "category": ["filesystem", "vfs"],
                "flags": {"filesystem_access": True},
                "description": "Copy folder from VFS to real filesystem recursively"
            },

            # Docker (Flag-based)
            {
                "function": docker_run,
                "name": "docker_run",
                "category": ["docker", "execute"],
                "flags": {"requires_docker": True},
                "is_async": True
            },
            {
                "function": docker_start_app,
                "name": "docker_start_app",
                "category": ["docker", "web"],
                "flags": {"requires_docker": True},
                "is_async": True
            },
            {
                "function": docker_stop_app,
                "name": "docker_stop_app",
                "category": ["docker", "web"],
                "flags": {"requires_docker": True},
                "is_async": True
            },
            {
                "function": docker_logs,
                "name": "docker_logs",
                "category": ["docker", "read"],
                "flags": {"requires_docker": True},
                "is_async": True
            },
            {
                "function": docker_status,
                "name": "docker_status",
                "category": ["docker", "read"],
                "flags": {"requires_docker": True}
            },

            # Memory/RAG
            {"function": recall, "name": "recall", "category": ["memory", "rag"], "is_async": True},
            {"function": history, "name": "history", "category": ["memory", "history"]},

            # Situation/Behavior
            {"function": set_agent_situation, "name": "set_agent_situation", "category": ["situation"]},
            {"function": check_permissions, "name": "check_permissions", "category": ["situation", "rules"]},
        ]

        # Register all tools
        for tool_def in tools:
            self.add_tool(**tool_def)

        session.tools_initialized = True

        return tools

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
            "system_tool_definitions": 0,
            "user_tool_definitions": 0,
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
        full_sys_msg = f"{base_system_msg}"
        metrics["system_prompt"] = litellm.token_counter(model=model, text=full_sys_msg)

        # 3. Tools Definitions Berechnung
        # Wir sammeln alle Tools + Standard VFS Tools um die Definition-Größe zu berechnen
        from toolboxv2.mods.isaa.base.Agent.execution_engine import STATIC_TOOLS, DISCOVERY_TOOLS

        # System Tools die immer injected werden
        all_tools = STATIC_TOOLS + DISCOVERY_TOOLS

        # LiteLLM Token Counter kann Tools nicht direkt, wir dumpen das JSON als Näherungswert
        # (Dies ist oft genauer als man denkt, da Definitionen als Text/JSON injected werden)
        tools_json = json.dumps(all_tools)
        metrics["system_tool_definitions"] = litellm.token_counter(model=model, text=tools_json)
        tools_json = json.dumps(self.tool_manager.get_all_litellm())
        metrics["user_tool_definitions"] = litellm.token_counter(model=model, text=tools_json)

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
        print_row("Tools (Sys)", metrics["system_tool_definitions"], C_BLUE)
        print_row("Tools (User)", metrics["user_tool_definitions"], C_BLUE)
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
        if self.amd.enable_docker:
            await self.session_manager.cleanup_docker_containers()
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
