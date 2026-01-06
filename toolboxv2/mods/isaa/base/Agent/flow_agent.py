"""
FlowAgent V2 - Production-ready Agent System

Refactored architecture:
- SessionManager: Session lifecycle with ChatSession integration
- ToolManager: Unified tool registry (local, MCP, A2A)
- CheckpointManager: Full state persistence
- BindManager: Agent-to-agent binding

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
from functools import wraps
from typing import Any, AsyncGenerator, Callable

from pydantic import BaseModel, ValidationError

from toolboxv2 import get_logger
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
        **kwargs
    ) -> str | Any:
        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM required")
        
        model = kwargs.pop('model', None) or (
            self.amd.fast_llm_model if model_preference == "fast" else self.amd.complex_llm_model
        )
        use_stream = stream if stream is not None else self.stream
        
        llm_kwargs = {'model': model, 'messages': messages.copy(), 'stream': use_stream, **kwargs}
        
        if with_context and self.active_session:
            session = self.session_manager.get(self.active_session)
            if session and session._initialized:
                vfs_context = session.build_vfs_context()
                system_msg = self.amd.get_system_message_with_persona()
                llm_kwargs['messages'] = [{"role": "system", "content": f"{system_msg}\n\n{vfs_context}"}] + llm_kwargs['messages']
        
        if 'api_key' not in llm_kwargs:
            llm_kwargs['api_key'] = self._get_api_key_for_model(model)
        
        llm_start = time.perf_counter()
        
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
        
        enhanced_prompt = f"{prompt}\n\nReturn YAML with fields:\n" + "\n".join(fields_desc)
        
        messages = (message_context or []) + [{"role": "user", "content": enhanced_prompt}]
        
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
    # CORE: a_run - MAKER/RLM inspired orchestration
    # =========================================================================

    async def a_run(
        self,
        query: str,
        session_id: str = "default",
        remember: bool = True,
        use_sandbox: bool = False,
        learn_on_success: bool = False,
        max_tool_iterations: int = 10,
        max_tools_per_call: int = 5,
        intermediate_callback: Callable[[str, dict], None] | None = None,
        **kwargs
    ) -> str:
        """
        Main entry point for agent execution.
        
        Architecture inspired by MAKER (maximal decomposition) + RLM (recursive context).
        
        Flow:
        1. Auto Intent Detection → ImmediateResponse or ToolRequired
        2. If need_tools → Select tool categories
        3. If >N tools → Select max N specific tools
        4. ReAct Loop until final response or max iterations
        5. Validate with initial model
        6. Optional: sandbox mode (clean VFS, transfer on success)
        7. Optional: non-blocking learning on success
        
        Args:
            query: User query
            session_id: Session identifier
            remember: Store in conversation history
            use_sandbox: Run in clean VFS, transfer on success
            learn_on_success: Learn patterns after successful completion
            max_tool_iterations: Max ReAct loop iterations
            max_tools_per_call: Max tools to provide per LLM call
            intermediate_callback: Callback for intermediate responses
                                   signature: (response: str, metadata: dict) -> None
            **kwargs: Additional parameters
            
        Returns:
            Final response string
        """
        from pydantic import BaseModel, Field
        from typing import Literal
        
        self.active_session = session_id
        self.is_running = True
        run_start = time.perf_counter()
        
        # Pydantic models for structured responses
        class IntentAnalysis(BaseModel):
            """Initial intent analysis result"""
            immediate_response: str | None = Field(None, description="Direct answer if no tools needed")
            need_tools: bool = Field(False, description="Whether tools are required")
            need_decomposition: bool = Field(False, description="Whether task needs decomposition")
            suggested_categories: list[str] = Field(default_factory=list, description="Tool categories needed")
            reasoning: str = Field("", description="Brief reasoning for the decision")
        
        class ToolSelection(BaseModel):
            """Selected tools for execution"""
            selected_tools: list[str] = Field(default_factory=list, description="Tool names to use")
            execution_plan: str = Field("", description="Brief plan for tool usage")
        
        class ReActStep(BaseModel):
            """Single ReAct reasoning step"""
            thought: str = Field("", description="Current reasoning")
            action: str | None = Field(None, description="Tool to call (null if final answer)")
            action_input: dict = Field(default_factory=dict, description="Tool arguments")
            is_final: bool = Field(False, description="True if ready to give final answer")
            final_answer: str | None = Field(None, description="Final answer if is_final=True")
        
        class ValidationResult(BaseModel):
            """Final validation result"""
            is_valid: bool = Field(True, description="Whether response is valid/complete")
            needs_more_tools: bool = Field(False, description="Whether more tool calls needed")
            feedback: str = Field("", description="Feedback if invalid")
            final_response: str = Field("", description="Final validated response")
        
        try:
            # === PHASE 0: Session Setup ===
            if use_sandbox:
                # Create temporary sandbox session
                sandbox_session_id = f"{session_id}_sandbox_{uuid.uuid4().hex[:8]}"
                session = await self.session_manager.get_or_create(sandbox_session_id)
                original_session = await self.session_manager.get_or_create(session_id)
            else:
                session = await self.session_manager.get_or_create(session_id)
                original_session = None
            
            # Add user message
            if remember:
                await session.add_message({"role": "user", "content": query})
            
            # Get RAG context
            rag_context = ""
            try:
                rag_context = await session.get_reference(query)
            except Exception:
                pass
            
            # Build base context
            history = session.get_history_for_llm(last_n=5)
            available_categories = self.tool_manager.list_categories()
            
            # === PHASE 1: Intent Detection ===
            intent_prompt = f"""
<query>{query}</query>

<context>
{rag_context[:2000] if rag_context else "No additional context."}
</context>

<available_tool_categories>
{', '.join(available_categories) if available_categories else 'No tools available'}
</available_tool_categories>

Analyze this query and decide:
1. Can you answer immediately without tools? If yes, provide the answer.
2. Do you need tools? If yes, which categories?
3. Is this a complex task that needs decomposition into sub-tasks?
"""
            
            intent_result = await self.a_format_class(
                pydantic_model=IntentAnalysis,
                prompt=intent_prompt,
                message_context=history,
                model_preference="fast",
                auto_context=True
            )
            
            intent = IntentAnalysis(**intent_result)
            
            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="intent_analysis",
                    node_name="a_run",
                    session_id=session_id,
                    status=NodeStatus.COMPLETED,
                    metadata={"need_tools": intent.need_tools, "categories": intent.suggested_categories}
                ))
            
            # === PHASE 2: Immediate Response (no tools needed) ===
            if not intent.need_tools and intent.immediate_response:
                final_response = intent.immediate_response
                
                if remember:
                    await session.add_message({"role": "assistant", "content": final_response})
                
                if use_sandbox and original_session:
                    # Transfer to original session
                    await original_session.add_message({"role": "user", "content": query})
                    await original_session.add_message({"role": "assistant", "content": final_response})
                    await self.session_manager.close_session(sandbox_session_id)
                
                return final_response
            
            # === PHASE 3: Tool Category Selection ===
            selected_categories = intent.suggested_categories or []
            
            if not selected_categories and available_categories:
                # Let model select categories
                category_prompt = f"""
<query>{query}</query>
<available_categories>{', '.join(available_categories)}</available_categories>

Select 1-3 most relevant tool categories for this task.
"""
                # Simple extraction
                cat_response = await self.a_run_llm_completion(
                    messages=[{"role": "user", "content": category_prompt}],
                    model_preference="fast",
                    with_context=False,
                    stream=False,
                    max_tokens=100,
                    task_id="category_selection"
                )
                # Parse categories from response
                for cat in available_categories:
                    if cat.lower() in cat_response.lower():
                        selected_categories.append(cat)
                        if len(selected_categories) >= 3:
                            break
            
            # === PHASE 4: Tool Selection (if too many tools) ===
            available_tools = self.tool_manager.get_by_category(*selected_categories) if selected_categories else self.tool_manager.get_all()
            
            if len(available_tools) > max_tools_per_call:
                # Let model select specific tools
                tool_list = "\n".join([f"- {t.name}: {t.description[:100]}" for t in available_tools[:30]])
                
                tool_select_result = await self.a_format_class(
                    pydantic_model=ToolSelection,
                    prompt=f"""
<query>{query}</query>
<available_tools>
{tool_list}
</available_tools>

Select up to {max_tools_per_call} most relevant tools for this specific task.
""",
                    model_preference="fast",
                    auto_context=False
                )
                
                tool_selection = ToolSelection(**tool_select_result)
                selected_tool_names = tool_selection.selected_tools[:max_tools_per_call]
                
                # Filter to selected tools
                tools_for_llm = [
                    t.litellm_schema for t in available_tools 
                    if t.name in selected_tool_names and t.litellm_schema
                ]
            else:
                tools_for_llm = [t.litellm_schema for t in available_tools if t.litellm_schema]
            
            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="tool_selection",
                    node_name="a_run",
                    session_id=session_id,
                    status=NodeStatus.COMPLETED,
                    metadata={"tool_count": len(tools_for_llm)}
                ))
            
            # === PHASE 5: ReAct Loop ===
            react_messages = history.copy()
            react_messages.append({"role": "user", "content": query})
            
            if rag_context:
                react_messages.insert(0, {"role": "system", "content": f"Relevant context:\n{rag_context[:3000]}"})
            
            tool_results_history = []  # For final validation
            iteration = 0
            final_response = None
            
            while iteration < max_tool_iterations:
                iteration += 1
                
                # ReAct step
                react_response = await self.a_run_llm_completion(
                    messages=react_messages,
                    tools=tools_for_llm if tools_for_llm else None,
                    tool_choice="auto" if tools_for_llm else None,
                    model_preference="fast",
                    get_response_message=True,
                    stream=False,
                    task_id=f"react_step_{iteration}"
                )
                
                # Check for tool calls
                if hasattr(react_response, 'tool_calls') and react_response.tool_calls:
                    # Execute each tool call
                    react_messages.append({
                        "role": "assistant",
                        "content": react_response.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                            } for tc in react_response.tool_calls
                        ]
                    })
                    
                    for tc in react_response.tool_calls:
                        tool_name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except:
                            args = {}
                        
                        # Execute tool
                        try:
                            result = await self.arun_function(tool_name, **args)
                            result_str = json.dumps(result, default=str, ensure_ascii=False)[:5000]
                            tool_results_history.append({"tool": tool_name, "args": args, "result": result_str, "success": True})
                        except Exception as e:
                            result_str = json.dumps({"error": str(e)})
                            tool_results_history.append({"tool": tool_name, "args": args, "error": str(e), "success": False})
                        
                        # Add tool result to messages
                        react_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_str
                        })
                        
                        if self.progress_tracker:
                            await self.progress_tracker.emit_event(ProgressEvent(
                                event_type="react_tool_call",
                                node_name="a_run",
                                session_id=session_id,
                                status=NodeStatus.COMPLETED,
                                tool_name=tool_name,
                                iteration=iteration
                            ))
                    
                    # Intermediate callback
                    if intermediate_callback and react_response.content:
                        try:
                            intermediate_callback(react_response.content, {
                                "iteration": iteration,
                                "tools_called": [tc.function.name for tc in react_response.tool_calls],
                                "phase": "react_loop"
                            })
                        except Exception:
                            pass
                
                else:
                    # No tool calls - this is the response
                    content = react_response.content if hasattr(react_response, 'content') else str(react_response)
                    
                    if content:
                        final_response = content
                        break
            
            # === PHASE 6: Final Validation ===
            if final_response is None:
                final_response = "I was unable to complete the task within the allowed iterations."
            
            # Validate with initial model
            validation_prompt = f"""
<original_query>{query}</original_query>

<proposed_response>{final_response}</proposed_response>

<tool_execution_summary>
{len(tool_results_history)} tools executed.
Successful: {sum(1 for t in tool_results_history if t.get('success'))}
Failed: {sum(1 for t in tool_results_history if not t.get('success'))}
</tool_execution_summary>

Validate this response:
1. Is it complete and accurate?
2. Does it fully answer the query?
3. Should more tools be called?

If valid, provide the final polished response.
"""
            
            validation_result = await self.a_format_class(
                pydantic_model=ValidationResult,
                prompt=validation_prompt,
                model_preference=kwargs.get("validation_model", "fast"),
                auto_context=False
            )
            
            validation = ValidationResult(**validation_result)
            
            if validation.is_valid and validation.final_response:
                final_response = validation.final_response
            elif validation.needs_more_tools and iteration < max_tool_iterations:
                # Could recurse here, but for safety we just use current response
                pass
            
            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="validation_complete",
                    node_name="a_run",
                    session_id=session_id,
                    status=NodeStatus.COMPLETED,
                    metadata={"is_valid": validation.is_valid, "iterations": iteration}
                ))
            
            # === PHASE 7: Finalize ===
            if remember:
                await session.add_message({"role": "assistant", "content": final_response})
            
            # Sandbox transfer
            if use_sandbox and original_session:
                await original_session.add_message({"role": "user", "content": query})
                await original_session.add_message({"role": "assistant", "content": final_response})
                # Transfer successful VFS files
                for filename, vfs_file in session.vfs.files.items():
                    if not vfs_file.readonly:
                        original_session.vfs.create(filename, vfs_file.content)
                await self.session_manager.close_session(sandbox_session_id)
            
            # Non-blocking learning
            if learn_on_success and validation.is_valid:
                asyncio.create_task(self._learn_from_success(
                    query=query,
                    response=final_response,
                    tool_history=tool_results_history,
                    session=session
                ))
            
            run_duration = time.perf_counter() - run_start
            
            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="a_run_complete",
                    node_name="a_run",
                    session_id=session_id,
                    status=NodeStatus.COMPLETED,
                    duration=run_duration,
                    metadata={
                        "iterations": iteration,
                        "tools_used": len(tool_results_history),
                        "sandbox_mode": use_sandbox,
                        "learned": learn_on_success and validation.is_valid
                    }
                ))
            
            return final_response
            
        except Exception as e:
            logger.error(f"a_run failed: {e}")
            import traceback
            traceback.print_exc()
            
            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="a_run_error",
                    node_name="a_run",
                    session_id=session_id,
                    status=NodeStatus.FAILED,
                    error_details={"message": str(e), "type": type(e).__name__}
                ))
            
            return f"Error: {str(e)}"
        finally:
            self.is_running = False
            self.active_session = None

    async def _learn_from_success(
        self,
        query: str,
        response: str,
        tool_history: list[dict],
        session: 'AgentSession'
    ):
        """
        Non-blocking learning from successful execution.
        Updates RuleSet with learned patterns.
        """
        try:
            if not tool_history:
                return
            
            # Extract patterns
            successful_tools = [t["tool"] for t in tool_history if t.get("success")]
            
            if successful_tools:
                # Learn tool sequence pattern
                pattern = f"For queries like '{query[:50]}...', tools {', '.join(successful_tools[:3])} were effective"
                session.rule_set.learn_pattern(
                    pattern=pattern,
                    source_situation=session.rule_set.current_situation or "general",
                    confidence=0.6,
                    category="tool_usage"
                )
            
            # Update rule success counts
            for rule in session.rule_set.get_active_rules():
                session.rule_set.record_rule_success(rule.id)
                
        except Exception as e:
            logger.warning(f"Learning failed (non-critical): {e}")

    # =========================================================================
    # CORE: a_stream
    # =========================================================================

    async def a_stream(
        self,
        query: str,
        session_id: str = "default",
        remember: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response."""
        self.active_session = session_id
        self.is_running = True
        
        try:
            session = await self.session_manager.get_or_create(session_id)
            
            if remember:
                await session.add_message({"role": "user", "content": query})
            
            messages = session.get_history_for_llm(last_n=10)
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
            yield f"Error: {str(e)}"
        finally:
            self.is_running = False
            self.active_session = None

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
        
        await self.session_manager.close_all()
        self.executor.shutdown(wait=True)
        
        if self.a2a_server:
            await self.a2a_server.close()
        if self.mcp_server:
            await self.mcp_server.close()
        
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
