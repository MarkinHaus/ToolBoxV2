"""
Z.AI LiteLLM Provider - Uses Anthropic SDK as backend with OpenAI format output

This provider enables using Z.AI's GLM models through LiteLLM with automatic
conversion between Anthropic and OpenAI formats.

Features:
- Completion (sync/async)
- Streaming (sync/async) with proper chunk conversion
- Tool Calling with automatic schema conversion
- Usage tracking
- Reasoning/thinking content support

Usage:
    from zai_litellm_provider import setup_zai_provider
    import litellm

    setup_zai_provider()

    response = litellm.completion(
        model="zai/GLM-4.7",
        messages=[{"role": "user", "content": "Hello!"}],
        tools=[...],  # OpenAI format tools
        stream=True
    )
"""

import os
import json
import time
import uuid
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Iterator,
    AsyncIterator,
    Union,
)
from dataclasses import dataclass, field

import asyncio
import litellm
from litellm import CustomLLM
from litellm.types.utils import (
    GenericStreamingChunk,
    ModelResponse,
    Choices,
    Message,
    Usage,
    StreamingChoices,
    Delta,
    ChatCompletionMessageToolCall,
    Function,
)

# Anthropic SDK imports
try:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import (
        Message as AnthropicMessage,
        ContentBlock,
        TextBlock,
        ToolUseBlock,
        RawMessageStartEvent,
        RawContentBlockStartEvent,
        RawContentBlockDeltaEvent,
        RawContentBlockStopEvent,
        RawMessageDeltaEvent,
        RawMessageStopEvent,
    )
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️ anthropic package not installed. Install with: pip install anthropic")


# === Configuration ===
ZAI_API_KEY = os.getenv("ZAI_API_KEY", "")
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/anthropic")


import contextlib

@contextlib.contextmanager
def _suppress_loop_closed():
    """Suppress 'Event loop is closed' during cleanup on Windows."""
    try:
        yield
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise
    except Exception:
        pass  # Best-effort cleanup, never propagate

@dataclass
class ToolCallAccumulator:
    """Accumulates tool call data during streaming"""
    id: str = ""
    name: str = ""
    arguments: str = ""
    index: int = 0



@dataclass
class StreamState:
    """Tracks state during streaming response processing"""
    content: str = ""
    tool_calls: Dict[int, ToolCallAccumulator] = field(default_factory=dict)
    current_block_index: int = 0
    current_block_type: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: Optional[str] = None
    model: str = ""
    message_id: str = ""
    # Map Anthropic block index to OpenAI tool index (0-based)
    tool_map: Dict[int, int] = field(default_factory=dict)
    next_tool_index: int = 0

class ZAIProvider(CustomLLM):
    """
    Custom LLM Provider for Z.AI using Anthropic SDK

    Automatically converts between Anthropic and OpenAI formats:
    - OpenAI tools → Anthropic tools
    - Anthropic response → OpenAI response
    - Streaming events → OpenAI streaming chunks
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.api_key = api_key or ZAI_API_KEY
        self.base_url = base_url or ZAI_BASE_URL
        self.debug = debug

        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic package required. Install with: pip install anthropic")

        # Initialize Anthropic clients
        self._sync_client: Optional[Anthropic] = None
        self._sync_client: Optional[Anthropic] = None
        self._async_client: Optional[AsyncAnthropic] = None

    def _debug_log(self, msg: str, data: Any = None):
        """Debug logging helper"""
        if self.debug:
            print(f"[ZAI DEBUG] {msg}")
            if data is not None:
                print(f"  → {json.dumps(data, indent=2, default=str)[:500]}")

    @property
    def sync_client(self) -> Anthropic:
        """Lazy initialization of sync client"""
        if self._sync_client is None:
            self._sync_client = Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._sync_client

    @property
    def async_client(self) -> AsyncAnthropic:
        """Lazy initialization of async client — recreates if event loop changed.

        httpx.AsyncClient binds its transport to the loop at creation time.
        If a sub-agent or background task runs in a different loop context,
        the old client's SSL transport will crash on cleanup with
        'Event loop is closed'. Detect this and rebuild.
        """
        current_loop_id: Optional[int] = None
        try:
            current_loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            pass  # No running loop (sync context) — use cached if exists

        needs_rebuild = (
            self._async_client is None
            or (current_loop_id is not None and self._async_client_loop_id != current_loop_id)
        )

        if needs_rebuild:
            # Discard old client — do NOT await close(), old loop may be dead
            self._async_client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self._async_client_loop_id = current_loop_id

        return self._async_client

    # =========================================================================
    # FORMAT CONVERSION: OpenAI → Anthropic
    # =========================================================================

    def _extract_model(self, model: str) -> str:
        """Remove provider prefix from model name"""
        return model.split("/", 1)[-1] if "/" in model else model

    def _convert_messages_to_anthropic(
        self,
        messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert OpenAI messages format to Anthropic format

        OpenAI format:
        - system: {"role": "system", "content": "..."}
        - user: {"role": "user", "content": "..."}
        - assistant: {"role": "assistant", "content": "...", "tool_calls": [...]}
        - tool: {"role": "tool", "tool_call_id": "...", "content": "..."}

        Anthropic format:
        - system: Separate parameter
        - user: {"role": "user", "content": "..." or [...]}
        - assistant: {"role": "assistant", "content": [...with tool_use blocks...]}
        - tool result: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}

        Returns: (system_message, converted_messages)
        """
        system_message = None
        converted = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                # Anthropic uses separate system parameter
                if system_message:
                    system_message += "\n\n" + str(content)
                else:
                    system_message = str(content) if content else ""
                continue

            if role == "assistant":
                # Check for tool calls in assistant message
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    # Convert to Anthropic content array with tool_use blocks
                    anthropic_content = []

                    # Add text content first if present
                    if content:
                        anthropic_content.append({"type": "text", "text": str(content)})

                    # Add tool_use blocks
                    for tc in tool_calls:
                        # Handle both dict and object formats
                        if isinstance(tc, dict):
                            tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:24]}")
                            func = tc.get("function", {})
                            func_name = func.get("name", "")
                            func_args = func.get("arguments", "{}")
                        else:
                            tc_id = getattr(tc, "id", f"call_{uuid.uuid4().hex[:24]}")
                            func = getattr(tc, "function", None)
                            func_name = getattr(func, "name", "") if func else ""
                            func_args = getattr(func, "arguments", "{}") if func else "{}"

                        # Parse arguments string to dict for Anthropic's "input" field
                        try:
                            input_dict = json.loads(func_args) if isinstance(func_args, str) else func_args
                        except json.JSONDecodeError:
                            input_dict = {}

                        anthropic_content.append({
                            "type": "tool_use",
                            "id": tc_id,
                            "name": func_name,
                            "input": input_dict  # Anthropic uses "input", not "arguments"
                        })

                    converted.append({
                        "role": "assistant",
                        "content": anthropic_content
                    })
                else:
                    # Simple assistant message
                    converted.append({
                        "role": "assistant",
                        "content": str(content) if content else ""
                    })

            elif role == "tool":
                # Convert OpenAI tool response to Anthropic tool_result
                # OpenAI uses "tool_call_id", Anthropic uses "tool_use_id"
                tool_call_id = msg.get("tool_call_id", "")
                tool_content = msg.get("content", "")

                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,  # Anthropic field name
                        "content": str(tool_content) if tool_content else ""
                    }]
                })

            elif role == "user":
                # Handle multimodal content if present
                if isinstance(content, list):
                    anthropic_content = []
                    for item in content:
                        item_type = item.get("type", "")

                        if item_type == "text":
                            anthropic_content.append({
                                "type": "text",
                                "text": item.get("text", "")
                            })
                        elif item_type == "image_url":
                            # Convert image URL to Anthropic format
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                # Base64 image
                                media_type, data = self._parse_data_url(url)
                                anthropic_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data
                                    }
                                })
                            else:
                                # URL image
                                anthropic_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": url
                                    }
                                })

                    converted.append({
                        "role": "user",
                        "content": anthropic_content if anthropic_content else ""
                    })
                else:
                    converted.append({
                        "role": "user",
                        "content": str(content) if content else ""
                    })

            else:
                # Pass through other roles (shouldn't happen normally)
                converted.append({
                    "role": role,
                    "content": str(content) if content else ""
                })

        return system_message, converted

        return system_message, converted

    def _parse_data_url(self, data_url: str) -> tuple[str, str]:
        """Parse data URL into media_type and base64 data"""
        # Format: data:image/png;base64,iVBORw0...
        if not data_url.startswith("data:"):
            return "image/png", data_url

        header, data = data_url.split(",", 1)
        media_type = header.split(";")[0].replace("data:", "")
        return media_type, data

    def _convert_tools_to_anthropic(
        self,
        tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Convert tools to Anthropic format - AUTO-DETECTS input format

        Anthropic REQUIRES this exact format:
        {
            "name": "get_weather",
            "description": "...",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

        Accepts input formats:

        1. OpenAI format (nested with "function" wrapper):
        {"type": "function", "function": {"name": ..., "parameters": {...}}}

        2. Flat format with "parameters":
        {"name": ..., "parameters": {...}}

        3. Anthropic native format (with "input_schema"):
        {"name": ..., "input_schema": {...}}
        """
        if not tools:
            return None

        converted = []
        for tool in tools:
            name = ""
            description = ""
            schema = {"type": "object", "properties": {}}

            # Format 1: OpenAI nested format {"type": "function", "function": {...}}
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                name = func.get("name", "")
                description = func.get("description", "")
                schema = func.get("parameters", schema)

            # Format 3: Anthropic format (has input_schema)
            elif "input_schema" in tool:
                name = tool.get("name", "")
                description = tool.get("description", "")
                schema = tool["input_schema"]

            # Format 2: Flat format with "parameters"
            elif "name" in tool and "parameters" in tool:
                name = tool.get("name", "")
                description = tool.get("description", "")
                schema = tool["parameters"]

            # Minimal format (just name)
            elif "name" in tool:
                name = tool.get("name", "")
                description = tool.get("description", "")
                schema = tool.get("input_schema", tool.get("parameters", schema))

            else:
                continue  # Skip invalid tools

            # Build Anthropic tool with input_schema (NOT parameters!)
            converted.append({
                "name": name,
                "description": description,
                "input_schema": schema
            })

        return converted if converted else None

    def _convert_tool_choice_to_anthropic(
        self,
        tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """
        Convert OpenAI tool_choice to Anthropic/Z.AI format

        OpenAI formats:
        - "auto" → {"type": "auto"}
        - "none" → {"type": "none"}
        - "required" → {"type": "any"}
        - {"type": "function", "function": {"name": "..."}} → {"type": "tool", "name": "..."}

        Z.AI/Anthropic format (what actually works):
        - {"type": "auto"} or {"input": "auto"}
        - {"type": "any"}
        - {"type": "tool", "name": "..."}
        """
        if tool_choice is None:
            return {"type": "auto"}

        # String shortcuts
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"type": "auto"}
            elif tool_choice == "none":
                return {"type": "none"}
            elif tool_choice == "required":
                return {"type": "any"}
            else:
                # Assume it's a tool name
                return {"type": "tool", "name": tool_choice}

        # Dict format
        elif isinstance(tool_choice, dict):
            # Already in Anthropic format with "type"
            if "type" in tool_choice:
                tc_type = tool_choice["type"]
                if tc_type == "function":
                    # OpenAI specific function format
                    func = tool_choice.get("function", {})
                    return {"type": "tool", "name": func.get("name", "")}
                else:
                    # Pass through (auto, any, none, tool)
                    return tool_choice

            # Z.AI format with "input" key
            elif "input" in tool_choice:
                return tool_choice  # Pass through as-is

            # Has "name" directly (shorthand for specific tool)
            elif "name" in tool_choice:
                return {"type": "tool", "name": tool_choice["name"]}

        # Default
        return {"type": "auto"}

    # =========================================================================
    # FORMAT CONVERSION: Anthropic → OpenAI
    # =========================================================================

    def _generate_response_id(self) -> str:
        """Generate OpenAI-style response ID"""
        return f"chatcmpl-{uuid.uuid4().hex[:29]}"

    def _convert_stop_reason(self, stop_reason: Optional[str]) -> str:
        """Convert Anthropic stop_reason to OpenAI finish_reason"""
        mapping = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
        }
        return mapping.get(stop_reason or "", "stop")

    def _convert_anthropic_response(
        self,
        response: "AnthropicMessage",
        model: str
    ) -> ModelResponse:
        """
        Convert Anthropic Message response to OpenAI ModelResponse

        Anthropic response format:
        {
            "id": "msg_...",
            "model": "...",
            "stop_reason": "tool_use",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "..."},
                {"type": "tool_use", "id": "toolu_...", "name": "...", "input": {...}}
            ]
        }

        OpenAI response format:
        {
            "choices": [{
                "message": {
                    "content": "...",
                    "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
                },
                "finish_reason": "tool_calls"
            }]
        }
        """
        content_text = ""
        tool_calls = []

        for block in response.content:
            # Handle text blocks
            if hasattr(block, "type") and block.type == "text":
                content_text += getattr(block, "text", "")

            # Handle tool_use blocks
            elif hasattr(block, "type") and block.type == "tool_use":
                # Anthropic uses "input", OpenAI uses "arguments" (as JSON string)
                tool_input = getattr(block, "input", {})
                tool_calls.append({
                    "id": getattr(block, "id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", ""),
                        "arguments": json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
                    }
                })

        # Build message
        message = Message(
            content=content_text if content_text else None,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
            function_call=None,
        )

        # Build usage
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=getattr(response.usage, "input_tokens", 0),
                completion_tokens=getattr(response.usage, "output_tokens", 0),
                total_tokens=getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0),
            )

        # Build response
        return ModelResponse(
            id=self._generate_response_id(),
            created=int(time.time()),
            model=model,
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason=self._convert_stop_reason(response.stop_reason),
                    index=0,
                    message=message,
                )
            ],
            usage=usage,
        )

    def _convert_stream_event_to_chunk(
        self,
        event: Any,
        state: StreamState,
        model: str
    ) -> Optional[GenericStreamingChunk]:
        """
        Convert Anthropic streaming event to OpenAI GenericStreamingChunk

        CRITICAL: LiteLLM expects tool_use as a single ChatCompletionToolCallChunk dict:
        {
            "id": "call_xxx",
            "type": "function",
            "function": {"name": "...", "arguments": "..."},
            "index": 0
        }
        """
        DEBUG_STREAM = self.debug

        def log(msg, **kwargs):
            if DEBUG_STREAM:
                data = json.dumps(kwargs, default=str) if kwargs else ""
                print(f"[STREAM] {msg} {data}")

        event_type = getattr(event, "type", None)
        log(f"Event: {event_type}", index=getattr(event, "index", "N/A"))

        if not event_type:
            return None

        # === MESSAGE START ===
        if event_type == "message_start":
            msg = getattr(event, "message", None)
            if msg:
                state.message_id = getattr(msg, "id", "")
                state.model = getattr(msg, "model", model)
                usage = getattr(msg, "usage", None)
                if usage:
                    state.input_tokens = getattr(usage, "input_tokens", 0)
            return None

        # === CONTENT BLOCK START ===
        elif event_type == "content_block_start":
            state.current_block_index = getattr(event, "index", 0)
            block = getattr(event, "content_block", None)

            if block:
                block_type = getattr(block, "type", "")
                state.current_block_type = block_type
                log(f"Block Start [{state.current_block_index}]", type=block_type)

                if block_type == "tool_use":
                    # Map Anthropic block index to OpenAI tool index (0-based)
                    anthropic_index = state.current_block_index
                    openai_index = state.next_tool_index
                    state.tool_map[anthropic_index] = openai_index
                    state.next_tool_index += 1

                    tool_id = getattr(block, "id", f"call_{uuid.uuid4().hex[:24]}")
                    tool_name = getattr(block, "name", "")

                    log(f"Tool Init: Anthropic[{anthropic_index}] -> OpenAI[{openai_index}]",
                        id=tool_id, name=tool_name)

                    # Store accumulator
                    state.tool_calls[anthropic_index] = ToolCallAccumulator(
                        id=tool_id,
                        name=tool_name,
                        arguments="",
                        index=anthropic_index
                    )

                    # FIXED: tool_use must be a SINGLE ChatCompletionToolCallChunk dict
                    # NOT a list! LiteLLM expects this exact structure.
                    tool_use = {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": ""
                        },
                        "index": openai_index
                    }

                    log("Yielding Tool Start Chunk", tool_use=tool_use)

                    return self._build_generic_chunk(
                        text="",
                        tool_use=tool_use,
                        is_finished=False,
                        finish_reason=None,
                        index=0,
                        usage=self._build_usage_dict(state)
                    )

                elif block_type == "text":
                    state.current_block_type = "text"

            return None

        # === CONTENT BLOCK DELTA ===
        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            event_index = getattr(event, "index", state.current_block_index)

            if delta:
                delta_type = getattr(delta, "type", "")
                log(f"Delta [{event_index}]", type=delta_type)

                # Text delta
                if delta_type == "text_delta":
                    chunk_text = getattr(delta, "text", "")
                    state.content += chunk_text

                    if chunk_text:
                        return self._build_generic_chunk(
                            text=chunk_text,
                            tool_use=None,
                            is_finished=False,
                            finish_reason=None,
                            index=0,
                            usage=self._build_usage_dict(state)
                        )

                # Tool arguments delta
                elif delta_type == "input_json_delta":
                    partial_json = getattr(delta, "partial_json", "")

                    if event_index in state.tool_calls:
                        state.tool_calls[event_index].arguments += partial_json
                        openai_index = state.tool_map.get(event_index, 0)

                        log(f"Tool Delta: Anthropic[{event_index}] -> OpenAI[{openai_index}]",
                            partial=partial_json[:50])

                        # FIXED: Single dict with function.arguments delta
                        # id and name can be None for delta chunks
                        tool_use = {
                            "id": None,
                            "type": "function",
                            "function": {
                                "name": None,
                                "arguments": partial_json
                            },
                            "index": openai_index
                        }

                        return self._build_generic_chunk(
                            text="",
                            tool_use=tool_use,
                            is_finished=False,
                            finish_reason=None,
                            index=0,
                            usage=self._build_usage_dict(state)
                        )

            return None

        # === CONTENT BLOCK STOP ===
        elif event_type == "content_block_stop":
            log("Block Stop", index=getattr(event, "index", -1))
            return None

        # === MESSAGE DELTA (finish reason + final usage) ===
        elif event_type == "message_delta":
            delta = getattr(event, "delta", None)
            usage = getattr(event, "usage", None)

            if delta:
                stop_reason = getattr(delta, "stop_reason", None)
                if stop_reason:
                    state.finish_reason = self._convert_stop_reason(stop_reason)
                    log("Message Delta Stop", reason=stop_reason, converted=state.finish_reason)

            if usage:
                state.output_tokens = getattr(usage, "output_tokens", 0)
                state.input_tokens = getattr(usage, "input_tokens", state.input_tokens)

            # Only emit if we have a finish reason
            if state.finish_reason:
                return self._build_generic_chunk(
                    text="",
                    tool_use=None,
                    is_finished=True,
                    finish_reason=state.finish_reason,
                    index=0,
                    usage=self._build_usage_dict(state)
                )
            return None

        # === MESSAGE STOP ===
        elif event_type == "message_stop":
            log("Message Stop", final_reason=state.finish_reason)
            return self._build_generic_chunk(
                text="",
                tool_use=None,
                is_finished=True,
                finish_reason=state.finish_reason or "stop",
                index=0,
                usage=self._build_usage_dict(state)
            )

        return None

    def _build_usage_dict(self, state: StreamState) -> Dict[str, int]:
        """Build usage dictionary from state"""
        return {
            "prompt_tokens": state.input_tokens,
            "completion_tokens": state.output_tokens,
            "total_tokens": state.input_tokens + state.output_tokens
        }

    def _build_generic_chunk(
        self,
        text: str,
        tool_use: Optional[Dict[str, Any]],  # CHANGED: Single dict, not List
        is_finished: bool,
        finish_reason: Optional[str],
        index: int,
        usage: Dict[str, int]
    ) -> GenericStreamingChunk:
        """Build a GenericStreamingChunk with proper tool_use format for LiteLLM"""
        chunk: GenericStreamingChunk = {
            "text": text,
            "is_finished": is_finished,
            "finish_reason": finish_reason,
            "index": index,
            "usage": usage
        }

        # LiteLLM expects tool_use as single ChatCompletionToolCallChunk, not a list!
        if tool_use is not None:
            chunk["tool_use"] = tool_use

        return chunk

    # =========================================================================
    # ALLOWED PARAMETERS
    # =========================================================================

    def _get_allowed_params(self) -> set:
        """Parameters that can be passed to Anthropic API"""
        return {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "stop",  # → stop_sequences
            "tools",
            "tool_choice",
            "stream",
            "metadata",
        }

    def _prepare_anthropic_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare kwargs for Anthropic API call"""
        allowed = self._get_allowed_params()
        result = {}

        for k, v in kwargs.items():
            if k in allowed and v is not None:
                # Rename 'stop' to 'stop_sequences' for Anthropic
                if k == "stop":
                    if isinstance(v, list):
                        result["stop_sequences"] = v
                    elif isinstance(v, str):
                        result["stop_sequences"] = [v]
                else:
                    result[k] = v

        return result

    # =========================================================================
    # NON-STREAMING COMPLETION
    # =========================================================================

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Synchronous non-streaming completion"""
        actual_model = self._extract_model(model)

        # Debug: Log all incoming kwargs
        self._debug_log("Raw kwargs keys", list(kwargs.keys()))

        # LiteLLM puts tools in different places - extract from all possible locations
        tools_input = kwargs.pop("tools", None)

        # Check optional_params (LiteLLM often puts tools here)
        optional_params = kwargs.get("optional_params", {})
        if tools_input is None and optional_params:
            tools_input = optional_params.get("tools", None)
            self._debug_log("Found tools in optional_params", tools_input is not None)

        # Check litellm_params
        litellm_params = kwargs.get("litellm_params", {})
        if tools_input is None and litellm_params:
            tools_input = litellm_params.get("tools", None)
            self._debug_log("Found tools in litellm_params", tools_input is not None)

        # Also check for tool_choice in same locations
        tool_choice_input = kwargs.pop("tool_choice", None)
        if tool_choice_input is None and optional_params:
            tool_choice_input = optional_params.get("tool_choice", None)
        if tool_choice_input is None and litellm_params:
            tool_choice_input = litellm_params.get("tool_choice", None)

        self._debug_log("Tools input before conversion", tools_input)

        # Convert messages
        system_message, anthropic_messages = self._convert_messages_to_anthropic(messages)

        # Convert tools
        tools = self._convert_tools_to_anthropic(tools_input)

        self._debug_log("Tools after conversion", tools)

        # Convert tool_choice
        tool_choice = self._convert_tool_choice_to_anthropic(tool_choice_input)

        # Prepare kwargs - filter out LiteLLM internal params
        clean_kwargs = self._prepare_anthropic_kwargs(kwargs)
        clean_kwargs.pop("stream", None)  # Ensure non-streaming

        # Set default max_tokens if not provided
        max_tokens = kwargs.get("max_tokens") or optional_params.get("max_tokens", 4096)
        clean_kwargs["max_tokens"] = max_tokens

        # Build request
        request_kwargs = {
            "model": actual_model,
            "messages": anthropic_messages,
            **clean_kwargs
        }

        if system_message:
            request_kwargs["system"] = system_message
        if tools:
            request_kwargs["tools"] = tools
        if tool_choice and tools:  # Only add tool_choice if we have tools
            request_kwargs["tool_choice"] = tool_choice

        # Debug logging
        self._debug_log("Final Anthropic request", {
            "model": actual_model,
            "tools_count": len(tools) if tools else 0,
            "tool_choice": tool_choice if tools else None,
            "messages_count": len(anthropic_messages),
            "has_system": system_message is not None
        })

        # Make request
        response = self.sync_client.messages.create(**request_kwargs)

        self._debug_log("Anthropic response", {
            "stop_reason": response.stop_reason,
            "content_types": [c.type for c in response.content]
        })

        return self._convert_anthropic_response(response, actual_model)

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Asynchronous non-streaming completion"""
        actual_model = self._extract_model(model)

        # LiteLLM puts tools in different places - extract from all possible locations
        tools_input = kwargs.pop("tools", None)
        optional_params = kwargs.get("optional_params", {})
        litellm_params = kwargs.get("litellm_params", {})

        if tools_input is None and optional_params:
            tools_input = optional_params.get("tools", None)
        if tools_input is None and litellm_params:
            tools_input = litellm_params.get("tools", None)

        tool_choice_input = kwargs.pop("tool_choice", None)
        if tool_choice_input is None and optional_params:
            tool_choice_input = optional_params.get("tool_choice", None)
        if tool_choice_input is None and litellm_params:
            tool_choice_input = litellm_params.get("tool_choice", None)

        self._debug_log("Async tools input", tools_input)

        # Convert messages
        system_message, anthropic_messages = self._convert_messages_to_anthropic(messages)

        # Convert tools
        tools = self._convert_tools_to_anthropic(tools_input)
        tool_choice = self._convert_tool_choice_to_anthropic(tool_choice_input)

        # Prepare kwargs
        clean_kwargs = self._prepare_anthropic_kwargs(kwargs)
        clean_kwargs.pop("stream", None)

        max_tokens = kwargs.get("max_tokens") or optional_params.get("max_tokens", 4096)
        clean_kwargs["max_tokens"] = max_tokens

        # Build request
        request_kwargs = {
            "model": actual_model,
            "messages": anthropic_messages,
            **clean_kwargs
        }

        if system_message:
            request_kwargs["system"] = system_message
        if tools:
            request_kwargs["tools"] = tools
        if tool_choice and tools:
            request_kwargs["tool_choice"] = tool_choice

        self._debug_log("Async completion request", {
            "model": actual_model,
            "tools_count": len(tools) if tools else 0,
            "tool_choice": tool_choice if tools else None
        })

        # Make request
        response = await self.async_client.messages.create(**request_kwargs)

        self._debug_log("Anthropic response", {
            "stop_reason": response.stop_reason,
            "content_types": [c.type for c in response.content]
        })

        return self._convert_anthropic_response(response, actual_model)

    # =========================================================================
    # STREAMING COMPLETION
    # =========================================================================

    def streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs
    ) -> Iterator[GenericStreamingChunk]:
        """Synchronous streaming completion"""
        actual_model = self._extract_model(model)

        # LiteLLM puts tools in different places - extract from all possible locations
        tools_input = kwargs.pop("tools", None)
        optional_params = kwargs.get("optional_params", {})
        litellm_params = kwargs.get("litellm_params", {})

        if tools_input is None and optional_params:
            tools_input = optional_params.get("tools", None)
        if tools_input is None and litellm_params:
            tools_input = litellm_params.get("tools", None)

        tool_choice_input = kwargs.pop("tool_choice", None)
        if tool_choice_input is None and optional_params:
            tool_choice_input = optional_params.get("tool_choice", None)
        if tool_choice_input is None and litellm_params:
            tool_choice_input = litellm_params.get("tool_choice", None)

        # Convert messages
        system_message, anthropic_messages = self._convert_messages_to_anthropic(messages)

        # Convert tools
        tools = self._convert_tools_to_anthropic(tools_input)
        tool_choice = self._convert_tool_choice_to_anthropic(tool_choice_input)

        # Prepare kwargs
        clean_kwargs = self._prepare_anthropic_kwargs(kwargs)
        clean_kwargs.pop("stream", None)

        max_tokens = kwargs.get("max_tokens") or optional_params.get("max_tokens", 4096)
        clean_kwargs["max_tokens"] = max_tokens

        # Build request
        request_kwargs = {
            "model": actual_model,
            "messages": anthropic_messages,
            "stream": True,
            **clean_kwargs
        }

        if system_message:
            request_kwargs["system"] = system_message
        if tools:
            request_kwargs["tools"] = tools
        if tool_choice and tools:
            request_kwargs["tool_choice"] = tool_choice

        self._debug_log("Streaming request", {
            "model": actual_model,
            "tools_count": len(tools) if tools else 0
        })

        # Initialize stream state
        state = StreamState(model=actual_model)

        # Make streaming request
        stream = self.sync_client.messages.create(**request_kwargs)

        for event in stream:
            chunk = self._convert_stream_event_to_chunk(event, state, actual_model)
            if chunk:
                yield chunk

    async def astreaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Asynchronous streaming completion — resilient to event loop closure."""
        actual_model = self._extract_model(model)

        # LiteLLM puts tools in different places - extract from all possible locations
        tools_input = kwargs.pop("tools", None)
        optional_params = kwargs.get("optional_params", {})
        litellm_params = kwargs.get("litellm_params", {})

        if tools_input is None and optional_params:
            tools_input = optional_params.get("tools", None)
        if tools_input is None and litellm_params:
            tools_input = litellm_params.get("tools", None)

        tool_choice_input = kwargs.pop("tool_choice", None)
        if tool_choice_input is None and optional_params:
            tool_choice_input = optional_params.get("tool_choice", None)
        if tool_choice_input is None and litellm_params:
            tool_choice_input = litellm_params.get("tool_choice", None)

        # Convert messages
        system_message, anthropic_messages = self._convert_messages_to_anthropic(messages)

        # Convert tools
        tools = self._convert_tools_to_anthropic(tools_input)
        tool_choice = self._convert_tool_choice_to_anthropic(tool_choice_input)

        # Prepare kwargs
        clean_kwargs = self._prepare_anthropic_kwargs(kwargs)
        clean_kwargs.pop("stream", None)

        max_tokens = kwargs.get("max_tokens") or optional_params.get("max_tokens", 4096)
        clean_kwargs["max_tokens"] = max_tokens

        # Build request
        request_kwargs = {
            "model": actual_model,
            "messages": anthropic_messages,
            "stream": True,
            **clean_kwargs
        }

        if system_message:
            request_kwargs["system"] = system_message
        if tools:
            request_kwargs["tools"] = tools
        if tool_choice and tools:
            request_kwargs["tool_choice"] = tool_choice

        # Initialize stream state
        state = StreamState(model=actual_model)

        # Make streaming request
        stream = await self.async_client.messages.create(**request_kwargs)

        # ── Resilient stream consumption ──────────────────────────
        # LiteLLM MidStreamFallback or sub-agent cancellation can
        # abandon this generator.  When httpx tries to close the
        # underlying SSL transport it calls loop.call_soon() — but
        # if the loop is already closing (Windows SelectorEventLoop
        # race), that raises RuntimeError("Event loop is closed").
        # We catch it, emit a final chunk from accumulated state,
        # and exit cleanly so litellm can proceed with its fallback.

        try:
            async for event in stream:
                chunk = self._convert_stream_event_to_chunk(event, state, actual_model)
                if chunk:
                    yield chunk
        except (RuntimeError, Exception) as e:
            err_str = str(e)
            if "Event loop is closed" in err_str or "different event loop" in err_str:
                self._debug_log(f"Stream interrupted (httpx cleanup): {err_str[:120]}")
                # Emit final chunk with whatever we accumulated
                yield self._build_generic_chunk(
                    text="",
                    tool_use=None,
                    is_finished=True,
                    finish_reason=state.finish_reason or "stop",
                    index=0,
                    usage=self._build_usage_dict(state),
                )
            else:
                raise
        finally:
            # Best-effort stream cleanup — never let it raise
            with _suppress_loop_closed():
                if hasattr(stream, 'close'):
                    try:
                        await stream.close()
                    except Exception:
                        pass


# =============================================================================
# TOOL FORMAT CONVERSION UTILITIES
# =============================================================================

def convert_tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert tools from any format to Anthropic format

    Accepts:
    - OpenAI format: {"type": "function", "function": {"name": ..., "parameters": ...}}
    - Flat format: {"name": ..., "parameters": ...}
    - Anthropic format: {"name": ..., "input_schema": ...} (passthrough)

    Returns Anthropic format:
    [{"name": ..., "description": ..., "input_schema": {...}}]

    Example:
        # All these work:
        tools = convert_tools_to_anthropic([
            {"type": "function", "function": {"name": "foo", "parameters": {...}}},
            {"name": "bar", "parameters": {...}},
            {"name": "baz", "input_schema": {...}},
        ])
    """
    if not tools:
        return []

    converted = []
    for tool in tools:
        # OpenAI nested format
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            converted.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })
        # Already Anthropic format
        elif "input_schema" in tool:
            converted.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool["input_schema"]
            })
        # Flat format with parameters
        elif "name" in tool and "parameters" in tool:
            converted.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool["parameters"]
            })
        # Minimal format
        elif "name" in tool:
            converted.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", {"type": "object", "properties": {}})
            })

    return converted


def convert_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert tools from any format to OpenAI format

    Accepts:
    - Anthropic format: {"name": ..., "input_schema": ...}
    - Flat format: {"name": ..., "parameters": ...}
    - OpenAI format: {"type": "function", "function": {...}} (passthrough)

    Returns OpenAI format:
    [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]

    Example:
        tools = convert_tools_to_openai([
            {"name": "get_weather", "input_schema": {...}},
            {"name": "search", "parameters": {...}},
        ])
    """
    if not tools:
        return []

    converted = []
    for tool in tools:
        # Already OpenAI format
        if tool.get("type") == "function" and "function" in tool:
            converted.append(tool)
        # Anthropic format (input_schema)
        elif "input_schema" in tool:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"]
                }
            })
        # Flat format (parameters)
        elif "name" in tool:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", tool.get("input_schema", {"type": "object", "properties": {}}))
                }
            })

    return converted


def normalize_tools(tools: List[Dict[str, Any]], target: str = "auto") -> List[Dict[str, Any]]:
    """
    Normalize tools to a consistent format

    Args:
        tools: List of tools in any format
        target: "openai", "anthropic", or "auto" (detect from first tool)

    Returns:
        Normalized tools list
    """
    if not tools:
        return []

    if target == "auto":
        # Detect format from first tool
        first = tools[0]
        if first.get("type") == "function":
            target = "openai"
        elif "input_schema" in first:
            target = "anthropic"
        else:
            target = "anthropic"  # Default to Anthropic

    if target == "openai":
        return convert_tools_to_openai(tools)
    else:
        return convert_tools_to_anthropic(tools)


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_zai_provider(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider_name: str = "zglm",
    debug: bool = False,
) -> ZAIProvider:
    """
    Register Z.AI provider with LiteLLM

    Args:
        api_key: Z.AI API key (defaults to ZAI_API_KEY env var)
        base_url: Z.AI base URL (defaults to ZAI_BASE_URL env var)
        provider_name: Provider prefix for model names (default: "zai")
        debug: Enable debug logging

    Returns:
        ZAIProvider instance

    Usage:
        setup_zai_provider()

        response = litellm.completion(
            model="zai/GLM-4.7",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """
    provider = ZAIProvider(api_key=api_key, base_url=base_url, debug=debug)

    if litellm.custom_provider_map is None:
        litellm.custom_provider_map = []

    litellm.custom_provider_map.append({
        "provider": provider_name,
        "custom_handler": provider,

    })

    print(f"✅ Registered Z.AI provider as '{provider_name}' with base_url: {base_url or ZAI_BASE_URL}")
    return provider



class StreamingToolCallAccumulator:
    """
    Accumulates tool calls from streaming chunks

    Usage with your existing _process_streaming_response pattern:

        accumulator = StreamingToolCallAccumulator()

        async for chunk in response:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            result += content

            # Accumulate tool calls
            accumulator.process_delta(delta)

        # Get final tool calls
        tool_calls = accumulator.get_tool_calls()
    """

    def __init__(self):
        self._tool_calls: Dict[int, Dict[str, Any]] = {}

    def process_delta(self, delta: Any) -> None:
        """Process a streaming delta and accumulate tool call info"""
        tool_calls = getattr(delta, "tool_calls", None)
        if not tool_calls:
            return

        for tc in tool_calls:
            idx = tc.index if hasattr(tc, "index") else tc.get("index", 0)

            if idx not in self._tool_calls:
                self._tool_calls[idx] = {
                    "id": "",
                    "type": "function",
                    "function": {
                        "name": "",
                        "arguments": ""
                    }
                }

            # Update ID if present
            tc_id = tc.id if hasattr(tc, "id") else tc.get("id")
            if tc_id:
                self._tool_calls[idx]["id"] = tc_id

            # Update function info
            func = tc.function if hasattr(tc, "function") else tc.get("function", {})
            if func:
                func_name = func.name if hasattr(func, "name") else func.get("name")
                func_args = func.arguments if hasattr(func, "arguments") else func.get("arguments")

                if func_name:
                    self._tool_calls[idx]["function"]["name"] = func_name
                if func_args:
                    self._tool_calls[idx]["function"]["arguments"] += func_args

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get accumulated tool calls as a list"""
        if not self._tool_calls:
            return []
        return [self._tool_calls[idx] for idx in sorted(self._tool_calls.keys())]

    def has_tool_calls(self) -> bool:
        """Check if any tool calls were accumulated"""
        return bool(self._tool_calls)

    def clear(self) -> None:
        """Reset accumulator"""
        self._tool_calls.clear()


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    print("🚀 Z.AI LiteLLM Provider Test\n")
    print("=" * 60)

    # Setup provider with debug enabled
    provider = setup_zai_provider(debug=False)

    # Test model
    TEST_MODEL = "zglm/GLM-4.7"

    # Tool definitions in different formats
    tools_openai = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    tools_flat = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    ]

    tools_anthropic = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    ]

    async def test():
            # === Test 0: Direct Provider Test (bypass LiteLLM) ===

        print("\n📝 Test 0: Direct Provider Test (bypass LiteLLM)")
        print("-" * 40)

        try:
            response = await provider.acompletion(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
                tools=tools_anthropic,
                tool_choice={"type": "auto"},
                max_tokens=200
            )
            msg = response.choices[0].message
            print(f"✅ Content: {msg.content}")
            print(f"   Tool calls: {msg.tool_calls}")
            print(f"   Finish reason: {response.choices[0].finish_reason}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

        # === Test 1: Simple completion ===
        print("\n📝 Test 1: Simple Completion (via LiteLLM)")
        print("-" * 40)

        try:
            response = await litellm.acompletion(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "Say 'Hello from Z.AI!' in exactly 5 words."}],
                max_tokens=50
            )
            print(f"✅ Response: {response.choices[0].message.content}")
            print(f"   Usage: {response.usage}")
        except Exception as e:
            print(f"❌ Error: {e}")

        # === Test 2: Tool Calling via LiteLLM ===
        print("\n📝 Test 2: Tool Calling via LiteLLM (OpenAI format)")
        print("-" * 40)

        try:
            response = await litellm.acompletion(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "What's the weather like in Munich?"}],
                tools=tools_openai,
                tool_choice="auto",
                max_tokens=200
            )

            msg = response.choices[0].message
            print(f"✅ Content: {msg.content}")
            print(f"   Tool calls: {msg.tool_calls}")
            print(f"   Finish reason: {response.choices[0].finish_reason}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

        # === Test 3: Direct Provider with different tool formats ===
        print("\n📝 Test 3: Direct Provider - Flat format tools")
        print("-" * 40)

        try:
            response = await provider.acompletion(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "What's the weather in Paris?"}],
                tools=tools_flat,
                tool_choice="auto",
                max_tokens=200
            )
            msg = response.choices[0].message
            print(f"✅ Content: {msg.content}")
            print(f"   Tool calls: {msg.tool_calls}")
        except Exception as e:
            print(f"❌ Error: {e}")

        # === Test 4: Tool conversion verification ===
        print("\n📝 Test 4: Tool Format Conversion")
        print("-" * 40)

        print("OpenAI format input:")
        print(f"  {json.dumps(tools_openai[0], indent=2)[:150]}...")
        converted = convert_tools_to_anthropic(tools_openai)
        print("\nConverted to Anthropic:")
        print(f"  {json.dumps(converted[0], indent=2)[:200]}...")

        # in FlowAgent test
        # === Test 5: Streaming with tool calls ===
        from toolboxv2 import get_app
        from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
        app = get_app()
        isaa = app.get_mod("isaa")
        test_agent: FlowAgent = await isaa.get_agent("self-test")
        test_agent.amd.fast_llm_model = TEST_MODEL
        test_agent.amd.complex_llm_model = TEST_MODEL
        test_agent.add_tool(lambda location, unit="celsius": f"Current weather in {location} is 25°C", "get_weather", "Get the current weather for a location")

        print("\n📝 Test 5: Non-Streaming with tool calls")
        response = await test_agent.a_run_llm_completion(
            messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
            tools=tools_openai,
            tool_choice="auto",
            max_tokens=200,
            stream=False,
            get_response_message=True,
            with_context=False,
        )
        print(f"✅ Content: {response}")

        print("\n📝 Test 5.5: Streaming with tool calls")
        response = await test_agent.a_run_llm_completion(
            messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
            tools=tools_openai,
            tool_choice="auto",
            max_tokens=200,
            stream=True,
            get_response_message=True,
            with_context=False,
        )
        print(f"✅ Content: {response}")

        # === Test 6: Streaming with tool calls ===

        print("\n📝 Test 5: Streaming with tool calls")
        async for chunk in test_agent.a_stream_verbose("What's the weather in Berlin? nutze load tools um tas tool zu laden und danach das tool zu nutzen."):
            print(chunk, end="")

        # === Test 7: Agent run with tool calls ===
        print("\n📝 Test 6: Agent run with tool calls")
        response = await test_agent.a_run(
            query="What's the weather in Berlin?",
            session_id="tool-test-custom-glm-provider",
            user_id="default"
        )

        print(f"✅ Content: {response}")

        print("\n" + "=" * 60)
        print("✅ All tests completed!")

    asyncio.run(test())

    #for model in ["zglm/GLM-4.7", "zglm/GLM-4.7-flash",  "zglm/glm-4.6", "zglm/glm-4.5", "zglm/glm-4.6v"]:
    #    try:
    #        start_time = time.perf_counter()
    #        response = provider.completion(
    #            model=model,
    #            messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
    #            tools=tools_anthropic,
    #            tool_choice={"type": "auto"},
    #            max_tokens=200
    #        )
    #        msg = response.choices[0].message
    #        print(f"time: {time.perf_counter() - start_time:.2f}")
    #        print(f"✅ Content {model}: {msg.content}")
    #        print(f"   Tool calls: {msg.tool_calls}")
    #        print(f"   Finish reason: {response.choices[0].finish_reason}")
    #    except Exception as e:
    #        print(f"❌ Error: {model} {e}")
    #        import traceback
    #        traceback.print_exc()
#
#
