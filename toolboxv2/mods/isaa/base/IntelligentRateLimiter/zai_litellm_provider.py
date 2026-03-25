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
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ openai package not installed. Install with: pip install openai")
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
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4/")


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

        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai package required. Install with: pip install openai")

        self._sync_client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None

    def _debug_log(self, msg: str, data: Any = None):
        """Debug logging helper"""
        if self.debug:
            print(f"[ZAI DEBUG] {msg}")
            if data is not None:
                print(f"  → {json.dumps(data, indent=2, default=str)[:500]}")


    @property
    def sync_client(self) -> OpenAI:
        """Lazy initialization of sync client"""
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._sync_client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Lazy initialization of async client — resilient to event loop closure."""
        current_loop_id: Optional[int] = None
        try:
            current_loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            pass

        needs_rebuild = (
            self._async_client is None
            or (current_loop_id is not None and getattr(self, "_async_client_loop_id", None) != current_loop_id)
        )

        if needs_rebuild:
            self._async_client = AsyncOpenAI(
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

    def _parse_data_url(self, data_url: str) -> tuple[str, str]:
        """Parse data URL into media_type and base64 data"""
        # Format: data:image/png;base64,iVBORw0...
        if not data_url.startswith("data:"):
            return "image/png", data_url

        header, data = data_url.split(",", 1)
        media_type = header.split(";")[0].replace("data:", "")
        return media_type, data

    def _convert_openai_chunk_to_generic(self, chunk: Any) -> GenericStreamingChunk:
        """Convert native OpenAI stream chunk to GenericStreamingChunk for LiteLLM"""
        chunk_dict = chunk.model_dump()
        choices = chunk_dict.get("choices", [])
        choice = choices[0] if choices else {}
        delta = choice.get("delta", {})

        # Tool Call sicher parsen
        tool_use = None
        if delta.get("tool_calls"):
            tc = delta["tool_calls"][0]
            tool_use = {
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": tc.get("function", {}).get("name"),
                    "arguments": tc.get("function", {}).get("arguments", "")
                },
                "index": tc.get("index", 0)
            }

        return {
            "text": delta.get("content") or "",
            "is_finished": choice.get("finish_reason") is not None,
            "finish_reason": choice.get("finish_reason"),
            "index": choice.get("index", 0),
            "usage": chunk_dict.get("usage") or {},
            "tool_use": tool_use
        }

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

    def _prepare_openai_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extrahiert Tools sicher und behält NUR für das OpenAI SDK gültige Parameter."""

        # 1. Strikte Liste aller Parameter, die OpenAI in chat.completions.create() akzeptiert
        allowed_openai_params = {
            "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
            "max_tokens", "n", "presence_penalty", "response_format",
            "seed", "stop", "temperature", "top_p", "tools",
            "tool_choice", "user"
        }

        clean_kwargs = {}

        # 2. Nur erlaubte Parameter aus den direkten kwargs übernehmen
        for k, v in kwargs.items():
            if k in allowed_openai_params and v is not None:
                clean_kwargs[k] = v

        # 3. LiteLLM packt oft Parameter in 'optional_params'. Diese auch filtern:
        opt_params = kwargs.get("optional_params", {})
        if isinstance(opt_params, dict):
            for k, v in opt_params.items():
                if k in allowed_openai_params and v is not None:
                    clean_kwargs[k] = v

        # 4. Tools standardisieren (übersetzt Anthropic/Flat Formate in sauberes OpenAI Format)
        tools_input = clean_kwargs.get("tools")
        if tools_input:
            clean_kwargs["tools"] = convert_tools_to_openai(tools_input)

        # 5. Tool Choice standardisieren
        tool_choice = clean_kwargs.get("tool_choice")
        if tool_choice:
            # Fallback falls Agenten fälschlicherweise Anthropic {"type": "auto"} übergeben
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "auto":
                clean_kwargs["tool_choice"] = "auto"
            else:
                clean_kwargs["tool_choice"] = tool_choice

        # 6. Default Tokens setzen, falls der Agent keine spezifiziert hat
        if "max_tokens" not in clean_kwargs:
            clean_kwargs["max_tokens"] = 4096

        return clean_kwargs

    # =========================================================================
    # NON-STREAMING COMPLETION
    # =========================================================================

    def completion(self, model: str, messages: List[Dict[str, Any]], api_base: Optional[str] = None,
                   custom_llm_provider: Optional[str] = None, **kwargs) -> ModelResponse:
        actual_model = self._extract_model(model)
        request_kwargs = self._prepare_openai_kwargs(kwargs)

        self._debug_log("Sync Request", {"model": actual_model, "tools": bool(request_kwargs.get("tools"))})

        response = self.sync_client.chat.completions.create(
            model=actual_model,
            messages=messages,
            stream=False,
            **request_kwargs
        )
        return ModelResponse(**response.model_dump())

    async def acompletion(self, model: str, messages: List[Dict[str, Any]], api_base: Optional[str] = None,
                          custom_llm_provider: Optional[str] = None, **kwargs) -> ModelResponse:
        actual_model = self._extract_model(model)
        request_kwargs = self._prepare_openai_kwargs(kwargs)

        response = await self.async_client.chat.completions.create(
            model=actual_model,
            messages=messages,
            stream=False,
            **request_kwargs
        )
        return ModelResponse(**response.model_dump())

    def streaming(self, model: str, messages: List[Dict[str, Any]], api_base: Optional[str] = None,
                  custom_llm_provider: Optional[str] = None, **kwargs) -> Iterator[GenericStreamingChunk]:
        actual_model = self._extract_model(model)
        request_kwargs = self._prepare_openai_kwargs(kwargs)

        stream = self.sync_client.chat.completions.create(
            model=actual_model,
            messages=messages,
            stream=True,
            **request_kwargs
        )
        for chunk in stream:
            yield self._convert_openai_chunk_to_generic(chunk)

    async def astreaming(self, model: str, messages: List[Dict[str, Any]], api_base: Optional[str] = None,
                         custom_llm_provider: Optional[str] = None, **kwargs) -> AsyncIterator[
        GenericStreamingChunk]:
        actual_model = self._extract_model(model)
        request_kwargs = self._prepare_openai_kwargs(kwargs)

        stream = await self.async_client.chat.completions.create(
            model=actual_model,
            messages=messages,
            stream=True,
            **request_kwargs
        )

        try:
            async for chunk in stream:
                yield self._convert_openai_chunk_to_generic(chunk)
        except (RuntimeError, Exception) as e:
            err_str = str(e)
            if "Event loop is closed" in err_str or "different event loop" in err_str:
                self._debug_log(f"Stream interrupted (loop closed): {err_str[:120]}")
            else:
                raise


# =============================================================================
# TOOL FORMAT CONVERSION UTILITIES
# =============================================================================

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

    print(f"✅ Registered{' Z.AI' if provider_name == 'zglm' else ''} provider as '{provider_name}' with base_url: {base_url or ZAI_BASE_URL}")
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
    provider = setup_zai_provider(debug=True)

    # Test model
    TEST_MODEL = "zglm/GLM-5"

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
        converted = tools_openai
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
