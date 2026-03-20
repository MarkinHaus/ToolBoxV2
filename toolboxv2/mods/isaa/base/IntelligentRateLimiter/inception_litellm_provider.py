"""
InceptionLabs LiteLLM Provider - Mercury 2

Async-only provider for chat completions with streaming (normal + diffusion mode).
Sends all required default fields that Inception's API expects.

Usage:
    from inception_litellm_provider import setup_inception_provider
    import litellm

    setup_inception_provider()

    # Async completion
    response = await litellm.acompletion(
        model="inception/mercury-2",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    # Async streaming
    response = await litellm.acompletion(
        model="inception/mercury-2",
        messages=[...],
        stream=True,
    )

    # Diffusion mode streaming
    response = await litellm.acompletion(
        model="inception/mercury-2",
        messages=[...],
        stream=True,
        diffusing=True,
    )
"""

import os
import json
import time
import uuid
import asyncio
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Iterator,
    AsyncIterator,
)
from dataclasses import dataclass, field

import litellm
litellm.suppress_debug_info = True
from litellm import CustomLLM
from litellm.types.utils import (
    GenericStreamingChunk,
    ModelResponse,
    Choices,
    Message,
    Usage,
)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️ httpx not installed. Install with: pip install httpx")


# === Configuration ===
INCEPTION_API_KEY = os.getenv("INCEPTION_API_KEY", "")
INCEPTION_BASE_URL = os.getenv("INCEPTION_BASE_URL", "https://api.inceptionlabs.ai")

# =====================================================================
# DEFAULT FIELDS — the API expects these in every chat request
# Derived from successful request logs on the Inception dashboard
# =====================================================================
CHAT_DEFAULTS = {
    "mode": "chat",
    "diffusing": False,
    "realtime": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "repetition_penalty": 1,
    "temperature": 0.75,
    "top_k": -1,
    "top_p": 1,
    "reasoning_summary": True,
    "reasoning_summary_wait": False,
}


@dataclass
class StreamState:
    """Tracks state during SSE stream processing"""
    content: str = ""
    tool_calls: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: Optional[str] = None
    model: str = ""


class InceptionProvider(CustomLLM):
    """
    LiteLLM Custom Provider for InceptionLabs Mercury 2.

    Async completion + streaming (normal & diffusion).
    Sends all required default fields the API expects.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.api_key = api_key or INCEPTION_API_KEY
        self.base_url = (base_url or INCEPTION_BASE_URL).rstrip("/")
        self.debug = debug

        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx required. Install with: pip install httpx")

        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._async_client_loop_id: Optional[int] = None

    def _log(self, msg: str, data: Any = None):
        if self.debug:
            print(f"[INCEPTION] {msg}")
            if data is not None:
                try:
                    print(f"  → {json.dumps(data, indent=2, default=str)[:1000]}")
                except Exception:
                    print(f"  → {str(data)[:1000]}")

    # ── Client management ─────────────────────────────────────────

    @property
    def sync_client(self) -> "httpx.Client":
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        return self._sync_client

    @property
    def async_client(self) -> "httpx.AsyncClient":
        current_loop_id: Optional[int] = None
        try:
            current_loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            pass

        needs_rebuild = (
            self._async_client is None
            or (current_loop_id is not None and self._async_client_loop_id != current_loop_id)
        )
        if needs_rebuild:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
            self._async_client_loop_id = current_loop_id
        return self._async_client

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # ── Helpers ────────────────────────────────────────────────────

    def _extract_model(self, model: str) -> str:
        """Remove provider prefix: 'inception/mercury-2' → 'mercury-2'"""
        return model.split("/", 1)[-1] if "/" in model else model

    @staticmethod
    def _generate_id() -> str:
        return f"chatcmpl-{uuid.uuid4().hex[:29]}"

    # ── Request body builder ──────────────────────────────────────

    def _build_chat_body(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build request body with ALL required default fields.

        The Inception API returns 403/400 if certain fields are missing.
        We start from CHAT_DEFAULTS and overlay user params.
        """
        # Collect params from all LiteLLM sources
        optional_params = kwargs.get("optional_params", {})
        litellm_params = kwargs.get("litellm_params", {})
        all_params = {**kwargs, **optional_params, **litellm_params}

        # Start with defaults
        body: Dict[str, Any] = {**CHAT_DEFAULTS}

        # Required fields
        body["model"] = model
        body["messages"] = messages
        body["stream"] = stream

        # Max tokens
        body["max_tokens"] = all_params.get("max_tokens", 8192)

        # Override defaults with user-provided values
        overridable = {
            "temperature", "top_p", "top_k", "frequency_penalty",
            "presence_penalty", "repetition_penalty", "stop",
            "reasoning_effort", "reasoning_summary", "reasoning_summary_wait",
            "diffusing", "realtime", "mode",
            "tools", "tool_choice", "response_format",
            "stream_options",
        }
        for key in overridable:
            val = all_params.get(key)
            if val is not None:
                body[key] = val

        # Diffusing requires stream=true
        if body.get("diffusing") and not body.get("stream"):
            body["stream"] = True

        self._log("Request body", {k: v for k, v in body.items() if k != "messages"})
        return body

    # ── Response parsing ──────────────────────────────────────────

    def _parse_response(self, data: Dict[str, Any], model: str) -> ModelResponse:
        """Parse Inception JSON response → LiteLLM ModelResponse."""
        choices_data = data.get("choices", [])
        choices = []

        for i, c in enumerate(choices_data):
            msg_data = c.get("message", {})
            tool_calls_raw = msg_data.get("tool_calls")

            tool_calls = None
            if tool_calls_raw:
                tool_calls = []
                for tc in tool_calls_raw:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    tool_calls.append({
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": args if isinstance(args, str) else json.dumps(args),
                        },
                    })

            message = Message(
                content=msg_data.get("content"),
                role=msg_data.get("role", "assistant"),
                tool_calls=tool_calls if tool_calls else None,
                function_call=None,
            )

            choices.append(Choices(
                finish_reason=c.get("finish_reason", "stop"),
                index=i,
                message=message,
            ))

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return ModelResponse(
            id=data.get("id", self._generate_id()),
            created=data.get("created", int(time.time())),
            model=model,
            object="chat.completion",
            choices=choices,
            usage=usage,
        )

    # ── SSE stream parsing ────────────────────────────────────────

    def _parse_sse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single SSE data line → dict or None."""
        line = line.strip()
        if not line or not line.startswith("data: "):
            return None
        payload = line[6:]
        if payload == "[DONE]":
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def _sse_chunk_to_generic(
        self,
        data: Dict[str, Any],
        state: StreamState,
    ) -> Optional[GenericStreamingChunk]:
        """Convert parsed SSE chunk → GenericStreamingChunk."""
        usage = data.get("usage")
        if usage:
            state.input_tokens = usage.get("prompt_tokens", state.input_tokens)
            state.output_tokens = usage.get("completion_tokens", state.output_tokens)

        if not data.get("choices"):
            return None

        choice = data["choices"][0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        text = delta.get("content") or ""

        if text:
            state.content += text
        if finish_reason:
            state.finish_reason = finish_reason

        # Tool calls in streaming
        tool_use = None
        tc_deltas = delta.get("tool_calls")
        if tc_deltas:
            for tc in tc_deltas:
                idx = tc.get("index", 0)
                func = tc.get("function", {})

                if idx not in state.tool_calls:
                    state.tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": "",
                    }
                if tc.get("id"):
                    state.tool_calls[idx]["id"] = tc["id"]
                if func.get("name"):
                    state.tool_calls[idx]["name"] = func["name"]
                if func.get("arguments"):
                    state.tool_calls[idx]["arguments"] += func["arguments"]

                tool_use = {
                    "id": tc.get("id") or None,
                    "type": "function",
                    "function": {
                        "name": func.get("name") or None,
                        "arguments": func.get("arguments", ""),
                    },
                    "index": idx,
                }

        chunk: GenericStreamingChunk = {
            "text": text,
            "is_finished": finish_reason is not None,
            "finish_reason": finish_reason,
            "index": 0,
            "usage": {
                "prompt_tokens": state.input_tokens,
                "completion_tokens": state.output_tokens,
                "total_tokens": state.input_tokens + state.output_tokens,
            },
        }
        if tool_use is not None:
            chunk["tool_use"] = tool_use

        return chunk

    # =========================================================================
    # SYNC COMPLETION (required by LiteLLM even if we focus on async)
    # =========================================================================

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ) -> ModelResponse:
        actual_model = self._extract_model(model)
        body = self._build_chat_body(actual_model, messages, stream=False, **kwargs)

        self._log("POST /v1/chat/completions (sync)")
        resp = self.sync_client.post("/v1/chat/completions", json=body)

        if resp.status_code != 200:
            self._log(f"ERROR {resp.status_code}", resp.text[:500])
        resp.raise_for_status()

        return self._parse_response(resp.json(), actual_model)

    # =========================================================================
    # ASYNC COMPLETION
    # =========================================================================

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ) -> ModelResponse:
        actual_model = self._extract_model(model)
        body = self._build_chat_body(actual_model, messages, stream=False, **kwargs)

        self._log("POST /v1/chat/completions (async)")
        resp = await self.async_client.post("/v1/chat/completions", json=body)

        if resp.status_code != 200:
            self._log(f"ERROR {resp.status_code}", resp.text[:500])
        resp.raise_for_status()

        return self._parse_response(resp.json(), actual_model)

    # =========================================================================
    # SYNC STREAMING
    # =========================================================================

    def streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ) -> Iterator[GenericStreamingChunk]:
        actual_model = self._extract_model(model)
        body = self._build_chat_body(actual_model, messages, stream=True, **kwargs)
        body.setdefault("stream_options", {"include_usage": True})

        self._log("POST /v1/chat/completions (sync stream)")
        state = StreamState(model=actual_model)

        with self.sync_client.stream("POST", "/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                error_body = resp.read().decode()
                self._log(f"ERROR {resp.status_code}", error_body[:500])
            resp.raise_for_status()
            for line in resp.iter_lines():
                parsed = self._parse_sse_line(line)
                if parsed is None:
                    continue
                chunk = self._sse_chunk_to_generic(parsed, state)
                if chunk:
                    yield chunk

    # =========================================================================
    # ASYNC STREAMING (normal + diffusion mode)
    # =========================================================================

    async def astreaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[GenericStreamingChunk]:
        actual_model = self._extract_model(model)
        body = self._build_chat_body(actual_model, messages, stream=True, **kwargs)
        body.setdefault("stream_options", {"include_usage": True})

        self._log("POST /v1/chat/completions (async stream)")
        state = StreamState(model=actual_model)

        try:
            async with self.async_client.stream("POST", "/v1/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    error_body = (await resp.aread()).decode()
                    self._log(f"ERROR {resp.status_code}", error_body[:500])
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    parsed = self._parse_sse_line(line)
                    if parsed is None:
                        continue
                    chunk = self._sse_chunk_to_generic(parsed, state)
                    if chunk:
                        yield chunk
        except (RuntimeError, Exception) as e:
            err = str(e)
            if "Event loop is closed" in err or "different event loop" in err:
                self._log(f"Stream interrupted: {err[:120]}")
                yield {
                    "text": "",
                    "is_finished": True,
                    "finish_reason": state.finish_reason or "stop",
                    "index": 0,
                    "usage": {
                        "prompt_tokens": state.input_tokens,
                        "completion_tokens": state.output_tokens,
                        "total_tokens": state.input_tokens + state.output_tokens,
                    },
                }
            else:
                raise


# =============================================================================
# STREAMING TOOL CALL ACCUMULATOR
# =============================================================================

class StreamingToolCallAccumulator:
    """Accumulates tool calls from streaming chunks."""

    def __init__(self):
        self._tool_calls: Dict[int, Dict[str, Any]] = {}

    def process_delta(self, delta: Any) -> None:
        tool_calls = getattr(delta, "tool_calls", None)
        if not tool_calls:
            return
        for tc in tool_calls:
            idx = tc.index if hasattr(tc, "index") else tc.get("index", 0)
            if idx not in self._tool_calls:
                self._tool_calls[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            tc_id = tc.id if hasattr(tc, "id") else tc.get("id")
            if tc_id:
                self._tool_calls[idx]["id"] = tc_id
            func = tc.function if hasattr(tc, "function") else tc.get("function", {})
            if func:
                name = func.name if hasattr(func, "name") else func.get("name")
                args = func.arguments if hasattr(func, "arguments") else func.get("arguments")
                if name:
                    self._tool_calls[idx]["function"]["name"] = name
                if args:
                    self._tool_calls[idx]["function"]["arguments"] += args

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        return [self._tool_calls[i] for i in sorted(self._tool_calls)] if self._tool_calls else []

    def has_tool_calls(self) -> bool:
        return bool(self._tool_calls)

    def clear(self) -> None:
        self._tool_calls.clear()


# =============================================================================
# SETUP
# =============================================================================

def setup_inception_provider(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider_name: str = "inception",
    debug: bool = False,
) -> InceptionProvider:
    """
    Register InceptionLabs provider with LiteLLM.

    Usage:
        setup_inception_provider()

        # Async completion
        response = await litellm.acompletion(
            model="inception/mercury-2",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        # Async streaming
        response = await litellm.acompletion(
            model="inception/mercury-2",
            messages=[...],
            stream=True,
        )

        # Diffusion streaming
        response = await litellm.acompletion(
            model="inception/mercury-2",
            messages=[...],
            stream=True,
            diffusing=True,
        )
    """
    provider = InceptionProvider(api_key=api_key, base_url=base_url, debug=debug)

    if litellm.custom_provider_map is None:
        litellm.custom_provider_map = []

    litellm.custom_provider_map.append({
        "provider": provider_name,
        "custom_handler": provider,
    })

    base = base_url or INCEPTION_BASE_URL
    print(f"✅ Registered InceptionLabs provider as '{provider_name}' → {base}")
    return provider


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio as _asyncio
    from dotenv import load_dotenv
    load_dotenv()

    print("🚀 InceptionLabs LiteLLM Provider Test\n" + "=" * 60)

    provider = setup_inception_provider(debug=True)
    TEST_MODEL = "inception/mercury-2"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    async def run_tests():
        # Test 1: Async completion
        print("\n📝 Test 1: Async Completion")
        print("-" * 40)
        try:
            resp = await provider.acompletion(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "Say hello in 5 words."}],
                max_tokens=50,
            )
            print(f"✅ {resp.choices[0].message.content}")
            print(f"   Usage: {resp.usage}")
        except Exception as e:
            print(f"❌ {e}")

        # Test 2: Async streaming (normal)
        print("\n📝 Test 2: Async Streaming (normal)")
        print("-" * 40)
        try:
            collected = ""
            async for chunk in provider.astreaming(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "Count to 5."}],
                max_tokens=100,
            ):
                text = chunk.get("text", "")
                if text:
                    print(text, end="", flush=True)
                    collected += text
            print(f"\n✅ Streaming done ({len(collected)} chars)")
        except Exception as e:
            print(f"❌ {e}")

        # Test 3: Async streaming (diffusion mode)
        print("\n📝 Test 3: Async Streaming (DIFFUSION mode)")
        print("-" * 40)
        try:
            collected = ""
            async for chunk in provider.astreaming(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "What is 2+2? Explain briefly."}],
                max_tokens=200,
                diffusing=True,
            ):
                text = chunk.get("text", "")
                if text:
                    # In diffusion mode, delta.content replaces the full content
                    collected = text
                    print(f"\r{text[:80]}", end="", flush=True)
            print(f"\n✅ Diffusion done. Final: {collected[:200]}")
        except Exception as e:
            print(f"❌ {e}")

        # Test 4: Async completion with tools
        print("\n📝 Test 4: Async Completion with Tools")
        print("-" * 40)
        try:
            resp = await provider.acompletion(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
                tools=tools,
                max_tokens=200,
            )
            msg = resp.choices[0].message
            print(f"✅ Content: {msg.content}")
            print(f"   Tool calls: {msg.tool_calls}")
            print(f"   Finish: {resp.choices[0].finish_reason}")
        except Exception as e:
            print(f"❌ {e}")

        # Test 5: Async completion with reasoning_effort=instant
        print("\n📝 Test 5: Instant Mode")
        print("-" * 40)
        try:
            resp = await provider.acompletion(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "What is 2+2?"}],
                reasoning_effort="instant",
                max_tokens=50,
            )
            print(f"✅ {resp.choices[0].message.content}")
        except Exception as e:
            print(f"❌ {e}")

        # Test 6: Async completion with structured output
        print("\n📝 Test 6: Structured Output")
        print("-" * 40)
        try:
            resp = await provider.acompletion(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "Analyze the sentiment: 'This is great!'"}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Sentiment",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                                "confidence": {"type": "number"},
                            },
                            "required": ["sentiment", "confidence"],
                        },
                    },
                },
                max_tokens=200,
            )
            print(f"✅ {resp.choices[0].message.content}")
        except Exception as e:
            print(f"❌ {e}")

        # Test 7: Via LiteLLM routing (async)
        print("\n📝 Test 7: Via LiteLLM acompletion")
        print("-" * 40)
        try:
            resp = await litellm.acompletion(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "Say hi in 3 words."}],
                max_tokens=50,
            )
            print(f"✅ {resp.choices[0].message.content}")
        except Exception as e:
            print(f"❌ {e}")

        # Test 8: Via LiteLLM routing (async streaming)
        print("\n📝 Test 8: Via LiteLLM async streaming")
        print("-" * 40)
        try:
            resp = await litellm.acompletion(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "Count to 3."}],
                stream=True,
                max_tokens=100,
            )
            async for chunk in resp:
                c = chunk.choices[0].delta.content
                if c:
                    print(c, end="", flush=True)
            print("\n✅ LiteLLM streaming done")
        except Exception as e:
            print(f"❌ {e}")

        print("\n" + "=" * 60)
        print("✅ All tests completed!")

    _asyncio.run(run_tests())
