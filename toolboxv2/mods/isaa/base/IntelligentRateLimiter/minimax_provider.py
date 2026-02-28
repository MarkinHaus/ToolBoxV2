"""
MiniMax Custom LiteLLM Provider (OpenAI-Compatible Style)
=========================================================

Uses https://api.minimax.io/v1/chat/completions (OpenAI-compatible endpoint).
Tested and confirmed working for: normal calls, streaming, tool use.

Usage:
    from minimax_provider import register_minimax
    register_minimax()

    import litellm
    resp = litellm.completion(model="minimax/MiniMax-M2.5", messages=[...])
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

import httpx
from litellm import CustomLLM
from litellm.types.utils import (
    Choices,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)

# ─── Config ──────────────────────────────────────────────────────────────────

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimax.io/v1/chat/completions"
DEFAULT_MAX_TOKENS = 4096
HTTP_TIMEOUT = 120.0


# ─── Stream State ────────────────────────────────────────────────────────────

@dataclass
class StreamState:
    """Accumulates state across SSE events during streaming."""
    model: str = ""
    response_id: str = ""
    content: str = ""
    finish_reason: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: Dict[int, Dict[str, Any]] = field(default_factory=dict)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gen_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:29]}"


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from MiniMax output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _extract_think_content(text: str) -> Optional[str]:
    """Extract content from <think>...</think> blocks (for reasoning_content)."""
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if matches:
        return "\n".join(m.strip() for m in matches if m.strip())
    return None


# ─── Provider ────────────────────────────────────────────────────────────────

class MiniMaxProvider(CustomLLM):
    """
    LiteLLM CustomLLM provider for MiniMax via OpenAI-compatible endpoint.

    Async-safe: recreates httpx.AsyncClient when the event loop changes,
    preventing 'Event loop is closed' errors in sub-agent / parallel usage.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.api_key = api_key or MINIMAX_API_KEY
        self.base_url = base_url or MINIMAX_BASE_URL
        self.debug = debug

        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._async_client_loop_id: Optional[int] = None

    def _log(self, msg: str, data: Any = None):
        if self.debug:
            print(f"[MiniMax] {msg}")
            if data is not None:
                print(f"  → {json.dumps(data, indent=2, default=str)[:800]}")

    # ── HTTP Clients ─────────────────────────────────────────────────────

    @property
    def sync_http(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=HTTP_TIMEOUT)
        return self._sync_client

    @property
    def async_http(self) -> httpx.AsyncClient:
        cur_loop_id: Optional[int] = None
        try:
            cur_loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            pass
        if (
            self._async_client is None
            or (cur_loop_id is not None
                and self._async_client_loop_id != cur_loop_id)
        ):
            self._async_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
            self._async_client_loop_id = cur_loop_id
        return self._async_client

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _check_response(resp: httpx.Response) -> None:
        """Raise with full error body so we can debug MiniMax 400s."""
        if resp.is_success:
            return
        try:
            body = resp.json()
        except Exception:
            body = resp.text[:500]
        raise httpx.HTTPStatusError(
            f"MiniMax API error {resp.status_code}: {json.dumps(body, ensure_ascii=False, default=str)[:500]}",
            request=resp.request,
            response=resp,
        )

    @staticmethod
    async def _check_response_async(resp: httpx.Response) -> None:
        """Async version — reads body before raising."""
        if resp.is_success:
            return
        # For streaming responses, read what we can
        try:
            await resp.aread()
            body = resp.json()
        except Exception:
            body = resp.text[:500] if resp.text else "(empty)"
        raise httpx.HTTPStatusError(
            f"MiniMax API error {resp.status_code}: {json.dumps(body, ensure_ascii=False, default=str)[:500]}",
            request=resp.request,
            response=resp,
        )

    # =====================================================================
    # PAYLOAD
    # =====================================================================

    _ALLOWED = frozenset({
        "model", "messages", "max_tokens", "temperature", "top_p",
        "stream", "stop", "tools", "tool_choice", "n",
        "frequency_penalty", "presence_penalty",
    })

    # Known MiniMax model names (case-sensitive!)
    _MODEL_MAP = {
        "minimax-m2.5": "MiniMax-M2.5",
        "minimax-m2.5-highspeed": "MiniMax-M2.5-highspeed",
        "minimax-m2.1": "MiniMax-M2.1",
        "minimax-m2": "MiniMax-M2",
        "codex-minimax-m2.5": "MiniMax-M2.5",
        "codex-minimax-m2.1": "MiniMax-M2.1",
    }

    def _extract_model(self, model: str) -> str:
        """Strip provider prefix and normalize to MiniMax's expected casing."""
        name = model.split("/", 1)[-1] if "/" in model else model
        # Lookup case-insensitive
        return self._MODEL_MAP.get(name.lower(), name)

    def _sanitize_messages(self, messages: list, model: str = "") -> List[Dict[str, Any]]:
        """
        Sanitize messages for MiniMax OpenAI-compat endpoint.

        Key fix: MiniMax-M2.1/M2 do NOT support role="system" in messages.
        We merge system messages into the first user message as a prefix.
        M2.5 supports system, but merging works universally — so we always merge.
        """
        system_parts: List[str] = []
        non_system: List[Dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            content = msg.get("content")

            # Normalize content to str
            if content is None:
                content = ""
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts) if text_parts else ""
            content = str(content)

            if role == "system":
                # Collect all system messages to merge later
                if content.strip():
                    system_parts.append(content.strip())
                continue

            if role not in ("user", "assistant", "tool"):
                continue

            out: Dict[str, Any] = {"role": role, "content": content}

            # Preserve tool-related fields
            if "tool_calls" in msg and msg["tool_calls"]:
                out["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                out["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                out["name"] = msg["name"]

            non_system.append(out)

        # Merge system into first user message
        if system_parts and non_system:
            system_text = "\n\n".join(system_parts)
            # Find first user message to prepend system text
            for i, m in enumerate(non_system):
                if m["role"] == "user":
                    non_system[i]["content"] = (
                        f"[System Instructions]\n{system_text}\n\n[User Message]\n{m['content']}"
                    )
                    break
            else:
                # No user message found — insert as user message at start
                non_system.insert(0, {
                    "role": "user",
                    "content": f"[System Instructions]\n{system_text}",
                })

        # MiniMax requires at least one message
        if not non_system:
            non_system = [{"role": "user", "content": "Hello"}]

        return non_system

    def _build_payload(
        self,
        model: str,
        messages: list,
        stream: bool,
        optional_params: dict,
        litellm_params: Optional[dict] = None,
    ) -> Dict[str, Any]:
        op = optional_params or {}
        lp = litellm_params or {}

        def _get(key, default=None):
            v = op.get(key)
            if v is None:
                v = lp.get(key)
            return v if v is not None else default

        # Sanitize messages for MiniMax compatibility
        clean_messages = self._sanitize_messages(messages, model)

        p: Dict[str, Any] = {
            "model": self._extract_model(model),
            "messages": clean_messages,
            "stream": stream,
            "max_tokens": _get("max_tokens", DEFAULT_MAX_TOKENS),
        }

        for k in ("temperature", "top_p", "n", "frequency_penalty", "presence_penalty"):
            v = _get(k)
            if v is not None:
                p[k] = v

        stop = _get("stop")
        if stop is not None:
            p["stop"] = [stop] if isinstance(stop, str) else stop

        tools = _get("tools")
        if tools:
            p["tools"] = self._validate_tools(tools)
            tc = _get("tool_choice")
            if tc is not None:
                p["tool_choice"] = tc

        return {k: v for k, v in p.items() if k in self._ALLOWED}

    def _validate_tools(self, tools: list) -> list:
        out = []
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                out.append(t)
            else:
                out.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters") or t.get("input_schema")
                                      or {"type": "object", "properties": {}},
                    },
                })
        return out

    # =====================================================================
    # RESPONSE PARSING
    # =====================================================================

    def _parse_response(self, body: dict, model: str) -> ModelResponse:
        choices = []
        for i, c in enumerate(body.get("choices", [])):
            m = c.get("message", {})
            raw_content = m.get("content", "") or ""

            # Extract reasoning from <think> blocks before stripping
            reasoning = _extract_think_content(raw_content)
            content = raw_content

            tc_raw = m.get("tool_calls")
            tc = None
            if tc_raw:
                tc = [
                    {
                        "id": x.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": x.get("function", {}).get("name", ""),
                            "arguments": x.get("function", {}).get("arguments", "{}"),
                        },
                    }
                    for x in tc_raw
                ]
            choices.append(Choices(
                finish_reason=c.get("finish_reason", "stop"),
                index=i,
                message=Message(
                    content=content or None,
                    role="assistant",
                    tool_calls=tc,
                    function_call=None,
                    reasoning_content=reasoning,
                ),
            ))

        u = body.get("usage", {})
        return ModelResponse(
            id=body.get("id", _gen_id()),
            created=body.get("created", int(time.time())),
            model=model,
            object="chat.completion",
            choices=choices,
            usage=Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            ),
        )

    # =====================================================================
    # STREAMING HELPERS
    # =====================================================================

    def _sse_to_generic(
        self, chunk: dict, state: StreamState,
    ) -> Optional[GenericStreamingChunk]:
        if not state.response_id:
            state.response_id = chunk.get("id", _gen_id())

        choices = chunk.get("choices", [])
        if not choices:
            return None

        ch = choices[0]
        delta = ch.get("delta", {})
        fin = ch.get("finish_reason")

        text = delta.get("content", "") or ""
        if text:
            state.content += text

        tool_use = None
        tcd_list = delta.get("tool_calls")
        if tcd_list:
            for tcd in tcd_list:
                idx = tcd.get("index", 0)
                func = tcd.get("function", {})
                if idx not in state.tool_calls:
                    state.tool_calls[idx] = {
                        "id": tcd.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                    }
                    tool_use = {
                        "id": state.tool_calls[idx]["id"],
                        "type": "function",
                        "function": {
                            "name": state.tool_calls[idx]["name"],
                            "arguments": func.get("arguments", ""),
                        },
                        "index": idx,
                    }
                else:
                    ad = func.get("arguments", "")
                    state.tool_calls[idx]["arguments"] += ad
                    tool_use = {
                        "id": None,
                        "type": "function",
                        "function": {"name": None, "arguments": ad},
                        "index": idx,
                    }

        is_done = fin is not None
        if is_done:
            state.finish_reason = fin

        u = chunk.get("usage", {})
        if u:
            state.input_tokens = u.get("prompt_tokens", state.input_tokens)
            state.output_tokens = u.get("completion_tokens", state.output_tokens)

        r: GenericStreamingChunk = {
            "text": text,
            "is_finished": is_done,
            "finish_reason": fin,
            "index": 0,
            "usage": {
                "prompt_tokens": state.input_tokens,
                "completion_tokens": state.output_tokens,
                "total_tokens": state.input_tokens + state.output_tokens,
            },
        }
        if tool_use is not None:
            r["tool_use"] = tool_use
        return r

    def _iter_sse(self, lines_iter):
        """Yield parsed JSON from raw SSE lines (sync)."""
        for line in lines_iter:
            t = line.strip() if isinstance(line, str) else line.decode("utf-8").strip()
            if not t or not t.startswith("data: "):
                continue
            d = t[6:]
            if d == "[DONE]":
                return
            try:
                yield json.loads(d)
            except json.JSONDecodeError:
                continue

    # =====================================================================
    # COMPLETION — exact LiteLLM CustomLLM signatures
    # =====================================================================

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout=None,
        client=None,
    ) -> ModelResponse:
        payload = self._build_payload(model, messages, False, optional_params, litellm_params)
        self._log("completion", payload)
        resp = self.sync_http.post(self.base_url, json=payload, headers=self._headers())
        self._check_response(resp)
        return self._parse_response(resp.json(), self._extract_model(model))

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout=None,
        client=None,
    ) -> ModelResponse:
        payload = self._build_payload(model, messages, False, optional_params, litellm_params)
        self._log("acompletion", payload)
        resp = await self.async_http.post(self.base_url, json=payload, headers=self._headers())
        self._check_response(resp)
        return self._parse_response(resp.json(), self._extract_model(model))

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout=None,
        client=None,
    ) -> Iterator[GenericStreamingChunk]:
        payload = self._build_payload(model, messages, True, optional_params, litellm_params)
        self._log("streaming", payload)
        state = StreamState(model=self._extract_model(model))

        with self.sync_http.stream(
            "POST", self.base_url, json=payload, headers=self._headers()
        ) as resp:
            if not resp.is_success:
                resp.read()
                self._check_response(resp)
            for obj in self._iter_sse(resp.iter_lines()):
                c = self._sse_to_generic(obj, state)
                if c:
                    yield c

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout=None,
        client=None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        payload = self._build_payload(model, messages, True, optional_params, litellm_params)
        self._log("astreaming", payload)
        state = StreamState(model=self._extract_model(model))

        try:
            async with self.async_http.stream(
                "POST", self.base_url, json=payload, headers=self._headers()
            ) as resp:
                if not resp.is_success:
                    await self._check_response_async(resp)
                async for line in resp.aiter_lines():
                    t = line.strip()
                    if not t or not t.startswith("data: "):
                        continue
                    d = t[6:]
                    if d == "[DONE]":
                        break
                    try:
                        obj = json.loads(d)
                    except json.JSONDecodeError:
                        continue
                    c = self._sse_to_generic(obj, state)
                    if c:
                        yield c
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                self._log(f"Stream interrupted: {e}")
                yield {
                    "text": "", "is_finished": True,
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




# =========================================================================
# REGISTRATION
# =========================================================================

_instance: Optional[MiniMaxProvider] = None


def register_minimax(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    debug: bool = False,
    provider_name:str = "minimax"
) -> MiniMaxProvider:
    """
    Register MiniMax as LiteLLM custom provider.
    Use model="minimax/MiniMax-M2.5" after registration.
    """
    import litellm

    global _instance
    _instance = MiniMaxProvider(api_key=api_key, base_url=base_url, debug=debug)
    litellm.custom_provider_map.append(
        {"provider": provider_name, "custom_handler": _instance}
    )
    return _instance
