"""ProviderAdapter ABC — base class for all LLM provider adapters."""
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import AsyncIterator, TYPE_CHECKING

from .types import (
    CompletionResult, StreamChunk, EmbedResult,
    ToolCallData, UsageData,
)
from .stream_metrics import StreamMetrics

if TYPE_CHECKING:
    import aiohttp


class ProviderAdapter(ABC):
    ALLOWED_PARAMS: frozenset = frozenset({
        'temperature', 'top_p', 'max_tokens', 'stop', 'n',
        'presence_penalty', 'frequency_penalty', 'seed',
        'tools', 'tool_choice', 'response_format',
    })

    def __init__(self, base_url: str, default_headers: dict | None = None):
        self.base_url = base_url.rstrip('/')
        self.default_headers = default_headers or {}

    def filter_params(self, kwargs: dict) -> dict:
        """Strip params not in ALLOWED_PARAMS. Remove tool_choice when tools absent."""
        filtered = {k: v for k, v in kwargs.items()
                    if k in self.ALLOWED_PARAMS and v is not None}
        # If tools is empty/None, drop tool_choice too
        if not filtered.get('tools'):
            filtered.pop('tools', None)
            filtered.pop('tool_choice', None)
        return filtered

    def build_url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def build_headers(self, api_key: str) -> dict:
        """Default: Bearer token. Subclass overrides for x-api-key etc."""
        headers = {"Content-Type": "application/json", **self.default_headers}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def build_payload(self, model: str, messages: list,
                      tools: list | None, **kwargs) -> dict:
        """Default: OpenAI format. Subclass overrides for Anthropic etc."""
        payload = {"model": model, "messages": messages}
        filtered = self.filter_params(kwargs)
        if tools:
            filtered['tools'] = tools
        payload.update(filtered)
        return payload

    @abstractmethod
    async def complete(self, session: 'aiohttp.ClientSession', api_key: str,
                       model: str, messages: list, tools: list | None,
                       **kwargs) -> CompletionResult: ...

    @abstractmethod
    async def stream(self, session: 'aiohttp.ClientSession', api_key: str,
                     model: str, messages: list, tools: list | None,
                     metrics: StreamMetrics | None = None,
                     **kwargs) -> AsyncIterator[StreamChunk]: ...

    async def embed(self, session: 'aiohttp.ClientSession', api_key: str,
                    model: str, texts: list[str], **kwargs) -> EmbedResult:
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")

    # --- Shared helpers (not abstract) ---

    def parse_sse_line(self, line: str) -> dict | None:
        """Parse 'data: {...}' → dict. Returns None for non-data/[DONE]."""
        if not line.startswith("data: "):
            return None
        payload = line[6:].strip()
        if payload == "[DONE]" or not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def parse_tool_calls(self, raw_tool_calls: list | None) -> list[ToolCallData]:
        """Parse OpenAI-format tool_calls. JSON string → dict."""
        if not raw_tool_calls:
            return []
        result = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            args_raw = func.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            result.append(ToolCallData(
                id=tc.get("id", ""),
                name=func.get("name", ""),
                arguments=args,
            ))
        return result

    def parse_usage(self, raw_usage: dict | None) -> UsageData:
        """Parse OpenAI-format usage dict."""
        if not raw_usage:
            return UsageData()
        return UsageData(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
            cache_read_tokens=raw_usage.get("cache_read_tokens", 0)
                              or raw_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            cache_creation_tokens=raw_usage.get("cache_creation_tokens", 0),
        )
