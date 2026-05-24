"""OpenAI-compatible adapter via raw aiohttp. Covers Groq, Gemini, Cerebras, etc."""
from __future__ import annotations
import time
from typing import AsyncIterator, TYPE_CHECKING

from ..adapter import ProviderAdapter
from ..types import (
    CompletionResult, StreamChunk, EmbedResult,
    ToolCallData, ToolCallDelta, UsageData, ProviderError,
)
from ..stream_metrics import StreamMetrics

if TYPE_CHECKING:
    import aiohttp


class OpenAICompatAdapter(ProviderAdapter):
    """Raw aiohttp adapter for any OpenAI-compatible API.

    Usage:
        groq = OpenAICompatAdapter("https://api.groq.com/openai/v1")
        gemini = OpenAICompatAdapter("https://generativelanguage.googleapis.com/v1beta/openai")
        ollama = OpenAICompatAdapter("http://localhost:11434/v1")
    """

    CHAT_PATH = "chat/completions"
    EMBED_PATH = "embeddings"

    async def complete(self, session: 'aiohttp.ClientSession', api_key: str,
                       model: str, messages: list, tools: list | None,
                       **kwargs) -> CompletionResult:
        url = self.build_url(self.CHAT_PATH)
        headers = self.build_headers(api_key)
        payload = self.build_payload(model, messages, tools, **kwargs)
        payload["stream"] = False

        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ProviderError(
                    f"HTTP {resp.status}: {body[:300]}",
                    status_code=resp.status, body=body, model=model,
                )
            raw = await resp.json()

        return self._parse_response(raw, model)

    async def stream(self, session: 'aiohttp.ClientSession', api_key: str,
                     model: str, messages: list, tools: list | None,
                     metrics: StreamMetrics | None = None,
                     **kwargs) -> AsyncIterator[StreamChunk]:
        url = self.build_url(self.CHAT_PATH)
        headers = self.build_headers(api_key)
        payload = self.build_payload(model, messages, tools, **kwargs)
        payload["stream"] = True

        if metrics:
            metrics.t_start = time.perf_counter()

        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ProviderError(
                    f"HTTP {resp.status}: {body[:300]}",
                    status_code=resp.status, body=body, model=model,
                )

            buffer = ""
            async for raw_bytes in resp.content.iter_any():
                buffer += raw_bytes.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    chunk_dict = self.parse_sse_line(line)
                    if chunk_dict is None:
                        continue

                    sc = self._chunk_to_stream_chunk(chunk_dict)

                    if metrics:
                        metrics.chunk_count += 1
                        if metrics.chunk_count == 1:
                            metrics.t_first_token = time.perf_counter()
                        if sc.usage and sc.usage.completion_tokens:
                            metrics.token_count = sc.usage.completion_tokens

                    yield sc

        if metrics:
            metrics.t_end = time.perf_counter()

    async def embed(self, session: 'aiohttp.ClientSession', api_key: str,
                    model: str, texts: list[str], **kwargs) -> EmbedResult:
        url = self.build_url(self.EMBED_PATH)
        headers = self.build_headers(api_key)
        payload: dict = {"model": model, "input": texts}
        if kwargs.get("dimensions"):
            payload["dimensions"] = kwargs["dimensions"]
        if kwargs.get("input_type"):
            payload["input_type"] = kwargs["input_type"]
        if kwargs.get("encoding_format"):
            payload["encoding_format"] = kwargs["encoding_format"]

        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ProviderError(
                    f"HTTP {resp.status}: {body[:300]}",
                    status_code=resp.status, body=body, model=model,
                )
            raw = await resp.json()

        embeddings = [item["embedding"] for item in raw.get("data", [])]
        return EmbedResult(
            embeddings=embeddings,
            usage=self.parse_usage(raw.get("usage")),
            model=raw.get("model", model),
        )

    # --- Response parsing ---

    def _parse_response(self, raw: dict, model: str) -> CompletionResult:
        """Parse a non-streaming OpenAI-format response."""
        choices = raw.get("choices", [])
        if not choices:
            return CompletionResult(
                content=None, tool_calls=None, finish_reason="stop",
                usage=self.parse_usage(raw.get("usage")), model=model, raw=raw,
            )
        choice = choices[0]
        msg = choice.get("message", {})
        return CompletionResult(
            content=msg.get("content"),
            tool_calls=self.parse_tool_calls(msg.get("tool_calls")) or None,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=self.parse_usage(raw.get("usage")),
            model=raw.get("model", model),
            raw=raw,
        )

    def _chunk_to_stream_chunk(self, chunk_dict: dict) -> StreamChunk:
        """Convert an SSE chunk dict to StreamChunk."""
        choices = chunk_dict.get("choices", [])
        if not choices:
            # Usage-only chunk (some providers send this at the end)
            return StreamChunk(usage=self.parse_usage(chunk_dict.get("usage")))

        choice = choices[0]
        delta = choice.get("delta", {})
        finish = choice.get("finish_reason")

        # Tool call delta
        tc_delta = None
        raw_tcs = delta.get("tool_calls")
        if raw_tcs:
            tc = raw_tcs[0]
            func = tc.get("function", {})
            tc_delta = ToolCallDelta(
                index=tc.get("index", 0),
                id=tc.get("id"),
                name=func.get("name"),
                arguments_delta=func.get("arguments", ""),
            )

        return StreamChunk(
            content=delta.get("content"),
            tool_call_delta=tc_delta,
            finish_reason=finish,
            usage=self.parse_usage(chunk_dict.get("usage")),
        )
