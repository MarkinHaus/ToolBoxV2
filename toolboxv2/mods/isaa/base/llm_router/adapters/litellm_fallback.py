"""LiteLLM fallback adapter. Lazy-imports litellm."""
from __future__ import annotations
from typing import AsyncIterator, TYPE_CHECKING

from ..adapter import ProviderAdapter
from ..types import (
    CompletionResult, StreamChunk, EmbedResult,
    ToolCallData, ToolCallDelta, UsageData, ProviderError,
)

if TYPE_CHECKING:
    import aiohttp


class LiteLLMFallbackAdapter(ProviderAdapter):
    """Lazy-imports litellm. Used when no native adapter matches."""

    def __init__(self):
        super().__init__(base_url="")
        self._litellm = None

    def _ensure(self):
        if self._litellm is None:
            import litellm
            litellm.drop_params = True
            self._litellm = litellm

    def _convert_usage(self, usage) -> UsageData:
        if not usage:
            return UsageData()
        return UsageData(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )

    def _convert_tool_calls(self, tool_calls) -> list[ToolCallData] | None:
        if not tool_calls:
            return None
        import json
        result = []
        for tc in tool_calls:
            func = tc.function
            args_raw = func.arguments or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {"_raw": args_raw}
            result.append(ToolCallData(id=tc.id or "", name=func.name or "", arguments=args))
        return result

    async def complete(self, session: 'aiohttp.ClientSession', api_key: str,
                       model: str, messages: list, tools: list | None,
                       **kwargs) -> CompletionResult:
        self._ensure()
        kw = {k: v for k, v in kwargs.items() if v is not None}
        if tools:
            kw["tools"] = tools
        try:
            resp = await self._litellm.acompletion(model=model, messages=messages, **kw)
        except Exception as e:
            raise ProviderError(str(e), status_code=getattr(e, "status_code", 0),
                                body=str(e), model=model) from e
        choice = resp.choices[0]
        msg = choice.message
        return CompletionResult(
            content=msg.content,
            tool_calls=self._convert_tool_calls(getattr(msg, "tool_calls", None)),
            finish_reason=choice.finish_reason or "stop",
            usage=self._convert_usage(resp.usage),
            model=getattr(resp, "model", model),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )

    async def stream(self, session: 'aiohttp.ClientSession', api_key: str,
                     model: str, messages: list, tools: list | None,
                     metrics=None, **kwargs) -> AsyncIterator[StreamChunk]:
        self._ensure()
        kw = {k: v for k, v in kwargs.items() if v is not None}
        if tools:
            kw["tools"] = tools
        try:
            resp = await self._litellm.acompletion(model=model, messages=messages,
                                                    stream=True, **kw)
        except Exception as e:
            raise ProviderError(str(e), status_code=getattr(e, "status_code", 0),
                                body=str(e), model=model) from e
        async for chunk in resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            finish = chunk.choices[0].finish_reason if chunk.choices else None
            tc_delta = None
            if delta and getattr(delta, "tool_calls", None):
                tc = delta.tool_calls[0]
                tc_delta = ToolCallDelta(
                    index=getattr(tc, "index", 0),
                    id=getattr(tc, "id", None),
                    name=getattr(tc.function, "name", None) if tc.function else None,
                    arguments_delta=getattr(tc.function, "arguments", "") if tc.function else "",
                )
            yield StreamChunk(
                content=getattr(delta, "content", None) if delta else None,
                tool_call_delta=tc_delta,
                finish_reason=finish,
                usage=self._convert_usage(getattr(chunk, "usage", None)),
            )

    async def embed(self, session: 'aiohttp.ClientSession', api_key: str,
                    model: str, texts: list[str], **kwargs) -> EmbedResult:
        self._ensure()
        try:
            resp = await self._litellm.aembedding(model=model, input=texts, **kwargs)
        except Exception as e:
            raise ProviderError(str(e), status_code=getattr(e, "status_code", 0),
                                body=str(e), model=model) from e
        embeddings = [item["embedding"] for item in resp.data]
        return EmbedResult(
            embeddings=embeddings,
            usage=self._convert_usage(resp.usage),
            model=getattr(resp, "model", model),
        )
