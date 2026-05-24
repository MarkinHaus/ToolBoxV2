"""CompletionRouter — central entry point for LLM completions."""
from __future__ import annotations
from typing import AsyncIterator

from .types import CompletionResult, StreamChunk, EmbedResult, ProviderError, UsageData
from .adapter import ProviderAdapter
from .budget import BudgetTracker
from .stream_metrics import StreamMetrics


class CompletionRouter:

    def __init__(self, session=None, rate_limiter=None, strict_mode: bool = False):
        self._session = session
        self._rate_limiter = rate_limiter
        self._adapters: dict[str, ProviderAdapter] = {}
        self._strict_mode = strict_mode
        self._litellm_fallback: ProviderAdapter | None = None
        self._budget = BudgetTracker()
        self._logger = None

    def _get_logger(self):
        if self._logger is None:
            try:
                from toolboxv2 import get_logger
                self._logger = get_logger()
            except ImportError:
                import logging
                self._logger = logging.getLogger("llm_router")
        return self._logger

    @property
    def session(self):
        """Lazy session resolution."""
        if self._session is None:
            from toolboxv2.utils.system.session import Session
            self._session = Session()
        return self._session.session

    @property
    def budget(self) -> BudgetTracker:
        return self._budget

    def register(self, prefix: str, adapter: ProviderAdapter):
        self._adapters[prefix] = adapter

    def resolve(self, model_string: str) -> tuple[ProviderAdapter, str, str]:
        """Returns (adapter, actual_model, prefix).
        actual_model has the prefix stripped."""
        for prefix, adapter in self._adapters.items():
            if model_string.startswith(prefix + "/"):
                actual_model = model_string[len(prefix) + 1:]
                return adapter, actual_model, prefix
        if self._strict_mode:
            raise ProviderError(
                f"No adapter for model '{model_string}' and strict_mode=True",
                status_code=0, body="", model=model_string,
            )
        return self._get_litellm_fallback(), model_string, ""

    async def complete(self, model: str, messages: list,
                       tools: list | None = None, **kw) -> CompletionResult:
        adapter, actual_model, prefix = self.resolve(model)
        # Budget pre-check (estimate input tokens as chars//4)
        est = sum(len(str(m)) for m in messages) // 4
        self._budget.check(model, est)
        # Rate limiter
        api_key = kw.pop("api_key", "")
        used_model = model
        if self._rate_limiter:
            used_model, limiter_key = await self._rate_limiter.acquire(
                model=model,
                estimated_input_tokens=sum(len(str(m)) for m in messages) // 4,
            )
            if limiter_key:
                api_key = limiter_key
            if used_model != model:
                adapter, actual_model, prefix = self.resolve(used_model)
        try:
            result = await adapter.complete(self.session, api_key, actual_model,
                                            messages, tools, **kw)
        except Exception as e:
            if self._rate_limiter:
                await self._rate_limiter.handle_rate_limit_error(
                    model=used_model, error=e, is_rate_limit="429" in str(e),
                )
            raise
        self._budget.track(model, result.usage)
        if self._rate_limiter:
            self._rate_limiter.report_success(model=used_model, tokens_used=result.usage.total_tokens)
        return result

    async def stream(self, model: str, messages: list,
                     tools: list | None = None,
                     collect_metrics: bool = False, **kw) -> AsyncIterator[StreamChunk]:
        metrics = StreamMetrics() if collect_metrics else kw.pop("metrics", None)
        adapter, actual_model, prefix = self.resolve(model)
        est = sum(len(str(m)) for m in messages) // 4
        self._budget.check(model, est)
        api_key = kw.pop("api_key", "")
        used_model = model
        if self._rate_limiter:
            used_model, limiter_key = await self._rate_limiter.acquire(
                model=model,
                estimated_input_tokens=sum(len(str(m)) for m in messages) // 4,
            )
            if limiter_key:
                api_key = limiter_key
            if used_model != model:
                adapter, actual_model, prefix = self.resolve(used_model)

        try:
            async for chunk in adapter.stream(self.session, api_key, actual_model,
                                              messages, tools, metrics=metrics, **kw):
                yield chunk
                if chunk.usage:
                    self._budget.track(model, chunk.usage)
        except Exception as e:
            if self._rate_limiter:
                await self._rate_limiter.handle_rate_limit_error(
                    model=used_model, error=e, is_rate_limit="429" in str(e),
                )
            raise
        token_used = None
        if metrics:
            import time
            metrics.t_end = time.perf_counter()
            if metrics.token_count == 0 and metrics.chunk_count > 0:
                token_used = metrics.token_count = metrics.chunk_count

            self._get_logger().debug(
                f"[stream] {model} TTFT={metrics.ttft_ms:.0f}ms "
                f"TPS={metrics.tps:.0f} total={metrics.total_ms:.0f}ms"
            )
        if self._rate_limiter:
            self._rate_limiter.report_success(model=used_model, tokens_used=token_used)

    async def stream_with_metrics(self, model: str, messages: list,
                                  tools: list | None = None, **kw):
        """Returns (async_iterator, metrics) tuple."""
        metrics = StreamMetrics()

        async def _inner():
            async for chunk in self.stream(model, messages, tools,
                                           metrics=metrics, **kw):
                yield chunk

        return _inner(), metrics

    async def embed(self, model: str, texts: list[str], **kw) -> EmbedResult:
        adapter, actual_model, prefix = self.resolve(model)
        api_key = kw.pop("api_key", "")
        return await adapter.embed(self.session, api_key, actual_model, texts, **kw)

    def _get_litellm_fallback(self) -> ProviderAdapter:
        if self._litellm_fallback is None:
            from .adapters.litellm_fallback import LiteLLMFallbackAdapter
            self._litellm_fallback = LiteLLMFallbackAdapter()
        return self._litellm_fallback
