"""CompletionRouter — central entry point for LLM completions."""
from __future__ import annotations

import os
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
        self._api_key_map: dict[str, str] = {}
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

    def register(self, prefix: str, adapter: ProviderAdapter, env_key_name: str = None):
        self._adapters[prefix] = adapter
        if env_key_name:
            self._api_key_map[prefix] = env_key_name

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

    def quick_key(self, prefix: str, api_key: str) -> str:
        if prefix not in self._api_key_map:
            return api_key
        #if api_key:
        #    return api_key
        env_key_name = self._api_key_map[prefix]
        _api_key = os.getenv(env_key_name)
        if _api_key is not None and _api_key != api_key:
            return _api_key
        return api_key

    async def _resolve_fallback(self, used_model: str, error: Exception) -> str | None:
        """Ermittle das aktuelle Fallback-Modell nach einem Provider-Fehler.

        Robust gegen Struktur-Varianten des Rate-Limiters:
        - handle_rate_limit_error() liefert (backoff, fallback_model) direkt
        - fallback: ModelFallbackManager.get_active_model() via .fallback_manager
        """
        if not self._rate_limiter:
            return None
        s_err = str(error)
        # 400 = Content-/Request-Fehler (z.B. Bild-Parse 1210): Ein anderes Modell
        # bekommt dasselbe kaputte Payload — Modell-Hopping ist verschwendet.
        if 'HTTP 400' in s_err or '"code":"1210"' in s_err:
            return None
        try:
            res = await self._rate_limiter.handle_rate_limit_error(
                model=used_model, error=error, is_rate_limit="429" in str(error),
            )
            if res:
                fb = res[1] if isinstance(res, tuple) and len(res) >= 2 else None
                if fb and fb.lower() != used_model.lower():
                    return fb
        except Exception:
            pass
        fb_mgr = getattr(self._rate_limiter, "fallback_manager", None)
        if fb_mgr is None:
            return None
        try:
            active, is_fb = await fb_mgr.get_active_model(used_model)
            if is_fb and active and active.lower() != used_model.lower():
                return active
        except Exception:
            pass
        return None

    @staticmethod
    def _max_retries() -> int:
        try:
            return max(0, int(os.getenv("ISAA_LLM_RETRIES", "3")))
        except ValueError:
            return 3

    async def _pick(self, model: str, messages: list, api_key: str, adapter, actual_model: str):
        """Modell und Key fuer den naechsten Versuch vom Rate-Limiter: acquire() wartet einen
        gesetzten Backoff ab und liefert ggf. das aktive Fallback-Modell und einen anderen Key."""
        if not self._rate_limiter:
            return model, api_key, adapter, actual_model
        used_model, limiter_key = await self._rate_limiter.acquire(
            model=model, estimated_input_tokens=sum(len(str(m)) for m in messages) // 4)
        adapter, actual_model, _prefix = self.resolve(used_model)
        return used_model, limiter_key or api_key, adapter, actual_model

    async def _retry(self, used_model: str, error: Exception, attempt: int) -> bool:
        """Wiederholen ueber den Rate-Limiter (IRL): Rate-Limit (429) und voruebergehende Fehler
        (Verbindungsabbruch, 5xx) meldet der Router an handle_rate_limit_error(); der IRL zaehlt sie,
        setzt den Backoff (Retry-After aus der Meldung bzw. exponentiell), sperrt ggf. den Key und
        loest den Fallback aus. Das naechste _pick() wartet dann und waehlt neu. Andere Fehler
        (400, 401, Schema) werden nicht wiederholt."""
        if not self._rate_limiter or attempt >= self._max_retries():
            return False
        from toolboxv2.mods.isaa.base.IntelligentRateLimiter.intelligent_rate_limiter import classify_error
        kind = classify_error(error)
        if kind is None:
            return False
        await self._rate_limiter.handle_rate_limit_error(
            model=used_model, error=error, is_rate_limit=kind == "rate_limit")
        self._get_logger().warning(
            f"[llm_router] {used_model}: {kind} ({type(error).__name__}: {str(error)[:200]}) "
            f"- Versuch {attempt + 1}/{self._max_retries()} ueber den Rate-Limiter")
        return True

    async def complete(self, model: str, messages: list,
                       tools: list | None = None, **kw) -> CompletionResult:
        adapter, actual_model, prefix = self.resolve(model)

        # Prompt-Caching injection
        cache_enabled = kw.pop("cache", True)  # default ON
        cache_ttl = kw.pop("cache_ttl", "5m")
        if cache_enabled:
            from .cache_control import inject_cache_breakpoints
            messages, tools = inject_cache_breakpoints(messages, tools, prefix, cache_ttl)

        # Budget pre-check (estimate input tokens as chars//4)
        est = sum(len(str(m)) for m in messages) // 4
        self._budget.check(model, est)
        base_key = self.quick_key(prefix, kw.pop("api_key", ""))
        attempt = 0
        while True:
            used_model, api_key, adapter, actual_model = await self._pick(
                model, messages, base_key, adapter, actual_model)
            try:
                result = await adapter.complete(self.session, api_key, actual_model,
                                                messages, tools, **kw)
                break
            except Exception as e:
                if await self._retry(used_model, e, attempt):
                    attempt += 1
                    continue
                if attempt:
                    raise                                # der Rate-Limiter hatte seine Versuche
                fallback_model = await self._resolve_fallback(used_model, e)
                # Fallback-Retry: wenn der Limiter ein anderes aktives Modell liefert,
                # direkt 1x damit wiederholen statt den Task craschen zu lassen.
                if fallback_model and fallback_model.lower() != used_model.lower() and not kw.get("_fallback_retry"):
                    try:
                        return await self.complete(fallback_model, messages, tools=tools,
                                                   _fallback_retry=True,
                                                   **{k: v for k, v in kw.items() if k != "_fallback_retry"})
                    except Exception:
                        pass
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
        # Prompt-Caching injection
        cache_enabled = kw.pop("cache", True)  # default ON
        cache_ttl = kw.pop("cache_ttl", "5m")
        if cache_enabled:
            from .cache_control import inject_cache_breakpoints
            messages, tools = inject_cache_breakpoints(messages, tools, prefix, cache_ttl)

        est = sum(len(str(m)) for m in messages) // 4
        self._budget.check(model, est)
        base_key = self.quick_key(prefix, kw.pop("api_key", ""))
        attempt = 0
        while True:
            used_model, api_key, adapter, actual_model = await self._pick(
                model, messages, base_key, adapter, actual_model)
            started = False
            try:
                async for chunk in adapter.stream(self.session, api_key, actual_model,
                                                  messages, tools, metrics=metrics, **kw):
                    started = True
                    yield chunk
                    if chunk.usage:
                        self._budget.track(model, chunk.usage)
                break
            except Exception as e:
                # Nur VOR dem ersten Chunk wiederholen (typisch: tote Keep-Alive-Verbindung, 429);
                # danach hat der Aufrufer schon Teile der Antwort - doppelte Ausgabe waere schlimmer.
                if not started and await self._retry(used_model, e, attempt):
                    attempt += 1
                    continue
                if attempt:
                    raise
                fallback_model = await self._resolve_fallback(used_model, e)
                if fallback_model and fallback_model.lower() != used_model.lower() and not kw.get("_fallback_retry"):
                    # 1x mit Fallback-Modell neu streamen statt craschen
                    try:
                        async for chunk in self.stream(fallback_model, messages,
                                                       tools=tools, metrics=metrics,
                                                       _fallback_retry=True,
                                                       **{k: v for k, v in kw.items() if k != "_fallback_retry"}):
                            yield chunk
                        return
                    except Exception:
                        pass
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
        api_key = self.quick_key(prefix, kw.pop("api_key", ""))
        return await adapter.embed(self.session, api_key, actual_model, texts, **kw)

    def _get_litellm_fallback(self) -> ProviderAdapter:
        if self._litellm_fallback is None:
            from .adapters.litellm_fallback import LiteLLMFallbackAdapter
            self._litellm_fallback = LiteLLMFallbackAdapter()
        return self._litellm_fallback
