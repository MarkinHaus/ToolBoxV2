"""Router wiederholt ueber den IntelligentRateLimiter (IRL): Verbindungsabbruch, 5xx und 429 gehen an
handle_rate_limit_error() (Backoff, Fallback, Key-Sperre), das naechste acquire() wartet und waehlt
neu. Im Stream nur vor dem ersten Chunk. Anfragefehler (400/401) werden nicht wiederholt."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp

from toolboxv2.mods.isaa.base.IntelligentRateLimiter import intelligent_rate_limiter as irl
from toolboxv2.mods.isaa.base.llm_router.router import CompletionRouter
from toolboxv2.mods.isaa.base.llm_router.types import CompletionResult, ProviderError, StreamChunk, UsageData


class Flaky:
    """Adapter: die ersten `fails` Aufrufe werfen `error` (im Stream optional nach einem Chunk)."""

    def __init__(self, error, fails=1, after_chunk=False):
        self.error, self.fails, self.after_chunk, self.calls = error, fails, after_chunk, 0

    async def complete(self, session, api_key, model, messages, tools, **kw):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.error
        return CompletionResult(content="ok", tool_calls=None, finish_reason="stop", usage=UsageData(), model=model)

    async def stream(self, session, api_key, model, messages, tools, metrics=None, **kw):
        self.calls += 1
        if self.calls <= self.fails:
            if self.after_chunk:
                yield StreamChunk(content="teil")
            raise self.error
        yield StreamChunk(content="ok")


def router(adapter):
    limiter = irl.IntelligentRateLimiter(config_path=Path(tempfile.mkdtemp()) / "limits.json",
                                         persist_learned_limits=False, use_default_fallback_chains=False)
    r = CompletionRouter(session=MagicMock(), rate_limiter=limiter, strict_mode=True)
    r._session = MagicMock()
    type(r._session).session = property(lambda self: MagicMock())
    r.register("t", adapter)
    return r, limiter


async def collect(r):
    return [c.content async for c in r.stream("t/m", [{"role": "user", "content": "x"}], cache=False)]


class RetryTest(unittest.TestCase):
    def setUp(self):
        self.waits = []

        async def no_wait(seconds, *a, **k):              # Backoff des IRL mitschreiben statt warten
            self.waits.append(seconds)
        self.patch = patch.object(irl.asyncio, "sleep", no_wait)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_server_disconnected_retried_with_irl_backoff(self):
        a = Flaky(aiohttp.ServerDisconnectedError())
        r, limiter = router(a)
        res = asyncio.run(r.complete("t/m", [{"role": "user", "content": "x"}], cache=False))
        self.assertEqual((res.content, a.calls), ("ok", 2))
        self.assertTrue(self.waits and self.waits[0] >= 0.5, self.waits)   # Backoff kommt vom IRL

    def test_429_uses_retry_after_from_message(self):
        a = Flaky(ProviderError("HTTP 429: Please retry in 7s", 429, "", "m"))
        r, _ = router(a)
        asyncio.run(r.complete("t/m", [], cache=False))
        self.assertEqual(a.calls, 2)
        self.assertTrue(any(5.0 <= w <= 9.0 for w in self.waits), self.waits)   # 7 s +- Jitter des IRL

    def test_503_gives_up_after_limit(self):
        a = Flaky(ProviderError("HTTP 503: down", 503, "", "m"), fails=99)
        r, _ = router(a)
        with self.assertRaises(ProviderError):
            asyncio.run(r.complete("t/m", [], cache=False))
        self.assertEqual(a.calls, 1 + CompletionRouter._max_retries())

    def test_request_errors_not_retried(self):
        for status in (400, 401):
            a = Flaky(ProviderError(f"HTTP {status}", status, "", "m"))
            r, _ = router(a)
            with self.assertRaises(ProviderError):
                asyncio.run(r.complete("t/m", [], cache=False))
            self.assertEqual(a.calls, 1, status)

    def test_stream_retried_before_first_chunk_only(self):
        a = Flaky(aiohttp.ServerDisconnectedError())
        self.assertEqual(asyncio.run(collect(router(a)[0])), ["ok"])
        b = Flaky(aiohttp.ServerDisconnectedError(), after_chunk=True)
        with self.assertRaises(aiohttp.ServerDisconnectedError):
            asyncio.run(collect(router(b)[0]))
        self.assertEqual(b.calls, 1)


class ClassifyTest(unittest.TestCase):
    def test_classification(self):
        transient = [aiohttp.ServerDisconnectedError(), ConnectionResetError(), asyncio.TimeoutError(),
                     ProviderError("HTTP 502", 502, "", "m"),
                     RuntimeError("litellm.APIConnectionError: Server disconnected")]
        try:
            raise RuntimeError("Agent crashed") from aiohttp.ServerDisconnectedError()
        except RuntimeError as wrapped:
            transient.append(wrapped)
        for e in transient:
            self.assertEqual(irl.classify_error(e), "transient", repr(e))
        self.assertEqual(irl.classify_error(ProviderError("HTTP 429", 429, "", "m")), "rate_limit")
        for e in (ProviderError("HTTP 400", 400, "", "m"), ValueError("bad tool schema"), KeyError("x")):
            self.assertIsNone(irl.classify_error(e), repr(e))


if __name__ == "__main__":
    unittest.main()
