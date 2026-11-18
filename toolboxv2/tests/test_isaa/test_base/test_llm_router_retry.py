"""Router wiederholt voruebergehende Fehler (Verbindungsabbruch, 5xx) statt den Agenten abstuerzen zu
lassen - im Stream nur vor dem ersten Chunk. Anfragefehler und Rate-Limits werden nicht wiederholt."""
import asyncio
import unittest
from unittest.mock import MagicMock, patch

import aiohttp

from toolboxv2.mods.isaa.base.llm_router import retry
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
    r = CompletionRouter(session=MagicMock(), strict_mode=True)
    r._session = MagicMock()
    type(r._session).session = property(lambda self: MagicMock())
    r.register("t", adapter)
    return r


def run(coro):
    return asyncio.run(coro)


async def collect(r):
    return [c.content async for c in r.stream("t/m", [{"role": "user", "content": "x"}], cache=False)]


@patch("toolboxv2.mods.isaa.base.llm_router.router.backoff", lambda attempt: 0)
class RetryTest(unittest.TestCase):
    def test_server_disconnected_is_retried(self):
        a = Flaky(aiohttp.ServerDisconnectedError())
        res = run(router(a).complete("t/m", [{"role": "user", "content": "x"}], cache=False))
        self.assertEqual((res.content, a.calls), ("ok", 2))

    def test_503_retried_then_gives_up(self):
        a = Flaky(ProviderError("HTTP 503: down", 503, "", "m"), fails=99)
        with self.assertRaises(ProviderError):
            run(router(a).complete("t/m", [], cache=False))
        self.assertEqual(a.calls, 1 + retry.max_retries())

    def test_request_errors_and_rate_limits_not_retried(self):
        for status in (400, 401, 429):
            a = Flaky(ProviderError(f"HTTP {status}", status, "", "m"))
            with self.assertRaises(ProviderError):
                run(router(a).complete("t/m", [], cache=False))
            self.assertEqual(a.calls, 1, status)

    def test_stream_retried_before_first_chunk(self):
        a = Flaky(aiohttp.ServerDisconnectedError())
        self.assertEqual(run(collect(router(a))), ["ok"])
        self.assertEqual(a.calls, 2)

    def test_stream_not_retried_after_output(self):
        a = Flaky(aiohttp.ServerDisconnectedError(), after_chunk=True)
        with self.assertRaises(aiohttp.ServerDisconnectedError):
            run(collect(router(a)))
        self.assertEqual(a.calls, 1)


class ClassifyTest(unittest.TestCase):
    def test_classification(self):
        yes = [aiohttp.ServerDisconnectedError(), ConnectionResetError(), asyncio.TimeoutError(),
               ProviderError("HTTP 502", 502, "", "m"), RuntimeError("litellm.APIConnectionError: Server disconnected")]
        no = [ProviderError("HTTP 400", 400, "", "m"), ProviderError("HTTP 429", 429, "", "m"),
              ValueError("bad tool schema"), KeyError("x")]
        try:
            raise RuntimeError("Agent crashed") from aiohttp.ServerDisconnectedError()
        except RuntimeError as wrapped:
            yes.append(wrapped)
        for e in yes:
            self.assertTrue(retry.is_transient(e), repr(e))
        for e in no:
            self.assertFalse(retry.is_transient(e), repr(e))


if __name__ == "__main__":
    unittest.main()
