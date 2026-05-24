"""Tests for llm_router — unittest only."""
import unittest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toolboxv2.mods.isaa.base.llm_router.types import (
    UsageData, ToolCallData, CompletionResult, StreamChunk,
    ToolCallDelta, EmbedResult, ProviderError, BudgetExceededError,
)
from toolboxv2.mods.isaa.base.llm_router.adapter import ProviderAdapter
from toolboxv2.mods.isaa.base.llm_router.router import CompletionRouter
from toolboxv2.mods.isaa.base.llm_router.budget import BudgetTracker
from toolboxv2.mods.isaa.base.llm_router.stream_accumulator import StreamAccumulator
from toolboxv2.mods.isaa.base.llm_router.model_info import ctx_limit, supports_tools


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# --- Concrete adapter for testing (ABC can't be instantiated) ---
class DummyAdapter(ProviderAdapter):
    async def complete(self, session, api_key, model, messages, tools, **kw):
        return CompletionResult(
            content="ok", tool_calls=None, finish_reason="stop",
            usage=UsageData(10, 5, 15), model=model,
        )

    async def stream(self, session, api_key, model, messages, tools,
                     metrics=None, **kw):
        yield StreamChunk(content="hi", finish_reason="stop",
                          usage=UsageData(10, 5, 15))


class TestTypes(unittest.TestCase):
    def test_usage_data_defaults(self):
        u = UsageData()
        self.assertEqual(u.prompt_tokens, 0)
        self.assertEqual(u.cache_read_tokens, 0)
        self.assertEqual(u.cache_creation_tokens, 0)
        self.assertEqual(u.total_tokens, 0)

    def test_tool_call_data_fields(self):
        tc = ToolCallData(id="c1", name="search", arguments={"q": "test"})
        self.assertEqual(tc.name, "search")
        self.assertIsInstance(tc.arguments, dict)

    def test_completion_result_with_tools(self):
        tc = ToolCallData(id="c1", name="fn", arguments={})
        r = CompletionResult(content=None, tool_calls=[tc],
                             finish_reason="tool_calls",
                             usage=UsageData(10, 5, 15), model="m")
        self.assertIsNone(r.content)
        self.assertEqual(len(r.tool_calls), 1)

    def test_provider_error_fields(self):
        e = ProviderError("fail", status_code=429, body="rate limited", model="m")
        self.assertEqual(e.status_code, 429)
        self.assertEqual(e.model, "m")
        self.assertIn("fail", str(e))


class TestAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = DummyAdapter(base_url="https://api.example.com")

    def test_filter_params_strips_unknown(self):
        result = self.adapter.filter_params({"temperature": 0.7, "bogus": 42, "max_tokens": 100})
        self.assertIn("temperature", result)
        self.assertIn("max_tokens", result)
        self.assertNotIn("bogus", result)

    def test_filter_params_removes_none_values(self):
        result = self.adapter.filter_params({"temperature": None, "max_tokens": 100})
        self.assertNotIn("temperature", result)
        self.assertIn("max_tokens", result)

    def test_filter_params_removes_tool_choice_when_no_tools(self):
        result = self.adapter.filter_params({"tool_choice": "auto", "tools": None})
        self.assertNotIn("tool_choice", result)
        self.assertNotIn("tools", result)

    def test_filter_params_keeps_tool_choice_with_tools(self):
        result = self.adapter.filter_params({"tool_choice": "auto", "tools": [{"type": "function"}]})
        self.assertIn("tool_choice", result)
        self.assertIn("tools", result)

    def test_build_headers_bearer_token(self):
        h = self.adapter.build_headers("sk-123")
        self.assertEqual(h["Authorization"], "Bearer sk-123")
        self.assertEqual(h["Content-Type"], "application/json")

    def test_build_url(self):
        self.assertEqual(self.adapter.build_url("/v1/chat"), "https://api.example.com/v1/chat")
        self.assertEqual(self.adapter.build_url("v1/chat"), "https://api.example.com/v1/chat")

    def test_parse_sse_line_valid(self):
        data = '{"id":"x","choices":[{"delta":{"content":"hi"}}]}'
        result = self.adapter.parse_sse_line(f"data: {data}")
        self.assertEqual(result["id"], "x")

    def test_parse_sse_line_done(self):
        self.assertIsNone(self.adapter.parse_sse_line("data: [DONE]"))

    def test_parse_sse_line_empty(self):
        self.assertIsNone(self.adapter.parse_sse_line(""))
        self.assertIsNone(self.adapter.parse_sse_line("data: "))

    def test_parse_sse_line_non_data(self):
        self.assertIsNone(self.adapter.parse_sse_line("event: ping"))

    def test_parse_tool_calls_valid(self):
        raw = [{"id": "c1", "function": {"name": "search", "arguments": '{"q":"hi"}'}}]
        result = self.adapter.parse_tool_calls(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "search")
        self.assertEqual(result[0].arguments, {"q": "hi"})

    def test_parse_tool_calls_malformed_json_fallback(self):
        raw = [{"id": "c1", "function": {"name": "fn", "arguments": "{broken"}}]
        result = self.adapter.parse_tool_calls(raw)
        self.assertIn("_raw", result[0].arguments)

    def test_parse_tool_calls_empty(self):
        self.assertEqual(self.adapter.parse_tool_calls(None), [])
        self.assertEqual(self.adapter.parse_tool_calls([]), [])

    def test_parse_usage(self):
        raw = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        u = self.adapter.parse_usage(raw)
        self.assertEqual(u.prompt_tokens, 100)
        self.assertEqual(u.total_tokens, 150)

    def test_parse_usage_none(self):
        u = self.adapter.parse_usage(None)
        self.assertEqual(u.total_tokens, 0)


class TestRouter(unittest.TestCase):
    def _make_router(self):
        router = CompletionRouter(session=MagicMock(), strict_mode=True)
        # Mock session property to return a mock aiohttp session
        router._session = MagicMock()
        type(router._session).session = property(lambda self: MagicMock())
        adapter = DummyAdapter(base_url="https://api.example.com")
        router.register("test", adapter)
        return router, adapter

    def test_register_and_resolve(self):
        router, adapter = self._make_router()
        resolved, model, prefix = router.resolve("test/gpt-4")
        self.assertIs(resolved, adapter)
        self.assertEqual(prefix, "test")

    def test_resolve_strips_prefix(self):
        router, _ = self._make_router()
        _, model, _ = router.resolve("test/gpt-4")
        self.assertEqual(model, "gpt-4")

    def test_resolve_unknown_strict_raises(self):
        router, _ = self._make_router()
        with self.assertRaises(ProviderError):
            router.resolve("unknown/model")

    def test_resolve_unknown_fallback(self):
        router = CompletionRouter(session=MagicMock(), strict_mode=False)
        router._session = MagicMock()
        # Patch the litellm fallback import
        with patch("llm_router.router.CompletionRouter._get_litellm_fallback") as mock_fb:
            mock_adapter = MagicMock()
            mock_fb.return_value = mock_adapter
            resolved, model, prefix = router.resolve("unknown/model")
            self.assertIs(resolved, mock_adapter)
            self.assertEqual(model, "unknown/model")
            self.assertEqual(prefix, "")

    def test_complete_calls_adapter(self):
        result = _run(self._async_test_complete())
        self.assertEqual(result.content, "ok")

    async def _async_test_complete(self):
        router, adapter = self._make_router()
        return await router.complete("test/gpt-4", [{"role": "user", "content": "hi"}],
                                     api_key="sk-123")

    def test_complete_tracks_budget(self):
        _run(self._async_test_budget())

    async def _async_test_budget(self):
        router, _ = self._make_router()
        await router.complete("test/gpt-4", [{"role": "user", "content": "hi"}],
                              api_key="sk-123")
        stats = router.budget.get_stats("global")
        self.assertGreater(stats["total_tokens"], 0)

    def test_stream_yields_chunks(self):
        chunks = _run(self._async_test_stream())
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "hi")

    async def _async_test_stream(self):
        router, _ = self._make_router()
        chunks = []
        async for chunk in router.stream("test/gpt-4", [{"role": "user", "content": "hi"}],
                                         api_key="sk-123"):
            chunks.append(chunk)
        return chunks


class TestBudgetTracker(unittest.TestCase):
    def setUp(self):
        self.bt = BudgetTracker()

    def test_track_updates_global(self):
        self.bt.track("groq/llama", UsageData(10, 5, 15))
        stats = self.bt.get_stats("global")
        self.assertEqual(stats["total_tokens"], 15)

    def test_track_updates_provider_scope(self):
        self.bt.track("groq/llama", UsageData(10, 5, 15))
        stats = self.bt.get_stats("groq")
        self.assertEqual(stats["total_tokens"], 15)

    def test_track_updates_model_scope(self):
        self.bt.track("groq/llama", UsageData(10, 5, 15))
        stats = self.bt.get_stats("groq/llama")
        self.assertEqual(stats["total_tokens"], 15)

    def test_check_under_limit_passes(self):
        self.bt.set_limit("global", max_tokens=1000)
        self.bt.check("groq/llama", 100)  # should not raise

    def test_check_over_limit_raises(self):
        self.bt.set_limit("global", max_tokens=100)
        self.bt.track("groq/llama", UsageData(0, 0, 90))
        with self.assertRaises(BudgetExceededError) as ctx:
            self.bt.check("groq/llama", 20)
        self.assertEqual(ctx.exception.scope, "global")

    def test_get_stats(self):
        self.bt.set_limit("global", max_tokens=500, max_cost_usd=1.0)
        self.bt.track("groq/llama", UsageData(100, 50, 150))
        stats = self.bt.get_stats("global")
        self.assertEqual(stats["total_tokens"], 150)
        self.assertEqual(stats["limit_tokens"], 500)
        self.assertEqual(stats["limit_cost_usd"], 1.0)

    def test_reset(self):
        self.bt.track("groq/llama", UsageData(10, 5, 15))
        self.bt.reset()
        stats = self.bt.get_stats("global")
        self.assertEqual(stats["total_tokens"], 0)

    def test_cost_tracking_with_rates(self):
        self.bt.set_cost_rate("groq/llama", 0.5, 1.0)  # $/Mtok
        self.bt.track("groq/llama", UsageData(1_000_000, 1_000_000, 2_000_000))
        stats = self.bt.get_stats("global")
        self.assertAlmostEqual(stats["total_cost_usd"], 1.5)


class TestStreamAccumulator(unittest.TestCase):
    def test_content_accumulation(self):
        acc = StreamAccumulator()
        acc.feed(StreamChunk(content="hel"))
        acc.feed(StreamChunk(content="lo"))
        result = acc.build(model="m")
        self.assertEqual(result.content, "hello")

    def test_tool_call_accumulation(self):
        acc = StreamAccumulator()
        acc.feed(StreamChunk(tool_call_delta=ToolCallDelta(index=0, id="c1", name="fn")))
        acc.feed(StreamChunk(tool_call_delta=ToolCallDelta(index=0, arguments_delta='{"q":')))
        acc.feed(StreamChunk(tool_call_delta=ToolCallDelta(index=0, arguments_delta='"hi"}')))
        result = acc.build(model="m")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "fn")

    def test_build_parses_tool_args(self):
        acc = StreamAccumulator()
        acc.feed(StreamChunk(tool_call_delta=ToolCallDelta(index=0, id="c1", name="fn",
                                                           arguments_delta='{"x":1}')))
        result = acc.build()
        self.assertEqual(result.tool_calls[0].arguments, {"x": 1})

    def test_build_with_usage(self):
        acc = StreamAccumulator()
        acc.feed(StreamChunk(content="ok", finish_reason="stop",
                             usage=UsageData(10, 5, 15)))
        result = acc.build(model="m")
        self.assertEqual(result.usage.total_tokens, 15)
        self.assertEqual(result.finish_reason, "stop")

    def test_reset(self):
        acc = StreamAccumulator()
        acc.feed(StreamChunk(content="old"))
        acc.reset()
        acc.feed(StreamChunk(content="new"))
        result = acc.build()
        self.assertEqual(result.content, "new")


class TestModelInfo(unittest.TestCase):
    def test_ctx_limit_known(self):
        self.assertEqual(ctx_limit("gemini/gemini-2.5-pro"), 1_000_000)
        self.assertEqual(ctx_limit("groq/llama-3.3-70b-versatile"), 128_000)

    def test_ctx_limit_wildcard(self):
        self.assertEqual(ctx_limit("ollama/mistral"), 128_000)
        self.assertEqual(ctx_limit("openrouter/some-model"), 128_000)

    def test_ctx_limit_unknown_default(self):
        self.assertEqual(ctx_limit("totally/unknown-model"), 128_000)

    def test_supports_tools(self):
        self.assertTrue(supports_tools("anthropic/claude-sonnet-4-6"))
        self.assertFalse(supports_tools("totally/unknown-model"))


if __name__ == "__main__":
    unittest.main()
