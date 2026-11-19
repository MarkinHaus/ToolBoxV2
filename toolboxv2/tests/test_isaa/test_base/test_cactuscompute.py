"""Tests for the CactusComputeAdapter (local Needle 3 on-device model).

unittest only. The real `cactus-needle` package is never required: a fake
`needle` module is injected so the adapter's conversion / mapping / caching
logic is tested in isolation (mirrors the mock style of test_adapters.py).
"""
import unittest
import asyncio
import sys
import os
import types
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Imports work both in-repo (toolboxv2.mods.isaa.base.llm_router...) and from an
# isolated copy of the package (llm_router...). The isolated form avoids the
# isaa __init__ import chain (see page/HANDOFF.md).
try:
    from llm_router.types import (
        CompletionResult, StreamChunk, ToolCallData, ProviderError,
    )
    from llm_router.adapters.cactuscompute import CactusComputeAdapter
except ImportError:  # pragma: no cover - in-repo path
    from toolboxv2.mods.isaa.base.llm_router.types import (
        CompletionResult, StreamChunk, ToolCallData, ProviderError,
    )
    from toolboxv2.mods.isaa.base.llm_router.adapters.cactuscompute import (
        CactusComputeAdapter,
    )


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _collect_async_gen(coro_factory):
    async def _inner():
        items = []
        async for item in coro_factory():
            items.append(item)
        return items
    return _run(_inner())


# --- A fake `needle` module -------------------------------------------------

class _FakeNeedle:
    """Stand-in for needle.Needle. Records construction + returns a scripted
    response from complete()."""
    last_instance = None

    def __init__(self, tools=None, system=None, weights=None, **kw):
        self.tools = tools
        self.system = system
        self.weights = weights
        self.kw = kw
        self.completed_with = []
        self.reset_calls = 0
        self.closed = False
        # scripted response (overwritten per-test)
        self.response = {
            "type": "respond",
            "success": True,
            "error": None,
            "function_calls": [],
            "suppressed_calls": [],
            "reasoning": "",
            "confidence": None,
        }
        _FakeNeedle.last_instance = self

    def complete(self, text="", max_new_tokens=512):
        self.completed_with.append(text)
        return self.response

    def reset(self):
        self.reset_calls += 1

    def close(self):
        self.closed = True


def _make_fake_needle_module():
    mod = types.ModuleType("needle")

    def tool(fn=None, **kw):
        # decorator usable as @needle.tool or @needle.tool(triggers=[...])
        if fn is None:
            return lambda f: f
        return fn

    mod.Needle = _FakeNeedle
    mod.tool = tool
    return mod


class CactusAdapterTestBase(unittest.TestCase):
    def setUp(self):
        _FakeNeedle.last_instance = None
        # Reset the process-global Needle worker so tests don't bleed into each other.
        self._reset_worker()
        self.fake_mod = _make_fake_needle_module()
        self.adapter = CactusComputeAdapter()
        # Force the adapter to use the fake module (no install / no real engine)
        self._patch = patch.object(self.adapter, "_import_needle",
                                   return_value=self.fake_mod)
        self._patch.start()
        self.session = MagicMock()

    def tearDown(self):
        self._patch.stop()
        self._reset_worker()

    @staticmethod
    def _reset_worker():
        w = CactusComputeAdapter._WORKER
        if w is not None:
            try:
                w._q.put(None)  # stop the worker thread
            except Exception:
                pass
        CactusComputeAdapter._WORKER = None


# =========================================================================
# tools -> needle schema conversion
# =========================================================================

class TestToolSchemaConversion(CactusAdapterTestBase):
    def test_openai_function_wrapper_unwrapped(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "set_lights",
                "description": "Turn lights on/off",
                "parameters": {
                    "type": "object",
                    "properties": {"room": {"type": "string"}},
                    "required": ["room"],
                },
            },
        }]
        schema = self.adapter._to_needle_tools(tools)
        self.assertEqual(len(schema), 1)
        self.assertEqual(schema[0]["name"], "set_lights")
        self.assertEqual(schema[0]["description"], "Turn lights on/off")
        self.assertIn("parameters", schema[0])
        self.assertNotIn("type", schema[0])  # not the OpenAI wrapper type
        self.assertNotIn("function", schema[0])

    def test_already_flat_schema_passthrough(self):
        tools = [{
            "name": "fn",
            "description": "d",
            "parameters": {"type": "object", "properties": {}},
        }]
        schema = self.adapter._to_needle_tools(tools)
        self.assertEqual(schema[0]["name"], "fn")

    def test_none_tools_returns_none(self):
        self.assertIsNone(self.adapter._to_needle_tools(None))
        self.assertIsNone(self.adapter._to_needle_tools([]))


# =========================================================================
# messages -> query / system extraction
# =========================================================================

class TestMessageExtraction(CactusAdapterTestBase):
    def test_last_user_message_is_query(self):
        msgs = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "toggle the garage door"},
        ]
        query = self.adapter._extract_query(msgs)
        self.assertEqual(query, "toggle the garage door")

    def test_system_facts_collected(self):
        msgs = [
            {"role": "system", "content": "date: 2026-07-21"},
            {"role": "user", "content": "x"},
        ]
        system = self.adapter._extract_system(msgs)
        self.assertIn("date: 2026-07-21", system)

    def test_list_content_flattened_to_text(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ]},
        ]
        query = self.adapter._extract_query(msgs)
        self.assertIn("hello", query)
        self.assertIn("world", query)


# =========================================================================
# complete(): response -> CompletionResult mapping
# =========================================================================

class TestCompleteMapping(CactusAdapterTestBase):
    def _set_response(self, response):
        # Prime the fake so the NEXT constructed Needle returns this response.
        orig_init = _FakeNeedle.__init__

        def init(inner, *a, **k):
            orig_init(inner, *a, **k)
            inner.response = response
        self.addCleanup(lambda: setattr(_FakeNeedle, "__init__", orig_init))
        _FakeNeedle.__init__ = init

    def test_tool_call_maps_to_tool_calls(self):
        self._set_response({
            "type": "call",
            "success": True,
            "error": None,
            "function_calls": [
                {"name": "set_lights", "arguments": {"room": "living room", "on": True}},
            ],
            "suppressed_calls": [],
            "reasoning": "r",
            "confidence": 0.94,
        })
        tools = [{"name": "set_lights", "description": "d",
                  "parameters": {"type": "object", "properties": {}}}]
        result = _run(self.adapter.complete(
            self.session, "", "needle3",
            [{"role": "user", "content": "dim the living room"}], tools))

        self.assertIsInstance(result, CompletionResult)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(result.tool_calls[0].name, "set_lights")
        self.assertEqual(result.tool_calls[0].arguments["room"], "living room")
        self.assertEqual(result.finish_reason, "tool_calls")
        # Needle's reasoning is passed through as content for context.
        self.assertEqual(result.content, "r")

    def test_reasoning_passed_through_as_content(self):
        self._set_response({
            "type": "respond",
            "success": True,
            "error": None,
            "function_calls": [],
            "suppressed_calls": [],
            "reasoning": "no tool matched the request",
            "confidence": 0.0,
        })
        tools = [{"name": "fn", "description": "d",
                  "parameters": {"type": "object", "properties": {}}}]
        result = _run(self.adapter.complete(
            self.session, "", "needle3",
            [{"role": "user", "content": "tell me a joke"}], tools))
        # Even with an empty call, the reasoning is surfaced as content.
        self.assertEqual(result.content, "no tool matched the request")
        self.assertIsNone(result.tool_calls)
        self.assertEqual(result.finish_reason, "stop")

    def test_empty_reasoning_yields_none_content(self):
        self._set_response({
            "type": "respond",
            "success": True,
            "error": None,
            "function_calls": [],
            "suppressed_calls": [],
            "reasoning": "",
            "confidence": 0.0,
        })
        tools = [{"name": "fn", "description": "d",
                  "parameters": {"type": "object", "properties": {}}}]
        result = _run(self.adapter.complete(
            self.session, "", "needle3",
            [{"role": "user", "content": "x"}], tools))
        self.assertIsNone(result.content)

    def test_no_tools_short_circuits_without_engine(self):
        """A call with NO tools must NOT touch the native engine (that avoids the
        0-tool bind → re-init-with-tools crash). It returns the empty call."""
        # Prime a response that WOULD be returned if the engine were used — it
        # must NOT be, so content stays None (short-circuit refusal).
        self._set_response({
            "type": "respond", "success": True, "error": None,
            "function_calls": [], "suppressed_calls": [],
            "reasoning": "engine was wrongly invoked", "confidence": 0.0,
        })
        result = _run(self.adapter.complete(
            self.session, "", "needle3",
            [{"role": "user", "content": "hi"}], None))  # no tools
        self.assertIsNone(result.tool_calls)
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNone(result.content)
        # The engine must not have been constructed for a no-tools call.
        self.assertIsNone(_FakeNeedle.last_instance)

    def test_respond_with_empty_calls_is_refusal_stop(self):
        self._set_response({
            "type": "respond",
            "success": True,
            "error": None,
            "function_calls": [],
            "suppressed_calls": [],
            "reasoning": "",
            "confidence": 0.0,
        })
        result = _run(self.adapter.complete(
            self.session, "", "needle3",
            [{"role": "user", "content": "tell me a poem"}], None))
        self.assertIsNone(result.tool_calls)
        self.assertEqual(result.finish_reason, "stop")

    def test_confidence_in_raw(self):
        self._set_response({
            "type": "call",
            "success": True,
            "error": None,
            "function_calls": [{"name": "fn", "arguments": {}}],
            "suppressed_calls": [],
            "reasoning": "why",
            "confidence": 0.71,
        })
        tools = [{"name": "fn", "description": "d",
                  "parameters": {"type": "object", "properties": {}}}]
        result = _run(self.adapter.complete(
            self.session, "", "needle3",
            [{"role": "user", "content": "go"}], tools))
        self.assertEqual(result.raw.get("confidence"), 0.71)
        self.assertEqual(result.model, "needle3")


# =========================================================================
# stream() wraps complete() as a single chunk
# =========================================================================

class TestStreamWrapsComplete(CactusAdapterTestBase):
    def test_stream_yields_single_chunk(self):
        tools = [{"name": "fn", "description": "d",
                  "parameters": {"type": "object", "properties": {}}}]

        def factory():
            return self.adapter.stream(
                self.session, "", "needle3",
                [{"role": "user", "content": "go"}], tools)
        chunks = _collect_async_gen(factory)
        self.assertTrue(len(chunks) >= 1)
        self.assertTrue(all(isinstance(c, StreamChunk) for c in chunks))
        # a terminal chunk carries a finish_reason
        self.assertTrue(any(c.finish_reason for c in chunks))


# =========================================================================
# instance cache + lazy install
# =========================================================================

class TestInstanceCacheAndInstall(CactusAdapterTestBase):
    def test_same_toolset_reuses_needle_instance(self):
        tools = [{"name": "fn", "description": "d",
                  "parameters": {"type": "object", "properties": {}}}]
        _run(self.adapter.complete(self.session, "", "needle3",
                                   [{"role": "user", "content": "a"}], tools))
        first = _FakeNeedle.last_instance
        _run(self.adapter.complete(self.session, "", "needle3",
                                   [{"role": "user", "content": "b"}], tools))
        second = _FakeNeedle.last_instance
        self.assertIs(first, second)  # reused, not rebuilt

    def test_different_toolset_reuses_same_instance(self):
        """IDEMPOTENT: the native Needle engine is one-per-process and a second
        needle_init crashes. So the adapter builds the instance ONCE and reuses
        it for every call — even when the toolset changes. No second instance,
        no close()."""
        t1 = [{"name": "a", "description": "d",
               "parameters": {"type": "object", "properties": {}}}]
        t2 = [{"name": "b", "description": "d",
               "parameters": {"type": "object", "properties": {}}}]
        _run(self.adapter.complete(self.session, "", "needle3",
                                   [{"role": "user", "content": "x"}], t1))
        first = _FakeNeedle.last_instance
        _run(self.adapter.complete(self.session, "", "needle3",
                                   [{"role": "user", "content": "y"}], t2))
        second = _FakeNeedle.last_instance
        self.assertIs(first, second)         # SAME instance reused
        self.assertFalse(first.closed)       # never closed

    def test_instance_built_exactly_once(self):
        """Across many calls with varying toolsets, needle.Needle is constructed
        exactly ONCE (no re-init can ever happen)."""
        builds = {"n": 0}
        orig_init = _FakeNeedle.__init__

        def counting_init(inner, *a, **k):
            builds["n"] += 1
            orig_init(inner, *a, **k)
        self.addCleanup(lambda: setattr(_FakeNeedle, "__init__", orig_init))
        _FakeNeedle.__init__ = counting_init

        for i in range(5):
            tools = [{"name": f"t{i}", "description": "d",
                      "parameters": {"type": "object", "properties": {}}}]
            _run(self.adapter.complete(self.session, "", "needle3",
                                       [{"role": "user", "content": f"q{i}"}], tools))
        self.assertEqual(builds["n"], 1)     # built exactly once


class TestLazyImportInstall(unittest.TestCase):
    def test_import_needle_installs_when_missing(self):
        adapter = CactusComputeAdapter()
        fake_mod = _make_fake_needle_module()
        calls = {"import": 0, "install": 0}

        def fake_import():
            calls["import"] += 1
            if calls["install"] == 0:
                raise ImportError("no module named needle")
            return fake_mod

        def fake_install():
            calls["install"] += 1

        with patch.object(adapter, "_raw_import", side_effect=fake_import), \
             patch.object(adapter, "_pip_install", side_effect=fake_install):
            mod = adapter._import_needle()

        self.assertIs(mod, fake_mod)
        self.assertEqual(calls["install"], 1)  # installed exactly once
        self.assertGreaterEqual(calls["import"], 2)  # retried after install

    def test_embed_not_supported_raises(self):
        adapter = CactusComputeAdapter()
        with self.assertRaises(NotImplementedError):
            _run(adapter.embed(MagicMock(), "", "needle3", ["x"]))


# =========================================================================
# setup_default_adapters: `needle` prefix registration
# =========================================================================

class TestNeedleSetup(unittest.TestCase):
    def _router_and_setup(self):
        try:
            from llm_router.router import CompletionRouter
            from llm_router.adapters.setup import setup_default_adapters
        except ImportError:  # pragma: no cover - in-repo path
            from toolboxv2.mods.isaa.base.llm_router.router import CompletionRouter
            from toolboxv2.mods.isaa.base.llm_router.adapters.setup import (
                setup_default_adapters,
            )
        return CompletionRouter, setup_default_adapters

    def test_needle_always_registered_no_key(self):
        CompletionRouter, setup_default_adapters = self._router_and_setup()
        router = CompletionRouter(session=MagicMock(), strict_mode=True)
        setup_default_adapters(router, env={})  # no keys at all

        adapter, model, prefix = router.resolve("needle/needle3")
        self.assertIsInstance(adapter, CactusComputeAdapter)
        self.assertEqual(model, "needle3")  # prefix stripped
        self.assertEqual(prefix, "needle")

    def test_needle_model_info_supports_tools(self):
        try:
            from llm_router.model_info import supports_tools, ctx_limit
        except ImportError:  # pragma: no cover - in-repo path
            from toolboxv2.mods.isaa.base.llm_router.model_info import (
                supports_tools, ctx_limit,
            )
        self.assertTrue(supports_tools("needle/needle3"))
        self.assertGreater(ctx_limit("needle/needle3"), 0)


if __name__ == "__main__":
    unittest.main()
