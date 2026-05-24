"""Tests for provider adapters — unittest only, mock aiohttp."""
import unittest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toolboxv2.mods.isaa.base.llm_router.types import (
    UsageData, ToolCallData, CompletionResult, StreamChunk,
    ToolCallDelta, ProviderError,
)
from toolboxv2.mods.isaa.base.llm_router.stream_metrics import StreamMetrics
from toolboxv2.mods.isaa.base.llm_router.adapters.openai_compat import OpenAICompatAdapter
from toolboxv2.mods.isaa.base.llm_router.adapters.zai import ZAIAdapter
from toolboxv2.mods.isaa.base.llm_router.adapters.anthropic import AnthropicAdapter
from toolboxv2.mods.isaa.base.llm_router.adapters.minimax import MiniMaxAdapter


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _collect_async_gen(coro_factory):
    """Run an async generator and collect all yielded items."""
    async def _inner():
        items = []
        async for item in coro_factory():
            items.append(item)
        return items
    return _run(_inner())


# --- Mock helpers ---

def _mock_response(status=200, json_data=None, sse_lines=None):
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)

    if json_data is not None:
        resp.json = AsyncMock(return_value=json_data)

    if sse_lines is not None:
        # Build bytes from SSE lines
        raw = "\n".join(sse_lines).encode("utf-8")

        async def iter_any():
            yield raw

        resp.content = MagicMock()
        resp.content.iter_any = iter_any

    resp.text = AsyncMock(return_value=json.dumps(json_data or {}))
    return resp


def _oai_completion_json(content="hello", model="test-model",
                          tool_calls=None, finish_reason="stop"):
    """Standard OpenAI completion response."""
    msg = {"content": content, "role": "assistant"}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _oai_stream_lines(content_chunks=None, tool_call_chunks=None,
                        finish_reason="stop"):
    """Build SSE lines for OpenAI streaming."""
    lines = []
    if content_chunks:
        for i, text in enumerate(content_chunks):
            chunk = {
                "choices": [{"index": 0, "delta": {"content": text},
                             "finish_reason": None}],
            }
            lines.append(f"data: {json.dumps(chunk)}")
    if tool_call_chunks:
        for tc_chunk in tool_call_chunks:
            chunk = {
                "choices": [{"index": 0, "delta": {"tool_calls": [tc_chunk]},
                             "finish_reason": None}],
            }
            lines.append(f"data: {json.dumps(chunk)}")
    # Final chunk with finish_reason and usage
    final = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    lines.append(f"data: {json.dumps(final)}")
    lines.append("data: [DONE]")
    return lines


# =========================================================================
# OpenAICompatAdapter Tests
# =========================================================================

class TestOpenAICompatAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAICompatAdapter("https://api.example.com/v1")
        self.session = MagicMock()

    def test_complete_success(self):
        resp = _mock_response(json_data=_oai_completion_json("hi"))
        self.session.post = MagicMock(return_value=resp)

        result = _run(self.adapter.complete(self.session, "sk-x", "gpt-4",
                                             [{"role": "user", "content": "hi"}], None))
        self.assertEqual(result.content, "hi")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage.total_tokens, 15)

    def test_complete_with_tools(self):
        tc = [{"id": "c1", "function": {"name": "fn", "arguments": '{"x":1}'}}]
        resp = _mock_response(json_data=_oai_completion_json(
            content=None, tool_calls=tc, finish_reason="tool_calls"))
        self.session.post = MagicMock(return_value=resp)

        result = _run(self.adapter.complete(self.session, "sk-x", "gpt-4",
                                             [{"role": "user", "content": "hi"}], None))
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(result.tool_calls[0].name, "fn")
        self.assertEqual(result.tool_calls[0].arguments, {"x": 1})

    def test_complete_error_status(self):
        resp = _mock_response(status=429, json_data={"error": "rate limited"})
        self.session.post = MagicMock(return_value=resp)

        with self.assertRaises(ProviderError) as ctx:
            _run(self.adapter.complete(self.session, "sk-x", "gpt-4",
                                        [{"role": "user", "content": "hi"}], None))
        self.assertEqual(ctx.exception.status_code, 429)

    def test_stream_yields_chunks(self):
        lines = _oai_stream_lines(content_chunks=["hel", "lo"])
        resp = _mock_response(sse_lines=lines)
        self.session.post = MagicMock(return_value=resp)

        def factory():
            return self.adapter.stream(self.session, "sk-x", "gpt-4",
                                        [{"role": "user", "content": "hi"}], None)
        chunks = _collect_async_gen(factory)
        contents = [c.content for c in chunks if c.content]
        self.assertEqual(contents, ["hel", "lo"])

    def test_stream_tool_call_accumulation(self):
        tc_chunks = [
            {"index": 0, "id": "c1", "function": {"name": "fn", "arguments": '{"x"'}},
            {"index": 0, "function": {"arguments": ':1}'}},
        ]
        lines = _oai_stream_lines(tool_call_chunks=tc_chunks)
        resp = _mock_response(sse_lines=lines)
        self.session.post = MagicMock(return_value=resp)

        def factory():
            return self.adapter.stream(self.session, "sk-x", "gpt-4",
                                        [{"role": "user", "content": "hi"}], None)
        chunks = _collect_async_gen(factory)
        tc_deltas = [c.tool_call_delta for c in chunks if c.tool_call_delta]
        self.assertTrue(len(tc_deltas) >= 2)
        self.assertEqual(tc_deltas[0].id, "c1")
        self.assertEqual(tc_deltas[0].name, "fn")

    def test_embed_success(self):
        embed_resp = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
            "model": "text-embed",
        }
        resp = _mock_response(json_data=embed_resp)
        self.session.post = MagicMock(return_value=resp)

        result = _run(self.adapter.embed(self.session, "sk-x", "text-embed",
                                          ["hello", "world"]))
        self.assertEqual(len(result.embeddings), 2)
        self.assertEqual(result.embeddings[0], [0.1, 0.2, 0.3])

    def test_filter_params_applied(self):
        payload = self.adapter.build_payload("m", [], None,
                                              temperature=0.5, bogus=42)
        self.assertIn("temperature", payload)
        self.assertNotIn("bogus", payload)
        self.assertNotIn("stream", payload)  # stream managed explicitly


# =========================================================================
# ZAIAdapter Tests
# =========================================================================

class TestZAIAdapter(unittest.TestCase):
    def test_coding_plan_base_url(self):
        a = ZAIAdapter(use_coding_plan=True)
        self.assertIn("coding", a.base_url)

    def test_free_plan_base_url(self):
        a = ZAIAdapter(use_coding_plan=False)
        self.assertNotIn("coding", a.base_url)
        self.assertIn("paas", a.base_url)

    def test_model_prefix_stripping(self):
        # The router strips the prefix before passing to the adapter,
        # so the adapter gets "glm-4.7-flash" not "zai/glm-4.7-flash"
        a = ZAIAdapter(use_coding_plan=False)
        payload = a.build_payload("glm-4.7-flash", [{"role": "user", "content": "hi"}], None)
        self.assertEqual(payload["model"], "glm-4.7-flash")


# =========================================================================
# AnthropicAdapter Tests
# =========================================================================

class TestAnthropicAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = AnthropicAdapter()
        self.session = MagicMock()

    def test_build_headers_x_api_key(self):
        h = self.adapter.build_headers("sk-ant-123")
        self.assertEqual(h["x-api-key"], "sk-ant-123")
        self.assertNotIn("Authorization", h)
        self.assertEqual(h["anthropic-version"], "2023-06-01")

    def test_messages_conversion_system_prompt(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        payload = self.adapter.build_payload("claude-3", msgs, None)
        self.assertEqual(payload["system"], "You are helpful.")
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "user")

    def test_messages_conversion_tool_calls(self):
        msgs = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "function": {"name": "weather", "arguments": '{"city":"Berlin"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "25°C"},
        ]
        payload = self.adapter.build_payload("claude-3", msgs, None)
        # Assistant message should have tool_use block
        assistant_msg = payload["messages"][1]
        self.assertEqual(assistant_msg["role"], "assistant")
        blocks = assistant_msg["content"]
        tool_use_blocks = [b for b in blocks if b.get("type") == "tool_use"]
        self.assertEqual(len(tool_use_blocks), 1)
        self.assertEqual(tool_use_blocks[0]["name"], "weather")
        self.assertEqual(tool_use_blocks[0]["input"], {"city": "Berlin"})

        # Tool result should be in a user message
        tool_result_msg = payload["messages"][2]
        self.assertEqual(tool_result_msg["role"], "user")

    def test_parse_response_text(self):
        raw = {
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "model": "claude-3",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = self.adapter._parse_response(raw, "claude-3")
        self.assertEqual(result.content, "Hello!")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNone(result.tool_calls)

    def test_parse_response_tool_use(self):
        raw = {
            "content": [
                {"type": "tool_use", "id": "tc1", "name": "search",
                 "input": {"q": "test"}},
            ],
            "stop_reason": "tool_use",
            "model": "claude-3",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = self.adapter._parse_response(raw, "claude-3")
        self.assertIsNone(result.content)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(result.tool_calls[0].name, "search")
        self.assertEqual(result.finish_reason, "tool_calls")

    def test_parse_usage_with_cache(self):
        raw = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "model": "claude-3",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 20,
            },
        }
        result = self.adapter._parse_response(raw, "claude-3")
        self.assertEqual(result.usage.cache_read_tokens, 80)
        self.assertEqual(result.usage.cache_creation_tokens, 20)

    def test_stream_events(self):
        """Test Anthropic SSE event processing."""
        sse_lines = [
            'event: message_start',
            f'data: {json.dumps({"type": "message_start", "message": {"model": "claude-3", "usage": {"input_tokens": 10}}})}',
            'event: content_block_start',
            f'data: {json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})}',
            'event: content_block_delta',
            f'data: {json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}})}',
            'event: content_block_delta',
            f'data: {json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}})}',
            'event: content_block_stop',
            f'data: {json.dumps({"type": "content_block_stop", "index": 0})}',
            'event: message_delta',
            f'data: {json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}})}',
            'event: message_stop',
            f'data: {json.dumps({"type": "message_stop"})}',
        ]

        resp = _mock_response(sse_lines=sse_lines)
        self.session.post = MagicMock(return_value=resp)

        def factory():
            return self.adapter.stream(self.session, "sk-ant-x", "claude-3",
                                        [{"role": "user", "content": "hi"}], None)
        chunks = _collect_async_gen(factory)
        contents = [c.content for c in chunks if c.content]
        self.assertEqual(contents, ["Hello", " world"])

        # Check finish_reason in one of the chunks
        finish_chunks = [c for c in chunks if c.finish_reason]
        self.assertTrue(any(c.finish_reason == "stop" for c in finish_chunks))

    def test_tool_choice_conversion(self):
        self.assertEqual(
            AnthropicAdapter._convert_tool_choice("auto"),
            {"type": "auto"},
        )
        self.assertEqual(
            AnthropicAdapter._convert_tool_choice("none"),
            {"type": "none"},
        )
        self.assertEqual(
            AnthropicAdapter._convert_tool_choice("required"),
            {"type": "any"},
        )


# =========================================================================
# MiniMaxAdapter Tests
# =========================================================================

class TestMiniMaxAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = MiniMaxAdapter("https://api.minimax.io/v1")

    def test_model_name_mapping(self):
        self.assertEqual(self.adapter._map_model("minimax-m2.5"), "MiniMax-M2.5")
        self.assertEqual(self.adapter._map_model("MiniMax-M2.5"), "MiniMax-M2.5")
        self.assertEqual(self.adapter._map_model("unknown-model"), "unknown-model")

    def test_sanitize_messages(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        result = self.adapter._sanitize_messages(msgs)
        # System merged at index 0
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("Be helpful.", result[0]["content"])
        self.assertIn("Be concise.", result[0]["content"])
        # User at index 1
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[1]["content"], "Hi")

    def test_sanitize_none_content(self):
        msgs = [
            {"role": "user", "content": None},
        ]
        result = self.adapter._sanitize_messages(msgs)
        # content normalized to ""
        user_msg = [m for m in result if m["role"] == "user"][0]
        self.assertEqual(user_msg["content"], "")

    def test_think_tag_stripping(self):
        raw = _oai_completion_json(content="<think>reasoning</think>Answer here")
        result = self.adapter._parse_response(raw, "minimax-m2.5")
        self.assertEqual(result.content, "Answer here")
        self.assertEqual(result.raw["reasoning_content"], "reasoning")

    def test_orphaned_tool_result_fix(self):
        """Orphaned tool result (no matching assistant tool_call) → user message."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "orphan", "content": "result"},
        ]
        result = self.adapter._sanitize_messages(msgs)
        # The orphaned tool should become a user message
        roles = [m["role"] for m in result]
        self.assertNotIn("tool", roles)


# =========================================================================
# Setup Tests
# =========================================================================

class TestSetupDefaultAdapters(unittest.TestCase):
    def test_registers_only_available_providers(self):
        from llm_router.router import CompletionRouter
        router = CompletionRouter(session=MagicMock(), strict_mode=True)

        from llm_router.adapters.setup import setup_default_adapters
        # Only GROQ key set
        setup_default_adapters(router, env={"GROQ_API_KEY": "test"})

        # Groq should be registered
        adapter, model, prefix = router.resolve("groq/llama-3.3-70b")
        self.assertIsInstance(adapter, OpenAICompatAdapter)

        # Anthropic should NOT be registered (no key)
        with self.assertRaises(ProviderError):
            router.resolve("anthropic/claude-3")

    def test_ollama_always_registered(self):
        from llm_router.router import CompletionRouter
        router = CompletionRouter(session=MagicMock(), strict_mode=True)

        from llm_router.adapters.setup import setup_default_adapters
        setup_default_adapters(router, env={})  # no keys at all

        adapter, model, prefix = router.resolve("ollama/mistral")
        self.assertIsInstance(adapter, OpenAICompatAdapter)


# =========================================================================
# StreamMetrics Tests
# =========================================================================

class TestStreamMetrics(unittest.TestCase):
    def test_ttft_ms(self):
        m = StreamMetrics(t_start=1.0, t_first_token=1.05)
        self.assertAlmostEqual(m.ttft_ms, 50.0, places=1)

    def test_tps_calculation(self):
        m = StreamMetrics(t_start=1.0, t_first_token=1.1, t_end=2.1, token_count=100)
        # 100 tokens / 1.0s = 100 tps
        self.assertAlmostEqual(m.tps, 100.0, places=0)

    def test_tps_zero_when_no_tokens(self):
        m = StreamMetrics(t_start=1.0, t_first_token=1.1, t_end=2.0)
        self.assertEqual(m.tps, 0.0)

    def test_summary_dict(self):
        m = StreamMetrics(t_start=1.0, t_first_token=1.05, t_end=2.0,
                          chunk_count=10, token_count=50)
        s = m.summary()
        self.assertIn("ttft_ms", s)
        self.assertIn("tps", s)
        self.assertIn("total_ms", s)
        self.assertEqual(s["chunks"], 10)
        self.assertEqual(s["output_tokens"], 50)

    def test_metrics_none_no_overhead(self):
        """Verify stream works with metrics=None."""
        adapter = OpenAICompatAdapter("https://api.example.com/v1")
        session = MagicMock()

        lines = _oai_stream_lines(content_chunks=["ok"])
        resp = _mock_response(sse_lines=lines)
        session.post = MagicMock(return_value=resp)

        def factory():
            return adapter.stream(session, "sk-x", "m",
                                   [{"role": "user", "content": "hi"}], None,
                                   metrics=None)
        chunks = _collect_async_gen(factory)
        self.assertTrue(len(chunks) > 0)

    def test_metrics_populated_during_stream(self):
        """Verify metrics are filled when passed to stream."""
        adapter = OpenAICompatAdapter("https://api.example.com/v1")
        session = MagicMock()

        lines = _oai_stream_lines(content_chunks=["a", "b", "c"])
        resp = _mock_response(sse_lines=lines)
        session.post = MagicMock(return_value=resp)

        metrics = StreamMetrics()

        def factory():
            return adapter.stream(session, "sk-x", "m",
                                   [{"role": "user", "content": "hi"}], None,
                                   metrics=metrics)
        _collect_async_gen(factory)

        self.assertGreater(metrics.chunk_count, 0)
        self.assertGreater(metrics.t_start, 0)
        self.assertGreater(metrics.t_first_token, 0)
        self.assertGreater(metrics.t_end, 0)


if __name__ == "__main__":
    unittest.main()
