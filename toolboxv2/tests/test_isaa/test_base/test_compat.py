"""Tests for compat layer — verify shim objects match expected interface."""
import unittest
import json

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toolboxv2.mods.isaa.base.llm_router.types import (
    CompletionResult, StreamChunk, ToolCallData, ToolCallDelta, UsageData,
)
from toolboxv2.mods.isaa.base.llm_router.compat import (
    completion_result_to_message,
    completion_result_to_model_response,
    stream_chunk_to_shim,
)


class TestCompletionResultToMessage(unittest.TestCase):
    def test_text_only(self):
        result = CompletionResult(
            content="hello", tool_calls=None, finish_reason="stop",
            usage=UsageData(10, 5, 15), model="m",
        )
        msg = completion_result_to_message(result)
        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.content, "hello")
        self.assertIsNone(msg.tool_calls)

    def test_with_tool_calls(self):
        tc = ToolCallData(id="c1", name="search", arguments={"q": "test"})
        result = CompletionResult(
            content=None, tool_calls=[tc], finish_reason="tool_calls",
            usage=UsageData(10, 5, 15), model="m",
        )
        msg = completion_result_to_message(result)
        self.assertIsNone(msg.content)
        self.assertEqual(len(msg.tool_calls), 1)

        # Verify interface matches what ExecutionEngine expects
        shim_tc = msg.tool_calls[0]
        self.assertEqual(shim_tc.id, "c1")
        self.assertEqual(shim_tc.type, "function")
        self.assertEqual(shim_tc.function.name, "search")
        # arguments must be JSON STRING (not dict) for auto-resume json.loads()
        self.assertEqual(json.loads(shim_tc.function.arguments), {"q": "test"})

    def test_to_dict(self):
        tc = ToolCallData(id="c1", name="fn", arguments={"x": 1})
        result = CompletionResult(
            content=None, tool_calls=[tc], finish_reason="tool_calls",
            usage=UsageData(), model="m",
        )
        msg = completion_result_to_message(result)
        d = msg.tool_calls[0].to_dict()
        self.assertEqual(d["id"], "c1")
        self.assertEqual(d["function"]["name"], "fn")

    def test_reasoning_content(self):
        result = CompletionResult(
            content="answer", tool_calls=None, finish_reason="stop",
            usage=UsageData(), model="m",
            raw={"reasoning_content": "thinking..."},
        )
        msg = completion_result_to_message(result)
        self.assertEqual(msg.reasoning_content, "thinking...")


class TestModelResponse(unittest.TestCase):
    def test_choices_and_usage(self):
        result = CompletionResult(
            content="hi", tool_calls=None, finish_reason="stop",
            usage=UsageData(100, 50, 150), model="gpt-4",
        )
        resp = completion_result_to_model_response(result)
        self.assertEqual(len(resp.choices), 1)
        self.assertEqual(resp.choices[0].message.content, "hi")
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.usage.prompt_tokens, 100)
        self.assertEqual(resp.usage.completion_tokens, 50)
        self.assertEqual(resp.model, "gpt-4")

    def test_usage_asdict(self):
        result = CompletionResult(
            content="hi", tool_calls=None, finish_reason="stop",
            usage=UsageData(10, 5, 15), model="m",
        )
        resp = completion_result_to_model_response(result)
        d = resp.usage._asdict()
        self.assertEqual(d["total_tokens"], 15)


class TestStreamChunkShim(unittest.TestCase):
    def test_content_chunk(self):
        chunk = StreamChunk(content="hello")
        shim = stream_chunk_to_shim(chunk)
        self.assertEqual(shim.choices[0].delta.content, "hello")
        self.assertIsNone(shim.choices[0].delta.tool_calls)
        self.assertIsNone(shim.choices[0].finish_reason)

    def test_tool_call_delta(self):
        chunk = StreamChunk(
            tool_call_delta=ToolCallDelta(
                index=0, id="c1", name="fn", arguments_delta='{"x":'
            )
        )
        shim = stream_chunk_to_shim(chunk)
        tc = shim.choices[0].delta.tool_calls[0]
        self.assertEqual(tc["id"], "c1")
        self.assertEqual(tc["function"]["name"], "fn")
        self.assertEqual(tc["function"]["arguments"], '{"x":')

    def test_finish_with_usage(self):
        chunk = StreamChunk(
            finish_reason="stop",
            usage=UsageData(10, 5, 15),
        )
        shim = stream_chunk_to_shim(chunk)
        self.assertEqual(shim.choices[0].finish_reason, "stop")
        self.assertEqual(shim.usage.total_tokens, 15)

    def test_streaming_interface_for_process_streaming_response(self):
        """Verify the shim has .choices[0].delta.content — the exact path used in
        ExecutionEngine.stream_generator pump_stream."""
        chunk = StreamChunk(content="ok")
        shim = stream_chunk_to_shim(chunk)
        # This is the exact access pattern from stream_generator:
        delta = shim.choices[0].delta
        self.assertTrue(hasattr(delta, "content"))
        self.assertEqual(delta.content, "ok")
        self.assertTrue(hasattr(delta, "tool_calls"))
        self.assertTrue(hasattr(delta, "reasoning_content"))


if __name__ == "__main__":
    unittest.main()
