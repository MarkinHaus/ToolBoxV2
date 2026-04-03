"""
Tests für MiniMax LiteLLM Custom Provider
==========================================

Voraussetzung: export MINIMAX_API_KEY=your_key

Ausführen: python test_minimax_provider.py
"""

import asyncio
import json
import os
import unittest
import warnings

# Suppress Windows ProactorEventLoop transport warnings
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport")

# Provider muss vor litellm-Aufrufen registriert werden
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.minimax_provider import register_minimax

provider = register_minimax(debug=True)

import litellm

MODEL = "minimax/MiniMax-M2.5"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"],
        },
    },
}


class TestCompletion(unittest.TestCase):
    """Sync + Async non-streaming completion."""

    def test_sync_completion(self):
        resp = litellm.completion(
            model=MODEL,
            messages=[{"role": "system", "content": "antworte kurtz!"},{"role": "system", "content": "antworte kurtz!"},{"role": "user", "content": "Sag 'Hallo Test'. Nichts weiter."}],
            max_tokens=100,
        )
        text = resp.choices[0].message.content
        print(f"\n[sync completion] {text}")
        self.assertIsNotNone(text)
        self.assertTrue(len(text) > 0)

    def test_async_completion(self):
        async def _run():
            resp = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "system", "content": "antworte kurtz!"},{"role": "system", "content": "antworte kurtz!"},{"role": "user", "content": "Sag 'Hallo Async'. Nichts weiter."}],
                max_tokens=100,
            )
            return resp

        resp = asyncio.run(_run())
        text = resp.choices[0].message.content
        print(f"\n[async completion] {text}")
        self.assertIsNotNone(text)
        self.assertTrue(len(text) > 0)


class TestStreaming(unittest.TestCase):
    """Sync + Async streaming."""

    def test_sync_streaming(self):
        collected = ""
        resp = litellm.completion(
            model=MODEL,
            messages=[{"role": "system", "content": "antworte kurtz!"},{"role": "system", "content": "antworte kurtz!"},{"role": "user", "content": "Zähle von 1 bis 5."}],
            max_tokens=200,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            collected += delta

        print(f"\n[sync stream] {collected[:200]}")
        self.assertTrue(len(collected) > 0)

    def test_async_streaming(self):
        async def _run():
            collected = ""
            resp = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "system", "content": "antworte kurtz!"},{"role": "system", "content": "antworte kurtz!"},{"role": "user", "content": "Zähle von 1 bis 5."}],
                max_tokens=200,
                stream=True,
            )
            async for chunk in resp:
                delta = chunk.choices[0].delta.content or ""
                collected += delta
            return collected

        text = asyncio.run(_run())
        print(f"\n[async stream] {text[:200]}")
        self.assertTrue(len(text) > 0)


class TestToolUse(unittest.TestCase):
    """Tool use (sync + async)."""

    def test_sync_tool_use(self):
        resp = litellm.completion(
            model=MODEL,
            messages=[{"role": "system", "content": "antworte kurtz!"},{"role": "system", "content": "antworte kurtz!"},{"role": "user", "content": "Wie ist das Wetter in Berlin?"}],
            tools=[WEATHER_TOOL],
            max_tokens=300,
        )
        msg = resp.choices[0].message
        print(f"\n[sync tool] finish_reason={resp.choices[0].finish_reason}")
        print(f"  tool_calls={msg.tool_calls}")

        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertIsNotNone(msg.tool_calls)
        self.assertTrue(len(msg.tool_calls) > 0)

        tc = msg.tool_calls[0]
        self.assertEqual(tc["function"]["name"], "get_weather")
        args = json.loads(tc["function"]["arguments"])
        self.assertIn("city", args)

    def test_async_tool_use(self):
        async def _run():
            return await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "system", "content": "antworte kurtz!"},{"role": "system", "content": "antworte kurtz!"},{"role": "user", "content": "Wie ist das Wetter in Berlin?"}],
                tools=[WEATHER_TOOL],
                max_tokens=300,
            )

        resp = asyncio.run(_run())
        msg = resp.choices[0].message
        print(f"\n[async tool] finish_reason={resp.choices[0].finish_reason}")
        print(f"  tool_calls={msg.tool_calls}")

        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertIsNotNone(msg.tool_calls)
        self.assertTrue(len(msg.tool_calls) > 0)


class TestParallelAsync(unittest.TestCase):
    """Parallel async calls — verifies no event-loop conflicts."""

    def test_parallel_completions(self):
        async def _run():
            tasks = [
                litellm.acompletion(
                    model=MODEL,
                    messages=[{"role": "system", "content": "antworte kurtz!"}, {"role": "user", "content": f"Sag '{i}'."}],
                    max_tokens=150,
                )
                for i in range(3)
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_run())
        print(f"\n[parallel] Got {len(results)} responses")
        for i, r in enumerate(results):
            c = r.choices[0].message.content or "(no content)"
            print(f"  [{i}] {c[:60]}")
        self.assertEqual(len(results), 3)
        for r in results:
            msg = r.choices[0].message
            # Either content or tool_calls should be present
            self.assertTrue(
                msg.content is not None or msg.tool_calls is not None,
                "Response has neither content nor tool_calls",
            )


if __name__ == "__main__":
    if not os.environ.get("MINIMAX_API_KEY"):
        print("ERROR: MINIMAX_API_KEY nicht gesetzt!")
        exit(1)

    print(f"Model: {MODEL}")
    print("=" * 60)
    unittest.main(verbosity=2)
