import asyncio
import unittest
import time
import json
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

# Importiere die echten Klassen
from toolboxv2.mods.isaa.base.IntelligentRateLimiter import IntelligentRateLimiter, LiteLLMRateLimitHandler, FallbackReason
from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine, ExecutionContext, HistoryCompressor


class TestAgentPerformance(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.limiter = IntelligentRateLimiter(persist_learned_limits=False)
        self.handler = LiteLLMRateLimitHandler(rate_limiter=self.limiter)

    # --- BEREICH 1: RATE LIMITER STABILITÄT ---

    async def test_rate_limiter_blocks_correctly(self):
        """Testet, ob der Limiter bei Erreichen des RPM-Limits wirklich blockiert."""
        # Setze extrem niedriges Limit: 2 Requests pro Minute
        self.limiter.set_limits("test-provider/model", rpm=2)

        start_time = time.perf_counter()

        # Erster Request (sofort)
        await self.limiter.acquire("test-provider/model")
        # Zweiter Request (sofort)
        await self.limiter.acquire("test-provider/model")

        # Dritter Request MUSS blockieren (da 2 RPM erreicht)
        # Wir setzen einen Timeout, damit der Test nicht ewig läuft
        try:
            async with asyncio.timeout(0.5):
                await self.limiter.acquire("test-provider/model")
                self.fail("Limiter hätte blockieren müssen!")
        except TimeoutError:
            duration = time.perf_counter() - start_time
            print(f"✅ Limiter blockiert korrekt nach {duration:.2f}s")

    async def test_429_recovery_logic(self):
        """Simuliert einen echten 429 Fehler und prüft die Auto-Wait Wiederholung."""
        mock_litellm = AsyncMock()

        # Simuliere: Erster Aufruf wirft RateLimitError, zweiter Aufruf klappt
        error_msg = "Rate limit reached. Please retry in 1 seconds."
        mock_litellm.acompletion.side_effect = [
            Exception(error_msg),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Erfolg nach Wait"))])
        ]

        start_time = time.perf_counter()
        response = await self.handler.completion_with_rate_limiting(
            mock_litellm,
            model="openai/gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )

        duration = time.perf_counter() - start_time
        self.assertEqual(response.choices[0].message.content, "Erfolg nach Wait")
        self.assertGreaterEqual(duration, 1.0, "Der Limiter hätte mindestens 1s warten müssen")
        print(f"✅ Auto-Recovery erfolgreich nach {duration:.2f}s")

    async def test_key_rotation_on_failure(self):
        """Prüft, ob bei einem 429 Fehler zum nächsten Key gewechselt wird."""
        self.limiter.key_manager.mode = "drain"
        key1 = self.limiter.add_api_key("provider", "key-one")
        key2 = self.limiter.add_api_key("provider", "key-two")

        # Simuliere Fehler für Key 1
        await self.limiter.handle_rate_limit_error("provider/model", Exception("429"), api_key_hash=key1)

        # Nächster Key sollte Key 2 sein
        next_key = await self.limiter.key_manager.get_next_key("provider")
        self.assertEqual(next_key.key, "key-two")
        print("✅ Key-Rotation funktioniert nach Fehler")

    # --- BEREICH 2: PERFORMANCE & EVENT-LOOP BLOCKING ---

    async def test_event_loop_blocking_check(self):
        """
        Benchmark: Misst, ob interne Arbeit (Kompression/Locks) den Loop blockiert.
        Dies identifiziert das 'Agent pausiert grundlos' Problem.
        """
        history = []
        # Erzeuge massive History für Stress-Test
        for i in range(200):
            history.append({"role": "user", "content": "Sehr langer Text " * 50})
            history.append({"role": "assistant", "content": "Antwort " * 50})

        start_time = time.perf_counter()

        # Starte einen "Heartbeat" Task, der misst, wie oft der Loop pro Sekunde atmet
        heartbeat_delays = []

        async def heartbeat():
            for _ in range(10):
                t1 = time.perf_counter()
                await asyncio.sleep(0.01)
                heartbeat_delays.append(time.perf_counter() - t1)

        hb_task = asyncio.create_task(heartbeat())

        # Führe CPU-intensive interne Arbeit aus
        # Hier: Partial Compression einer riesigen History
        summary, compressed = HistoryCompressor.compress_partial(history, keep_last_n=5)

        await hb_task

        max_delay = max(heartbeat_delays)
        print(f"📊 Max Event-Loop Latenz während Kompression: {max_delay * 1000:.2f}ms")

        # Wenn max_delay > 100ms, dann blockiert der Agent spürbar die Ausführung anderer Tasks (z.B. Sub-Agenten)
        self.assertLess(max_delay, 0.5, "Event Loop wurde für mehr als 500ms blockiert! Das verursacht die Pausen.")

    async def test_lock_contention_stress(self):
        """Simuliert 10 parallele Sub-Agenten, die auf den RateLimiter zugreifen."""
        self.limiter.set_limits("fast-provider/model", rpm=100)

        async def mock_agent_call():
            await self.limiter.acquire("fast-provider/model")
            await asyncio.sleep(0.05)
            self.limiter.report_success("fast-provider/model")

        start = time.perf_counter()
        tasks = [mock_agent_call() for _ in range(20)]
        await asyncio.gather(*tasks)
        end = time.perf_counter()

        print(f"📊 20 parallele Zugriffe Dauer: {end - start:.4f}s")
        # Das sollte fast instant gehen. Wenn es Sekunden dauert, hängen die Locks fest.


# --- HILFSFUNKTION FÜR REAL-RUN SIMULATION ---

async def benchmark_internal_overhead():
    """Simuliert einen echten Run und trackt die Zeit pro Phase."""
    print("\n--- PERFORMANCE BENCHMARK (REAL CODE) ---")
    agent = MagicMock()  # Mock für Agent
    from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine
    engine = ExecutionEngine(agent)
    ctx = ExecutionContext()

    # Phase 1: System Prompt Build
    t1 = time.perf_counter()
    engine._build_system_prompt(ctx, MagicMock())
    print(f"Phase 1 (Prompt Build): {(time.perf_counter() - t1) * 1000:.2f}ms")

    # Phase 2: Tool Definition Gen
    t1 = time.perf_counter()
    engine._get_tool_definitions(ctx)
    print(f"Phase 2 (Tool Defs): {(time.perf_counter() - t1) * 1000:.2f}ms")

    # Phase 3: History Compressing (Stress)
    ctx.working_history = [{"role": "user", "content": "x"}] * 100
    t1 = time.perf_counter()
    HistoryCompressor.compress_partial(ctx.working_history)
    print(f"Phase 3 (Compression 100 msgs): {(time.perf_counter() - t1) * 1000:.2f}ms")


if __name__ == "__main__":
    asyncio.run(benchmark_internal_overhead())
    unittest.main()
