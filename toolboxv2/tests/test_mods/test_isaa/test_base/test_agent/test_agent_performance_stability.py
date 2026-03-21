import asyncio
import time
import unittest
import json
import os
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

# Echte Importe
from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine, ExecutionContext
from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
from toolboxv2.mods.isaa.base.Agent.types import AgentModelData
from toolboxv2.mods.isaa.base.IntelligentRateLimiter import IntelligentRateLimiter, LiteLLMRateLimitHandler


# =============================================================================
# VERBESSERTE MOCKS (Kompatibel mit litellm.acompletion)
# =============================================================================

def mock_acompletion_response(content="", tool_calls=None, finish_reason="stop"):
    """Erstellt eine awaitable Response, die exakt das Format von litellm hat."""
    mock_res = MagicMock()

    # Choice Objekt
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    choice.finish_reason = finish_reason
    choice.message.role = "assistant"

    mock_res.choices = [choice]

    # Usage Objekt (für Token-Zähler)
    mock_res.usage = MagicMock()
    mock_res.usage.prompt_tokens = 100
    mock_res.usage.completion_tokens = 50
    mock_res.usage.total_tokens = 150

    # Damit 'await response' funktioniert
    return mock_res


class HeartbeatMonitor:
    def __init__(self):
        self.count = 0
        self.running = True
        self.max_gap = 0
        self.last_tick = time.perf_counter()

    async def start(self):
        while self.running:
            now = time.perf_counter()
            gap = now - self.last_tick
            if gap > self.max_gap:
                self.max_gap = gap
            self.count += 1
            self.last_tick = now
            await asyncio.sleep(0.01)

    def stop(self):
        self.running = False


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestAgentPerformance(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Nutze 'openai/' Prefix, damit litellm nicht wegen fehlendem Provider meckert
        self.amd = AgentModelData(
            name="PerfTestAgent",
            fast_llm_model="openai/gpt-4o-mini",
            complex_llm_model="openai/gpt-4o",
            system_message="You are a test agent.",
            vfs_max_window_lines=100
        )
        self.agent = FlowAgent(self.amd, auto_load_checkpoint=False)
        self.engine = ExecutionEngine(self.agent)

        self.monitor = HeartbeatMonitor()
        self.monitor_task = asyncio.create_task(self.monitor.start())

    async def asyncTearDown(self):
        self.monitor.stop()
        await self.monitor_task
        await self.agent.close()

    @patch("litellm.acompletion")
    @patch("litellm.token_counter")
    async def test_internal_overhead_benchmark(self, mock_counter, mock_completion):
        """Misst den Overhead zwischen LLM-Calls."""
        mock_counter.return_value = 100

        # Sequenz: Think -> Final Answer -> Summary (für _commit_run)
        mock_completion.side_effect = [
            mock_acompletion_response(
                tool_calls=[MagicMock(id="1", function=MagicMock(name="think", arguments='{"thought":"plan"}'))]),
            mock_acompletion_response(tool_calls=[MagicMock(id="2", function=MagicMock(name="final_answer",
                                                                                       arguments='{"answer":"done","success":true}'))]),
            mock_acompletion_response(content="Summary")
        ]

        start_time = time.perf_counter()
        await self.engine.execute("Hallo Test", session_id="perf_sess", max_iterations=5)
        total_time = time.perf_counter() - start_time

        print(f"\n--- PERFORMANCE REPORT ---")
        print(f"Gesamtzeit: {total_time:.4f}s")
        print(f"Max Event-Loop Gap: {self.monitor.max_gap * 1000:.2f}ms")

        self.assertLess(self.monitor.max_gap, 0.3, "Der Event-Loop wurde zu lange blockiert!")

    @patch("litellm.acompletion")
    async def test_stability_auto_resume_loop(self, mock_completion):
        """Prüft ob der Agent bei 100 Continuation-Calls abbricht."""
        # Simuliere 'abgeschnittenen' Text
        mock_completion.return_value = mock_acompletion_response(content="Part...", finish_reason="length")

        # Wir begrenzen den Test hier künstlich, um nicht 100 echte Mocks zu warten,
        # oder lassen ihn durchlaufen (geht schnell mit Mocks)
        with patch("toolboxv2.mods.isaa.base.Agent.flow_agent.MAX_CONTINUATIONS", 5):
            res = await self.agent.a_run_llm_completion(
                messages=[{"role": "user", "content": "Erzähl viel"}],
                stream=False
            )

        print(f"\n--- STABILITY REPORT: Auto-Resume ---")
        self.assertIn("Part", res)

    @patch("litellm.acompletion")
    @patch("litellm.token_counter")
    async def test_context_displacement_corruption(self, mock_counter, mock_completion):
        """Sicherstellung der tool_call_id Integrität bei riesigem Content."""
        mock_counter.return_value = 50000  # Simuliere hohen Token-Verbrauch
        from litellm.types.utils import ChatCompletionMessageToolCall, Function
        tc = ChatCompletionMessageToolCall(
            id="call_999",
            type="function",
            function=Function(name="vfs_read", arguments='{"path":"huge.txt"}')
        )
        mock_completion.side_effect = [
            mock_acompletion_response(tool_calls=[tc]),
            mock_acompletion_response(tool_calls=[
                MagicMock(id="call_final", function=MagicMock(name="final_answer", arguments='{"answer":"ok"}'))]),
            mock_acompletion_response(content="Summary")
        ]

        # Simuliere extrem großen Tool-Output
        big_content = "DATA" * 40000
        self.agent.arun_function = AsyncMock(return_value=big_content)

        ctx = ExecutionContext()
        # Wichtig: Context config setzen damit Offloading getriggert wird
        ctx.context_config.immediate_offload_ratio = 0.01

        await self.engine.execute("Große Datei", session_id="budget_sess", ctx=ctx)

        # Prüfe ob tool_call_id erhalten blieb
        tool_msgs = [m for m in ctx.working_history if m.get("role") == "tool"]
        self.assertTrue(len(tool_msgs) > 0, "Keine Tool-Antwort in History gefunden!")

        last_tool = tool_msgs[0]
        print(f"\n--- BUDGET MGMT REPORT ---")
        print(f"Tool ID: {last_tool.get('tool_call_id')}")

        self.assertEqual(last_tool.get('tool_call_id'), "call_999", "Tool Call ID Match fehlgeschlagen!")
        self.assertIn("[DATA OFFLOADED", last_tool['content'])

    async def test_stress_test_rate_limiters(self):
        """Simuliert 10 Agenten, die gleichzeitig auf ein enges Limit (2 RPM) hämmern."""
        limiter = IntelligentRateLimiter(default_rpm=2)  # Extrem enges Limit
        handler = LiteLLMRateLimitHandler(rate_limiter=limiter)
        # default_rpm=2 greift nur fuer unbekannte Modelle.
        # openai/gpt-4o hat 500 RPM hardcoded -> explizit setzen:
        handler.set_limits("test/stress-model", rpm=2)

        # Mock für litellm.acompletion
        mock_litellm = MagicMock()

        async def mock_call(**kwargs):
            # Simuliere Netzwerk-Latenz
            await asyncio.sleep(0.5)
            res = MagicMock()
            res.choices = [MagicMock()]
            res.choices[0].message.content = "Success"
            res.usage.total_tokens = 100
            return res

        mock_litellm.acompletion = AsyncMock(side_effect=mock_call)

        print("🚀 Starte Stress-Test: 10 parallele Anfragen bei 2 RPM Limit...")
        start = time.perf_counter()

        # Starte 10 Aufgaben gleichzeitig
        tasks = [
            handler.completion_with_rate_limiting(mock_litellm, model="test/stress-model",
                                                  messages=[{"role": "user", "content": "test"}])
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        duration = time.perf_counter() - start
        success = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, Exception)]

        print(f"\n--- ERGEBNIS ---")
        print(f"Dauer: {duration:.2f}s (Sollte > 120s sein, wenn 2 RPM eingehalten werden)")
        print(f"Erfolgreich: {len(success)}/10")
        print(f"Fehler: {len(errors)}")

        for i, e in enumerate(errors):
            print(f" Error {i}: {e}")

    async def test_vfs_mount_blocking_analysis(self):
        """VFS Indexing Test."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            for i in range(100):
                with open(os.path.join(tmpdirname, f"f_{i}.py"), "w") as f:
                    f.write("print('test')")

            session = await self.agent.session_manager.get_or_create("vfs_bench")
            start = time.perf_counter()
            res = session.vfs.mount(tmpdirname, vfs_path="/bench")
            duration = time.perf_counter() - start

            print(f"\n--- VFS BENCHMARK ---")
            print(f"Mount Zeit: {duration * 1000:.2f}ms")
            self.assertTrue(res["success"])


async def stress_test_rate_limiter():
    """Simuliert 10 Agenten, die gleichzeitig auf ein enges Limit (2 RPM) hämmern."""
    limiter = IntelligentRateLimiter(default_rpm=2)  # Extrem enges Limit
    handler = LiteLLMRateLimitHandler(rate_limiter=limiter)
    # default_rpm=2 greift nur fuer unbekannte Modelle.
    # openai/gpt-4o hat 500 RPM hardcoded -> explizit setzen:
    handler.set_limits("test/stress-model", rpm=2)

    # Mock für litellm.acompletion
    mock_litellm = MagicMock()

    async def mock_call(**kwargs):
        # Simuliere Netzwerk-Latenz
        await asyncio.sleep(0.5)
        res = MagicMock()
        res.choices = [MagicMock()]
        res.choices[0].message.content = "Success"
        res.usage.total_tokens = 100
        return res

    mock_litellm.acompletion = AsyncMock(side_effect=mock_call)

    print("🚀 Starte Stress-Test: 10 parallele Anfragen bei 2 RPM Limit...")
    start = time.perf_counter()

    # Starte 10 Aufgaben gleichzeitig
    tasks = [
        handler.completion_with_rate_limiting(mock_litellm, model="test/stress-model",
                                              messages=[{"role": "user", "content": "test"}])
        for _ in range(10)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.perf_counter() - start
    success = [r for r in results if not isinstance(r, Exception)]
    errors = [r for r in results if isinstance(r, Exception)]

    print(f"\n--- ERGEBNIS ---")
    print(f"Dauer: {duration:.2f}s (Sollte > 120s sein, wenn 2 RPM eingehalten werden)")
    print(f"Erfolgreich: {len(success)}/10")
    print(f"Fehler: {len(errors)}")

    for i, e in enumerate(errors):
        print(f" Error {i}: {e}")
    else:
        return True

if __name__ == "__main__":
    unittest.main()
