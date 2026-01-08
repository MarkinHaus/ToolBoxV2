"""
Unit tests for VoiceStreamEngine.

Compatible with unittest and pytest.
"""

import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from toolboxv2.tests.test_mods.test_isaa.test_base.test_agent.conftest2 import AsyncTestCase
from toolboxv2.mods.isaa.base.Agent.voice_stream import  VoiceStreamEngine, BackgroundTaskManager, CapabilityDetection, QueryComplexity

import unittest
from unittest.mock import MagicMock, AsyncMock, patch


class TestVoiceStream(AsyncTestCase):
    def setUp(self):
        self.agent = MagicMock()
        self.agent.session_manager.get_or_create = AsyncMock()
        self.engine = VoiceStreamEngine(self.agent)

    def test_01_complexity_instant_short(self):
        res = self.async_run(self.engine._detect_complexity("hi", "s1"))
        self.assertEqual(res.complexity, "instant")

    def test_02_complexity_quick_tool(self):
        res = self.async_run(self.engine._detect_complexity("what time is it", "s1"))
        self.assertEqual(res.complexity, "quick_tool")
        res = self.async_run(self.engine._detect_complexity("erinnere mich in 5 minuten xyz zmachen", "s1"))
        self.assertEqual(res.complexity, "quick_tool")

    def test_03_complexity_hard_heuristic(self):
        # "create" keyword
        self.agent.tool_manager.entries = {'t1': 1}
        res = self.async_run(self.engine._detect_complexity("create a website", "s1"))
        self.assertEqual(res.complexity, "hard_task")

    def test_04_complexity_llm_fallback(self):
        # a_format_class ist async, muss AsyncMock sein!
        self.agent.a_format_class = AsyncMock(return_value={
            'complexity': 'hard_task', 'confidence': 1.0, 'reason': 'LLM'
        })
        # tool_manager.entries muss leer sein damit keine heuristics greifen
        self.agent.tool_manager.entries = {}
        self.agent.tool_manager.list_categories = MagicMock(return_value=[])

        res = self.async_run(self.engine._detect_complexity(
            "something ambiguous but complex must do coplex stuff doing it!", "s1"
        ))
        self.assertEqual(res.complexity, "hard_task")

    def test_05_handle_status_query_match(self):
        res = self.engine._check_status_query("status of bg_123")
        self.assertEqual(res, "bg_123")

    def test_06_handle_status_query_any(self):
        self.engine.background_manager.tasks["t1"] = MagicMock(status="running")
        res = self.engine._check_status_query("any progress?")
        self.assertEqual(res, "any")

    async def test_07_background_task_create(self):
        mgr = BackgroundTaskManager()

        async def job(): return "done"

        tid = mgr.create_task(job(), "q")
        self.assertIn(tid, mgr.tasks)

        await asyncio.sleep(0.01)
        self.assertEqual(mgr.get_result(tid), "done")

    async def test_08_background_task_fail(self):
        mgr = BackgroundTaskManager()

        async def fail(): raise ValueError("Boom")

        tid = mgr.create_task(fail(), "q")

        await asyncio.sleep(0.01)
        status = mgr.get_status(tid)
        self.assertEqual(status['status'], 'failed')

    async def _consume_generator(self, gen):
        res = []
        async for item in gen:
            res.append(item)
        return "".join(res)

    def test_09_stream_instant(self):
        # Session mock
        mock_session = MagicMock()
        mock_session.get_history_for_llm = MagicMock(return_value=[])  # NICHT async
        mock_session.get_reference = AsyncMock(return_value="")
        mock_session.add_message = AsyncMock()

        self.agent.session_manager.get_or_create = AsyncMock(return_value=mock_session)

        # LLM response mock
        async def mock_resp():
            m1 = MagicMock()
            m1.choices[0].delta.content = "Hel"
            m2 = MagicMock()
            m2.choices[0].delta.content = "lo"
            yield m1
            yield m2

        self.agent.llm_handler.completion_with_rate_limiting = AsyncMock(return_value=mock_resp())

        res = self.async_run(self._consume_generator(
            self.engine.stream("Hi", force_mode="instant")
        ))
        self.assertEqual(res, "Hello")

    # In test_10_stream_quick_tool:
    def test_10_stream_quick_tool(self):
        self.agent.tool_manager.get.return_value = MagicMock(name="time_tool")
        self.agent.arun_function = AsyncMock(return_value="12:00")
        self.agent.a_format_class = AsyncMock(return_value={"tool": "time_tool", "args": {}})
        async def mock_resp():
            m = MagicMock()
            m.choices[0].delta.content = "It is 12:00"
            yield m

        self.agent.llm_handler.completion_with_rate_limiting = AsyncMock(return_value=mock_resp())

        res = self.async_run(self._consume_generator(
            self.engine.stream("time", force_mode="quick_tool")
        ))
        self.assertIn("12:00", res)

    def test_11_stream_hard_nowait(self):
        self.agent.a_run.return_value = "Future Result"

        res = self.async_run(self._consume_generator(
            self.engine.stream("Complex", force_mode="hard_task", wait_for_hard=False)
        ))
        self.assertIn("take a moment", res)  # phrase "will_callback"
        self.assertIn("Task ID", res)

    def test_12_stream_hard_wait(self):
        async def delayed_run(**kwargs):
            return "Done"

        self.agent.a_run.side_effect = delayed_run

        res = self.async_run(self._consume_generator(
            self.engine.stream("Complex", force_mode="hard_task", wait_for_hard=True)
        ))
        self.assertIn("Done", res)

    def test_13_phrases_de(self):
        self.engine = VoiceStreamEngine(self.agent, language="de")
        self.assertIn("Moment", self.engine._get_phrase("thinking"))

    async def test_14_cancel_task(self):
        mgr = BackgroundTaskManager()

        async def forever(): await asyncio.sleep(10)

        tid = mgr.create_task(forever(), "q")

        success = mgr.cancel(tid)
        self.assertTrue(success)
        self.assertEqual(mgr.tasks[tid].status, "cancelled")

    def test_15_list_pending(self):
        mgr = BackgroundTaskManager()
        mgr.tasks["t1"] = MagicMock(status="running")
        mgr.tasks["t2"] = MagicMock(status="completed")
        self.assertEqual(len(mgr.list_pending()), 1)


    async def test_16_wait_for_task_timeout(self):
        mgr = BackgroundTaskManager()

        async def slow(): await asyncio.sleep(0.2); return "ok"

        tid = mgr.create_task(slow(), "q")

        res = await mgr.wait_for(tid, timeout=0.05)
        self.assertIsNone(res)
