"""
Tests for ManagedBrowser out_of_process extension.

Covers:
- Flag handling + state transitions
- OOP wrapper signature preservation
- Worker protocol (mocked queues)
- call_tool serialization
- get_agent guard in OOP mode

Does NOT require playwright installed (all browser interactions mocked).
"""

import asyncio
import inspect
import multiprocessing as mp
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from toolboxv2.mods.isaa.extras.web_helper.tooklit import (
    ManagedBrowser,
    WebAgentToolkit,
)


class TestManagedBrowserInit(unittest.TestCase):
    """Flag handling and initial state."""

    def test_default_is_in_process(self):
        b = ManagedBrowser()
        self.assertFalse(b.out_of_process)
        self.assertFalse(b.is_running)

    def test_oop_flag_set(self):
        b = ManagedBrowser(out_of_process=True)
        self.assertTrue(b.out_of_process)
        self.assertFalse(b.is_running)
        self.assertIsNone(b._worker)
        self.assertIsNone(b._cmd_q)

    def test_all_params_forwarded(self):
        b = ManagedBrowser(
            headless=False,
            auto_start=False,
            keep_open=False,
            verbose=True,
            out_of_process=True,
        )
        self.assertFalse(b.headless)
        self.assertFalse(b.auto_start)
        self.assertFalse(b.keep_open)
        self.assertTrue(b.verbose)
        self.assertTrue(b.out_of_process)


class TestGetAgentGuard(unittest.TestCase):
    """get_agent must raise in OOP mode."""

    def test_raises_in_oop_mode(self):
        b = ManagedBrowser(out_of_process=True, auto_start=False)

        async def _run():
            with self.assertRaises(RuntimeError) as ctx:
                await b.get_agent()
            self.assertIn("out_of_process", str(ctx.exception))

        asyncio.run(_run())

    def test_raises_when_not_started_inprocess(self):
        b = ManagedBrowser(out_of_process=False, auto_start=False)

        async def _run():
            with self.assertRaises(RuntimeError):
                await b.get_agent()

        asyncio.run(_run())


class TestCallToolGuard(unittest.TestCase):
    """call_tool must raise in in-process mode."""

    def test_raises_in_inprocess_mode(self):
        b = ManagedBrowser(out_of_process=False)

        async def _run():
            with self.assertRaises(RuntimeError) as ctx:
                await b.call_tool("goto", url="https://example.com")
            self.assertIn("out_of_process", str(ctx.exception))

        asyncio.run(_run())


class TestCallToolIPC(unittest.TestCase):
    """call_tool sends correct protocol and handles responses."""

    def _make_browser_with_mock_queues(self):
        b = ManagedBrowser(out_of_process=True, auto_start=False)
        b._started = True
        b._cmd_q = MagicMock()
        b._result_q = MagicMock()
        b._oop_lock = asyncio.Lock()
        b._call_counter = 0
        return b

    def test_success_response(self):
        b = self._make_browser_with_mock_queues()
        b._result_q.get = MagicMock(return_value={
            "type": "result", "id": 1,
            "data": {"url": "https://example.com", "title": "Example"},
        })

        async def _run():
            result = await b.call_tool("goto", url="https://example.com")
            self.assertEqual(result["url"], "https://example.com")

            # Verify correct message was sent
            call_args = b._cmd_q.put.call_args[0][0]
            self.assertEqual(call_args["type"], "call")
            self.assertEqual(call_args["tool"], "goto")
            self.assertEqual(call_args["kwargs"], {"url": "https://example.com"})

        asyncio.run(_run())

    def test_error_response_raises(self):
        b = self._make_browser_with_mock_queues()
        b._result_q.get = MagicMock(return_value={
            "type": "result", "id": 1,
            "error": "Element not found",
            "traceback": "Traceback...",
        })

        async def _run():
            with self.assertRaises(RuntimeError) as ctx:
                await b.call_tool("click", selector="#missing")
            self.assertIn("Element not found", str(ctx.exception))

        asyncio.run(_run())

    def test_counter_increments(self):
        b = self._make_browser_with_mock_queues()
        b._result_q.get = MagicMock(return_value={"type": "result", "id": 1, "data": {}})

        async def _run():
            await b.call_tool("goto", url="a")
            await b.call_tool("goto", url="b")
            self.assertEqual(b._call_counter, 2)

            ids = [b._cmd_q.put.call_args_list[i][0][0]["id"] for i in range(2)]
            self.assertEqual(ids, [1, 2])

        asyncio.run(_run())

    def test_auto_start_when_not_started(self):
        b = ManagedBrowser(out_of_process=True, auto_start=True)
        # Not started, auto_start=True → should call start()

        async def _run():
            with patch.object(b, "start", new_callable=AsyncMock) as mock_start:
                # After start, set _started so call_tool proceeds
                async def _fake_start():
                    b._started = True
                    b._cmd_q = MagicMock()
                    b._result_q = MagicMock()
                    b._result_q.get = MagicMock(return_value={"type": "result", "id": 1, "data": {}})
                    b._oop_lock = asyncio.Lock()
                mock_start.side_effect = _fake_start

                await b.call_tool("goto", url="https://example.com")
                mock_start.assert_called_once()

        asyncio.run(_run())


class TestOOPWrapperSignatures(unittest.TestCase):
    """Wrapper must preserve original function's signature."""

    def test_preserves_name_and_doc(self):
        async def original_func(url: str, wait_until: str = "networkidle") -> dict:
            """Navigate to URL."""
            pass

        browser_mock = MagicMock()
        wrapper = WebAgentToolkit._make_oop_wrapper(browser_mock, "goto", original_func)

        self.assertEqual(wrapper.__name__, "original_func")
        self.assertEqual(wrapper.__doc__, "Navigate to URL.")

    def test_preserves_parameters(self):
        async def original_func(url: str, wait_until: str = "networkidle") -> dict:
            """Navigate to URL."""
            pass

        browser_mock = MagicMock()
        wrapper = WebAgentToolkit._make_oop_wrapper(browser_mock, "goto", original_func)

        sig = inspect.signature(wrapper)
        params = list(sig.parameters.keys())
        self.assertIn("url", params)
        self.assertIn("wait_until", params)
        self.assertEqual(sig.parameters["wait_until"].default, "networkidle")

    def test_preserves_annotations(self):
        async def original_func(url: str, max_results: int = 5) -> dict:
            """Search."""
            pass

        browser_mock = MagicMock()
        wrapper = WebAgentToolkit._make_oop_wrapper(browser_mock, "web_search", original_func)

        sig = inspect.signature(wrapper)
        self.assertEqual(sig.parameters["url"].annotation, str)
        self.assertEqual(sig.parameters["max_results"].annotation, int)

    def test_wrapper_calls_browser_call_tool(self):
        async def original_func(url: str) -> dict:
            pass

        browser_mock = MagicMock()
        browser_mock.call_tool = AsyncMock(return_value={"url": "ok"})
        wrapper = WebAgentToolkit._make_oop_wrapper(browser_mock, "goto", original_func)

        async def _run():
            result = await wrapper(url="https://example.com")
            browser_mock.call_tool.assert_called_once_with("goto", url="https://example.com")
            self.assertEqual(result, {"url": "ok"})

        asyncio.run(_run())


class TestWrapToolsForOOP(unittest.TestCase):
    """_wrap_tools_for_oop replaces all funcs on a toolkit-like object."""

    def test_wraps_all_tools(self):
        # Minimal mock of toolkit structure
        class FakeTool:
            def __init__(self, name, func):
                self.name = name
                self.func = func

        async def tool_a(x: int) -> dict:
            return {"original": True}

        async def tool_b(y: str = "hi") -> dict:
            return {"original": True}

        class FakeToolkit:
            def __init__(self):
                self._tools = [FakeTool("a", tool_a), FakeTool("b", tool_b)]
                self.browser = MagicMock()
                self.browser.call_tool = AsyncMock(return_value={"wrapped": True})
                self._make_oop_wrapper = WebAgentToolkit._make_oop_wrapper
        toolkit = FakeToolkit()
        WebAgentToolkit._wrap_tools_for_oop(toolkit)

        # Verify both tools are now wrappers
        async def _run():
            result = await toolkit._tools[0].func(x=42)
            toolkit.browser.call_tool.assert_called_with("a", x=42)
            self.assertEqual(result, {"wrapped": True})

        asyncio.run(_run())

        # Verify signatures are preserved
        sig_b = inspect.signature(toolkit._tools[1].func)
        self.assertIn("y", sig_b.parameters)
        self.assertEqual(sig_b.parameters["y"].default, "hi")


class TestSetHeadless(unittest.TestCase):
    """set_headless restarts browser in both modes."""

    def test_toggles_flag(self):
        b = ManagedBrowser(headless=True, out_of_process=True, auto_start=False)
        self.assertTrue(b.headless)

        async def _run():
            with patch.object(b, "stop", new_callable=AsyncMock), \
                 patch.object(b, "start", new_callable=AsyncMock):
                await b.set_headless(False)

            self.assertFalse(b.headless)

        asyncio.run(_run())

    def test_restarts_if_running(self):
        b = ManagedBrowser(headless=True, out_of_process=True, auto_start=False)
        b._started = True

        async def _run():
            with patch.object(b, "stop", new_callable=AsyncMock) as mock_stop, \
                 patch.object(b, "start", new_callable=AsyncMock) as mock_start:
                await b.set_headless(False)
                mock_stop.assert_called_once()
                mock_start.assert_called_once()

        asyncio.run(_run())


class TestSingletonReset(unittest.TestCase):
    """Singleton lifecycle."""

    def test_get_instance_creates_once(self):
        ManagedBrowser.reset_instance()
        a = ManagedBrowser.get_instance(out_of_process=True)
        b = ManagedBrowser.get_instance()
        self.assertIs(a, b)
        self.assertTrue(a.out_of_process)
        ManagedBrowser.reset_instance()

    def test_reset_clears(self):
        ManagedBrowser.reset_instance()
        a = ManagedBrowser.get_instance()
        ManagedBrowser.reset_instance()
        b = ManagedBrowser.get_instance()
        self.assertIsNot(a, b)
        ManagedBrowser.reset_instance()


class TestOOPStartStop(unittest.TestCase):
    """Start/stop lifecycle in OOP mode with mocked process."""

    def test_start_creates_process(self):
        b = ManagedBrowser(out_of_process=True, auto_start=False)

        async def _run():
            with patch("toolboxv2.mods.isaa.extras.web_helper.tooklit.mp.get_context") as mock_ctx:
                mock_process = MagicMock()
                mock_queue_cmd = MagicMock()
                mock_queue_result = MagicMock()
                mock_queue_result.get = MagicMock(return_value={"type": "ready"})

                ctx_instance = MagicMock()
                ctx_instance.Queue.side_effect = [mock_queue_cmd, mock_queue_result]
                ctx_instance.Process.return_value = mock_process
                mock_ctx.return_value = ctx_instance

                await b.start()

                self.assertTrue(b.is_running)
                mock_process.start.assert_called_once()
                ctx_instance.Process.assert_called_once()

        asyncio.run(_run())

    def test_stop_sends_shutdown(self):
        b = ManagedBrowser(out_of_process=True, auto_start=False)
        b._started = True
        cmd_q = MagicMock()
        b._cmd_q = cmd_q
        b._result_q = MagicMock()
        b._result_q.get = MagicMock(return_value={"type": "stopped"})
        b._worker = MagicMock()
        b._worker.is_alive.return_value = True

        async def _run():
            await b.stop()

            cmd_q.put.assert_called_once_with({"type": "shutdown"})
            self.assertFalse(b.is_running)
            self.assertIsNone(b._worker)

        asyncio.run(_run())

    def test_double_start_is_noop(self):
        b = ManagedBrowser(out_of_process=True, auto_start=False)
        b._started = True  # already running

        async def _run():
            # Should not raise or create new process
            await b.start()
            self.assertTrue(b.is_running)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
