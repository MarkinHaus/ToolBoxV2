"""
Unit tests for ToolManager Health Check System.

Testet:
- live_test_inputs / cleanup_func / result_contract auf ToolEntry
- health_check_single: HEALTHY, DEGRADED, FAILED, SKIPPED, GUARANTEED
- health_check_all sweep
- is_healthy() inline check
- semantic_check on-demand (mit mock llm_caller)
- result_contract validation in execute()
- self-agent tools via isaa.get_agent("self")
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager():
    from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolManager
    return ToolManager()


def _sync_tool_ok(x: str = "test") -> str:
    return f"ok:{x}"


def _sync_tool_none(x: str = "test") -> None:
    return None


def _sync_tool_raises(x: str = "test"):
    raise RuntimeError("intentional failure")


async def _async_tool_ok(x: str = "test") -> str:
    return f"async_ok:{x}"


# ---------------------------------------------------------------------------
# Test: ToolEntry neue Felder vorhanden
# ---------------------------------------------------------------------------

class TestToolEntryFields(unittest.TestCase):
    def setUp(self):
        from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolEntry
        self.ToolEntry = ToolEntry

    def test_01_health_fields_exist(self):
        entry = self.ToolEntry(name="t", description="d", args_schema="()")
        self.assertTrue(hasattr(entry, "live_test_inputs"))
        self.assertTrue(hasattr(entry, "cleanup_func"))
        self.assertTrue(hasattr(entry, "result_contract"))
        self.assertTrue(hasattr(entry, "health_status"))
        self.assertTrue(hasattr(entry, "health_error"))
        self.assertTrue(hasattr(entry, "last_health_check"))

    def test_02_health_defaults(self):
        entry = self.ToolEntry(name="t", description="d", args_schema="()")
        self.assertEqual(entry.health_status, "UNKNOWN")
        self.assertIsNone(entry.health_error)
        self.assertIsNone(entry.last_health_check)
        self.assertEqual(entry.live_test_inputs, [])
        self.assertIsNone(entry.cleanup_func)
        self.assertIsNone(entry.result_contract)

    def test_03_guaranteed_healthy_flag_default(self):
        entry = self.ToolEntry(name="t", description="d", args_schema="()")
        self.assertFalse(entry.flags.get("guaranteed_healthy", False))


# ---------------------------------------------------------------------------
# Test: ToolHealthResult
# ---------------------------------------------------------------------------

class TestToolHealthResult(unittest.TestCase):
    def setUp(self):
        from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolHealthResult
        self.THR = ToolHealthResult

    def test_04_is_ok_healthy(self):
        r = self.THR(tool_name="t", status="HEALTHY")
        self.assertTrue(r.is_ok())

    def test_05_is_ok_guaranteed(self):
        r = self.THR(tool_name="t", status="GUARANTEED")
        self.assertTrue(r.is_ok())

    def test_06_is_ok_false_for_degraded(self):
        r = self.THR(tool_name="t", status="DEGRADED")
        self.assertFalse(r.is_ok())

    def test_07_to_agent_message_contains_violations(self):
        r = self.THR(
            tool_name="t", status="DEGRADED",
            contract_violations=["result is None but allow_none=False"]
        )
        msg = r.to_agent_message()
        self.assertIn("DEGRADED", msg)
        self.assertIn("allow_none=False", msg)


# ---------------------------------------------------------------------------
# Test: register() mit neuen Parametern
# ---------------------------------------------------------------------------

class TestRegisterWithHealthParams(unittest.TestCase):
    def setUp(self):
        self.tm = _make_manager()

    def test_08_register_stores_live_test_inputs(self):
        inputs = [{"x": "hello"}]
        self.tm.register(
            func=_sync_tool_ok,
            live_test_inputs=inputs,
        )
        entry = self.tm.get("_sync_tool_ok")
        self.assertEqual(entry.live_test_inputs, inputs)

    def test_09_register_stores_result_contract(self):
        contract = {"allow_none": False, "expected_type": "str"}
        self.tm.register(func=_sync_tool_ok, result_contract=contract)
        entry = self.tm.get("_sync_tool_ok")
        self.assertEqual(entry.result_contract, contract)

    def test_10_register_stores_cleanup_func(self):
        cleanup_called = []
        def cleanup(inputs, result):
            cleanup_called.append((inputs, result))

        self.tm.register(func=_sync_tool_ok, cleanup_func=cleanup)
        entry = self.tm.get("_sync_tool_ok")
        self.assertIs(entry.cleanup_func, cleanup)


# ---------------------------------------------------------------------------
# Test: health_check_single
# ---------------------------------------------------------------------------

class TestHealthCheckSingle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tm = _make_manager()

    async def test_11_healthy_with_test_inputs(self):
        self.tm.register(
            func=_sync_tool_ok,
            live_test_inputs=[{"x": "ping"}],
            result_contract={"allow_none": False, "expected_type": "str"},
        )
        result = await self.tm.health_check_single("_sync_tool_ok")
        self.assertEqual(result.status, "HEALTHY")
        self.assertTrue(result.is_ok())
        entry = self.tm.get("_sync_tool_ok")
        self.assertEqual(entry.health_status, "HEALTHY")
        self.assertIsNotNone(entry.last_health_check)

    async def test_12_skipped_no_test_inputs(self):
        self.tm.register(func=_sync_tool_ok)
        result = await self.tm.health_check_single("_sync_tool_ok")
        self.assertEqual(result.status, "SKIPPED")
        # SKIPPED ändert health_status nicht (bleibt UNKNOWN)
        entry = self.tm.get("_sync_tool_ok")
        self.assertEqual(entry.health_status, "UNKNOWN")

    async def test_13_failed_on_exception(self):
        self.tm.register(
            func=_sync_tool_raises,
            live_test_inputs=[{"x": "test"}],
        )
        result = await self.tm.health_check_single("_sync_tool_raises")
        self.assertEqual(result.status, "FAILED")
        self.assertIsNotNone(result.error)
        entry = self.tm.get("_sync_tool_raises")
        self.assertEqual(entry.health_status, "FAILED")

    async def test_14_degraded_contract_none_violation(self):
        self.tm.register(
            func=_sync_tool_none,
            live_test_inputs=[{"x": "test"}],
            result_contract={"allow_none": False},
        )
        result = await self.tm.health_check_single("_sync_tool_none")
        self.assertEqual(result.status, "DEGRADED")
        self.assertTrue(any("allow_none" in v for v in result.contract_violations))

    async def test_15_none_ok_when_allowed(self):
        self.tm.register(
            func=_sync_tool_none,
            live_test_inputs=[{"x": "test"}],
            result_contract={"allow_none": True},
        )
        result = await self.tm.health_check_single("_sync_tool_none")
        self.assertEqual(result.status, "HEALTHY")

    async def test_16_guaranteed_no_execution(self):
        # guaranteed_healthy=True → kein execution, sofort GUARANTEED
        execution_called = []
        def spy_func(x="test"):
            execution_called.append(x)
            return "ok"

        self.tm.register(
            func=spy_func,
            live_test_inputs=[{"x": "test"}],
            flags={"guaranteed_healthy": True},
        )
        result = await self.tm.health_check_single("spy_func")
        self.assertEqual(result.status, "GUARANTEED")
        self.assertEqual(execution_called, [])  # nie aufgerufen

    async def test_17_cleanup_called_with_inputs_and_result(self):
        cleanup_log = []
        def cleanup(inputs, result):
            cleanup_log.append({"inputs": inputs, "result": result})

        self.tm.register(
            func=_sync_tool_ok,
            live_test_inputs=[{"x": "ping"}],
            cleanup_func=cleanup,
        )
        await self.tm.health_check_single("_sync_tool_ok")
        self.assertEqual(len(cleanup_log), 1)
        self.assertEqual(cleanup_log[0]["inputs"], {"x": "ping"})
        self.assertEqual(cleanup_log[0]["result"], "ok:ping")

    async def test_18_async_tool_healthy(self):
        self.tm.register(
            func=_async_tool_ok,
            live_test_inputs=[{"x": "hello"}],
            result_contract={"expected_type": "str"},
        )
        result = await self.tm.health_check_single("_async_tool_ok")
        self.assertEqual(result.status, "HEALTHY")

    async def test_19_wrong_type_contract_degraded(self):
        self.tm.register(
            func=_sync_tool_ok,          # gibt str zurück
            live_test_inputs=[{"x": "test"}],
            result_contract={"expected_type": "dict"},  # erwartet dict
        )
        result = await self.tm.health_check_single("_sync_tool_ok")
        self.assertEqual(result.status, "DEGRADED")
        self.assertTrue(any("expected_type=dict" in v for v in result.contract_violations))

    async def test_20_tool_not_found(self):
        result = await self.tm.health_check_single("does_not_exist")
        self.assertEqual(result.status, "FAILED")
        self.assertIn("not found", result.error)


# ---------------------------------------------------------------------------
# Test: health_check_all & is_healthy
# ---------------------------------------------------------------------------

class TestHealthCheckAllAndIsHealthy(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tm = _make_manager()

    async def test_21_health_check_all_returns_all_tools(self):
        self.tm.register(func=_sync_tool_ok, live_test_inputs=[{"x": "t"}])
        self.tm.register(func=_sync_tool_raises, live_test_inputs=[{"x": "t"}])
        results = await self.tm.health_check_all()
        self.assertIn("_sync_tool_ok", results)
        self.assertIn("_sync_tool_raises", results)

    async def test_22_is_healthy_true_after_check(self):
        self.tm.register(func=_sync_tool_ok, live_test_inputs=[{"x": "t"}])
        await self.tm.health_check_single("_sync_tool_ok")
        self.assertTrue(self.tm.is_healthy("_sync_tool_ok"))

    async def test_23_is_healthy_false_after_failed(self):
        self.tm.register(func=_sync_tool_raises, live_test_inputs=[{"x": "t"}])
        await self.tm.health_check_single("_sync_tool_raises")
        self.assertFalse(self.tm.is_healthy("_sync_tool_raises"))

    async def test_24_is_healthy_true_for_guaranteed(self):
        self.tm.register(func=_sync_tool_ok, flags={"guaranteed_healthy": True})
        await self.tm.health_check_single("_sync_tool_ok")
        self.assertTrue(self.tm.is_healthy("_sync_tool_ok"))

    async def test_25_is_healthy_false_for_unknown(self):
        self.tm.register(func=_sync_tool_ok)  # kein test_input, kein guaranteed
        # health_status bleibt UNKNOWN nach SKIPPED
        self.assertFalse(self.tm.is_healthy("_sync_tool_ok"))

    async def test_26_healthy_tools_filter_pattern(self):
        """Zeigt das empfohlene Pattern statt get_healthy_tools()."""
        self.tm.register(func=_sync_tool_ok, live_test_inputs=[{"x": "t"}])
        self.tm.register(func=_sync_tool_raises, live_test_inputs=[{"x": "t"}])
        await self.tm.health_check_all()

        healthy = [e for e in self.tm.get_all() if self.tm.is_healthy(e.name)]
        names = [e.name for e in healthy]
        self.assertIn("_sync_tool_ok", names)
        self.assertNotIn("_sync_tool_raises", names)


# ---------------------------------------------------------------------------
# Test: semantic_check on-demand
# ---------------------------------------------------------------------------

class TestSemanticCheck(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tm = _make_manager()

    async def test_27_semantic_check_ok(self):
        self.tm.register(
            func=_sync_tool_ok,
            result_contract={
                "semantic_check_hint": "result should contain 'ok'"
            }
        )
        async def llm_caller(prompt: str) -> str:
            return "OK"

        issues = await self.tm.semantic_check("_sync_tool_ok", "ok:test", llm_caller)
        self.assertEqual(issues, [])

    async def test_28_semantic_check_finds_contradiction(self):
        self.tm.register(
            func=_sync_tool_ok,
            result_contract={
                "semantic_check_hint": "if status=timeout, output_path implies no files"
            }
        )
        async def llm_caller(prompt: str) -> str:
            return "- status=timeout but output_path is non-empty while files list is empty"

        fake_result = {"status": "timeout", "output": "/path/to/file", "files": []}
        issues = await self.tm.semantic_check("_sync_tool_ok", fake_result, llm_caller)
        self.assertEqual(len(issues), 1)
        self.assertIn("timeout", issues[0])

    async def test_29_semantic_check_no_hint_returns_empty(self):
        self.tm.register(func=_sync_tool_ok)  # kein semantic_check_hint

        async def llm_caller(prompt: str) -> str:
            raise AssertionError("should not be called")

        issues = await self.tm.semantic_check("_sync_tool_ok", "whatever", llm_caller)
        self.assertEqual(issues, [])


# ---------------------------------------------------------------------------
# Test: execute() mit result_contract validation
# ---------------------------------------------------------------------------

class TestExecuteContractValidation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tm = _make_manager()

    async def test_30_execute_raises_on_contract_violation(self):
        self.tm.register(
            func=_sync_tool_none,
            result_contract={"allow_none": False},
        )
        with self.assertRaises(ValueError) as ctx:
            await self.tm.execute("_sync_tool_none")
        self.assertIn("allow_none=False", str(ctx.exception))

    async def test_31_execute_includes_semantic_hint_in_error(self):
        self.tm.register(
            func=_sync_tool_none,
            result_contract={
                "allow_none": False,
                "semantic_check_hint": "None means sub-agent timed out silently"
            },
        )
        with self.assertRaises(ValueError) as ctx:
            await self.tm.execute("_sync_tool_none")
        msg = str(ctx.exception)
        self.assertIn("semantic_hint", msg)
        self.assertIn("sub-agent timed out silently", msg)

    async def test_32_execute_ok_no_violation(self):
        self.tm.register(
            func=_sync_tool_ok,
            result_contract={"allow_none": False, "expected_type": "str"},
        )
        result = await self.tm.execute("_sync_tool_ok", x="hello")
        self.assertEqual(result, "ok:hello")


# ---------------------------------------------------------------------------
# Test: self-agent tool-set (Integration)
# ---------------------------------------------------------------------------

class TestSelfAgentToolHealth(unittest.IsolatedAsyncioTestCase):
    """
    Testet das Tool-Set des realen self-Agents über isaa.get_agent("self").
    Jedes Tool muss entweder guaranteed_healthy=True haben ODER
    live_test_inputs providen um HEALTHY zu sein.
    """

    async def asyncSetUp(self):
        try:
            from toolboxv2 import get_app
            app = get_app()
            isaa = app.get_mod("isaa")
            self.agent = await isaa.get_agent("self")
            self.tm = self.agent.tool_manager
            self.available = True
        except Exception as e:
            self.available = False
            self.skip_reason = str(e)

    async def test_33_self_agent_tools_registered(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        self.assertGreater(self.tm.count(), 0, "self-agent hat keine Tools registriert")

    async def test_34_self_agent_no_tool_without_schema(self):
        """Kein Tool darf args_schema='()' haben UND function=None gleichzeitig."""
        if not self.available:
            self.skipTest(self.skip_reason)
        broken = [
            e.name for e in self.tm.get_all()
            if e.args_schema == "()" and e.function is None and e.source == "local"
        ]
        self.assertEqual(broken, [], f"Local tools ohne schema+function: {broken}")

    async def test_35_self_agent_sweep_no_hard_fails(self):
        """
        Sweep: alle Tools mit live_test_inputs dürfen nicht FAILED sein.
        SKIPPED (kein test_input) ist akzeptabel.
        """
        if not self.available:
            self.skipTest(self.skip_reason)

        results = await self.tm.health_check_all()
        hard_fails = [
            name for name, r in results.items()
            if r.status == "FAILED"
        ]
        self.assertEqual(
            hard_fails, [],
            f"Tools mit FAILED status: {hard_fails}"
        )

    async def test_36_self_agent_all_testable_tools_healthy(self):
        """
        Alle Tools MIT live_test_inputs müssen HEALTHY sein.
        Tools ohne → SKIPPED ist OK, aber sie sollten dann guaranteed_healthy tragen.
        """
        if not self.available:
            self.skipTest(self.skip_reason)

        results = await self.tm.health_check_all()
        not_healthy = {
            name:( r.status, r.result_preview, r.contract_violations )for name, r in results.items()
            if r.status not in ("HEALTHY", "GUARANTEED", "SKIPPED")
        }
        self.assertEqual(
            not_healthy, {},
            f"Unstabile Tools: {not_healthy}"
        )

    async def test_37_self_agent_is_healthy_consistent(self):
        """is_healthy() muss nach sweep konsistent mit status sein."""
        if not self.available:
            self.skipTest(self.skip_reason)

        await self.tm.health_check_all()
        for entry in self.tm.get_all():
            expected = entry.health_status in ("HEALTHY", "GUARANTEED")
            self.assertEqual(
                self.tm.is_healthy(entry.name),
                expected,
                f"is_healthy() inkonsistent für {entry.name}: status={entry.health_status}"
            )


if __name__ == "__main__":
    unittest.main()
