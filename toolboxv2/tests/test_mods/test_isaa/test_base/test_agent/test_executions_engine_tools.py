"""
Bullet-proof Tests: ExecutionEngine Static & Discovery Tools

Testet STATIC_TOOLS (think, final_answer, shift_focus),
DISCOVERY_TOOLS (list_tools, load_tools) und Tool-Call-Integrität.

Enthält deterministischen wait_for Semantic Validator
(ohne LLM — pure logic).

Run:
    python -m unittest test_execution_engine_tools -v
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from toolboxv2.mods.isaa.base.Agent import execution_engine


# =============================================================================
# Helpers / Fixtures
# =============================================================================

def _make_agent():
    agent = MagicMock()
    agent.amd = MagicMock()
    agent.amd.name = "test_agent"
    # think braucht a_run_llm_completion
    agent.a_run_llm_completion = AsyncMock(return_value=MagicMock(
        content="## 1. Situation Assessment\nStable.\n"
                "## 2. Key Insights\nNone.\n"
                "## 3. Concrete Tips\nContinue.\n"
                "## 4. Partial Solutions\nN/A\n"
                "## 5. Pitfalls\nNone.\n"
    ))
    agent.arun_function = AsyncMock(return_value="tool_result")

    session = MagicMock()
    session.add_message = AsyncMock()
    agent.session_manager.get_or_create = AsyncMock(return_value=session)
    return agent


def _make_engine(is_sub=False):
    return execution_engine.ExecutionEngine(_make_agent(), is_sub_agent=is_sub)


def _make_ctx():
    ctx = execution_engine.ExecutionContext()
    ctx.run_id = "test_run"
    return ctx


def _tc(name: str, args: dict, tc_id: str = "call_001"):
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _collect_orphan_ids(history: list) -> list[str]:
    """
    Findet tool_call_ids, für die kein zugehöriger tool-Response existiert.
    Orphan = assistant macht tool_call mit ID X, aber kein {'role':'tool', 'tool_call_id': X} folgt.
    """
    assistant_ids = set()
    tool_response_ids = set()
    for msg in history:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                assistant_ids.add(tc["id"])
        if msg.get("role") == "tool":
            tool_response_ids.add(msg.get("tool_call_id", ""))
    return list(assistant_ids - tool_response_ids)


# =============================================================================
# wait_for Semantic Validator (deterministisch, kein LLM)
# =============================================================================

def validate_wait_for_result(sub_result: dict) -> list[str]:
    """
    Deterministischer Semantic-Check für ein einzelnes wait_for SubAgentResult-Dict.

    Erkennt den bekannten Bug:
        status=timeout + output_dir non-empty + files=[]
        → misleading: Agent denkt es gibt Partial-Results, gibt es aber nicht.

    Args:
        sub_result: dict mit keys: status, output_dir, files, error

    Returns:
        Liste von Widersprüchen (leer = OK)
    """
    issues = []
    status = sub_result.get("status", "")
    output_dir = sub_result.get("output_dir") or sub_result.get("output", "")
    files = sub_result.get("files") or sub_result.get("files_written", [])
    error = sub_result.get("error", "")

    # Bug Pattern: timeout + path behauptet + keine Files
    if status == "timeout" and output_dir and not files:
        issues.append(
            f"MISLEADING: status=timeout aber output_dir='{output_dir}' "
            f"ohne files — Agent wird fälschlich nach Partial-Results suchen"
        )

    # Zusätzliche Konsistenz-Checks
    if status in ("timeout", "failed", "error") and not error:
        issues.append(
            f"INCOMPLETE: status={status} aber kein error-Text — "
            f"Agent hat keine Diagnose-Basis"
        )

    if status == "success" and not files and not output_dir:
        issues.append(
            "SUSPICIOUS: status=success aber weder files noch output_dir — "
            "Agent kann Ergebnis nicht verifizieren"
        )

    if isinstance(files, list) and len(files) > 0 and not output_dir:
        issues.append(
            "INCONSISTENT: files vorhanden aber kein output_dir — "
            "Pfad-Referenz fehlt"
        )

    return issues


# =============================================================================
# Class 1 — think
# =============================================================================

class TestThinkTool(unittest.IsolatedAsyncioTestCase):

    async def test_01_think_is_not_final(self):
        e, ctx = _make_engine(), _make_ctx()
        _, is_final = await e._execute_tool_call(ctx, _tc("think", {"thought": "plan"}))
        self.assertFalse(is_final)

    async def test_02_think_returns_string(self):
        e, ctx = _make_engine(), _make_ctx()
        result, _ = await e._execute_tool_call(ctx, _tc("think", {"thought": "plan"}))
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    async def test_03_think_increments_max_iterations(self):
        e, ctx = _make_engine(), _make_ctx()
        before = ctx.max_iterations
        await e._execute_tool_call(ctx, _tc("think", {"thought": "plan"}))
        self.assertEqual(ctx.max_iterations, before + 1)

    async def test_04_think_records_in_auto_focus(self):
        e, ctx = _make_engine(), _make_ctx()
        before = len(ctx.auto_focus.actions)
        await e._execute_tool_call(ctx, _tc("think", {"thought": "insight"}))
        self.assertGreater(len(ctx.auto_focus.actions), before)

    async def test_05_think_adds_to_working_history(self):
        e, ctx = _make_engine(), _make_ctx()
        before = len(ctx.working_history)
        await e._execute_tool_call(ctx, _tc("think", {"thought": "x"}))
        self.assertGreater(len(ctx.working_history), before)

    async def test_06_think_result_structure_has_sections(self):
        """LLM-Response muss Situation Assessment enthalten."""
        e, ctx = _make_engine(), _make_ctx()
        result, _ = await e._execute_tool_call(ctx, _tc("think", {"thought": "plan"}))
        self.assertIn("Situation", result,
            "think-result muss 'Situation Assessment' enthalten")

    async def test_07_think_empty_thought_still_works(self):
        """Leerer thought-String darf nicht crashen."""
        e, ctx = _make_engine(), _make_ctx()
        try:
            result, is_final = await e._execute_tool_call(ctx, _tc("think", {"thought": ""}))
            self.assertFalse(is_final)
        except Exception as ex:
            self.fail(f"Leerer thought crasht: {ex}")

    async def test_08_think_missing_thought_key_no_crash(self):
        """think ohne 'thought' key darf nicht crashen (args={})."""
        e, ctx = _make_engine(), _make_ctx()
        try:
            result, is_final = await e._execute_tool_call(ctx, _tc("think", {}))
            self.assertFalse(is_final)
        except Exception as ex:
            self.fail(f"Fehlendes thought-key crasht: {ex}")

    async def test_09_think_llm_error_returns_string_not_exception(self):
        """Wenn LLM-Call fehlschlägt: result = error-string, kein crash."""
        e, ctx = _make_engine(), _make_ctx()
        e.agent.a_run_llm_completion = AsyncMock(side_effect=RuntimeError("LLM down"))
        result, is_final = await e._execute_tool_call(ctx, _tc("think", {"thought": "x"}))
        self.assertIsInstance(result, str)
        self.assertFalse(is_final)

    async def test_10_think_working_history_entry_role_tool(self):
        """History-Eintrag muss role=tool haben."""
        e, ctx = _make_engine(), _make_ctx()
        await e._execute_tool_call(ctx, _tc("think", {"thought": "x"}, "id_think"))
        last = ctx.working_history[-1]
        self.assertEqual(last["role"], "tool")
        self.assertEqual(last["tool_call_id"], "id_think")

    async def test_11_think_multiple_calls_accumulate_iterations(self):
        e, ctx = _make_engine(), _make_ctx()
        ctx.max_iterations = 10
        await e._execute_tool_call(ctx, _tc("think", {"thought": "a"}))
        await e._execute_tool_call(ctx, _tc("think", {"thought": "b"}))
        self.assertEqual(ctx.max_iterations, 12)


# =============================================================================
# Class 2 — final_answer
# =============================================================================

class TestFinalAnswerTool(unittest.IsolatedAsyncioTestCase):

    async def test_12_final_answer_is_final(self):
        e, ctx = _make_engine(), _make_ctx()
        _, is_final = await e._execute_tool_call(
            ctx, _tc("final_answer", {"answer": "Done", "success": True}))
        self.assertTrue(is_final)

    async def test_13_final_answer_result_equals_answer(self):
        e, ctx = _make_engine(), _make_ctx()
        result, _ = await e._execute_tool_call(
            ctx, _tc("final_answer", {"answer": "Result XYZ"}))
        self.assertEqual(result, "Result XYZ")

    async def test_14_final_answer_not_in_working_history(self):
        e, ctx = _make_engine(), _make_ctx()
        before = len(ctx.working_history)
        await e._execute_tool_call(ctx, _tc("final_answer", {"answer": "Done"}))
        self.assertEqual(len(ctx.working_history), before,
            "final_answer darf niemals in working_history landen")

    async def test_15_final_answer_success_false_still_final(self):
        e, ctx = _make_engine(), _make_ctx()
        _, is_final = await e._execute_tool_call(
            ctx, _tc("final_answer", {"answer": "Failed", "success": False}))
        self.assertTrue(is_final)

    async def test_16_final_answer_empty_answer_string(self):
        e, ctx = _make_engine(), _make_ctx()
        result, is_final = await e._execute_tool_call(
            ctx, _tc("final_answer", {"answer": ""}))
        self.assertTrue(is_final)
        self.assertEqual(result, "")

    async def test_17_final_answer_missing_answer_key_no_crash(self):
        """Fehlendes 'answer' key → leerer string, trotzdem is_final."""
        e, ctx = _make_engine(), _make_ctx()
        try:
            result, is_final = await e._execute_tool_call(
                ctx, _tc("final_answer", {}))
            self.assertTrue(is_final)
        except Exception as ex:
            self.fail(f"Fehlendes answer-key crasht: {ex}")

    async def test_18_final_answer_not_in_auto_focus(self):
        """final_answer soll NICHT in AutoFocus aufgezeichnet werden."""
        e, ctx = _make_engine(), _make_ctx()
        before = list(ctx.auto_focus.actions)
        await e._execute_tool_call(ctx, _tc("final_answer", {"answer": "Done"}))
        # AutoFocus-Einträge dürfen sich nicht erhöht haben
        new_entries = ctx.auto_focus.actions[len(before):]
        self.assertEqual(new_entries, [],
            "final_answer darf nicht in AutoFocus landen")

    async def test_19_final_answer_long_answer(self):
        """Sehr langer answer-String darf nicht crashen."""
        e, ctx = _make_engine(), _make_ctx()
        long_answer = "A" * 10_000
        result, is_final = await e._execute_tool_call(
            ctx, _tc("final_answer", {"answer": long_answer}))
        self.assertTrue(is_final)
        self.assertEqual(result, long_answer)


# =============================================================================
# Class 3 — shift_focus
# =============================================================================

class TestShiftFocusTool(unittest.IsolatedAsyncioTestCase):

    async def test_20_shift_focus_not_final(self):
        e, ctx = _make_engine(), _make_ctx()
        _, is_final = await e._execute_tool_call(ctx, _tc("shift_focus", {
            "summary_of_achievements": "done",
            "next_objective": "next",
        }))
        self.assertFalse(is_final)

    async def test_21_shift_focus_increments_iterations_by_10(self):
        e, ctx = _make_engine(), _make_ctx()
        ctx.max_iterations = 10
        await e._execute_tool_call(ctx, _tc("shift_focus", {
            "summary_of_achievements": "part 1 done",
            "next_objective": "part 2",
        }))
        self.assertEqual(ctx.max_iterations, 20)

    async def test_22_shift_focus_adds_to_history(self):
        e, ctx = _make_engine(), _make_ctx()
        before = len(ctx.working_history)
        await e._execute_tool_call(ctx, _tc("shift_focus", {
            "summary_of_achievements": "x",
            "next_objective": "y",
        }))
        self.assertGreater(len(ctx.working_history), before)

    async def test_23_shift_focus_empty_fields_no_crash(self):
        e, ctx = _make_engine(), _make_ctx()
        try:
            _, is_final = await e._execute_tool_call(ctx, _tc("shift_focus", {
                "summary_of_achievements": "",
                "next_objective": "",
            }))
            self.assertFalse(is_final)
        except Exception as ex:
            self.fail(f"Leere shift_focus-Felder crashen: {ex}")

    async def test_24_shift_focus_missing_keys_no_crash(self):
        e, ctx = _make_engine(), _make_ctx()
        try:
            _, is_final = await e._execute_tool_call(ctx, _tc("shift_focus", {}))
            self.assertFalse(is_final)
        except Exception as ex:
            self.fail(f"Fehlende shift_focus-Keys crashen: {ex}")

    async def test_25_shift_focus_multiple_calls_accumulate(self):
        e, ctx = _make_engine(), _make_ctx()
        ctx.max_iterations = 5
        for i in range(3):
            await e._execute_tool_call(ctx, _tc("shift_focus", {
                "summary_of_achievements": f"phase {i}",
                "next_objective": f"phase {i+1}",
            }))
        self.assertEqual(ctx.max_iterations, 35)

    async def test_26_shift_focus_history_entry_contains_summary(self):
        """History-Eintrag nach shift_focus soll Summary-Info enthalten."""
        e, ctx = _make_engine(), _make_ctx()
        await e._execute_tool_call(ctx, _tc("shift_focus", {
            "summary_of_achievements": "UNIQUE_SUMMARY_TOKEN_XYZ",
            "next_objective": "next",
        }))
        history_text = str(ctx.working_history)
        self.assertIn("UNIQUE_SUMMARY_TOKEN_XYZ", history_text)


# =============================================================================
# Class 4 — list_tools / load_tools
# =============================================================================

class TestDiscoveryTools(unittest.IsolatedAsyncioTestCase):

    async def test_27_list_tools_not_final(self):
        e, ctx = _make_engine(), _make_ctx()
        _, is_final = await e._execute_tool_call(ctx, _tc("list_tools", {}))
        self.assertFalse(is_final)

    async def test_28_list_tools_returns_string(self):
        e, ctx = _make_engine(), _make_ctx()
        result, _ = await e._execute_tool_call(ctx, _tc("list_tools", {}))
        self.assertIsInstance(result, str)

    async def test_29_list_tools_increments_iterations(self):
        e, ctx = _make_engine(), _make_ctx()
        before = ctx.max_iterations
        await e._execute_tool_call(ctx, _tc("list_tools", {}))
        self.assertEqual(ctx.max_iterations, before + 1)

    async def test_30_list_tools_with_category_filter(self):
        """list_tools mit category-Filter darf nicht crashen."""
        e, ctx = _make_engine(), _make_ctx()
        try:
            result, _ = await e._execute_tool_call(
                ctx, _tc("list_tools", {"category": "discord"}))
            self.assertIsInstance(result, str)
        except Exception as ex:
            self.fail(f"list_tools mit category crasht: {ex}")

    async def test_31_list_tools_adds_to_history(self):
        e, ctx = _make_engine(), _make_ctx()
        before = len(ctx.working_history)
        await e._execute_tool_call(ctx, _tc("list_tools", {}))
        self.assertGreater(len(ctx.working_history), before)

    async def test_32_load_tools_not_final(self):
        e, ctx = _make_engine(), _make_ctx()
        _, is_final = await e._execute_tool_call(
            ctx, _tc("load_tools", {"tools": ["vfs_read"]}))
        self.assertFalse(is_final)

    async def test_33_load_tools_string_input(self):
        """load_tools akzeptiert einzelnen String (nicht nur Liste)."""
        e, ctx = _make_engine(), _make_ctx()
        try:
            _, is_final = await e._execute_tool_call(
                ctx, _tc("load_tools", {"tools": "vfs_read"}))
            self.assertFalse(is_final)
        except Exception as ex:
            self.fail(f"load_tools mit string crasht: {ex}")

    async def test_34_load_tools_names_alias(self):
        """load_tools akzeptiert 'names' als alias für 'tools'."""
        e, ctx = _make_engine(), _make_ctx()
        try:
            _, is_final = await e._execute_tool_call(
                ctx, _tc("load_tools", {"names": ["vfs_read"]}))
            self.assertFalse(is_final)
        except Exception as ex:
            self.fail(f"load_tools mit names-alias crasht: {ex}")

    async def test_35_load_tools_empty_list_no_crash(self):
        e, ctx = _make_engine(), _make_ctx()
        try:
            _, is_final = await e._execute_tool_call(
                ctx, _tc("load_tools", {"tools": []}))
            self.assertFalse(is_final)
        except Exception as ex:
            self.fail(f"load_tools mit lerer Liste crasht: {ex}")

    async def test_36_load_tools_increments_iterations(self):
        e, ctx = _make_engine(), _make_ctx()
        before = ctx.max_iterations
        await e._execute_tool_call(ctx, _tc("load_tools", {"tools": ["x"]}))
        self.assertEqual(ctx.max_iterations, before + 1)


# =============================================================================
# Class 5 — Tool Call Integrity (JSON, History, Orphans)
# =============================================================================

class TestToolCallIntegrity(unittest.IsolatedAsyncioTestCase):

    async def test_37_malformed_json_args_no_crash(self):
        """Malformed JSON in arguments darf nicht crashen."""
        e, ctx = _make_engine(), _make_ctx()
        tc = MagicMock()
        tc.id = "call_bad"
        tc.function.name = "think"
        tc.function.arguments = "{not valid json!!}"
        try:
            result, is_final = await e._execute_tool_call(ctx, tc)
            self.assertFalse(is_final)
        except Exception as ex:
            self.fail(f"Malformed JSON crasht: {ex}")

    async def test_38_history_entry_has_tool_call_id(self):
        e, ctx = _make_engine(), _make_ctx()
        tc_id = "call_integrity_01"
        await e._execute_tool_call(ctx, _tc("think", {"thought": "x"}, tc_id))
        last = ctx.working_history[-1]
        self.assertEqual(last["role"], "tool")
        self.assertIn("tool_call_id", last, "tool_call_id fehlt (MiniMax 400!)")
        self.assertEqual(last["tool_call_id"], tc_id)

    async def test_39_minimax_style_id_preserved(self):
        e, ctx = _make_engine(), _make_ctx()
        mm_id = "call_function_hn4sxvjzffmj_1"
        await e._execute_tool_call(ctx, _tc("think", {"thought": "t"}, mm_id))
        last = ctx.working_history[-1]
        self.assertEqual(last["tool_call_id"], mm_id)

    async def test_40_history_entry_has_content(self):
        e, ctx = _make_engine(), _make_ctx()
        await e._execute_tool_call(ctx, _tc("think", {"thought": "x"}, "call_c"))
        last = ctx.working_history[-1]
        self.assertIn("content", last)
        self.assertIsNotNone(last["content"])

    async def test_41_sequence_no_orphan_ids(self):
        """think → list_tools → think produziert keine Orphan-IDs."""
        e, ctx = _make_engine(), _make_ctx()
        calls = [
            ("think",      {"thought": "step 1"}, "call_001"),
            ("list_tools", {},                    "call_002"),
            ("think",      {"thought": "step 3"}, "call_003"),
        ]
        for name, args, tc_id in calls:
            ctx.working_history.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tc_id, "type": "function",
                    "function": {"name": name, "arguments": "{}"}
                }],
            })
            await e._execute_tool_call(ctx, _tc(name, args, tc_id))

        orphans = _collect_orphan_ids(ctx.working_history)
        self.assertEqual(orphans, [],
            f"Orphan tool_call_ids nach Sequenz: {orphans}")

    async def test_42_final_answer_leaves_no_orphan(self):
        """final_answer (nicht in history) darf keine Orphan-ID erzeugen."""
        e, ctx = _make_engine(), _make_ctx()
        fa_id = "call_final_01"
        ctx.working_history.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": fa_id, "type": "function",
                "function": {"name": "final_answer", "arguments": "{}"}
            }],
        })
        await e._execute_tool_call(ctx, _tc("final_answer", {"answer": "done"}, fa_id))
        # final_answer landet NICHT in history → fa_id hat kein tool-response
        # das ist KORREKT — aber wir verifizieren, dass die engine das weiß
        tool_ids = {m.get("tool_call_id") for m in ctx.working_history
                    if m.get("role") == "tool"}
        self.assertNotIn(fa_id, tool_ids,
            "final_answer darf keinen tool-response-Eintrag erzeugen")

    async def test_43_unknown_tool_no_exception(self):
        """Unbekanntes Tool wirft keine Exception nach außen."""
        e, ctx = _make_engine(), _make_ctx()
        e.agent.arun_function = AsyncMock(side_effect=RuntimeError("not loaded"))
        try:
            result, is_final = await e._execute_tool_call(
                ctx, _tc("completely_unknown_xyz", {"x": 1}))
            self.assertFalse(is_final)
            self.assertIsInstance(result, str)
        except Exception as ex:
            self.fail(f"Unbekanntes Tool darf nicht crashen: {ex}")

    async def test_44_tools_used_tracked(self):
        """Jeder Tool-Call muss ctx.tools_used ergänzen."""
        e, ctx = _make_engine(), _make_ctx()
        await e._execute_tool_call(ctx, _tc("think", {"thought": "x"}))
        await e._execute_tool_call(ctx, _tc("list_tools", {}))
        self.assertIn("think", ctx.tools_used)
        self.assertIn("list_tools", ctx.tools_used)

    async def test_45_loop_detection_called(self):
        """loop_detector.record() wird bei jedem Tool-Call aufgerufen."""
        e, ctx = _make_engine(), _make_ctx()
        ctx.loop_detector.record = MagicMock(return_value=False)
        await e._execute_tool_call(ctx, _tc("think", {"thought": "x"}, "lp1"))
        ctx.loop_detector.record.assert_called_once()


# =============================================================================
# Class 6 — wait_for Semantic Validator (deterministisch, kein LLM)
# =============================================================================

class TestWaitForSemanticValidator(unittest.TestCase):
    """
    Deterministischer Validator für wait_for-Results.
    Erkennt den bekannten Bug: timeout + output_path + leere files.
    Kein LLM-Aufruf — pure logic.
    """

    def test_46_clean_success_no_issues(self):
        result = {
            "status": "success",
            "output_dir": "/sub/workspace/output",
            "files": ["report.md", "data.json"],
            "error": "",
        }
        issues = validate_wait_for_result(result)
        self.assertEqual(issues, [])

    def test_47_timeout_with_path_and_no_files_is_misleading(self):
        """DER bekannte Bug aus dem Output-Dump."""
        result = {
            "status": "timeout",
            "output": "/sub/workspace/vault_analysis",
            "files": [],
            "error": "Timeout after 120 seconds",
        }
        issues = validate_wait_for_result(result)
        self.assertEqual(len(issues), 1)
        self.assertIn("MISLEADING", issues[0])
        self.assertIn("vault_analysis", issues[0])

    def test_48_timeout_with_path_and_no_files_key_at_all(self):
        """files-key fehlt komplett (nicht nur leer)."""
        result = {
            "status": "timeout",
            "output_dir": "/some/path",
            "error": "Timeout after 60 seconds",
        }
        issues = validate_wait_for_result(result)
        self.assertTrue(any("MISLEADING" in i for i in issues))

    def test_49_timeout_without_path_is_ok(self):
        """Timeout ohne output_dir — kein irreführender Path."""
        result = {
            "status": "timeout",
            "output_dir": "",
            "files": [],
            "error": "Timeout after 120 seconds",
        }
        issues = validate_wait_for_result(result)
        self.assertFalse(any("MISLEADING" in i for i in issues))

    def test_50_failed_without_error_text_incomplete(self):
        result = {
            "status": "failed",
            "output_dir": "",
            "files": [],
            "error": "",
        }
        issues = validate_wait_for_result(result)
        self.assertTrue(any("INCOMPLETE" in i for i in issues))

    def test_51_success_no_files_no_dir_suspicious(self):
        result = {
            "status": "success",
            "output_dir": "",
            "files": [],
            "error": "",
        }
        issues = validate_wait_for_result(result)
        self.assertTrue(any("SUSPICIOUS" in i for i in issues))

    def test_52_files_without_output_dir_inconsistent(self):
        result = {
            "status": "success",
            "output_dir": "",
            "files": ["result.md"],
            "error": "",
        }
        issues = validate_wait_for_result(result)
        self.assertTrue(any("INCONSISTENT" in i for i in issues))

    def test_53_two_agents_both_timeout_misleading(self):
        """Exakt der Output-Dump: beide Sub-Agents, beide misleading."""
        agent_results = [
            {
                "status": "timeout",
                "output": "/sub/workspace/vault_analysis",
                "files": [],
                "error": "Timeout after 120 seconds",
            },
            {
                "status": "timeout",
                "output": "/sub/workspace/vault_analysis",
                "files": [],
                "error": "Timeout after 120 seconds",
            },
        ]
        for r in agent_results:
            issues = validate_wait_for_result(r)
            self.assertTrue(
                any("MISLEADING" in i for i in issues),
                f"Sub-agent result sollte als MISLEADING erkannt werden: {r}"
            )

    def test_54_clean_timeout_no_path_no_files(self):
        """Timeout komplett ohne Path/Files — klar kommuniziert, kein Bug."""
        result = {
            "status": "timeout",
            "output_dir": None,
            "files": [],
            "error": "Timeout after 120 seconds",
        }
        issues = validate_wait_for_result(result)
        self.assertFalse(any("MISLEADING" in i for i in issues))

    def test_55_validator_is_pure_no_side_effects(self):
        """Validator mutiert das Input-Dict nicht."""
        result = {
            "status": "timeout",
            "output": "/path",
            "files": [],
            "error": "Timeout",
        }
        original = dict(result)
        validate_wait_for_result(result)
        self.assertEqual(result, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
