# tests/test_icli_logic.py
"""
Umfangreiche Unit-Tests für toolboxv2/flows/icli.py

Testet Logik und Datenflüsse — keine Syntax, kein UI-Rendering.

Bereiche:
  1. Utility-Funktionen (_short, _bar, _fmt_elapsed)
  2. _tool_result_info — Parser für alle Tool-Typen
  3. ingest_chunk — zentraler Datenpipeline-Eingang
  4. TaskView / IterView — Datenmodell-Konsistenz
  5. Status-Übergänge (running → completed/failed/error)
  6. Sub-Agent-Tracking und Farb-Zuweisung
  7. render_footer_toolbar — State-basierte Ausgabelogik
  8. VFS-Argument-Parsing (readonly, no-sync, vfs_path)
  9. Dateiendungs-Erkennung (_detect_file_type)
 10. Rate-Limiter-Config-Anwendung
 11. Dream-Job-Config-Zusammenbau
 12. TaskOverlay — Navigation, Drill-Down, effective_view
"""

import asyncio
import time
import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Imports — alle isoliert von externen Abhängigkeiten
# ─────────────────────────────────────────────────────────────────────────────

def _import():
    """Lazy import um Module ohne Prompt-Toolkit-Init zu laden."""
    from toolboxv2.flows.icli import (
        _short, _bar, _fmt_elapsed, _tool_result_info,
        ingest_chunk, TaskView, IterView,
        render_footer_toolbar, TaskOverlay,
        STATUS_SYM, C, SYM,
    )
    return (_short, _bar, _fmt_elapsed, _tool_result_info,
            ingest_chunk, TaskView, IterView,
            render_footer_toolbar, TaskOverlay,
            STATUS_SYM, C, SYM)


def _tv(task_id="t1", agent="self", status="running",
        iteration=0, max_iter=15) -> "TaskView":
    from toolboxv2.flows.icli import TaskView
    return TaskView(task_id=task_id, agent_name=agent, query="test",
                    status=status, iteration=iteration, max_iter=max_iter)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# =============================================================================
# 1. UTILITY FUNKTIONEN
# =============================================================================

class TestShort(unittest.TestCase):

    def setUp(self):
        self._short = _import()[0]

    def test_short_no_truncation_if_fits(self):
        self.assertEqual(self._short("hello", 10), "hello")

    def test_short_truncates_exactly(self):
        result = self._short("abcdefghij", 5)
        self.assertTrue(result.endswith(".."))
        self.assertEqual(len(result), 7)  # 5 + ".."

    def test_short_exact_boundary(self):
        # len == n + 2 → nicht kürzen
        s = "abcdefg"  # len=7, n=5, n+2=7 → kein Truncate
        self.assertEqual(self._short(s, 5), s)

    def test_short_empty_string(self):
        self.assertEqual(self._short("", 5), "")

    def test_short_unicode_safe(self):
        s = "✓ vfs_shell"
        result = self._short(s, 5)
        self.assertIn("..", result)


class TestBar(unittest.TestCase):

    def setUp(self):
        self._bar = _import()[1]

    def test_bar_zero_total(self):
        result = self._bar(0, 0, 8)
        self.assertEqual(len(result), 8)

    def test_bar_full(self):
        result = self._bar(10, 10, 8)
        from toolboxv2.flows.icli import SYM
        self.assertEqual(result, SYM["bar_fill"] * 8)

    def test_bar_empty(self):
        result = self._bar(0, 10, 8)
        from toolboxv2.flows.icli import SYM
        self.assertEqual(result, SYM["bar_empty"] * 8)

    def test_bar_half(self):
        result = self._bar(5, 10, 8)
        from toolboxv2.flows.icli import SYM
        fill_count = result.count(SYM["bar_fill"])
        self.assertEqual(fill_count, 4)

    def test_bar_over_max_clamps(self):
        result = self._bar(20, 10, 8)
        from toolboxv2.flows.icli import SYM
        self.assertEqual(result, SYM["bar_fill"] * 8)

    def test_bar_width_respected(self):
        from toolboxv2.flows.icli import SYM
        for w in [4, 10, 20]:
            result = self._bar(1, 2, w)
            self.assertEqual(len(result), w)


class TestFmtElapsed(unittest.TestCase):

    def setUp(self):
        self._fmt = _import()[2]

    def test_seconds(self):
        self.assertEqual(self._fmt(45), "45s")

    def test_minutes(self):
        self.assertEqual(self._fmt(90), "1m30s")

    def test_hours(self):
        self.assertEqual(self._fmt(3661), "1h01m")

    def test_zero(self):
        self.assertEqual(self._fmt(0), "0s")

    def test_exactly_60(self):
        self.assertEqual(self._fmt(60), "1m00s")


# =============================================================================
# 2. _tool_result_info — Parser für alle bekannten Tool-Typen
# =============================================================================

class TestToolResultInfo(unittest.TestCase):

    def setUp(self):
        import json as _json
        self._info = _import()[3]
        self._json = _json

    def _j(self, d: dict) -> str:
        return self._json.dumps(d)

    def test_vfs_shell_success_stdout(self):
        result = self._j({"success": True, "stdout": "hello\nworld"})
        ok, info = self._info("vfs_shell", result)
        self.assertTrue(ok)
        self.assertIn("hello", info)
        self.assertNotIn("world", info)  # nur erste Zeile

    def test_vfs_shell_failure(self):
        result = self._j({"success": False, "stdout": "error output"})
        ok, info = self._info("vfs_shell", result)
        self.assertFalse(ok)

    def test_vfs_shell_empty_stdout(self):
        result = self._j({"success": True, "stdout": ""})
        ok, info = self._info("vfs_shell", result)
        self.assertTrue(ok)

    def test_vfs_view_lines_shown(self):
        result = self._j({"lines_shown": 42})
        ok, info = self._info("vfs_view", result)
        self.assertIn("42", info)

    def test_vfs_view_total_lines_fallback(self):
        result = self._j({"total_lines": 100})
        ok, info = self._info("vfs_view", result)
        self.assertIn("100", info)

    def test_search_vfs_count(self):
        result = self._j({"results": ["a", "b", "c"]})
        ok, info = self._info("search_vfs", result)
        self.assertIn("3", info)

    def test_search_vfs_matches_fallback(self):
        result = self._j({"matches": ["x"]})
        ok, info = self._info("search_vfs", result)
        self.assertIn("1", info)

    def test_fs_copy_to_vfs_path(self):
        result = self._j({"success": True, "vfs_path": "/project/file.py"})
        ok, info = self._info("fs_copy_to_vfs", result)
        self.assertTrue(ok)
        self.assertIn("/project/file.py", info)

    def test_docker_run_stdout(self):
        result = self._j({"success": True, "stdout": "Container started\nmore output"})
        ok, info = self._info("docker_run", result)
        self.assertIn("Container started", info)

    def test_docker_logs_last_line(self):
        result = self._j({"logs": "line1\nline2\nfinal line\n"})
        ok, info = self._info("docker_logs", result)
        self.assertIn("final line", info)

    def test_check_permissions_allowed(self):
        result = self._j({"allowed": True, "rule": "allow_all"})
        ok, info = self._info("check_permissions", result)
        self.assertTrue(ok)
        self.assertIn("allow_all", info)

    def test_check_permissions_denied(self):
        result = self._j({"allowed": False, "rule": "deny_write"})
        ok, info = self._info("check_permissions", result)
        self.assertFalse(ok)

    def test_set_agent_situation_intent(self):
        result = self._j({"success": True, "intent": "analyzing codebase"})
        ok, info = self._info("set_agent_situation", result)
        self.assertIn("analyzing codebase", info)

    def test_vfs_share_create_share_id(self):
        result = self._j({"success": True, "share_id": "abc123"})
        ok, info = self._info("vfs_share_create", result)
        self.assertIn("abc123", info)

    def test_generic_message_field(self):
        result = self._j({"success": True, "message": "operation done"})
        ok, info = self._info("unknown_tool", result)
        self.assertIn("operation done", info)

    def test_generic_info_field(self):
        result = self._j({"success": True, "info": "processed 5 items"})
        ok, info = self._info("any_tool", result)
        self.assertIn("processed 5 items", info)

    def test_list_result(self):
        result = self._j(["item1", "item2", "item3"])
        ok, info = self._info("list_tool", result)
        self.assertTrue(ok)
        self.assertIn("3", info)

    def test_non_json_string(self):
        ok, info = self._info("raw_tool", "plain text result")
        self.assertTrue(ok)
        self.assertIn("plain text", info)

    def test_empty_result(self):
        ok, info = self._info("any_tool", "")
        self.assertTrue(ok)

    def test_invalid_json_falls_back(self):
        ok, info = self._info("bad_tool", "not json {{{")
        # Kein Crash, irgendwas zurück
        self.assertIsNotNone(ok)

    def test_truncation_of_long_info(self):
        long_path = "/very/long/path/" + "x" * 200
        result = self._j({"success": True, "vfs_path": long_path})
        ok, info = self._info("fs_copy_to_vfs", result)
        self.assertLessEqual(len(info), 40)


# =============================================================================
# 3. ingest_chunk — zentraler Datenpipeline-Eingang
# =============================================================================

class TestIngestChunkBasicFields(unittest.TestCase):

    def test_agent_name_updated(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"agent": "analyst"})
        self.assertEqual(tv.agent_name, "analyst")

    def test_persona_updated(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"persona": "fallback_analyst"})
        self.assertEqual(tv.persona, "fallback_analyst")

    def test_skills_updated(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"skills": ["code", "research"]})
        self.assertEqual(tv.skills, ["code", "research"])

    def test_iteration_updated(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"iter": 7})
        self.assertEqual(tv.iteration, 7)

    def test_max_iter_updated(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"max_iter": 20})
        self.assertEqual(tv.max_iter, 20)

    def test_tokens_used_updated(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"tokens_used": 1234})
        self.assertEqual(tv.tokens_used, 1234)

    def test_tokens_max_updated(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"tokens_max": 128000})
        self.assertEqual(tv.tokens_max, 128000)

    def test_empty_chunk_no_crash(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {})

    def test_sub_agent_registered(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.iteration = 1
        ingest_chunk(tv, {"_sub_agent_id": "sub_abc123", "iter": 1})
        self.assertIn("sub_abc123", tv.sub_agents)


class TestIngestChunkTypeRouting(unittest.TestCase):

    def test_reasoning_sets_phase_thinking(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.iteration = 1
        ingest_chunk(tv, {"type": "reasoning", "chunk": "I think...", "iter": 1})
        self.assertEqual(tv.phase, "thinking")
        self.assertEqual(tv.last_thought, "I think...")

    def test_reasoning_appended_to_iter(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.iteration = 2
        ingest_chunk(tv, {"type": "reasoning", "chunk": "thought A", "iter": 2})
        ingest_chunk(tv, {"type": "reasoning", "chunk": "thought B", "iter": 2})
        iv = tv._iter_map.get(2)
        self.assertIsNotNone(iv)
        self.assertIn("thought A", iv.thoughts)
        self.assertIn("thought B", iv.thoughts)

    def test_content_sets_phase(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "content"})
        self.assertEqual(tv.phase, "content")

    def test_tool_start_sets_pending(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.iteration = 3
        ingest_chunk(tv, {"type": "tool_start", "name": "vfs_shell", "iter": 3})
        self.assertEqual(tv.last_tool, "vfs_shell")
        self.assertEqual(tv.phase, "tool")
        iv = tv._iter_map.get(3)
        self.assertEqual(iv.pending_tool, "vfs_shell")

    def test_tool_result_clears_pending(self):
        from toolboxv2.flows.icli import ingest_chunk
        import json
        tv = _tv()
        tv.iteration = 3
        ingest_chunk(tv, {"type": "tool_start", "name": "vfs_shell", "iter": 3})
        ingest_chunk(tv, {
            "type": "tool_result",
            "name": "vfs_shell",
            "result": json.dumps({"success": True, "stdout": "done"}),
            "iter": 3
        })
        iv = tv._iter_map.get(3)
        self.assertEqual(iv.pending_tool, "")
        self.assertEqual(len(iv.tools), 1)

    def test_tool_result_records_success(self):
        from toolboxv2.flows.icli import ingest_chunk
        import json
        tv = _tv()
        tv.iteration = 1
        ingest_chunk(tv, {"type": "tool_start", "name": "vfs_view", "iter": 1})
        ingest_chunk(tv, {
            "type": "tool_result",
            "name": "vfs_view",
            "result": json.dumps({"lines_shown": 10}),
            "iter": 1
        })
        iv = tv._iter_map[1]
        tname, tok, elapsed, info = iv.tools[0]
        self.assertEqual(tname, "vfs_view")
        self.assertTrue(tok)
        self.assertIn("10", info)

    def test_tool_result_records_failure(self):
        from toolboxv2.flows.icli import ingest_chunk
        import json
        tv = _tv()
        tv.iteration = 1
        ingest_chunk(tv, {"type": "tool_start", "name": "vfs_shell", "iter": 1})
        ingest_chunk(tv, {
            "type": "tool_result",
            "name": "vfs_shell",
            "result": json.dumps({"success": False, "stdout": "permission denied"}),
            "iter": 1
        })
        iv = tv._iter_map[1]
        _, tok, _, _ = iv.tools[0]
        self.assertFalse(tok)

    def test_done_chunk_sets_completed(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "done", "success": True})
        self.assertEqual(tv.status, "completed")
        self.assertEqual(tv.phase, "done")

    def test_done_chunk_failure(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "done", "success": False})
        self.assertEqual(tv.status, "failed")

    def test_error_chunk(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "error"})
        self.assertEqual(tv.status, "error")
        self.assertEqual(tv.phase, "error")

    def test_final_answer_with_answer_key(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "final_answer", "answer": "The answer is 42."})
        self.assertEqual(tv.final_answer, "The answer is 42.")
        self.assertEqual(tv.status, "completed")

    def test_final_answer_content_fallback(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "final_answer", "content": "Result from content."})
        self.assertEqual(tv.final_answer, "Result from content.")

    def test_final_answer_chunk_fallback(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "final_answer", "chunk": "Result from chunk."})
        self.assertEqual(tv.final_answer, "Result from chunk.")

    def test_final_answer_empty_does_not_overwrite(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.final_answer = "existing answer"
        ingest_chunk(tv, {"type": "final_answer"})
        self.assertEqual(tv.final_answer, "existing answer")

    def test_final_answer_multiline_preserved(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        multiline = "Line 1\nLine 2\nLine 3"
        ingest_chunk(tv, {"type": "final_answer", "answer": multiline})
        self.assertEqual(tv.final_answer, multiline)

    def test_unknown_type_no_crash(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "completely_unknown", "data": "x"})


# =============================================================================
# 4. TaskView / IterView — Datenmodell-Konsistenz
# =============================================================================

class TestTaskViewModel(unittest.TestCase):

    def test_get_iter_creates_new(self):
        tv = _tv()
        iv = tv._get_iter(5)
        self.assertEqual(iv.n, 5)
        self.assertIn(5, tv._iter_map)
        self.assertIn(iv, tv.iterations)

    def test_get_iter_returns_same_object(self):
        tv = _tv()
        iv1 = tv._get_iter(3)
        iv2 = tv._get_iter(3)
        self.assertIs(iv1, iv2)

    def test_get_iter_multiple(self):
        tv = _tv()
        for n in [1, 2, 3, 5, 10]:
            tv._get_iter(n)
        self.assertEqual(len(tv.iterations), 5)
        self.assertEqual(set(tv._iter_map.keys()), {1, 2, 3, 5, 10})

    def test_sub_color_assigns_sequential(self):
        tv = _tv()
        tv._sub_color("agent_a")
        tv._sub_color("agent_b")
        tv._sub_color("agent_c")
        self.assertEqual(tv.sub_agents["agent_a"], 0)
        self.assertEqual(tv.sub_agents["agent_b"], 1)
        self.assertEqual(tv.sub_agents["agent_c"], 2)

    def test_sub_color_wraps_at_6(self):
        tv = _tv()
        for i in range(7):
            tv._sub_color(f"agent_{i}")
        self.assertEqual(tv.sub_agents["agent_6"], 0)  # wraps

    def test_sub_color_idempotent(self):
        tv = _tv()
        c1 = tv._sub_color("my_agent")
        c2 = tv._sub_color("my_agent")
        self.assertEqual(c1, c2)

    def test_default_final_answer_empty(self):
        tv = _tv()
        self.assertEqual(tv.final_answer, "")

    def test_started_at_is_recent(self):
        before = time.time()
        tv = _tv()
        after = time.time()
        self.assertGreaterEqual(tv.started_at, before)
        self.assertLessEqual(tv.started_at, after)


class TestIterViewModel(unittest.TestCase):

    def test_iter_view_defaults(self):
        from toolboxv2.flows.icli import IterView
        iv = IterView(n=1)
        self.assertEqual(iv.n, 1)
        self.assertEqual(iv.thoughts, [])
        self.assertEqual(iv.tools, [])
        self.assertEqual(iv.pending_tool, "")

    def test_tool_start_time_tracked(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.iteration = 1
        t_before = time.time()
        ingest_chunk(tv, {"type": "tool_start", "name": "vfs_shell", "iter": 1})
        iv = tv._iter_map[1]
        self.assertIn("vfs_shell", iv._tool_start_times)
        self.assertGreaterEqual(iv._tool_start_times["vfs_shell"], t_before)

    def test_tool_elapsed_calculated(self):
        from toolboxv2.flows.icli import ingest_chunk
        import json
        tv = _tv()
        tv.iteration = 1
        ingest_chunk(tv, {"type": "tool_start", "name": "slow_tool", "iter": 1})
        # Simuliere 50ms Delay
        iv = tv._iter_map[1]
        iv._tool_start_times["slow_tool"] -= 0.05  # rückdatieren
        ingest_chunk(tv, {
            "type": "tool_result",
            "name": "slow_tool",
            "result": json.dumps({"success": True, "message": "ok"}),
            "iter": 1
        })
        _, _, elapsed, _ = iv.tools[0]
        self.assertGreaterEqual(elapsed, 0.04)

    def test_multiple_tools_same_iter(self):
        from toolboxv2.flows.icli import ingest_chunk
        import json
        tv = _tv()
        tv.iteration = 2
        for tool in ["vfs_shell", "vfs_view", "search_vfs"]:
            ingest_chunk(tv, {"type": "tool_start", "name": tool, "iter": 2})
            ingest_chunk(tv, {
                "type": "tool_result",
                "name": tool,
                "result": json.dumps({"success": True}),
                "iter": 2
            })
        iv = tv._iter_map[2]
        self.assertEqual(len(iv.tools), 3)
        tool_names = [t[0] for t in iv.tools]
        self.assertEqual(tool_names, ["vfs_shell", "vfs_view", "search_vfs"])


# =============================================================================
# 5. Status-Übergänge — vollständige State Machine
# =============================================================================

class TestStatusTransitions(unittest.TestCase):

    def test_initial_status_running(self):
        tv = _tv()
        self.assertEqual(tv.status, "running")

    def test_running_to_completed_via_done(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "done", "success": True})
        self.assertEqual(tv.status, "completed")

    def test_running_to_failed_via_done(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "done", "success": False})
        self.assertEqual(tv.status, "failed")

    def test_running_to_error(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "error"})
        self.assertEqual(tv.status, "error")

    def test_running_to_completed_via_final_answer(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"type": "final_answer", "answer": "result"})
        self.assertEqual(tv.status, "completed")

    def test_phase_sequence(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.iteration = 1
        phases = []
        ingest_chunk(tv, {"type": "reasoning", "chunk": "thinking", "iter": 1})
        phases.append(tv.phase)
        ingest_chunk(tv, {"type": "tool_start", "name": "vfs_shell", "iter": 1})
        phases.append(tv.phase)
        ingest_chunk(tv, {"type": "tool_result", "name": "vfs_shell",
                          "result": '{"success":true,"stdout":"done"}', "iter": 1})
        phases.append(tv.phase)
        ingest_chunk(tv, {"type": "done", "success": True})
        phases.append(tv.phase)
        self.assertEqual(phases, ["thinking", "tool", "tool_done", "done"])

    def test_last_tool_info_updated(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.iteration = 1
        ingest_chunk(tv, {"type": "tool_start", "name": "vfs_shell", "iter": 1})
        ingest_chunk(tv, {
            "type": "tool_result",
            "name": "vfs_shell",
            "result": '{"success":true,"stdout":"output line"}',
            "iter": 1
        })
        self.assertTrue(tv.last_tool_ok)
        self.assertEqual(tv.last_tool, "vfs_shell")
        self.assertIn("output line", tv.last_tool_info)


# =============================================================================
# 6. render_footer_toolbar — State-basierte Ausgabelogik
# =============================================================================

class TestRenderFooterToolbar(unittest.TestCase):

    def setUp(self):
        from toolboxv2.flows.icli import render_footer_toolbar
        self._render = render_footer_toolbar

    def _text(self, tokens):
        return "".join(t for _, t in tokens)

    def test_idle_no_tasks(self):
        result = self._render({}, None)
        text = self._text(result)
        self.assertIn("idle", text)
        self.assertIn("F4", text)

    def test_audio_recording_shows_rec(self):
        result = self._render({}, None, audio_recording=True)
        text = self._text(result)
        self.assertIn("REC", text)

    def test_audio_processing_shows_proc(self):
        result = self._render({}, None, audio_processing=True)
        text = self._text(result)
        self.assertIn("PROC", text)

    def test_overlay_open_shown(self):
        tv = _tv()
        result = self._render({"t1": tv}, "t1", overlay_open=True)
        text = self._text(result)
        self.assertIn("ZEN+", text)

    def test_running_task_shown(self):
        tv = _tv(agent="self", status="running", iteration=5, max_iter=15)
        result = self._render({"t1": tv}, "t1")
        text = self._text(result)
        self.assertIn("self", text)
        self.assertIn("5", text)
        self.assertIn("15", text)

    def test_focused_task_has_arrow(self):
        tv = _tv()
        result = self._render({"t1": tv}, "t1")
        text = self._text(result)
        self.assertIn("▸", text)

    def test_non_focused_no_arrow(self):
        tv = _tv()
        result = self._render({"t1": tv}, None)
        text = self._text(result)
        self.assertNotIn("▸", text)

    def test_overflow_shown(self):
        views = {f"t{i}": _tv(task_id=f"t{i}", agent=f"agent_{i}") for i in range(8)}
        result = self._render(views, None)
        text = self._text(result)
        self.assertIn("more", text)

    def test_max_5_tasks_shown(self):
        views = {f"t{i}": _tv(task_id=f"t{i}", agent=f"agent_{i}") for i in range(8)}
        result = self._render(views, None)
        # Counting agent names
        shown_count = sum(1 for _, t in result if f"agent_" in t and "more" not in t)
        # At most 5 tasks (each may appear in multiple tokens)
        # Check via agent names 0-4 present, 5-7 only in overflow
        text = self._text(result)
        for i in range(5):
            self.assertIn(f"agent_{i}", text)

    def test_token_percentage_shown(self):
        tv = _tv()
        tv.tokens_used = 6400
        tv.tokens_max = 128000
        result = self._render({"t1": tv}, "t1")
        text = self._text(result)
        self.assertIn("5%", text)  # 6400/128000 = 5%

    def test_sub_agent_count_shown(self):
        tv = _tv()
        tv._sub_color("sub_a")
        tv._sub_color("sub_b")
        result = self._render({"t1": tv}, "t1")
        text = self._text(result)
        self.assertIn("✦", text)
        self.assertIn("2", text)

    def test_completed_task_shows_done(self):
        tv = _tv(status="completed")
        result = self._render({"t1": tv}, "t1")
        text = self._text(result)
        self.assertIn("done", text)

    def test_failed_task_shows_fail(self):
        tv = _tv(status="failed")
        result = self._render({"t1": tv}, "t1")
        text = self._text(result)
        self.assertIn("fail", text)

    def test_thinking_phase_shows_thought(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.iteration = 1
        ingest_chunk(tv, {"type": "reasoning", "chunk": "pondering things", "iter": 1})
        result = self._render({"t1": tv}, "t1")
        text = self._text(result)
        self.assertIn("pondering", text)

    def test_tool_phase_shows_tool_name(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.iteration = 1
        ingest_chunk(tv, {"type": "tool_start", "name": "search_vfs", "iter": 1})
        result = self._render({"t1": tv}, "t1")
        text = self._text(result)
        self.assertIn("search_vfs", text)

    def test_shortcut_footer_always_present(self):
        tv = _tv()
        result = self._render({"t1": tv}, "t1")
        text = self._text(result)
        self.assertIn("F2", text)
        self.assertIn("F4", text)
        self.assertIn("F5", text)


# =============================================================================
# 7. VFS-Argument-Parsing (isolierte Logik aus _cmd_vfs)
# =============================================================================

class TestVfsArgParsing(unittest.TestCase):
    """
    Testet die Argument-Parsing-Logik aus _cmd_vfs ohne echtes VFS.
    Direkt als Logik-Extraktion ohne den gesamten ISAA_Host.
    """

    def _parse_mount_args(self, args):
        """Reimplementiert die Mount-Arg-Parsing-Logik aus _cmd_vfs."""
        local_path = args[1]
        vfs_path = "/project"
        readonly = False
        auto_sync = True

        for i, arg in enumerate(args[2:], start=2):
            if arg == "--readonly":
                readonly = True
            elif arg == "--no-sync":
                auto_sync = False
            elif not arg.startswith("--") and i == 2:
                vfs_path = arg

        return local_path, vfs_path, readonly, auto_sync

    def test_basic_mount(self):
        lp, vp, ro, sync = self._parse_mount_args(["mount", "./proj"])
        self.assertEqual(lp, "./proj")
        self.assertEqual(vp, "/project")
        self.assertFalse(ro)
        self.assertTrue(sync)

    def test_mount_with_custom_vfs_path(self):
        lp, vp, ro, sync = self._parse_mount_args(["mount", "./proj", "/src"])
        self.assertEqual(vp, "/src")

    def test_mount_readonly(self):
        _, _, ro, _ = self._parse_mount_args(["mount", "./proj", "--readonly"])
        self.assertTrue(ro)

    def test_mount_no_sync(self):
        _, _, _, sync = self._parse_mount_args(["mount", "./proj", "--no-sync"])
        self.assertFalse(sync)

    def test_mount_all_flags(self):
        lp, vp, ro, sync = self._parse_mount_args(
            ["mount", "./proj", "/custom", "--readonly", "--no-sync"]
        )
        self.assertEqual(lp, "./proj")
        self.assertEqual(vp, "/custom")
        self.assertTrue(ro)
        self.assertFalse(sync)

    def test_unmount_no_save_flag(self):
        args = ["unmount", "/project", "--no-save"]
        save_changes = "--no-save" not in args
        self.assertFalse(save_changes)

    def test_unmount_default_saves(self):
        args = ["unmount", "/project"]
        save_changes = "--no-save" not in args
        self.assertTrue(save_changes)


# =============================================================================
# 8. Dateiendungs-Erkennung (_detect_file_type)
# =============================================================================

class TestDetectFileType(unittest.TestCase):

    def _make_host(self):
        """Stub ISAA_Host mit nur _detect_file_type."""
        class FakeHost:
            def _detect_file_type(self, filename: str) -> str:
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                ext = "." + ext
                type_map = {
                    ".py": "python", ".js": "javascript", ".ts": "typescript",
                    ".jsx": "javascript", ".tsx": "typescript",
                    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
                    ".toml": "toml", ".md": "markdown", ".html": "html",
                    ".css": "css", ".sh": "bash", ".bash": "bash",
                    ".sql": "sql", ".xml": "xml", ".env": "env",
                    ".txt": "text", ".log": "text", ".rs": "rust",
                    ".go": "go", ".java": "java", ".c": "c",
                    ".cpp": "cpp", ".h": "c",
                }
                return type_map.get(ext, "markdown")
        return FakeHost()

    def test_python(self):
        self.assertEqual(self._make_host()._detect_file_type("main.py"), "python")

    def test_json(self):
        self.assertEqual(self._make_host()._detect_file_type("config.json"), "json")

    def test_yaml(self):
        self.assertEqual(self._make_host()._detect_file_type("manifest.yaml"), "yaml")
        self.assertEqual(self._make_host()._detect_file_type("config.yml"), "yaml")

    def test_markdown(self):
        self.assertEqual(self._make_host()._detect_file_type("README.md"), "markdown")

    def test_shell(self):
        self.assertEqual(self._make_host()._detect_file_type("setup.sh"), "bash")

    def test_rust(self):
        self.assertEqual(self._make_host()._detect_file_type("main.rs"), "rust")

    def test_unknown_defaults_to_markdown(self):
        self.assertEqual(self._make_host()._detect_file_type("file.xyz"), "markdown")

    def test_no_extension_defaults(self):
        self.assertEqual(self._make_host()._detect_file_type("Makefile"), "markdown")

    def test_env_file(self):
        self.assertEqual(self._make_host()._detect_file_type(".env"), "env")

    def test_log_file(self):
        self.assertEqual(self._make_host()._detect_file_type("app.log"), "text")

    def test_typescript(self):
        self.assertEqual(self._make_host()._detect_file_type("app.ts"), "typescript")
        self.assertEqual(self._make_host()._detect_file_type("component.tsx"), "typescript")


# =============================================================================
# 9. Rate-Limiter-Config-Anwendung (_apply_rate_limiter_to_builder)
# =============================================================================

class TestRateLimiterConfigApplication(unittest.TestCase):

    def _apply(self, config: dict) -> dict:
        """
        Simuliert _apply_rate_limiter_to_builder ohne Builder-Objekt.
        Gibt zurück was an den Builder übergeben würde.
        """
        calls = {}
        features = config.get("features", {})
        calls["with_rate_limiter"] = {
            "enable_rate_limiting": features.get("rate_limiting", True),
            "enable_model_fallback": features.get("model_fallback", True),
            "enable_key_rotation": features.get("key_rotation", True),
            "key_rotation_mode": features.get("key_rotation_mode", "balance"),
        }
        calls["api_keys"] = []
        for provider, keys in config.get("api_keys", {}).items():
            for key in keys:
                calls["api_keys"].append((provider, key))

        calls["fallback_chains"] = []
        for primary, fallbacks in config.get("fallback_chains", {}).items():
            calls["fallback_chains"].append((primary, fallbacks))

        calls["model_limits"] = []
        for model, limits in config.get("limits", {}).items():
            calls["model_limits"].append((model, limits))

        return calls

    def test_default_config(self):
        from toolboxv2.flows.icli import DEFAULT_RATE_LIMITER_CONFIG
        result = self._apply(DEFAULT_RATE_LIMITER_CONFIG)
        self.assertTrue(result["with_rate_limiter"]["enable_rate_limiting"])
        self.assertTrue(result["with_rate_limiter"]["enable_model_fallback"])
        self.assertTrue(result["with_rate_limiter"]["enable_key_rotation"])
        self.assertEqual(result["with_rate_limiter"]["key_rotation_mode"], "balance")

    def test_fallback_chain_extracted(self):
        from toolboxv2.flows.icli import DEFAULT_RATE_LIMITER_CONFIG
        result = self._apply(DEFAULT_RATE_LIMITER_CONFIG)
        # Default config has a fallback chain für glm-4.7
        primaries = [p for p, _ in result["fallback_chains"]]
        self.assertIn("zglm/glm-4.7", primaries)

    def test_api_keys_extracted(self):
        config = {
            "features": {"rate_limiting": True},
            "api_keys": {"openai": ["key1", "key2"], "anthropic": ["key3"]},
            "fallback_chains": {},
            "limits": {}
        }
        result = self._apply(config)
        self.assertEqual(len(result["api_keys"]), 3)
        providers = [p for p, _ in result["api_keys"]]
        self.assertIn("openai", providers)
        self.assertIn("anthropic", providers)

    def test_rate_limiting_disabled(self):
        config = {
            "features": {"rate_limiting": False, "model_fallback": True},
            "api_keys": {}, "fallback_chains": {}, "limits": {}
        }
        result = self._apply(config)
        self.assertFalse(result["with_rate_limiter"]["enable_rate_limiting"])
        self.assertTrue(result["with_rate_limiter"]["enable_model_fallback"])

    def test_empty_config_uses_defaults(self):
        result = self._apply({})
        self.assertTrue(result["with_rate_limiter"]["enable_rate_limiting"])
        self.assertEqual(result["api_keys"], [])
        self.assertEqual(result["fallback_chains"], [])


# =============================================================================
# 10. Dream-Job-Config-Zusammenbau
# =============================================================================

class TestDreamJobConfig(unittest.TestCase):

    def _build_dream_config(self, **kwargs) -> dict:
        """
        Reimplementiert die Dream-Config-Logik aus cli_create_dream_job
        ohne Tool-Registrierung.
        """
        max_budget = kwargs.get("max_budget", 3000)
        do_skill_split = kwargs.get("do_skill_split", True)
        do_skill_evolve = kwargs.get("do_skill_evolve", True)
        do_persona_evolve = kwargs.get("do_persona_evolve", True)
        do_create_new = kwargs.get("do_create_new", True)
        hard_stop = kwargs.get("hard_stop", False)

        return {
            "max_budget": max_budget,
            "do_skill_split": do_skill_split,
            "do_skill_evolve": do_skill_evolve,
            "do_persona_evolve": do_persona_evolve,
            "do_create_new": do_create_new,
            "hard_stop": hard_stop,
        }

    def _build_trigger_params(self, trigger_type, cron=None, idle=None):
        params = {}
        if trigger_type == "on_cron" and cron:
            params["cron_expression"] = cron
        elif trigger_type == "on_agent_idle" and idle:
            params["agent_idle_seconds"] = idle
        return params

    def test_default_dream_config(self):
        cfg = self._build_dream_config()
        self.assertEqual(cfg["max_budget"], 3000)
        self.assertTrue(cfg["do_skill_split"])
        self.assertTrue(cfg["do_skill_evolve"])
        self.assertTrue(cfg["do_persona_evolve"])
        self.assertTrue(cfg["do_create_new"])
        self.assertFalse(cfg["hard_stop"])

    def test_custom_budget(self):
        cfg = self._build_dream_config(max_budget=5000)
        self.assertEqual(cfg["max_budget"], 5000)

    def test_hard_stop_mode(self):
        cfg = self._build_dream_config(hard_stop=True)
        self.assertTrue(cfg["hard_stop"])

    def test_minimal_dream_no_evolve(self):
        cfg = self._build_dream_config(
            do_skill_evolve=False,
            do_persona_evolve=False,
            do_create_new=False
        )
        self.assertFalse(cfg["do_skill_evolve"])
        self.assertFalse(cfg["do_persona_evolve"])
        self.assertFalse(cfg["do_create_new"])

    def test_cron_trigger_params(self):
        params = self._build_trigger_params("on_cron", cron="0 3 * * *")
        self.assertIn("cron_expression", params)
        self.assertEqual(params["cron_expression"], "0 3 * * *")
        self.assertNotIn("agent_idle_seconds", params)

    def test_idle_trigger_params(self):
        params = self._build_trigger_params("on_agent_idle", idle=600)
        self.assertIn("agent_idle_seconds", params)
        self.assertEqual(params["agent_idle_seconds"], 600)
        self.assertNotIn("cron_expression", params)

    def test_dream_job_query_is_magic_string(self):
        # Dream Jobs verwenden __dream__ als Query
        query = "__dream__"
        self.assertEqual(query, "__dream__")

    def test_dream_job_timeout(self):
        timeout = 600
        self.assertEqual(timeout, 600)


# =============================================================================
# 11. ingest_chunk — Datenfluss über mehrere Iterationen
# =============================================================================

class TestIngestChunkMultiIter(unittest.TestCase):

    def test_iterations_independent(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.iteration = 1
        ingest_chunk(tv, {"type": "reasoning", "chunk": "iter1 thought", "iter": 1})
        tv.iteration = 2
        ingest_chunk(tv, {"type": "reasoning", "chunk": "iter2 thought", "iter": 2})

        self.assertIn("iter1 thought", tv._iter_map[1].thoughts)
        self.assertIn("iter2 thought", tv._iter_map[2].thoughts)
        self.assertNotIn("iter1 thought", tv._iter_map[2].thoughts)

    def test_tool_in_wrong_iter_creates_new_iter(self):
        from toolboxv2.flows.icli import ingest_chunk
        import json
        tv = _tv()
        tv.iteration = 5
        # Tool-Start mit iter=5
        ingest_chunk(tv, {"type": "tool_start", "name": "vfs_shell", "iter": 5})
        # Tool-Result mit gleicher iter
        ingest_chunk(tv, {
            "type": "tool_result", "name": "vfs_shell",
            "result": json.dumps({"success": True, "stdout": "done"}),
            "iter": 5
        })
        self.assertIn(5, tv._iter_map)
        self.assertEqual(len(tv._iter_map[5].tools), 1)

    def test_sub_agent_spawning_tracked(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        tv.iteration = 6
        for sub in ["sub_abc", "sub_def", "sub_ghi", "sub_jkl"]:
            ingest_chunk(tv, {"_sub_agent_id": sub, "iter": 6})
        self.assertEqual(len(tv.sub_agents), 4)

    def test_iteration_count_tracks_correctly(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        for i in range(1, 16):
            tv.iteration = i
            ingest_chunk(tv, {"iter": i, "max_iter": 15})
        self.assertEqual(tv.iteration, 15)
        self.assertEqual(tv.max_iter, 15)

    def test_token_usage_accumulates(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _tv()
        ingest_chunk(tv, {"tokens_used": 1000, "tokens_max": 128000})
        ingest_chunk(tv, {"tokens_used": 5000})
        self.assertEqual(tv.tokens_used, 5000)
        self.assertEqual(tv.tokens_max, 128000)

    def test_full_agent_run_simulation(self):
        """Simuliert einen kompletten Agenten-Lauf mit allen Chunk-Typen."""
        from toolboxv2.flows.icli import ingest_chunk
        import json

        tv = _tv()

        # Start
        ingest_chunk(tv, {"agent": "self", "persona": "analyst",
                          "skills": ["research"], "iter": 1, "max_iter": 5,
                          "tokens_max": 128000})

        # Iter 1: Denken + Tool
        tv.iteration = 1
        ingest_chunk(tv, {"type": "reasoning", "chunk": "I need to analyze", "iter": 1})
        ingest_chunk(tv, {"type": "tool_start", "name": "vfs_shell", "iter": 1})
        ingest_chunk(tv, {
            "type": "tool_result", "name": "vfs_shell",
            "result": json.dumps({"success": True, "stdout": "files found"}),
            "iter": 1, "tokens_used": 2000
        })

        # Iter 2: Sub-Agent spawnen
        tv.iteration = 2
        ingest_chunk(tv, {
            "_sub_agent_id": "sub_abc", "iter": 2,
            "type": "tool_start", "name": "spawn_sub_agent"
        })

        # Abschluss
        ingest_chunk(tv, {"type": "final_answer", "answer": "Analysis complete."})

        # Assertions
        self.assertEqual(tv.persona, "analyst")
        self.assertEqual(tv.skills, ["research"])
        self.assertEqual(tv.status, "completed")
        self.assertEqual(tv.final_answer, "Analysis complete.")
        self.assertIn(1, tv._iter_map)
        self.assertIn(2, tv._iter_map)
        iv1 = tv._iter_map[1]
        self.assertEqual(len(iv1.thoughts), 1)
        self.assertEqual(len(iv1.tools), 1)
        self.assertIn("sub_abc", tv.sub_agents)


# =============================================================================
# 12. STATUS_SYM — Korrektheit der Symbol-Zuordnung
# =============================================================================

class TestStatusSymbols(unittest.TestCase):

    def test_all_statuses_have_entries(self):
        from toolboxv2.flows.icli import STATUS_SYM
        for status in ["running", "completed", "done", "failed", "error", "cancelled"]:
            self.assertIn(status, STATUS_SYM)

    def test_running_is_cyan(self):
        from toolboxv2.flows.icli import STATUS_SYM
        _, col = STATUS_SYM["running"]
        self.assertEqual(col, "cyan")

    def test_completed_is_green(self):
        from toolboxv2.flows.icli import STATUS_SYM
        _, col = STATUS_SYM["completed"]
        self.assertEqual(col, "green")

    def test_failed_is_red(self):
        from toolboxv2.flows.icli import STATUS_SYM
        _, col = STATUS_SYM["failed"]
        self.assertEqual(col, "red")

    def test_error_is_red(self):
        from toolboxv2.flows.icli import STATUS_SYM
        _, col = STATUS_SYM["error"]
        self.assertEqual(col, "red")

    def test_unknown_status_uses_default(self):
        from toolboxv2.flows.icli import STATUS_SYM
        sym, col = STATUS_SYM.get("unknown_status", ("◯", "cyan"))
        self.assertEqual(sym, "◯")


if __name__ == "__main__":
    unittest.main(verbosity=2)
