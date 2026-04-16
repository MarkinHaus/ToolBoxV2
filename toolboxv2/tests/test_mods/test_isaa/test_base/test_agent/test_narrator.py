"""
Tests for AgentLiveNarrator.
Run:  python -m unittest test_narrator -v
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch
import os

from tests.a_util import async_test

os.environ["TOOLBOX_TESTING"] = "true"

# ---------------------------------------------------------------------------
# Minimal AgentLiveState stub (no real import needed)
# ---------------------------------------------------------------------------

@dataclass
class _FakeLiveState:
    narrator_msg: str = ""


# ---------------------------------------------------------------------------
# Import target (patch env BEFORE importing the module)
# ---------------------------------------------------------------------------

import os
os.environ.setdefault("BLITZMODEL", "groq/test-model")
os.environ.setdefault("AGENT_NARRATOR_ENABLED", "true")
os.environ.setdefault("NARRATOR_LANG", "de")

import toolboxv2.mods.isaa.base.Agent.narrator as N  # the file we're testing
from toolboxv2.mods.isaa.base.Agent.narrator import (
    AgentLiveNarrator,
    NarratorMiniState,
    _compress_diff,
    _estimate_tokens,
    _lang,
    _mock,
    _build_normal_prompt,
    _build_think_prompt,
    _call_blitz,
    NARRATOR_MIN_INTERVAL,
    NARRATOR_STALE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_narrator(enabled=True, **kwargs):
    """Erstellt den Narrator für Tests, zwingt ihn aber offline (Mock)."""

    # 1. Zentral den Mock-Handler erstellen
    mock_handler = MagicMock()
    # GANZ WICHTIG: Die Methode für den HTTP-Aufruf MUSS ein AsyncMock sein!
    mock_handler.async_completion = AsyncMock(return_value="Mocked LLM Response")
    mock_handler.acompletion = AsyncMock(return_value="Mocked LLM Response")

    # 2. Handler an die Initialisierung übergeben
    kwargs['handler'] = mock_handler

    # (Hier kommt dein bisheriger Code zur Erstellung von live/agent)
    live = MagicMock()  # oder wie auch immer du AgentLiveState hier erstellst
    agent = MagicMock()

    # 3. Narrator mit dem Mock-Handler erstellen
    nar = AgentLiveNarrator(live, agent, do_narator=enabled, **kwargs)

    return nar, live


def _history(*roles_contents) -> list[dict]:
    """Build a minimal history list from (role, content) pairs."""
    return [{"role": r, "content": c} for r, c in roles_contents]


def _fake_blitz_result(thought="teste tool", plan="", d=0, r=0,
                       tokens_in=10, tokens_out=5) -> dict:
    data = {"t": thought, "d": d, "r": r}
    if plan:
        data["plan"] = plan
    return {"data": data, "raw": json.dumps(data),
            "in": tokens_in, "out": tokens_out}


def run(coro):
    """Run a coroutine synchronously in tests using a safe event loop."""
    return async_test(coro)


# ===========================================================================
# 1. Language detection
# ===========================================================================

class TestLangDetection(unittest.TestCase):

    def test_forced_de(self):
        with patch.object(N, "NARRATOR_LANG", "de"):
            self.assertEqual(_lang("anything"), "de")

    def test_forced_en(self):
        with patch.object(N, "NARRATOR_LANG", "en"):
            self.assertEqual(_lang("anything"), "en")

    def test_auto_de(self):
        with patch.object(N, "NARRATOR_LANG", "auto"):
            self.assertEqual(_lang("was ist das und wie funktioniert das"), "de")

    def test_auto_en(self):
        with patch.object(N, "NARRATOR_LANG", "auto"):
            self.assertEqual(_lang("what is the capital of france"), "en")


# ===========================================================================
# 2. Mock string helpers
# ===========================================================================

class TestMockStrings(unittest.TestCase):

    def test_mock_returns_string(self):
        s = _mock("init", "de")
        self.assertIsInstance(s, str)
        self.assertTrue(len(s) > 0)

    def test_mock_tool_placeholder(self):
        s = _mock("tool_start", "de", tool="calc")
        self.assertIn("calc", s)

    def test_mock_en_pool(self):
        s = _mock("llm_pre", "en")
        self.assertIsInstance(s, str)

    def test_mock_unknown_key_fallback(self):
        s = _mock("nonexistent_key", "de")
        self.assertEqual(s, "…")


# ===========================================================================
# 3. History compression / diff
# ===========================================================================

class TestCompressDiff(unittest.TestCase):

    def test_returns_only_new_messages(self):
        hist = _history(("user", "a"), ("assistant", "b"), ("user", "c"))
        diff = _compress_diff(hist, cursor=1)
        self.assertEqual(len(diff), 2)
        self.assertEqual(diff[0]["content"], "b")

    def test_skips_system_messages(self):
        hist = _history(("user", "x"), ("system", "loop warning"), ("assistant", "y"))
        diff = _compress_diff(hist, cursor=0)
        roles = [m["role"] for m in diff]
        self.assertNotIn("system", roles)

    def test_truncates_long_content(self):
        long = "x" * 500
        hist = [{"role": "user", "content": long}]
        diff = _compress_diff(hist, cursor=0)
        self.assertLessEqual(len(diff[0]["content"]), N.DIFF_CONTENT_MAXCHARS)

    def test_empty_diff_when_cursor_at_end(self):
        hist = _history(("user", "hello"))
        diff = _compress_diff(hist, cursor=1)
        self.assertEqual(diff, [])

    def test_multipart_content_flattened(self):
        hist = [{"role": "user", "content": [{"text": "hello"}, {"text": " world"}]}]
        diff = _compress_diff(hist, cursor=0)
        self.assertIn("hello", diff[0]["content"])
        self.assertIn("world", diff[0]["content"])

    def test_estimate_tokens_positive(self):
        msgs = _history(("user", "hello world"))
        est = _estimate_tokens(msgs)
        self.assertGreater(est, 0)


# ===========================================================================
# 4. Prompt builders
# ===========================================================================

class TestPromptBuilders(unittest.TestCase):

    def test_normal_prompt_de_has_plan(self):
        mini = NarratorMiniState(plan_summary="berechne 50**991")
        system, msgs = _build_normal_prompt("de", [], mini)
        self.assertIn("berechne 50**991", msgs[0]["content"])

    def test_normal_prompt_en_tool_hint(self):
        mini = NarratorMiniState()
        system, msgs = _build_normal_prompt("en", [], mini, tool_hint="calc: 42")
        self.assertIn("calc: 42", msgs[0]["content"])

    def test_normal_prompt_includes_diff(self):
        mini = NarratorMiniState()
        diff = [{"role": "assistant", "content": "some result"}]
        system, msgs = _build_normal_prompt("de", diff, mini)
        self.assertIn("some result", msgs[0]["content"])

    def test_think_prompt_includes_thinking(self):
        mini = NarratorMiniState()
        system, msgs = _build_think_prompt("de", "ich denke über xyz nach", mini)
        self.assertIn("ich denke über xyz nach", msgs[0]["content"])

    def test_think_prompt_includes_previous_plan(self):
        mini = NarratorMiniState(plan_summary="vorheriger plan")
        system, msgs = _build_think_prompt("en", "thinking...", mini)
        self.assertIn("vorheriger plan", msgs[0]["content"])

    def test_think_prompt_truncates_long_thinking(self):
        mini = NarratorMiniState()
        long_think = "x" * 1000
        system, msgs = _build_think_prompt("de", long_think, mini)
        self.assertLessEqual(len(msgs[0]["content"]), N.THINK_CONTENT_MAXCHARS + 100)


# ===========================================================================
# 5. on_init / on_llm_pre_call / on_tool_start  (sync, no blitz)
# ===========================================================================

class TestSyncEventHandlers(unittest.TestCase):

    def test_on_init_sets_thought(self):
        nar, live = _make_narrator()
        nar.on_init("was ist 50**991")
        self.assertTrue(len(live.narrator_msg) > 0)

    def test_on_init_sets_lang_de(self):
        nar, _ = _make_narrator()
        nar.on_init("was ist das und wie")
        self.assertEqual(nar._lang, "de")

    def test_on_llm_pre_call_sets_thought(self):
        nar, live = _make_narrator()
        nar._lang = "de"
        nar.on_llm_pre_call([])
        self.assertTrue(len(live.narrator_msg) > 0)

    def test_on_llm_pre_call_advances_cursor(self):
        nar, _ = _make_narrator()
        nar._lang = "de"
        hist = _history(("user", "a"), ("assistant", "b"))
        nar.on_llm_pre_call(hist)
        self.assertEqual(nar._history_cursor, 2)

    def test_on_tool_start_normal(self):
        nar, live = _make_narrator()
        nar._lang = "de"
        nar.on_tool_start("python")
        self.assertIn("python", live.narrator_msg)

    def test_on_tool_start_think(self):
        nar, live = _make_narrator()
        nar._lang = "de"
        nar.on_tool_start("think")
        # should use "think" mock pool, not tool_start pool
        self.assertNotIn("{tool}", live.narrator_msg)
        self.assertTrue(len(live.narrator_msg) > 0)

    def test_on_summarise_sets_thought(self):
        nar, live = _make_narrator()
        nar._lang = "de"
        nar.on_summarise()
        self.assertTrue(len(live.narrator_msg) > 0)

    def test_mock_direct(self):
        nar, live = _make_narrator()
        nar._lang = "en"
        nar.mock("init")
        self.assertTrue(len(live.narrator_msg) > 0)


# ===========================================================================
# 6. reset()
# ===========================================================================

class TestReset(unittest.TestCase):

    def test_reset_clears_mini_state(self):
        nar, _ = _make_narrator()
        nar._mini.plan_summary = "some plan"
        nar._mini.drift = True
        nar.reset("neue anfrage")
        self.assertEqual(nar._mini.plan_summary, "")
        self.assertFalse(nar._mini.drift)

    def test_reset_resets_cursor(self):
        nar, _ = _make_narrator()
        nar._history_cursor = 42
        nar.reset()
        self.assertEqual(nar._history_cursor, 0)

    def test_reset_does_not_clear_token_log(self):
        nar, _ = _make_narrator()
        nar._token_log.append((time.monotonic(), 100, 10))
        nar.reset()
        self.assertEqual(len(nar._token_log), 1)  # PM window survives reset


# ===========================================================================
# 7. Budget / sliding window
# ===========================================================================

class TestBudget(unittest.TestCase):

    def test_budget_ok_when_empty(self):
        nar, _ = _make_narrator()
        self.assertTrue(nar._budget_ok([]))

    def test_budget_fails_on_input_overflow(self):
        nar, _ = _make_narrator()
        now = time.monotonic()
        # fill the log with a lot of input tokens (within the 60s window)
        nar._token_log = [(now - 1, N.NARRATOR_MAX_INPUT_TOKENS_PM, 0)]
        self.assertFalse(nar._budget_ok([]))

    def test_budget_fails_on_output_overflow(self):
        nar, _ = _make_narrator()
        now = time.monotonic()
        nar._token_log = [(now - 1, 0, N.NARRATOR_MAX_OUTPUT_TOKENS_PM)]
        self.assertFalse(nar._budget_ok([]))

    def test_budget_prunes_stale_entries(self):
        nar, _ = _make_narrator()
        old_time = time.monotonic() - 120  # 2 minutes ago → outside window
        nar._token_log = [(old_time, N.NARRATOR_MAX_INPUT_TOKENS_PM, 0)]
        self.assertTrue(nar._budget_ok([]))
        self.assertEqual(len(nar._token_log), 0)  # pruned

    def test_record_tokens_appends(self):
        nar, _ = _make_narrator()
        nar._record_tokens(50, 10)
        self.assertEqual(len(nar._token_log), 1)
        self.assertEqual(nar._token_log[0][1], 50)
        self.assertEqual(nar._token_log[0][2], 10)


# ===========================================================================
# 8. schedule_tool_end  (async BG task logic)
# ===========================================================================

class TestScheduleToolEnd(unittest.TestCase):

    def test_sets_mock_immediately_even_if_disabled(self):
        nar, live = _make_narrator(enabled=False)
        nar._lang = "de"
        nar.schedule_tool_end("calc", "42", [])
        # mock must still be set synchronously
        self.assertTrue(len(live.narrator_msg) > 0)

    def test_no_task_when_disabled(self):
        nar, _ = _make_narrator(enabled=False)
        nar._lang = "de"
        nar.schedule_tool_end("calc", "42", [])
        self.assertEqual(len(nar._pending_tasks), 0)

    def test_no_task_within_rate_limit(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        nar._last_blitz_time = time.monotonic()  # just called
        nar.schedule_tool_end("calc", "42", [])
        self.assertEqual(len(nar._pending_tasks), 0)

    @async_test
    async def test_task_scheduled_when_ready(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        nar._last_blitz_time = 0.0  # long ago
        nar._token_log = []

        async def _run():
            nar.schedule_tool_end("calc", "42", [])
            return nar._pending_tasks

        task = await _run()
        self.assertIsNotNone(task)
        # clean up
        await asyncio.gather(*task)

        for t in task:
            if task and not t.done():
                self.assertTrue(t.cancel())


# ===========================================================================
# 9. schedule_think_result
# ===========================================================================

class TestScheduleThinkResult(unittest.TestCase):

    def test_dedup_skips_same_chunk(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        thinking = "ich denke über lösung nach"
        # set hash manually as if we already processed this
        import hashlib
        nar._mini.last_think_hash = hashlib.md5(thinking[:200].encode()).hexdigest()[:8]
        nar.schedule_think_result(thinking, [])
        self.assertEqual(len(nar._pending_tasks), 0)

    @async_test
    async def test_different_chunk_schedules_task(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        nar._token_log = []

        async def _run():
            nar.schedule_think_result("neue überlegung xyz", [])
            return nar._pending_tasks

        task = await _run()
        self.assertIsNotNone(task)

        await asyncio.gather(*task)

        for t in task:
            if task and not t.done():
                self.assertTrue(t.cancel())

    @async_test
    async def test_think_task_marked_as_think(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        nar._token_log = []

        async def _run():
            nar.schedule_think_result("neue analyse abc", [])
            return nar._pending_is_think

        is_think = await _run()
        self.assertTrue(is_think)


# ===========================================================================
# 10. _blitz_normal integration  (mock _call_blitz)
# ===========================================================================

class TestBlitzNormal(unittest.TestCase):

    @async_test
    async def test_updates_thought_on_success(self):
        nar, live = _make_narrator()
        nar._lang = "de"
        result = _fake_blitz_result(thought="berechne potenz")
        # ensure not stale
        nar._thought_set_time = 0.0

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_normal([], tool_hint="calc: 42")

        await _run()
        self.assertEqual(live.narrator_msg, "berechne potenz")

    @async_test
    async def test_discards_stale_result(self):
        nar, live = _make_narrator()
        nar._lang = "de"
        live.narrator_msg = "frischer mock"
        nar._thought_set_time = time.monotonic()  # just set → stale guard triggers

        result = _fake_blitz_result(thought="veraltetes ergebnis")

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_normal([], tool_hint="")

        await _run()
        self.assertEqual(live.narrator_msg, "frischer mock")  # unchanged

    @async_test
    async def test_sets_drift_flag(self):
        nar, _ = _make_narrator()
        nar._lang = "de"
        nar._thought_set_time = 0.0
        result = _fake_blitz_result(thought="abweichung erkannt", d=1)

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_normal([], tool_hint="")

        await _run()
        self.assertTrue(nar._mini.drift)

    @async_test
    async def test_sets_repeat_flag(self):
        nar, _ = _make_narrator()
        nar._lang = "de"
        nar._thought_set_time = 0.0
        result = _fake_blitz_result(thought="wiederholung", r=1)

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_normal([], tool_hint="")

        await _run()
        self.assertTrue(nar._mini.repeat)

    @async_test
    async def test_appends_drift_suffix_to_thought(self):
        nar, live = _make_narrator()
        nar._lang = "de"
        nar._thought_set_time = 0.0
        nar._mini.drift = True
        result = _fake_blitz_result(thought="irgendwas")

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_normal([], tool_hint="")

        await _run()
        self.assertIn("⚠", live.narrator_msg)

    @async_test
    async def test_records_tokens(self):
        nar, _ = _make_narrator()
        nar._lang = "de"
        nar._thought_set_time = 0.0
        result = _fake_blitz_result(thought="ok", tokens_in=15, tokens_out=7)

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_normal([], tool_hint="")

        await _run()
        self.assertEqual(nar._token_log[-1][1], 15)
        self.assertEqual(nar._token_log[-1][2], 7)

    @async_test
    async def test_no_crash_on_call_blitz_failure(self):
        nar, live = _make_narrator()
        nar._lang = "de"
        live.narrator_msg = "vorher"

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=None)):
                await nar._blitz_normal([], tool_hint="")

        await _run()
        self.assertEqual(live.narrator_msg, "vorher")  # unchanged, no crash


# ===========================================================================
# 11. _blitz_think integration
# ===========================================================================

class TestBlitzThink(unittest.TestCase):

    @async_test
    async def test_updates_plan_summary(self):
        nar, _ = _make_narrator()
        nar._lang = "de"
        result = _fake_blitz_result(thought="analysiere potenz", plan="50**991 via python")

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_think("ich denke über 50**991 nach")

        await _run()
        self.assertEqual(nar._mini.plan_summary, "50**991 via python")

    @async_test
    async def test_updates_thought(self):
        nar, live = _make_narrator()
        nar._lang = "de"
        result = _fake_blitz_result(thought="analysiere potenz", plan="plan")

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_think("überlegung")

        await _run()
        self.assertEqual(live.narrator_msg, "analysiere potenz")

    @async_test
    async def test_fallback_to_mock_on_failure(self):
        nar, live = _make_narrator()
        nar._lang = "de"

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=None)):
                await nar._blitz_think("überlegung")

        await _run()
        # should have set a think-mock string
        print(nar._thought_set_time - time.monotonic())
        self.assertTrue(len(live.narrator_msg) > 0)

    @async_test
    async def test_fallback_to_mock_on_empty_thought(self):
        nar, live = _make_narrator()
        nar._lang = "de"
        result = _fake_blitz_result(thought=".", plan="some plan")

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_think("überlegung")

        await _run()
        self.assertTrue(len(live.narrator_msg) > 0)


# ===========================================================================
# 12. public blitz() method
# ===========================================================================

class TestPublicBlitz(unittest.TestCase):

    @async_test
    async def test_returns_none_when_disabled(self):
        nar, _ = _make_narrator(enabled=False)

        async def _run():
            return await nar.blitz("sys", [{"role": "user", "content": "x"}])

        self.assertIsNone(await _run())

    @async_test
    async def test_returns_dict_without_schema(self):
        nar, _ = _make_narrator()
        result = _fake_blitz_result(thought="ok")

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                return await nar.blitz("sys", [{"role": "user", "content": "x"}])

        out = await _run()
        self.assertIsInstance(out, dict)
        self.assertIn("t", out)

    @async_test
    async def test_schema_validation_pass(self):
        nar, _ = _make_narrator()
        result = {"data": {"score": 7, "label": "gut"}, "raw": "", "in": 5, "out": 3}

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                return await nar.blitz(
                    "sys", [],
                    schema={"score": int, "label": str},
                )

        out = await _run()
        self.assertEqual(out["score"], 7)

    @async_test
    async def test_schema_validation_fail_missing_key(self):
        nar, _ = _make_narrator()
        result = {"data": {"score": 7}, "raw": "", "in": 5, "out": 3}

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                return await nar.blitz(
                    "sys", [],
                    schema={"score": int, "label": str},  # label missing
                )

        self.assertIsNone(await _run())

    @async_test
    async def test_schema_validation_fail_wrong_type(self):
        nar, _ = _make_narrator()
        result = {"data": {"score": "not-an-int"}, "raw": "", "in": 5, "out": 3}

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                return await nar.blitz(
                    "sys", [],
                    schema={"score": int},
                )

        self.assertIsNone(await _run())

    @async_test
    async def test_ratelimit_respected_budget_exceeded(self):
        nar, _ = _make_narrator()
        now = time.monotonic()
        nar._token_log = [(now - 1, N.NARRATOR_MAX_INPUT_TOKENS_PM, 0)]

        async def _run():
            return await nar.blitz(
                "sys", [],
                respect_ratelimit=True,
                history=[],
            )

        self.assertIsNone(await _run())

    @async_test
    async def test_ratelimit_records_tokens_on_success(self):
        nar, _ = _make_narrator()
        result = {"data": {"t": "ok"}, "raw": "", "in": 20, "out": 8}

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                return await nar.blitz(
                    "sys", [],
                    respect_ratelimit=True,
                    history=[],
                )

        await _run()
        self.assertEqual(nar._token_log[-1][1], 20)
        self.assertEqual(nar._token_log[-1][2], 8)


# ===========================================================================
# 13. Task cancel logic
# ===========================================================================

class TestTaskCancelLogic(unittest.TestCase):
    async def asyncTearDown(self):
        """Wird nach jedem Test ausgeführt, egal ob er erfolgreich war oder abstürzte."""
        if hasattr(self, 'nar') and getattr(self.nar, '_pending_tasks', None):
            for task in self.nar._pending_tasks:
                task.cancel()
            if self.nar._pending_tasks:
                await asyncio.gather(*self.nar._pending_tasks, return_exceptions=True)

        # Globale Variable säubern (geschützt, falls der Import Pfad mal nicht stimmt)
        try:
            from toolboxv2.mods.isaa.base.Agent import narrator
            if hasattr(narrator, "_blitz_handler"):
                narrator._blitz_handler = None
        except ImportError:
            pass

    @async_test
    async def test_normal_task_cancels_previous_normal(self):
        nar, _ = _make_narrator(enabled=True)
        self.nar = nar
        nar._lang = "de"
        nar._token_log = []

        # 1. Normal Task hinzufügen
        nar.schedule_tool_end("calc", "1", [])
        first_task = nar._pending_tasks[-1] if nar._pending_tasks else None

        nar._last_blitz_time = 0.0  # Reset Interval, um Limits zu ignorieren

        # 2. Weiteren Normal Task direkt im Anschluss hinzufügen
        nar.schedule_tool_end("python", "2", [])
        second_task = nar._pending_tasks[-1] if nar._pending_tasks else None

        self.assertIsNotNone(first_task, "Erster Task wurde nicht erstellt.")
        self.assertIsNotNone(second_task, "Zweiter Task wurde nicht erstellt.")
        self.assertNotEqual(first_task, second_task, "Der Task wurde fälschlicherweise nicht neu gestartet.")
        await asyncio.sleep(0)
        # first_task sollte nun gecancelt (oder sehr schnell fertig) sein
        self.assertTrue(first_task.cancelled() or first_task.done())

        if second_task and not second_task.done():
            self.assertFalse(second_task.cancelled(), "Der neue Task sollte aktiv bleiben.")

    @async_test
    async def test_think_task_not_cancelled_by_normal(self):
        nar, _ = _make_narrator(enabled=True)
        self.nar = nar
        nar._lang = "de"
        nar._token_log = []

        # 1. Think Task hinzufügen (höchste Priorität)
        nar.schedule_think_result("denke nach über abc", [])
        think_task = nar._pending_tasks[-1] if nar._pending_tasks else None

        self.assertIsNotNone(think_task)
        self.assertTrue(nar._pending_is_think, "_pending_is_think Flag wurde nicht gesetzt!")

        # 2. Versuch, einen Normal Task zu schulen -> Sollte think_task in Ruhe lassen
        nar._last_blitz_time = 0.0
        nar.schedule_tool_end("calc", "1", [])

        # Think task darf nicht abgebrochen worden sein
        self.assertFalse(think_task.cancelled(),
                         "Der Think Task wurde fälschlicherweise von einem Normal Task abgebrochen.")


# ===========================================================================
# 14. Extension: prompt builders
# ===========================================================================

class TestExtensionPromptBuilders(unittest.TestCase):

    def _mini(self, plan=""):
        return NarratorMiniState(plan_summary=plan)

    def test_skills_prompt_contains_index(self):
        from toolboxv2.mods.isaa.base.Agent.narrator import _build_skills_prompt
        system, msgs = _build_skills_prompt(
            "de", "mache xyz", [], "skill1|Skill Eins|kw1,kw2", self._mini()
        )
        self.assertIn("skill1", msgs[0]["content"])

    def test_skills_prompt_contains_query(self):
        from toolboxv2.mods.isaa.base.Agent.narrator import _build_skills_prompt
        system, msgs = _build_skills_prompt(
            "en", "write tests", [], "s|S|t", self._mini()
        )
        self.assertIn("write tests", msgs[0]["content"])

    def test_skills_prompt_includes_plan(self):
        from toolboxv2.mods.isaa.base.Agent.narrator import _build_skills_prompt
        system, msgs = _build_skills_prompt(
            "de", "q", [], "s|S|t", self._mini(plan="berechne xyz")
        )
        self.assertIn("berechne xyz", msgs[0]["content"])

    def test_ruleset_prompt_contains_diff(self):
        from toolboxv2.mods.isaa.base.Agent.narrator import _build_ruleset_prompt
        diff = [{"role": "assistant", "content": "tool result: success"}]
        from toolboxv2.mods.isaa.base.Agent.rule_set import (
            RuleSet
        )
        system, msgs = _build_ruleset_prompt("de", diff, self._mini(), RuleSet(auto_sync_vfs=False))
        self.assertIn("tool result", msgs[0]["content"])

    def test_ruleset_prompt_en_system(self):
        from toolboxv2.mods.isaa.base.Agent.narrator import _build_ruleset_prompt, _RULESET_SYSTEM_EN
        from toolboxv2.mods.isaa.base.Agent.rule_set import (
            RuleSet
        )
        system, _ = _build_ruleset_prompt("en", [], self._mini(), RuleSet(auto_sync_vfs=False))
        self.assertEqual(system, _RULESET_SYSTEM_EN)

    def test_memory_prompt_contains_query(self):
        from toolboxv2.mods.isaa.base.Agent.narrator import _build_memory_prompt
        system, msgs = _build_memory_prompt("de", "was ist 2+2", [])
        self.assertIn("was ist 2+2", msgs[0]["content"])

    def test_memory_prompt_contains_diff_content(self):
        from toolboxv2.mods.isaa.base.Agent.narrator import _build_memory_prompt
        diff = [{"role": "tool", "content": "ergebnis: 4"}]
        system, msgs = _build_memory_prompt("en", "q", diff)
        self.assertIn("ergebnis: 4", msgs[0]["content"])


# ===========================================================================
# 15. Extension: _compress_skills_to_index
# ===========================================================================

class TestCompressSkillsToIndex(unittest.TestCase):

    def _fake_skill(self, sid, name, triggers, active=True):
        s = MagicMock()
        s.id = sid
        s.name = name
        s.triggers = triggers
        s.is_active = MagicMock(return_value=active)
        return s

    def _fake_manager(self, skills: dict):
        m = MagicMock()
        m.skills = skills
        return m

    def test_basic_index(self):
        from toolboxv2.mods.isaa.base.Agent.narrator import _compress_skills_to_index
        mgr = self._fake_manager({
            "s1": self._fake_skill("s1", "Calc", ["calc", "math"]),
        })
        result = _compress_skills_to_index(mgr)
        self.assertIn("s1", result)
        self.assertIn("Calc", result)

    def test_inactive_skills_excluded(self):
        from toolboxv2.mods.isaa.base.Agent.narrator import _compress_skills_to_index
        mgr = self._fake_manager({
            "s1": self._fake_skill("s1", "Active", [], active=True),
            "s2": self._fake_skill("s2", "Inactive", [], active=False),
        })
        result = _compress_skills_to_index(mgr)
        self.assertIn("s1", result)
        self.assertNotIn("s2", result)

    def test_empty_manager_returns_placeholder(self):
        from toolboxv2.mods.isaa.base.Agent.narrator import _compress_skills_to_index
        mgr = self._fake_manager({})
        self.assertEqual(_compress_skills_to_index(mgr), "(no skills)")

    def test_triggers_limited_to_three(self):
        from toolboxv2.mods.isaa.base.Agent.narrator import _compress_skills_to_index
        mgr = self._fake_manager({
            "s1": self._fake_skill("s1", "X", ["a", "b", "c", "d", "e"]),
        })
        result = _compress_skills_to_index(mgr)
        line = result.strip().split("|")
        triggers_part = line[2]
        self.assertLessEqual(triggers_part.count(","), 2)  # max 3 triggers = 2 commas


# ===========================================================================
# 16. Extension: schedule_skills_update
# ===========================================================================

class TestScheduleSkillsUpdate(unittest.TestCase):

    def _fake_manager(self, skill_ids=("s1",)):
        mgr = MagicMock()
        skills = {}
        for sid in skill_ids:
            s = MagicMock()
            s.id = sid
            s.name = f"Skill {sid}"
            s.triggers = ["kw1"]
            s.is_active = MagicMock(return_value=True)
            skills[sid] = s
        mgr.skills = skills
        return mgr

    def test_no_task_when_disabled(self):
        nar, _ = _make_narrator(enabled=False)
        nar._lang = "de"
        mgr = self._fake_manager()
        nar.schedule_skills_update("q", [], mgr)
        self.assertEqual(len(nar._pending_tasks), 0)

    def test_no_task_when_no_skills(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        mgr = self._fake_manager(skill_ids=())
        nar.schedule_skills_update("q", [], mgr)
        self.assertEqual(len(nar._pending_tasks), 0)

    @async_test
    async def test_task_scheduled_when_skills_available(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        nar._token_log = []
        mgr = self._fake_manager()

        async def _run():
            nar.schedule_skills_update("berechne 50**991", [], mgr)
            return nar._pending_tasks

        task = await _run()
        self.assertIsNotNone(task)

        await asyncio.gather(*task)

        for t in task:
            if task and not t.done():
                self.assertTrue(t.cancel())

    @async_test
    async def test_blitz_skills_updates_live_skills(self):
        nar, live = _make_narrator(enabled=True)
        nar._lang = "de"
        nar._thought_set_time = 0.0

        mgr = self._fake_manager(("s1", "s2"))
        result = {"data": {"ids": ["s1"], "reason": "passend"}, "raw": "", "in": 5, "out": 3}

        async def _run():
            from toolboxv2.mods.isaa.base.Agent.narrator import _compress_skills_to_index
            idx = _compress_skills_to_index(mgr)
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_skills("q", [], idx, mgr)

        await _run()
        self.assertIn("Skill s1", nar.live.skills)

    @async_test
    async def test_blitz_skills_ignores_invalid_ids(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        mgr = self._fake_manager(("s1",))
        result = {"data": {"ids": ["nonexistent"], "reason": "x"}, "raw": "", "in": 5, "out": 3}

        async def _run():
            from toolboxv2.mods.isaa.base.Agent.narrator import _compress_skills_to_index
            idx = _compress_skills_to_index(mgr)
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_skills("q", [], idx, mgr)

        await _run()
        # No crash, live.skills stays as-is or empty
        self.assertIsInstance(nar.live.skills, list)

    @async_test
    async def test_blitz_skills_no_crash_on_none(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        mgr = self._fake_manager()

        async def _run():
            from toolboxv2.mods.isaa.base.Agent.narrator import _compress_skills_to_index
            idx = _compress_skills_to_index(mgr)
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=None)):
                await nar._blitz_skills("q", [], idx, mgr)

        await _run()  # must not raise


# ===========================================================================
# 17. Extension: schedule_ruleset_update
# ===========================================================================

class TestScheduleRulesetUpdate(unittest.TestCase):

    def _fake_session(self, has_ruleset=True, has_vfs=True):
        session = MagicMock()
        if has_ruleset:
            rs = MagicMock()
            rs.set_situation = MagicMock()
            rs.build_vfs_content = MagicMock(return_value="# Rules\nContent here")
            rs.get_vfs_filename = MagicMock(return_value="active_rules.md")
            rs.mark_clean = MagicMock()
            session.rule_set = rs
        else:
            session.rule_set = None
        if has_vfs:
            vfs = MagicMock()
            vfs.write = MagicMock()  # sync write
            vfs.set_rules_file = MagicMock()  # sync write
            session.vfs = vfs
        else:
            session.vfs = None
        return session

    def test_no_task_when_disabled(self):
        nar, _ = _make_narrator(enabled=False)
        session = self._fake_session()
        nar.schedule_ruleset_update([], session, MagicMock())
        self.assertEqual(len(nar._pending_tasks), 0)

    def test_no_task_when_no_ruleset(self):
        nar, _ = _make_narrator(enabled=True)
        session = self._fake_session(has_ruleset=False)
        nar.schedule_ruleset_update([], session, MagicMock())
        self.assertEqual(len(nar._pending_tasks), 0)

    @async_test
    async def test_task_scheduled_when_ruleset_present(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        nar._token_log = []
        session = self._fake_session()

        # 1. Den Task planen
        nar.schedule_ruleset_update([], session, MagicMock())

        # 2. Zugriff auf den Task, den nar intern erstellt hat
        task = nar._pending_tasks

        await asyncio.gather(*task)

        for t in task:
            self.assertTrue(t.cancel())
            self.assertTrue(t.done())

    @async_test
    async def test_blitz_ruleset_calls_set_situation(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        session = self._fake_session()
        ctx = MagicMock()
        result = {
            "data": {"situation": "python datei analyse", "intent": "fehler finden", "confidence": 0.8},
            "raw": "", "in": 10, "out": 5,
        }

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_ruleset([], session, session.rule_set, ctx)

        await _run()
        session.rule_set.set_situation.assert_called_once_with(
            "python datei analyse", "fehler finden"
        )

    @async_test
    async def test_blitz_ruleset_writes_vfs(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        session = self._fake_session()
        ctx = MagicMock()
        result = {
            "data": {"situation": "api arbeit", "intent": "daten senden", "confidence": 0.9},
            "raw": "", "in": 10, "out": 5,
        }

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_ruleset([], session, session.rule_set, ctx)

        await _run()
        session.vfs.set_rules_file.assert_called_once()

    @async_test
    async def test_blitz_ruleset_skips_low_confidence(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        session = self._fake_session()
        ctx = MagicMock()
        result = {
            "data": {"situation": "irgendwas", "intent": "irgendwas", "confidence": 0.1},
            "raw": "", "in": 5, "out": 3,
        }

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_ruleset([], session, session.rule_set, ctx)

        await _run()
        session.rule_set.set_situation.assert_not_called()

    @async_test
    async def test_blitz_ruleset_no_crash_on_none(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        session = self._fake_session()
        ctx = MagicMock()

        async def _run():
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=None)):
                await nar._blitz_ruleset([], session, session.rule_set, ctx)

        await _run()  # must not raise


# ===========================================================================
# 18. Extension: schedule_memory_extraction
# ===========================================================================

class TestScheduleMemoryExtraction(unittest.TestCase):

    def _fake_ctx(self, history=None):
        ctx = MagicMock()
        ctx.working_history = list(history or [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ])
        return ctx

    def _fake_session(self, has_memory=True):
        session = MagicMock()
        return session

    def test_no_task_when_disabled(self):
        nar, _ = _make_narrator(enabled=False)
        nar._lang = "de"
        ctx, session = self._fake_ctx(), self._fake_session()
        nar.schedule_memory_extraction("q", [], ctx, session)
        self.assertEqual(len(nar._pending_tasks), 0)

    def test_no_task_when_empty_diff(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        nar._token_log = []
        ctx, session = self._fake_ctx(), self._fake_session()
        # cursor at end of history → empty diff
        nar._history_cursor = 10
        nar.schedule_memory_extraction("q", [], ctx, session)
        self.assertEqual(len(nar._pending_tasks), 0)

    def test_no_task_within_rate_limit(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        nar._last_blitz_time = time.monotonic()
        hist = _history(("assistant", "result xyz"))
        ctx, session = self._fake_ctx(hist), self._fake_session()
        nar.schedule_memory_extraction("q", hist, ctx, session)
        self.assertEqual(len(nar._pending_tasks), 0)

    @async_test
    async def test_working_facts_injected_into_ctx(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        ctx = self._fake_ctx()
        session = self._fake_session()
        initial_len = len(ctx.working_history)

        result = {
            "data": {"working": ["Datei liegt in /tmp/x"], "user": [], "found": True},
            "raw": "", "in": 10, "out": 8,
        }

        async def _run():
            diff = [{"role": "assistant", "content": "tool result"}]
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_memory("q", diff, ctx, session)

        await _run()
        # working_history should have grown by 1 (injected system msg)
        self.assertEqual(len(ctx.working_history), initial_len + 1)
        injected = next(m for m in ctx.working_history if m["role"] == "system"
                        and "Datei liegt" in m.get("content", ""))
        self.assertIsNotNone(injected)

    @async_test
    async def test_user_facts_persisted(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        ctx = self._fake_ctx()
        session = self._fake_session(has_memory=True)

        result = {
            "data": {"working": [], "user": ["User bevorzugt Deutsch"], "found": True},
            "raw": "", "in": 8, "out": 5,
        }

        async def _run():
            diff = [{"role": "user", "content": "bitte auf deutsch antworten"}]
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_memory("q", diff, ctx, session)

        await _run()

    @async_test
    async def test_nothing_injected_when_not_found(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        ctx = self._fake_ctx()
        initial_len = len(ctx.working_history)
        session = self._fake_session()

        result = {
            "data": {"working": [], "user": [], "found": False},
            "raw": "", "in": 5, "out": 3,
        }

        async def _run():
            diff = [{"role": "assistant", "content": "ok"}]
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_memory("q", diff, ctx, session)

        await _run()
        self.assertEqual(len(ctx.working_history), initial_len)

    @async_test
    async def test_no_crash_when_memory_is_none(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        ctx = self._fake_ctx()
        session = self._fake_session(has_memory=False)

        result = {
            "data": {"working": [], "user": ["user fakt"], "found": True},
            "raw": "", "in": 5, "out": 3,
        }

        async def _run():
            diff = [{"role": "assistant", "content": "result"}]
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_memory("q", diff, ctx, session)

        await _run()  # must not raise

    @async_test
    async def test_no_crash_on_blitz_failure(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        ctx = self._fake_ctx()
        session = self._fake_session()

        async def _run():
            diff = [{"role": "assistant", "content": "result"}]
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=None)):
                await nar._blitz_memory("q", diff, ctx, session)

        await _run()  # must not raise

    @async_test
    async def test_records_tokens_on_success(self):
        nar, _ = _make_narrator(enabled=True)
        nar._lang = "de"
        ctx = self._fake_ctx()
        session = self._fake_session()

        result = {
            "data": {"working": ["fakt"], "user": [], "found": True},
            "raw": "", "in": 12, "out": 6,
        }

        async def _run():
            diff = [{"role": "assistant", "content": "data"}]
            with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", AsyncMock(return_value=result)):
                await nar._blitz_memory("q", diff, ctx, session)

        await _run()
        self.assertEqual(nar._token_log[-1][1], 12)
        self.assertEqual(nar._token_log[-1][2], 6)


class TestNarratorRobustness(unittest.TestCase):
    async def asyncTearDown(self):
        """Räumt offene Tasks am Ende auf, um Memory Leaks in Tests zu vermeiden."""
        if hasattr(self, 'nar') and getattr(self.nar, '_pending_tasks', None):
            for task in self.nar._pending_tasks:
                task.cancel()
            if self.nar._pending_tasks:
                await asyncio.gather(*self.nar._pending_tasks, return_exceptions=True)

    @async_test
    async def test_sequential_thinks_cancel_old(self):
        """Wenn zwei Thinks schnell hintereinander kommen, MUSS der alte sterben."""
        nar, _ = _make_narrator(enabled=True)
        self.nar = nar
        nar._lang = "de"
        nar._token_log = []

        # Wir mocken die API-Antwort, sodass sie "hängt" (10 Sekunden braucht)
        async def hanging_blitz(*args, **kwargs):
            await asyncio.sleep(1.0)
            return {"data": {"t": "mock", "plan": "plan"}, "in": 10, "out": 10}

        # Patch auf die interne Methode (Passe den Pfad ggf. an deine Dateistruktur an)
        with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", new=hanging_blitz):
            # 1. Erster Think Task (soll 10s hängen)
            nar.schedule_think_result("Denke nach über Teil 1", [])
            task1 = nar._pending_tasks[-1]

            await asyncio.sleep(0)  # Event Loop einen Tick geben
            self.assertFalse(task1.done())
            self.assertTrue(nar._pending_is_think)

            # 2. Zweiter Think Task (Agent ist schneller als der Narrator)
            # Das erzwingt ein self._cancel_pending(force=True)
            nar.schedule_think_result("Denke nach über Teil 2", [])
            task2 = nar._pending_tasks[-1]

            await asyncio.sleep(0)  # Cancel verarbeiten lassen

            # Task 1 MUSS jetzt abgebrochen sein, um Kapazitäten freizugeben!
            self.assertTrue(task1.cancelled() or task1.done(), "Alter Think-Task wurde nicht abgebrochen!")

            # Task 2 MUSS laufen
            self.assertFalse(task2.done(), "Neuer Think-Task läuft nicht!")
            self.assertTrue(nar._pending_is_think, "Flag ist verloren gegangen!")

    @async_test
    async def test_think_task_finishes_and_clears_flag(self):
        """Wenn ein Think-Task (zB in 50s) erfolgreich durchläuft, muss das Flag sauber erlöschen."""
        nar, _ = _make_narrator(enabled=True)
        self.nar = nar

        # Simuliere eine sehr schnelle API-Antwort
        async def fast_blitz(*args, **kwargs):
            return {"data": {"t": "mock", "plan": "plan"}, "in": 10, "out": 10}

        with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", new=fast_blitz):
            nar.schedule_think_result("Gedanke der sofort fertig wird", [])
            task = nar._pending_tasks[-1]

            self.assertTrue(nar._pending_is_think)

            # Wir warten kurz, bis der Task WIRKLICH fertig ist
            await asyncio.wait_for(task, timeout=2.0)

            # Das Flag muss jetzt weg sein, da kein Think-Task mehr läuft
            self.assertFalse(nar._pending_is_think, "Flag '_pending_is_think' klemmt auf True!")
            self.assertEqual(len(nar._pending_tasks), 0, "Es sind noch Ghost-Tasks in der Liste!")

    @async_test
    async def test_think_deduplication(self):
        """Schickt der Agent exakt denselben Gedankengang 2x, darf der 2. Aufruf nicht triggern."""
        nar, _ = _make_narrator(enabled=True)
        self.nar = nar

        # 1. Aufruf
        nar.schedule_think_result("Dies ist ein wiederholter Gedankengang.", [])
        self.assertEqual(len(nar._pending_tasks), 1)

        # 2. Aufruf mit dem IDENTISCHEN String
        nar.schedule_think_result("Dies ist ein wiederholter Gedankengang.", [])

        # Es darf kein neuer Task entstanden sein (Länge bleibt 1)
        self.assertEqual(len(nar._pending_tasks), 1, "Deduplizierung hat versagt! Task wurde doppelt angelegt.")

    @async_test
    async def test_mixed_workload_priorities(self):
        """Think läuft im Hintergrund -> Tool-Tasks spammen -> Think muss unangetastet bleiben."""
        nar, _ = _make_narrator(enabled=True)
        self.nar = nar

        async def slow_blitz(*args, **kwargs):
            await asyncio.sleep(2.0)
            return {"data": {"t": "mock"}, "in": 10, "out": 10}

        with patch("toolboxv2.mods.isaa.base.Agent.narrator._call_blitz", new=slow_blitz):
            # 1. Think Start
            nar.schedule_think_result("Langer Think Start", [])
            t_think = nar._pending_tasks[-1]

            # 2. Ein Tool-Event kommt rein
            nar._last_blitz_time = 0.0
            nar.schedule_tool_end("calc", "1", [])

            # Da Tool-Tasks Think-Tasks nicht killen, sollten jetzt ZWEI Tasks laufen
            await asyncio.sleep(0)
            self.assertFalse(t_think.cancelled(), "Think wurde von Tool-Task gecancelt!")

            # Merke den Tool Task
            t_tool1 = [t for t in nar._pending_tasks if t != t_think][-1]

            # 3. Zweites Tool-Event kommt sofort hinterher
            nar._last_blitz_time = 0.0
            nar.schedule_tool_end("python", "2", [])

            await asyncio.sleep(0)

            # Auswertung:
            self.assertFalse(t_think.cancelled(), "Think muss auch nach Spam überleben!")
            self.assertTrue(t_tool1.cancelled() or t_tool1.done(),
                            "Alter Tool-Task wurde nicht von neuem Tool-Task gecancelt!")

if __name__ == "__main__":
    unittest.main(verbosity=2)
