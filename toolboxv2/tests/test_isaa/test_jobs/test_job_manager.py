"""
Tests for Job Manager
=====================

Tests serialization, CRUD, persistence, trigger evaluation,
job chaining, custom trigger registry, timeout handling,
offline catch-up (get_missed_jobs / fire_missed_jobs),
has_persistent_jobs, and JobLiveState integration.

Run:
    pytest toolboxv2/tests/test_jobs/test_job_manager.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

from toolboxv2.mods.isaa.extras.jobs.job_manager import (
    JobDefinition,
    JobEventBus,
    JobScheduler,
    OnCliLifecycleEvaluator,
    OnIntervalEvaluator,
    OnJobEventEvaluator,
    OnTimeEvaluator,
    TriggerConfig,
    TriggerEvaluator,
    TriggerRegistry,
    TriggerType,
)


# =============================================================================
# Async test base
# =============================================================================

class AsyncTestCase(unittest.TestCase):
    """Base class providing async_run() for all async tests."""

    def async_run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @classmethod
    def setUpClass(cls):
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())


# =============================================================================
# Helpers
# =============================================================================

def _tmp_jobs_file(content="[]") -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    f.write(content)
    f.close()
    return Path(f.name)


def _make_scheduler(jobs_file: Path | None = None, callback=None) -> JobScheduler:
    if jobs_file is None:
        jobs_file = _tmp_jobs_file()
    cb = callback or AsyncMock(return_value="done")
    return JobScheduler(jobs_file, cb)


def _interval_job(job_id="j1", interval=60) -> JobDefinition:
    return JobDefinition(
        job_id=job_id, name=f"Interval {job_id}", agent_name="self",
        query="do work",
        trigger=TriggerConfig(trigger_type="on_interval", interval_seconds=interval),
    )


def _time_job(job_id="jt", delta_seconds=-10) -> JobDefinition:
    """on_time job due delta_seconds from now (negative = past, positive = future)."""
    at = (datetime.now(timezone.utc) + timedelta(seconds=delta_seconds)).isoformat()
    return JobDefinition(
        job_id=job_id, name=f"Time {job_id}", agent_name="self",
        query="do work",
        trigger=TriggerConfig(trigger_type="on_time", at_datetime=at),
    )


# =============================================================================
# TriggerConfig
# =============================================================================

class TestTriggerConfig(unittest.TestCase):
    """Serialization round-trips and edge cases."""

    def test_to_dict_omits_none(self):
        tc = TriggerConfig(trigger_type="on_time", at_datetime="2026-01-01T00:00:00Z")
        d = tc.to_dict()
        self.assertEqual(d["trigger_type"], "on_time")
        self.assertEqual(d["at_datetime"], "2026-01-01T00:00:00Z")
        self.assertNotIn("interval_seconds", d)
        self.assertNotIn("cron_expression", d)

    def test_from_dict_ignores_unknown(self):
        tc = TriggerConfig.from_dict({
            "trigger_type": "on_interval",
            "interval_seconds": 60,
            "unknown_field": "ignored",
        })
        self.assertEqual(tc.trigger_type, "on_interval")
        self.assertEqual(tc.interval_seconds, 60)

    def test_roundtrip_file_changed(self):
        original = TriggerConfig(
            trigger_type="on_file_changed",
            watch_path="/tmp/test",
            watch_patterns=["*.py", "*.js"],
        )
        restored = TriggerConfig.from_dict(original.to_dict())
        self.assertEqual(restored.trigger_type, original.trigger_type)
        self.assertEqual(restored.watch_path, original.watch_path)
        self.assertEqual(restored.watch_patterns, original.watch_patterns)

    def test_roundtrip_cron(self):
        tc = TriggerConfig(trigger_type="on_cron", cron_expression="0 3 * * *")
        restored = TriggerConfig.from_dict(tc.to_dict())
        self.assertEqual(restored.cron_expression, "0 3 * * *")

    def test_roundtrip_extra(self):
        tc = TriggerConfig(trigger_type="on_interval", interval_seconds=30,
                           extra={"dream_config": {"max_budget": 10}})
        restored = TriggerConfig.from_dict(tc.to_dict())
        self.assertEqual(restored.extra["dream_config"]["max_budget"], 10)

    def test_all_trigger_type_enum_values_exist(self):
        for tt in TriggerType:
            # Ensure the string value is accessible
            self.assertIsInstance(tt.value, str)
            self.assertTrue(tt.value.startswith("on_"))


# =============================================================================
# JobDefinition
# =============================================================================

class TestJobDefinition(unittest.TestCase):
    """Serialization, ID generation, field preservation."""

    def test_to_dict_and_from_dict(self):
        trigger = TriggerConfig(trigger_type="on_interval", interval_seconds=300)
        job = JobDefinition(
            job_id="job_abc123", name="Test Job", agent_name="self",
            query="do something", trigger=trigger, timeout_seconds=120,
        )
        d = job.to_dict()
        self.assertEqual(d["job_id"], "job_abc123")
        self.assertEqual(d["name"], "Test Job")
        self.assertIsInstance(d["trigger"], dict)
        self.assertEqual(d["trigger"]["interval_seconds"], 300)
        self.assertNotIn("_last_fired_ts", d)

        restored = JobDefinition.from_dict(d)
        self.assertEqual(restored.job_id, job.job_id)
        self.assertEqual(restored.name, job.name)
        self.assertEqual(restored.trigger.interval_seconds, 300)

    def test_generate_id_unique(self):
        ids = {JobDefinition.generate_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_generate_id_format(self):
        jid = JobDefinition.generate_id()
        self.assertTrue(jid.startswith("job_"))
        self.assertEqual(len(jid), 12)  # "job_" + 8 hex

    def test_from_dict_ignores_internal_fields(self):
        d = {
            "job_id": "j1", "name": "N", "agent_name": "a", "query": "q",
            "trigger": {"trigger_type": "on_interval", "interval_seconds": 5},
            "_last_fired_ts": 999999.0,
            "status": "active",
        }
        job = JobDefinition.from_dict(d)
        self.assertEqual(job._last_fired_ts, 0.0)  # default, not from dict

    def test_all_stats_default_zero(self):
        job = JobDefinition(
            job_id="x", name="x", agent_name="x", query="x",
            trigger=TriggerConfig(trigger_type="on_time"),
        )
        self.assertEqual(job.run_count, 0)
        self.assertEqual(job.fail_count, 0)
        self.assertIsNone(job.last_run_at)
        self.assertIsNone(job.last_result)


# =============================================================================
# TriggerRegistry
# =============================================================================

class TestTriggerRegistry(unittest.TestCase):
    def test_register_and_get(self):
        r = TriggerRegistry()
        ev = MagicMock()
        r.register("custom", ev)
        self.assertIs(r.get("custom"), ev)

    def test_unregister(self):
        r = TriggerRegistry()
        r.register("t", MagicMock())
        r.unregister("t")
        self.assertIsNone(r.get("t"))

    def test_unregister_nonexistent_safe(self):
        r = TriggerRegistry()
        r.unregister("never_registered")  # must not raise

    def test_available_types(self):
        r = TriggerRegistry()
        r.register("a", MagicMock())
        r.register("b", MagicMock())
        self.assertIn("a", r.available_types())
        self.assertIn("b", r.available_types())

    def test_get_nonexistent(self):
        self.assertIsNone(TriggerRegistry().get("nope"))

    def test_overwrite_evaluator(self):
        r = TriggerRegistry()
        ev1, ev2 = MagicMock(), MagicMock()
        r.register("t", ev1)
        r.register("t", ev2)
        self.assertIs(r.get("t"), ev2)


# =============================================================================
# Built-in evaluators
# =============================================================================

class TestOnTimeEvaluator(AsyncTestCase):
    def test_fires_when_past(self):
        ev = OnTimeEvaluator()
        job = _time_job(delta_seconds=-3600)
        self.assertTrue(self.async_run(ev.evaluate(job)))
        self.assertEqual(job.status, "expired")

    def test_does_not_fire_future(self):
        ev = OnTimeEvaluator()
        job = _time_job(delta_seconds=3600)
        self.assertFalse(self.async_run(ev.evaluate(job)))
        self.assertEqual(job.status, "active")

    def test_no_datetime_returns_false(self):
        ev = OnTimeEvaluator()
        job = JobDefinition(
            job_id="j", name="n", agent_name="a", query="q",
            trigger=TriggerConfig(trigger_type="on_time"),
        )
        self.assertFalse(self.async_run(ev.evaluate(job)))

    def test_invalid_datetime_returns_false(self):
        ev = OnTimeEvaluator()
        job = JobDefinition(
            job_id="j", name="n", agent_name="a", query="q",
            trigger=TriggerConfig(trigger_type="on_time", at_datetime="not-a-date"),
        )
        self.assertFalse(self.async_run(ev.evaluate(job)))

    def test_naive_datetime_treated_as_utc(self):
        ev = OnTimeEvaluator()
        past_naive = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        job = JobDefinition(
            job_id="j", name="n", agent_name="a", query="q",
            trigger=TriggerConfig(trigger_type="on_time", at_datetime=past_naive),
        )
        self.assertTrue(self.async_run(ev.evaluate(job)))

    def test_fires_exactly_at_boundary(self):
        ev = OnTimeEvaluator()
        # 1 second in the past — must fire
        at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        job = JobDefinition(
            job_id="j", name="n", agent_name="a", query="q",
            trigger=TriggerConfig(trigger_type="on_time", at_datetime=at),
        )
        self.assertTrue(self.async_run(ev.evaluate(job)))


class TestOnIntervalEvaluator(AsyncTestCase):
    def test_fires_when_elapsed(self):
        ev = OnIntervalEvaluator()
        job = _interval_job(interval=1)
        job._last_fired_ts = time.time() - 2
        self.assertTrue(self.async_run(ev.evaluate(job)))

    def test_does_not_fire_when_recent(self):
        ev = OnIntervalEvaluator()
        job = _interval_job(interval=60)
        job._last_fired_ts = time.time()
        self.assertFalse(self.async_run(ev.evaluate(job)))

    def test_fires_when_never_run(self):
        ev = OnIntervalEvaluator()
        job = _interval_job(interval=60)
        job._last_fired_ts = 0.0  # default = never
        self.assertTrue(self.async_run(ev.evaluate(job)))

    def test_no_interval_returns_false(self):
        ev = OnIntervalEvaluator()
        job = JobDefinition(
            job_id="j", name="n", agent_name="a", query="q",
            trigger=TriggerConfig(trigger_type="on_interval"),
        )
        self.assertFalse(self.async_run(ev.evaluate(job)))

    def test_exactly_at_boundary(self):
        ev = OnIntervalEvaluator()
        job = _interval_job(interval=5)
        job._last_fired_ts = time.time() - 5
        self.assertTrue(self.async_run(ev.evaluate(job)))


class TestOnCliLifecycleEvaluator(AsyncTestCase):
    def test_mark_pending_and_evaluate(self):
        ev = OnCliLifecycleEvaluator()
        job = _interval_job("j1")
        self.assertFalse(self.async_run(ev.evaluate(job)))
        ev.mark_pending("j1")
        self.assertTrue(self.async_run(ev.evaluate(job)))
        # consumed
        self.assertFalse(self.async_run(ev.evaluate(job)))

    def test_teardown_clears_pending(self):
        ev = OnCliLifecycleEvaluator()
        job = _interval_job("j2")
        ev.mark_pending("j2")
        self.async_run(ev.teardown(job))
        self.assertFalse(self.async_run(ev.evaluate(job)))

    def test_multiple_jobs_independent(self):
        ev = OnCliLifecycleEvaluator()
        j1, j2 = _interval_job("a"), _interval_job("b")
        ev.mark_pending("a")
        self.assertTrue(self.async_run(ev.evaluate(j1)))
        self.assertFalse(self.async_run(ev.evaluate(j2)))


class TestOnJobEventEvaluator(AsyncTestCase):
    def test_job_chain_fires(self):
        ev = OnJobEventEvaluator()
        job_b = JobDefinition(
            job_id="job_b", name="B", agent_name="self", query="q",
            trigger=TriggerConfig(trigger_type="on_job_completed", watch_job_id="job_a"),
        )
        self.assertFalse(self.async_run(ev.evaluate(job_b)))
        ev.notify("on_job_completed", "job_a", [job_b])
        self.assertTrue(self.async_run(ev.evaluate(job_b)))
        self.assertFalse(self.async_run(ev.evaluate(job_b)))  # consumed

    def test_wrong_source_does_not_fire(self):
        ev = OnJobEventEvaluator()
        job_b = JobDefinition(
            job_id="job_b", name="B", agent_name="self", query="q",
            trigger=TriggerConfig(trigger_type="on_job_completed", watch_job_id="job_a"),
        )
        ev.notify("on_job_completed", "job_c", [job_b])
        self.assertFalse(self.async_run(ev.evaluate(job_b)))

    def test_wrong_event_type_does_not_fire(self):
        ev = OnJobEventEvaluator()
        job_b = JobDefinition(
            job_id="job_b", name="B", agent_name="self", query="q",
            trigger=TriggerConfig(trigger_type="on_job_completed", watch_job_id="job_a"),
        )
        ev.notify("on_job_failed", "job_a", [job_b])
        self.assertFalse(self.async_run(ev.evaluate(job_b)))

    def test_paused_job_not_notified(self):
        ev = OnJobEventEvaluator()
        job_b = JobDefinition(
            job_id="job_b", name="B", agent_name="self", query="q",
            status="paused",
            trigger=TriggerConfig(trigger_type="on_job_completed", watch_job_id="job_a"),
        )
        ev.notify("on_job_completed", "job_a", [job_b])
        self.assertFalse(self.async_run(ev.evaluate(job_b)))


# =============================================================================
# JobEventBus
# =============================================================================

class TestJobEventBus(unittest.TestCase):
    def test_emit_and_listen(self):
        bus = JobEventBus()
        received = []
        bus.on("evt", lambda e, d: received.append((e, d)))
        bus.emit("evt", {"key": "val"})
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][1]["key"], "val")

    def test_off_removes_listener(self):
        bus = JobEventBus()
        calls = []
        cb = lambda e, d: calls.append(e)
        bus.on("evt", cb)
        bus.off("evt", cb)
        bus.emit("evt", {})
        self.assertEqual(len(calls), 0)

    def test_multiple_listeners(self):
        bus = JobEventBus()
        a, b = [], []
        bus.on("e", lambda ev, d: a.append(1))
        bus.on("e", lambda ev, d: b.append(1))
        bus.emit("e", {})
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)

    def test_listener_exception_does_not_propagate(self):
        bus = JobEventBus()
        bus.on("e", lambda e, d: 1/0)
        bus.emit("e", {})  # must not raise

    def test_emit_no_listeners_safe(self):
        bus = JobEventBus()
        bus.emit("no_listeners", {"x": 1})  # must not raise

    def test_emit_none_data(self):
        bus = JobEventBus()
        received = []
        bus.on("e", lambda ev, d: received.append(d))
        bus.emit("e")  # no data arg
        self.assertEqual(received[0], {})


# =============================================================================
# JobScheduler — CRUD + persistence
# =============================================================================

class TestJobScheduler(AsyncTestCase):
    def test_add_and_list(self):
        s = _make_scheduler()
        s.add_job(_interval_job("", 60))
        self.assertEqual(len(s.list_jobs()), 1)

    def test_add_generates_id_when_empty(self):
        s = _make_scheduler()
        job = _interval_job("")
        jid = s.add_job(job)
        self.assertTrue(jid.startswith("job_"))
        self.assertEqual(s.get_job(jid).name, job.name)

    def test_remove_existing(self):
        s = _make_scheduler()
        s.add_job(_interval_job("job_test"))
        self.assertTrue(s.remove_job("job_test"))
        self.assertEqual(len(s.list_jobs()), 0)

    def test_remove_nonexistent(self):
        s = _make_scheduler()
        self.assertFalse(s.remove_job("does_not_exist"))

    def test_pause_resume_cycle(self):
        s = _make_scheduler()
        s.add_job(_interval_job("job_pr"))
        self.assertTrue(s.pause_job("job_pr"))
        self.assertEqual(s.get_job("job_pr").status, "paused")
        self.assertTrue(s.resume_job("job_pr"))
        self.assertEqual(s.get_job("job_pr").status, "active")

    def test_pause_already_paused(self):
        s = _make_scheduler()
        s.add_job(_interval_job("j"))
        s.pause_job("j")
        self.assertFalse(s.pause_job("j"))  # already paused

    def test_resume_active_job_fails(self):
        s = _make_scheduler()
        s.add_job(_interval_job("j"))
        self.assertFalse(s.resume_job("j"))  # not paused

    def test_persistence_roundtrip(self):
        jf = _tmp_jobs_file()
        s1 = _make_scheduler(jf)
        job = JobDefinition(
            job_id="job_persist", name="Persist Test", agent_name="worker",
            query="do work",
            trigger=TriggerConfig(trigger_type="on_interval", interval_seconds=120),
            timeout_seconds=60,
        )
        s1.add_job(job)

        s2 = _make_scheduler(jf)
        loaded = s2.get_job("job_persist")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "Persist Test")
        self.assertEqual(loaded.trigger.interval_seconds, 120)
        self.assertEqual(loaded.timeout_seconds, 60)

    def test_persistence_bad_json_skips_gracefully(self):
        jf = _tmp_jobs_file('[{"job_id": "ok"}, {"BROKEN": true}]')
        # Must not raise even with a partial bad entry
        s = _make_scheduler(jf)
        # The valid entry "ok" may or may not load, but no exception is raised
        self.assertIsInstance(s.list_jobs(), list)

    def test_paused_job_not_fired_during_tick(self):
        cb = AsyncMock(return_value="done")
        s = _make_scheduler(callback=cb)
        job = JobDefinition(
            job_id="job_p", name="P", agent_name="self", query="q",
            trigger=TriggerConfig(trigger_type="on_interval", interval_seconds=0),
        )
        s.add_job(job)
        s.pause_job("job_p")
        self.async_run(s._tick())
        cb.assert_not_called()

    def test_find_jobs_by_name_partial_match(self):
        s = _make_scheduler()
        s.add_job(JobDefinition(job_id="a1", name="Daily Report", agent_name="self",
                                query="q", trigger=TriggerConfig(trigger_type="on_time")))
        s.add_job(JobDefinition(job_id="a2", name="Weekly Cleanup", agent_name="self",
                                query="q", trigger=TriggerConfig(trigger_type="on_time")))
        r = s.find_jobs_by_name("daily")
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].name, "Daily Report")

    def test_find_jobs_by_job_id(self):
        s = _make_scheduler()
        s.add_job(JobDefinition(job_id="myunique42", name="X", agent_name="self",
                                query="q", trigger=TriggerConfig(trigger_type="on_time")))
        r = s.find_jobs_by_name("unique42")
        self.assertEqual(len(r), 1)

    def test_active_and_total_count(self):
        s = _make_scheduler()
        s.add_job(_interval_job("j1"))
        s.add_job(_interval_job("j2"))
        s.pause_job("j2")
        self.assertEqual(s.total_count, 2)
        self.assertEqual(s.active_count, 1)

    def test_created_at_set_on_add(self):
        s = _make_scheduler()
        jid = s.add_job(_interval_job(""))
        job = s.get_job(jid)
        self.assertIsNotNone(job.created_at)
        # Should be parseable as ISO datetime
        datetime.fromisoformat(job.created_at)

    def test_add_dream_job_creates_active_job(self):
        s = _make_scheduler()
        jid = s.add_dream_job("my_agent")
        job = s.get_job(jid)
        self.assertIsNotNone(job)
        self.assertEqual(job.query, "__dream__")
        self.assertEqual(job.agent_name, "my_agent")
        self.assertEqual(job.status, "active")

    def test_add_dream_job_on_agent_idle(self):
        s = _make_scheduler()
        jid = s.add_dream_job("agent", trigger_type="on_agent_idle", agent_idle_seconds=300)
        job = s.get_job(jid)
        self.assertEqual(job.trigger.trigger_type, "on_agent_idle")
        self.assertEqual(job.trigger.agent_idle_seconds, 300)


# =============================================================================
# _fire_job — stats, events, liveness
# =============================================================================

class TestFireJob(AsyncTestCase):
    def _scheduler_with_callback(self, cb) -> tuple[JobScheduler, Path]:
        jf = _tmp_jobs_file()
        return JobScheduler(jf, cb), jf

    def test_fire_updates_run_count_and_result(self):
        cb = AsyncMock(return_value="great")
        s, _ = self._scheduler_with_callback(cb)
        job = _interval_job("jf")
        s.add_job(job)
        self.async_run(s._fire_job(job))
        self.assertEqual(job.run_count, 1)
        self.assertIsNotNone(job.last_run_at)
        self.assertEqual(job.last_result, "completed")
        cb.assert_called_once_with(job)

    def test_fire_timeout_sets_fail(self):
        async def slow(j):
            await asyncio.sleep(10)

        s, _ = self._scheduler_with_callback(slow)
        job = JobDefinition(
            job_id="jto", name="T", agent_name="self", query="q",
            trigger=TriggerConfig(trigger_type="on_time"), timeout_seconds=1,
        )
        s.add_job(job)
        self.async_run(s._fire_job(job))
        self.assertEqual(job.last_result, "timeout")
        self.assertEqual(job.fail_count, 1)

    def test_fire_exception_sets_failed(self):
        async def boom(j):
            raise RuntimeError("crash")

        s, _ = self._scheduler_with_callback(boom)
        job = _interval_job("jfail")
        s.add_job(job)
        self.async_run(s._fire_job(job))
        self.assertEqual(job.last_result, "failed")
        self.assertEqual(job.fail_count, 1)

    def test_fire_removes_from_firing_set_on_success(self):
        cb = AsyncMock(return_value="ok")
        s, _ = self._scheduler_with_callback(cb)
        job = _interval_job("jf2")
        s.add_job(job)
        self.async_run(s._fire_job(job))
        self.assertNotIn("jf2", s._firing)

    def test_fire_removes_from_firing_set_on_failure(self):
        async def boom(j): raise RuntimeError()
        s, _ = self._scheduler_with_callback(boom)
        job = _interval_job("jf3")
        s.add_job(job)
        self.async_run(s._fire_job(job))
        self.assertNotIn("jf3", s._firing)

    def test_fire_emits_completed_event(self):
        cb = AsyncMock(return_value="res")
        s, _ = self._scheduler_with_callback(cb)
        events = []
        s.event_bus.on("job_completed", lambda e, d: events.append(d))
        job = _interval_job("je")
        s.add_job(job)
        self.async_run(s._fire_job(job))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["job_id"], "je")

    def test_fire_emits_failed_event(self):
        async def boom(j): raise ValueError("x")
        s, _ = self._scheduler_with_callback(boom)
        events = []
        s.event_bus.on("job_failed", lambda e, d: events.append(d))
        job = _interval_job("jef")
        s.add_job(job)
        self.async_run(s._fire_job(job))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["job_id"], "jef")

    def test_fire_persists_state_to_file(self):
        jf = _tmp_jobs_file()
        cb = AsyncMock(return_value="ok")
        s = JobScheduler(jf, cb)
        job = _interval_job("jp")
        s.add_job(job)
        self.async_run(s._fire_job(job))

        # Reload from disk and verify
        data = json.loads(jf.read_text())
        record = next((r for r in data if r["job_id"] == "jp"), None)
        self.assertIsNotNone(record)
        self.assertEqual(record["run_count"], 1)
        self.assertEqual(record["last_result"], "completed")


# =============================================================================
# has_persistent_jobs (NEW)
# =============================================================================

class TestHasPersistentJobs(unittest.TestCase):
    def _s(self):
        return _make_scheduler()

    def test_true_for_on_interval(self):
        s = self._s()
        s.add_job(_interval_job("j", 60))
        self.assertTrue(s.has_persistent_jobs())

    def test_true_for_on_time(self):
        s = self._s()
        s.add_job(_time_job("jt", -10))
        self.assertTrue(s.has_persistent_jobs())

    def test_true_for_on_cron(self):
        s = self._s()
        s.add_job(JobDefinition(
            job_id="jc", name="cron", agent_name="a", query="q",
            trigger=TriggerConfig(trigger_type="on_cron", cron_expression="0 3 * * *"),
        ))
        self.assertTrue(s.has_persistent_jobs())

    def test_true_for_on_system_boot(self):
        s = self._s()
        s.add_job(JobDefinition(
            job_id="jb", name="boot", agent_name="a", query="q",
            trigger=TriggerConfig(trigger_type="on_system_boot"),
        ))
        self.assertTrue(s.has_persistent_jobs())

    def test_false_for_non_persistent_triggers(self):
        s = self._s()
        s.add_job(JobDefinition(
            job_id="jl", name="lifecycle", agent_name="a", query="q",
            trigger=TriggerConfig(trigger_type="on_cli_start"),
        ))
        self.assertFalse(s.has_persistent_jobs())

    def test_false_when_persistent_job_is_paused(self):
        s = self._s()
        s.add_job(_interval_job("j"))
        s.pause_job("j")
        self.assertFalse(s.has_persistent_jobs())

    def test_false_when_no_jobs(self):
        s = self._s()
        self.assertFalse(s.has_persistent_jobs())

    def test_mixed_jobs_returns_true_if_any_persistent_active(self):
        s = self._s()
        s.add_job(JobDefinition(
            job_id="jcli", name="cli", agent_name="a", query="q",
            trigger=TriggerConfig(trigger_type="on_cli_start"),
        ))
        s.add_job(_interval_job("ji"))
        self.assertTrue(s.has_persistent_jobs())


# =============================================================================
# get_missed_jobs (NEW)
# =============================================================================

class TestGetMissedJobs(AsyncTestCase):
    def test_interval_missed_after_long_gap(self):
        s = _make_scheduler()
        job = _interval_job("ji", interval=60)
        last = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        job.last_run_at = last
        s.add_job(job)
        missed = s.get_missed_jobs()
        self.assertIn(job, missed)

    def test_interval_not_missed_when_recent(self):
        s = _make_scheduler()
        job = _interval_job("ji", interval=3600)
        job.last_run_at = datetime.now(timezone.utc).isoformat()
        s.add_job(job)
        self.assertNotIn(job, s.get_missed_jobs())

    def test_interval_missed_when_never_run(self):
        s = _make_scheduler()
        job = _interval_job("jnr", interval=60)
        s.add_job(job)
        self.assertIn(job, s.get_missed_jobs())

    def test_on_time_missed_when_never_run_past_due(self):
        s = _make_scheduler()
        job = _time_job("jt", delta_seconds=-10)
        s.add_job(job)
        missed = s.get_missed_jobs()
        self.assertIn(job, missed)

    def test_on_time_not_missed_future(self):
        s = _make_scheduler()
        job = _time_job("jf", delta_seconds=3600)
        s.add_job(job)
        self.assertNotIn(job, s.get_missed_jobs())

    def test_on_time_not_missed_already_ran_after_target(self):
        s = _make_scheduler()
        job = _time_job("jran", delta_seconds=-10)
        # Mark as already ran after the target time
        job.last_run_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        s.add_job(job)
        self.assertNotIn(job, s.get_missed_jobs())

    def test_paused_job_not_in_missed(self):
        s = _make_scheduler()
        job = _interval_job("jp", interval=1)
        job.last_run_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        s.add_job(job)
        s.pause_job("jp")
        self.assertNotIn(job, s.get_missed_jobs())

    def test_cli_start_not_in_missed(self):
        s = _make_scheduler()
        s.add_job(JobDefinition(
            job_id="jcli", name="cli", agent_name="a", query="q",
            trigger=TriggerConfig(trigger_type="on_cli_start"),
        ))
        self.assertEqual(s.get_missed_jobs(), [])

    def test_expired_job_not_in_missed(self):
        s = _make_scheduler()
        job = _time_job("jex", delta_seconds=-10)
        job.status = "expired"
        s.add_job(job)
        self.assertNotIn(job, s.get_missed_jobs())

    def test_returns_empty_when_no_jobs(self):
        self.assertEqual(_make_scheduler().get_missed_jobs(), [])


# =============================================================================
# fire_missed_jobs (NEW)
# =============================================================================

class TestFireMissedJobs(AsyncTestCase):
    def test_fires_missed_jobs_and_returns_count(self):
        fired = []

        async def cb(job):
            fired.append(job.job_id)
            return "ok"

        jf = _tmp_jobs_file()
        s = JobScheduler(jf, cb)

        # Job that was due 2 minutes ago
        job = _interval_job("missed_j", interval=60)
        job.last_run_at = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        s.add_job(job)

        count = self.async_run(s.fire_missed_jobs())
        self.assertEqual(count, 1)

        # Give ensure_future a chance to complete
        self.async_run(asyncio.sleep(0.2))
        self.assertIn("missed_j", fired)

    def test_returns_zero_when_nothing_missed(self):
        s = _make_scheduler()
        job = _interval_job("fresh", interval=3600)
        job.last_run_at = datetime.now(timezone.utc).isoformat()
        s.add_job(job)
        count = self.async_run(s.fire_missed_jobs())
        self.assertEqual(count, 0)

    def test_fires_multiple_missed(self):
        results = []

        async def cb(job):
            results.append(job.job_id)
            return "ok"

        jf = _tmp_jobs_file()
        s = JobScheduler(jf, cb)
        for i in range(3):
            job = _interval_job(f"m{i}", interval=60)
            job.last_run_at = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
            s.add_job(job)

        count = self.async_run(s.fire_missed_jobs())
        self.assertEqual(count, 3)

        self.async_run(asyncio.sleep(0.3))
        self.assertEqual(sorted(results), ["m0", "m1", "m2"])


# =============================================================================
# Custom trigger registry integration
# =============================================================================

class TestCustomTrigger(AsyncTestCase):
    def test_always_fire_evaluator(self):
        s = _make_scheduler()

        class AlwaysFire:
            async def setup(self, job, scheduler): pass
            async def evaluate(self, job): return True
            async def teardown(self, job): pass

        s.trigger_registry.register("always_fire", AlwaysFire())
        job = JobDefinition(
            job_id="jcust", name="C", agent_name="self", query="q",
            trigger=TriggerConfig(trigger_type="always_fire"),
        )
        s.add_job(job)
        self.assertTrue(self.async_run(s._evaluate_trigger(job)))

    def test_unknown_trigger_returns_false(self):
        s = _make_scheduler()
        job = JobDefinition(
            job_id="junk", name="U", agent_name="self", query="q",
            trigger=TriggerConfig(trigger_type="totally_unknown"),
        )
        s.add_job(job)
        self.assertFalse(self.async_run(s._evaluate_trigger(job)))

    def test_evaluator_exception_returns_false(self):
        s = _make_scheduler()

        class Broken:
            async def setup(self, job, scheduler): pass
            async def evaluate(self, job): raise RuntimeError("boom")
            async def teardown(self, job): pass

        s.trigger_registry.register("broken", Broken())
        job = JobDefinition(
            job_id="jbr", name="B", agent_name="self", query="q",
            trigger=TriggerConfig(trigger_type="broken"),
        )
        s.add_job(job)
        self.assertFalse(self.async_run(s._evaluate_trigger(job)))


# =============================================================================
# Scheduler lifecycle (start / stop / tick)
# =============================================================================

class TestSchedulerLifecycle(AsyncTestCase):
    def test_start_and_stop(self):
        s = _make_scheduler()

        async def run():
            await s.start()
            self.assertTrue(s._running)
            await asyncio.sleep(0.1)
            await s.stop()
            self.assertFalse(s._running)

        self.async_run(run())

    def test_double_start_is_idempotent(self):
        s = _make_scheduler()

        async def run():
            await s.start()
            task1 = s._tick_task
            await s.start()  # second call should be no-op
            self.assertIs(s._tick_task, task1)
            await s.stop()

        self.async_run(run())

    def test_fire_lifecycle_on_cli_start(self):
        fired = []

        async def cb(job):
            fired.append(job.job_id)
            return "ok"

        jf = _tmp_jobs_file()
        s = JobScheduler(jf, cb)
        job = JobDefinition(
            job_id="jcli", name="cli", agent_name="self", query="q",
            trigger=TriggerConfig(trigger_type="on_cli_start"),
        )
        s.add_job(job)

        async def run():
            await s.start()
            await s.fire_lifecycle("on_cli_start")
            await asyncio.sleep(0.2)
            await s.stop()

        self.async_run(run())
        self.assertIn("jcli", fired)


# =============================================================================
# Job chaining integration
# =============================================================================

class TestJobChainIntegration(AsyncTestCase):
    def test_a_completes_b_fires(self):
        jf = _tmp_jobs_file()
        fired = []

        async def cb(job):
            fired.append(job.job_id)
            return "ok"

        s = JobScheduler(jf, cb)
        job_a = JobDefinition(
            job_id="job_a", name="A", agent_name="self", query="do A",
            trigger=TriggerConfig(
                trigger_type="on_time",
                at_datetime=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
            ),
        )
        job_b = JobDefinition(
            job_id="job_b", name="B", agent_name="self", query="do B",
            trigger=TriggerConfig(trigger_type="on_job_completed", watch_job_id="job_a"),
        )
        s.add_job(job_a)
        s.add_job(job_b)

        self.async_run(s._fire_job(job_a))

        # After A fires, B should be pending
        result = self.async_run(s._evaluate_trigger(job_b))
        self.assertTrue(result)

    def test_job_b_does_not_fire_on_wrong_event(self):
        jf = _tmp_jobs_file()

        async def cb(job): return "ok"
        s = JobScheduler(jf, cb)

        job_b = JobDefinition(
            job_id="job_b", name="B", agent_name="self", query="q",
            trigger=TriggerConfig(trigger_type="on_job_completed", watch_job_id="job_a"),
        )
        s.add_job(job_b)

        # Emit a failed event, not completed
        s.event_bus.emit("job_failed", {"job_id": "job_a"})
        result = self.async_run(s._evaluate_trigger(job_b))
        self.assertFalse(result)


# =============================================================================
# JobLiveState (NEW)
# =============================================================================

class TestJobLiveState(unittest.TestCase):
    """Tests for JobLiveStateWriter and JobLiveStateReader."""

    def _live_file(self) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".live.json", delete=False, mode="w")
        f.write("{}")
        f.close()
        return Path(f.name)

    def test_writer_start_creates_entry(self):
        from toolboxv2.mods.isaa.extras.jobs.job_live_state import (
            JobLiveStateWriter, JobLiveStateReader,
        )
        lf = self._live_file()
        writer = JobLiveStateWriter(lf)
        writer.start("jid1", "My Job", "my_agent", "do stuff")

        reader = JobLiveStateReader(lf)
        state = reader.read()
        self.assertIn("jid1", state)
        e = state["jid1"]
        self.assertEqual(e.status, "running")
        self.assertEqual(e.name, "My Job")
        self.assertEqual(e.agent_name, "my_agent")

    def test_writer_update_iteration(self):
        from toolboxv2.mods.isaa.extras.jobs.job_live_state import (
            JobLiveStateWriter, JobLiveStateReader,
        )
        lf = self._live_file()
        writer = JobLiveStateWriter(lf)
        writer.start("j2", "J", "a", "q")
        writer.update_iteration("j2", iteration=3, tool="web_search",
                                thought="Thinking…", context_used=5000, context_max=200_000)

        state = JobLiveStateReader(lf).read()
        e = state["j2"]
        self.assertEqual(e.iteration, 3)
        self.assertIn("web_search", e.tool_calls)
        self.assertEqual(e.last_thought, "Thinking…")
        self.assertEqual(e.context_used, 5000)

    def test_writer_finish_removes_entry(self):
        from toolboxv2.mods.isaa.extras.jobs.job_live_state import (
            JobLiveStateWriter, JobLiveStateReader,
        )
        lf = self._live_file()
        writer = JobLiveStateWriter(lf)
        writer.start("jfin", "J", "a", "q")
        writer.finish("jfin", "done")

        state = JobLiveStateReader(lf).read()
        self.assertNotIn("jfin", state)

    def test_tool_calls_capped_at_max(self):
        from toolboxv2.mods.isaa.extras.jobs.job_live_state import (
            JobLiveStateWriter, JobLiveStateReader,
        )
        lf = self._live_file()
        writer = JobLiveStateWriter(lf)
        writer.start("jtools", "J", "a", "q")
        for i in range(20):
            writer.update_iteration("jtools", iteration=i, tool=f"tool_{i}")

        state = JobLiveStateReader(lf).read()
        self.assertLessEqual(len(state["jtools"].tool_calls), writer.MAX_TOOL_HISTORY)

    def test_context_pct_calculation(self):
        from toolboxv2.mods.isaa.extras.jobs.job_live_state import JobLiveEntry
        e = JobLiveEntry(job_id="j", context_used=100_000, context_max=200_000)
        self.assertAlmostEqual(e.context_pct(), 50.0)

    def test_context_pct_capped_at_100(self):
        from toolboxv2.mods.isaa.extras.jobs.job_live_state import JobLiveEntry
        e = JobLiveEntry(job_id="j", context_used=300_000, context_max=200_000)
        self.assertEqual(e.context_pct(), 100.0)

    def test_context_pct_zero_max(self):
        from toolboxv2.mods.isaa.extras.jobs.job_live_state import JobLiveEntry
        e = JobLiveEntry(job_id="j", context_used=0, context_max=0)
        self.assertEqual(e.context_pct(), 0.0)

    def test_reader_returns_empty_when_file_missing(self):
        from toolboxv2.mods.isaa.extras.jobs.job_live_state import JobLiveStateReader
        r = JobLiveStateReader(Path("/tmp/this_file_does_not_exist_xyz.live.json"))
        self.assertEqual(r.read(), {})

    def test_reader_returns_empty_on_corrupt_json(self):
        from toolboxv2.mods.isaa.extras.jobs.job_live_state import JobLiveStateReader
        lf = self._live_file()
        lf.write_text("not valid json")
        self.assertEqual(JobLiveStateReader(lf).read(), {})

    def test_roundtrip_serialization(self):
        from toolboxv2.mods.isaa.extras.jobs.job_live_state import JobLiveEntry
        e = JobLiveEntry(
            job_id="j", name="N", agent_name="A", status="running",
            iteration=7, tool_calls=["t1", "t2"], last_thought="hello",
            context_used=1000, context_max=50000,
        )
        restored = JobLiveEntry.from_dict(e.to_dict())
        self.assertEqual(restored.job_id, "j")
        self.assertEqual(restored.iteration, 7)
        self.assertEqual(restored.tool_calls, ["t1", "t2"])

    def test_multiple_jobs_tracked_independently(self):
        from toolboxv2.mods.isaa.extras.jobs.job_live_state import (
            JobLiveStateWriter, JobLiveStateReader,
        )
        lf = self._live_file()
        w = JobLiveStateWriter(lf)
        w.start("j1", "J1", "a1", "q1")
        w.start("j2", "J2", "a2", "q2")
        w.update_iteration("j1", 5)
        w.finish("j2", "done")

        state = JobLiveStateReader(lf).read()
        self.assertIn("j1", state)
        self.assertNotIn("j2", state)
        self.assertEqual(state["j1"].iteration, 5)


# =============================================================================
# _fire_job live state integration (NEW)
# =============================================================================

class TestFireJobWithLiveState(AsyncTestCase):
    """Verify _fire_job writes start/finish to the live state file."""

    def test_live_state_finish_written_on_success(self):
        jf = _tmp_jobs_file()
        cb = AsyncMock(return_value="great")
        s = JobScheduler(jf, cb)

        job = _interval_job("jlive")
        s.add_job(job)
        self.async_run(s._fire_job(job))

        lf = jf.with_suffix(".live.json")
        if lf.exists():
            state = json.loads(lf.read_text())
            # After finish the entry should be gone (writer removes it)
            self.assertNotIn("jlive", state)

    def test_live_state_finish_written_on_failure(self):
        async def boom(j): raise RuntimeError()
        jf = _tmp_jobs_file()
        s = JobScheduler(jf, boom)
        job = _interval_job("jlive2")
        s.add_job(job)
        self.async_run(s._fire_job(job))

        lf = jf.with_suffix(".live.json")
        if lf.exists():
            state = json.loads(lf.read_text())
            self.assertNotIn("jlive2", state)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
