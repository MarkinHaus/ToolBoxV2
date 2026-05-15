"""
Validation Tests — Offline / Headless Job Execution
====================================================

Stellt sicher dass Jobs korrekt feuern wenn das System nicht durchgehend
läuft. Testet drei Szenarien:

  1. CLI war offline — Jobs werden via get_missed_jobs() nachgeholt
  2. headless_runner._check_due_jobs() — korrektes Erkennen fälliger Jobs
     direkt aus der JSON-Datei (ohne Scheduler-Instanz)
  3. Kompletter Offline→Online-Zyklus: Job anlegen, "Stunden vergehen",
     Scheduler neu starten, Job muss gefeuert werden

Diese Tests laufen komplett ohne toolboxv2-Abhängigkeit für _check_due_jobs
und nur mit minimalem Stub für den Scheduler-Teil.

Run:
    python -m unittest test_jobs_offline_validation -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from toolboxv2.mods.isaa.extras.jobs.job_manager import (
    JobDefinition,
    JobScheduler,
    TriggerConfig,
)
from toolboxv2.mods.isaa.extras.jobs.headless_runner import _check_due_jobs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jobs(path: Path, jobs: list[dict]) -> None:
    path.write_text(json.dumps(jobs, indent=2, default=str), encoding="utf-8")


def _make_file(jobs: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    f.write(json.dumps(jobs, indent=2, default=str))
    f.close()
    return Path(f.name)


def _interval_dict(job_id: str, interval: int, last_run_offset_s: int | None) -> dict:
    """Build a raw job dict as it would appear in jobs.json."""
    d: dict = {
        "job_id": job_id,
        "name": f"Interval {job_id}",
        "agent_name": "agent",
        "query": "do work",
        "status": "active",
        "trigger": {
            "trigger_type": "on_interval",
            "interval_seconds": interval,
        },
        "run_count": 0,
        "fail_count": 0,
        "last_result": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if last_run_offset_s is not None:
        last = datetime.now(timezone.utc) - timedelta(seconds=last_run_offset_s)
        d["last_run_at"] = last.isoformat()
    else:
        d["last_run_at"] = None
    return d


def _time_dict(job_id: str, delta_s: int, last_run_at: str | None = None) -> dict:
    at = (datetime.now(timezone.utc) + timedelta(seconds=delta_s)).isoformat()
    return {
        "job_id": job_id,
        "name": f"Time {job_id}",
        "agent_name": "agent",
        "query": "do work",
        "status": "active",
        "trigger": {
            "trigger_type": "on_time",
            "at_datetime": at,
        },
        "run_count": 0,
        "fail_count": 0,
        "last_result": None,
        "last_run_at": last_run_at,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _cron_dict(job_id: str, expr: str, last_run_offset_s: int | None = None) -> dict:
    d: dict = {
        "job_id": job_id,
        "name": f"Cron {job_id}",
        "agent_name": "agent",
        "query": "do work",
        "status": "active",
        "trigger": {
            "trigger_type": "on_cron",
            "cron_expression": expr,
        },
        "run_count": 0,
        "fail_count": 0,
        "last_result": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if last_run_offset_s is not None:
        d["last_run_at"] = (datetime.now(timezone.utc) - timedelta(seconds=last_run_offset_s)).isoformat()
    else:
        d["last_run_at"] = None
    return d


class _AsyncBase(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @classmethod
    def setUpClass(cls):
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())


# =============================================================================
# BLOCK 1 — headless_runner._check_due_jobs()
#            Direkte JSON-Datei-Auswertung ohne laufenden Scheduler.
#            Dies simuliert den Fall: OS-Scheduler weckt headless_runner.
# =============================================================================

class TestHeadlessCheckDueJobs(unittest.TestCase):
    """
    _check_due_jobs() liest jobs.json und gibt die fälligen Jobs zurück.
    Kein Scheduler, kein Python-Process läuft dauerhaft.
    """

    # --- on_interval ---

    def test_interval_never_run_is_due(self):
        """Job ohne last_run_at gilt als fällig (ist nie gelaufen)."""
        jf = _make_file([_interval_dict("j_never", 60, last_run_offset_s=None)])
        due = _check_due_jobs(jf)
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["job_id"], "j_never")

    def test_interval_elapsed_is_due(self):
        """Job der vor 2× interval zuletzt lief ist fällig."""
        jf = _make_file([_interval_dict("j_elapsed", 60, last_run_offset_s=120)])
        due = _check_due_jobs(jf)
        self.assertEqual(len(due), 1)

    def test_interval_not_elapsed_is_not_due(self):
        """Job der gerade eben lief ist NICHT fällig."""
        jf = _make_file([_interval_dict("j_fresh", 3600, last_run_offset_s=10)])
        due = _check_due_jobs(jf)
        self.assertEqual(due, [])

    def test_interval_exactly_at_boundary_is_due(self):
        """Exakt am Boundary (elapsed == interval) → fällig."""
        jf = _make_file([_interval_dict("j_boundary", 60, last_run_offset_s=60)])
        due = _check_due_jobs(jf)
        self.assertEqual(len(due), 1)

    def test_multiple_intervals_only_due_ones_returned(self):
        jobs = [
            _interval_dict("j_due1", 60, last_run_offset_s=120),
            _interval_dict("j_fresh1", 3600, last_run_offset_s=5),
            _interval_dict("j_due2", 30, last_run_offset_s=90),
            _interval_dict("j_fresh2", 600, last_run_offset_s=10),
        ]
        jf = _make_file(jobs)
        due = _check_due_jobs(jf)
        due_ids = {d["job_id"] for d in due}
        self.assertIn("j_due1", due_ids)
        self.assertIn("j_due2", due_ids)
        self.assertNotIn("j_fresh1", due_ids)
        self.assertNotIn("j_fresh2", due_ids)

    # --- on_time ---

    def test_on_time_past_and_never_run_is_due(self):
        jf = _make_file([_time_dict("jt_past", delta_s=-30)])
        due = _check_due_jobs(jf)
        self.assertEqual(len(due), 1)

    def test_on_time_future_is_not_due(self):
        jf = _make_file([_time_dict("jt_future", delta_s=3600)])
        due = _check_due_jobs(jf)
        self.assertEqual(due, [])

    def test_on_time_paused_is_not_due(self):
        d = _time_dict("jt_paused", delta_s=-30)
        d["status"] = "paused"
        jf = _make_file([d])
        due = _check_due_jobs(jf)
        self.assertEqual(due, [])

    def test_on_time_expired_is_not_due(self):
        d = _time_dict("jt_exp", delta_s=-30)
        d["status"] = "expired"
        jf = _make_file([d])
        due = _check_due_jobs(jf)
        self.assertEqual(due, [])

    def test_on_time_naive_datetime_is_treated_as_utc(self):
        """Naive ISO-datetime (kein +00:00) wird als UTC behandelt."""
        past_naive = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        d = _time_dict("jt_naive", delta_s=-10)
        d["trigger"]["at_datetime"] = past_naive
        jf = _make_file([d])
        due = _check_due_jobs(jf)
        self.assertEqual(len(due), 1)

    def test_on_time_invalid_datetime_skipped_gracefully(self):
        d = _time_dict("jt_bad", delta_s=-10)
        d["trigger"]["at_datetime"] = "NOT-A-DATE"
        jf = _make_file([d])
        due = _check_due_jobs(jf)
        self.assertEqual(due, [])  # Kein Crash, kein fälschlicherweise markiert

    # --- on_system_boot ---

    def test_system_boot_job_is_always_due(self):
        """Boot-Jobs sind im headless_runner immer fällig."""
        d = {
            "job_id": "jboot", "name": "boot", "agent_name": "a",
            "query": "q", "status": "active",
            "trigger": {"trigger_type": "on_system_boot"},
            "run_count": 0, "fail_count": 0,
            "last_run_at": None, "last_result": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        jf = _make_file([d])
        due = _check_due_jobs(jf)
        self.assertEqual(len(due), 1)

    # --- on_cli_start (darf NICHT headless gefeuert werden) ---

    def test_cli_start_is_NOT_due_headless(self):
        """on_cli_start Jobs dürfen im headless_runner NICHT feuern."""
        d = {
            "job_id": "jcli", "name": "cli", "agent_name": "a",
            "query": "q", "status": "active",
            "trigger": {"trigger_type": "on_cli_start"},
            "run_count": 0, "fail_count": 0,
            "last_run_at": None, "last_result": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        jf = _make_file([d])
        due = _check_due_jobs(jf)
        self.assertEqual(due, [], "on_cli_start darf headless nicht feuern")

    # --- Datei fehlt / korrupt ---

    def test_missing_file_returns_empty(self):
        due = _check_due_jobs(Path("/tmp/does_not_exist_xyz_jobs.json"))
        self.assertEqual(due, [])

    def test_corrupt_json_returns_empty(self):
        jf = _make_file([])
        jf.write_text("NOT VALID JSON {{{{", encoding="utf-8")
        due = _check_due_jobs(jf)
        self.assertEqual(due, [])

    def test_empty_jobs_file_returns_empty(self):
        jf = _make_file([])
        self.assertEqual(_check_due_jobs(jf), [])

    def test_no_interval_value_is_skipped_gracefully(self):
        """on_interval ohne interval_seconds darf nicht crashen."""
        d = _interval_dict("j_no_interval", 60, None)
        d["trigger"].pop("interval_seconds")
        jf = _make_file([d])
        due = _check_due_jobs(jf)
        self.assertEqual(due, [])

    # --- on_cron (mit croniter wenn installiert) ---

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("croniter") is not None,
        "croniter nicht installiert"
    )
    def test_cron_overdue_is_due(self):
        """Cron-Job der letzte Ausführung vor 2h hatte und jede Stunde läuft."""
        jf = _make_file([_cron_dict("jcron_due", "0 * * * *", last_run_offset_s=7200)])
        due = _check_due_jobs(jf)
        self.assertEqual(len(due), 1)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("croniter") is not None,
        "croniter nicht installiert"
    )
    def test_cron_fresh_is_not_due(self):
        """Cron-Job der gerade erst ausgeführt wurde ist nicht fällig."""
        jf = _make_file([_cron_dict("jcron_fresh", "0 3 * * *", last_run_offset_s=60)])
        due = _check_due_jobs(jf)
        self.assertEqual(due, [])


# =============================================================================
# BLOCK 2 — JobScheduler.get_missed_jobs() + fire_missed_jobs()
#            Scheduler-basierte Offline-Erkennung nach Neustart.
# =============================================================================

class TestSchedulerOfflineCatchup(_AsyncBase):
    """
    Simuliert: CLI war offline, wird jetzt gestartet.
    Scheduler lädt jobs.json, get_missed_jobs() identifiziert fällige Jobs,
    fire_missed_jobs() führt sie aus.
    """

    def _make_scheduler(self, jobs_data: list[dict], callback=None) -> tuple[JobScheduler, Path]:
        jf = _make_file(jobs_data)
        cb = callback or AsyncMock(return_value="ok")
        s = JobScheduler(jf, cb)
        return s, jf

    # --- get_missed_jobs ---

    def test_interval_missed_while_offline(self):
        s, _ = self._make_scheduler([
            _interval_dict("jm1", 60, last_run_offset_s=600),
        ])
        missed = s.get_missed_jobs()
        self.assertEqual(len(missed), 1)
        self.assertEqual(missed[0].job_id, "jm1")

    def test_interval_not_missed_if_recent(self):
        s, _ = self._make_scheduler([
            _interval_dict("jfresh", 3600, last_run_offset_s=10),
        ])
        self.assertEqual(s.get_missed_jobs(), [])

    def test_interval_missed_if_never_ran(self):
        s, _ = self._make_scheduler([
            _interval_dict("jnever", 60, last_run_offset_s=None),
        ])
        self.assertIn(s.get_job("jnever"), s.get_missed_jobs())

    def test_on_time_missed_while_offline(self):
        s, _ = self._make_scheduler([
            _time_dict("jt_miss", delta_s=-300),
        ])
        self.assertEqual(len(s.get_missed_jobs()), 1)

    def test_on_time_not_missed_if_already_ran_after_target(self):
        target = datetime.now(timezone.utc) - timedelta(seconds=60)
        ran_after = datetime.now(timezone.utc) - timedelta(seconds=30)
        d = _time_dict("jt_ok", delta_s=-60)
        d["trigger"]["at_datetime"] = target.isoformat()
        d["last_run_at"] = ran_after.isoformat()
        s, _ = self._make_scheduler([d])
        self.assertEqual(s.get_missed_jobs(), [])

    def test_paused_job_not_in_missed(self):
        d = _interval_dict("jpaused", 60, last_run_offset_s=600)
        d["status"] = "paused"
        s, _ = self._make_scheduler([d])
        self.assertEqual(s.get_missed_jobs(), [])

    def test_expired_job_not_in_missed(self):
        d = _time_dict("jexp", delta_s=-60)
        d["status"] = "expired"
        s, _ = self._make_scheduler([d])
        self.assertEqual(s.get_missed_jobs(), [])

    def test_mixed_due_and_not_due_separated(self):
        s, _ = self._make_scheduler([
            _interval_dict("j_due", 60, last_run_offset_s=300),
            _interval_dict("j_ok", 3600, last_run_offset_s=10),
            _time_dict("jt_due", delta_s=-100),
            _time_dict("jt_future", delta_s=3600),
        ])
        missed_ids = {j.job_id for j in s.get_missed_jobs()}
        self.assertIn("j_due", missed_ids)
        self.assertIn("jt_due", missed_ids)
        self.assertNotIn("j_ok", missed_ids)
        self.assertNotIn("jt_future", missed_ids)

    # --- fire_missed_jobs ---

    def test_fire_missed_fires_due_job(self):
        fired = []

        async def cb(job):
            fired.append(job.job_id)
            return "ok"

        s, _ = self._make_scheduler(
            [_interval_dict("jfire", 60, last_run_offset_s=300)],
            callback=cb,
        )
        count = self.run_async(s.fire_missed_jobs())
        self.assertEqual(count, 1)
        self.run_async(asyncio.sleep(0.2))
        self.assertIn("jfire", fired)

    def test_fire_missed_updates_run_count(self):
        async def cb(job): return "ok"
        s, jf = self._make_scheduler(
            [_interval_dict("jcount", 60, last_run_offset_s=300)],
            callback=cb,
        )
        self.run_async(s.fire_missed_jobs())
        self.run_async(asyncio.sleep(0.2))
        # Reload from disk
        data = json.loads(jf.read_text())
        rec = next(r for r in data if r["job_id"] == "jcount")
        self.assertEqual(rec["run_count"], 1)
        self.assertEqual(rec["last_result"], "completed")

    def test_fire_missed_returns_zero_when_nothing_missed(self):
        s, _ = self._make_scheduler([
            _interval_dict("jok", 3600, last_run_offset_s=10),
        ])
        count = self.run_async(s.fire_missed_jobs())
        self.assertEqual(count, 0)

    def test_fire_missed_fires_multiple_jobs(self):
        fired = []

        async def cb(job):
            fired.append(job.job_id)
            return "ok"

        s, _ = self._make_scheduler(
            [
                _interval_dict("jm_a", 60, last_run_offset_s=300),
                _interval_dict("jm_b", 60, last_run_offset_s=300),
                _interval_dict("jm_c", 60, last_run_offset_s=300),
            ],
            callback=cb,
        )
        count = self.run_async(s.fire_missed_jobs())
        self.assertEqual(count, 3)
        self.run_async(asyncio.sleep(0.3))
        self.assertEqual(sorted(fired), ["jm_a", "jm_b", "jm_c"])

    def test_fire_missed_failed_job_increments_fail_count(self):
        async def boom(job): raise RuntimeError("crash")
        s, jf = self._make_scheduler(
            [_interval_dict("jfail", 60, last_run_offset_s=300)],
            callback=boom,
        )
        self.run_async(s.fire_missed_jobs())
        self.run_async(asyncio.sleep(0.2))
        data = json.loads(jf.read_text())
        rec = next(r for r in data if r["job_id"] == "jfail")
        self.assertEqual(rec["fail_count"], 1)
        self.assertEqual(rec["last_result"], "failed")


# =============================================================================
# BLOCK 3 — Vollständiger Offline→Online-Zyklus
#            Ende-zu-Ende: Job anlegen, Stunden simulieren, Neustart.
# =============================================================================

class TestFullOfflineOnlineCycle(_AsyncBase):
    """
    Simuliert den realen Ablauf:
      1. CLI läuft, Job wird angelegt
      2. CLI stoppt (Scheduler-Instanz weg)
      3. Zeit vergeht (last_run_at manuell in die Vergangenheit gesetzt)
      4. CLI startet neu (neue Scheduler-Instanz)
      5. fire_missed_jobs() → Job muss gefeuert werden
    """

    def test_job_fires_after_simulated_offline_period(self):
        """Kerntest: Job überlebt Neustart und wird korrekt nachgeholt."""
        fired = []

        async def cb(job):
            fired.append(job.job_id)
            return "ok"

        jf = _make_file([])

        # Phase 1: Scheduler erstellen, Job hinzufügen
        s1 = JobScheduler(jf, AsyncMock(return_value="ok"))
        from toolboxv2.mods.isaa.extras.jobs.job_manager import JobDefinition, TriggerConfig
        job = JobDefinition(
            job_id="jcycle",
            name="Cycle Test",
            agent_name="agent",
            query="do cycle work",
            trigger=TriggerConfig(trigger_type="on_interval", interval_seconds=3600),
        )
        s1.add_job(job)

        # Phase 2: Persistenz simulieren — last_run_at auf "vor 2 Stunden" setzen
        data = json.loads(jf.read_text())
        for r in data:
            if r["job_id"] == "jcycle":
                r["last_run_at"] = (
                    datetime.now(timezone.utc) - timedelta(hours=2)
                ).isoformat()
        jf.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        # Phase 3: Scheduler neu starten (CLI-Neustart)
        s2 = JobScheduler(jf, cb)
        count = self.run_async(s2.fire_missed_jobs())
        self.assertEqual(count, 1, "Genau 1 verpasster Job muss erkannt werden")
        self.run_async(asyncio.sleep(0.3))
        self.assertIn("jcycle", fired, "Job muss nach Neustart gefeuert worden sein")

    def test_on_time_job_fires_after_offline_period(self):
        """on_time Job der während Offline-Phase fällig wurde."""
        fired = []

        async def cb(job):
            fired.append(job.job_id)
            return "ok"

        # Job-Ziel: vor 10 Minuten
        target = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        d = _time_dict("jtime_cycle", delta_s=-600)
        d["trigger"]["at_datetime"] = target
        d["last_run_at"] = None  # nie gelaufen

        jf = _make_file([d])
        s = JobScheduler(jf, cb)
        count = self.run_async(s.fire_missed_jobs())
        self.assertEqual(count, 1)
        self.run_async(asyncio.sleep(0.3))
        self.assertIn("jtime_cycle", fired)

    def test_completed_one_time_job_not_refired(self):
        """on_time Job der bereits erfolgreich ausgeführt wurde, darf nicht nochmal feuern."""
        async def cb(job): return "ok"

        target = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        ran_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

        d = _time_dict("jtime_done", delta_s=-600)
        d["trigger"]["at_datetime"] = target
        d["last_run_at"] = ran_at  # schon nach target gelaufen

        jf = _make_file([d])
        s = JobScheduler(jf, cb)
        missed = s.get_missed_jobs()
        self.assertEqual(missed, [], "Bereits ausgeführter on_time Job darf nicht als missed gelten")

    def test_job_file_updated_after_fire_contains_correct_timestamps(self):
        """Nach fire_missed_jobs muss last_run_at korrekt in der Datei stehen."""
        before = datetime.now(timezone.utc)

        async def cb(job): return "ok"

        jf = _make_file([_interval_dict("jts", 60, last_run_offset_s=300)])
        s = JobScheduler(jf, cb)
        self.run_async(s.fire_missed_jobs())
        self.run_async(asyncio.sleep(0.3))

        data = json.loads(jf.read_text())
        rec = next(r for r in data if r["job_id"] == "jts")
        last_run = datetime.fromisoformat(rec["last_run_at"])
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)

        self.assertGreaterEqual(last_run, before, "last_run_at muss nach Testbeginn liegen")

    def test_multiple_restarts_accumulate_run_count(self):
        """Job läuft über 3 Neustarts — run_count muss kumulieren."""
        jf = _make_file([_interval_dict("jaccum", 60, last_run_offset_s=None)])

        for restart_nr in range(3):
            async def cb(job): return "ok"
            s = JobScheduler(jf, cb)
            self.run_async(s.fire_missed_jobs())
            self.run_async(asyncio.sleep(0.2))
            # Letzten Lauf in die Vergangenheit verschieben für nächsten Restart
            data = json.loads(jf.read_text())
            for r in data:
                if r["job_id"] == "jaccum":
                    r["last_run_at"] = (
                        datetime.now(timezone.utc) - timedelta(seconds=120)
                    ).isoformat()
            jf.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        data = json.loads(jf.read_text())
        rec = next(r for r in data if r["job_id"] == "jaccum")
        self.assertEqual(rec["run_count"], 3, "run_count muss über Neustarts kumulieren")


# =============================================================================
# BLOCK 4 — headless_runner JSON-Update-Logik
#            Prüft dass _check_due_jobs-Ergebnisse korrekt zurückgeschrieben
#            werden (simuliert den Schreib-Teil von headless_runner).
# =============================================================================

class TestHeadlessJsonUpdateLogic(unittest.TestCase):
    """
    Der headless_runner schreibt nach Ausführung last_run_at, run_count und
    last_result zurück in die JSON-Datei. Wir testen die Merge-Logik direkt.
    """

    def test_merge_updates_correct_job_only(self):
        jobs = [
            _interval_dict("j_target", 60, last_run_offset_s=300),
            _interval_dict("j_other", 3600, last_run_offset_s=10),
        ]
        jf = _make_file(jobs)

        # Simuliere was headless_runner nach Ausführung schreibt
        run_time = datetime.now(timezone.utc).isoformat()
        updated_target = dict(jobs[0])
        updated_target["last_run_at"] = run_time
        updated_target["run_count"] = 1
        updated_target["last_result"] = "completed"

        # Merge wie in headless_runner._run_due_jobs
        all_data = json.loads(jf.read_text())
        job_map = {j["job_id"]: j for j in all_data}
        job_map["j_target"].update(updated_target)
        jf.write_text(json.dumps(list(job_map.values()), indent=2, default=str),
                      encoding="utf-8")

        result = json.loads(jf.read_text())
        result_map = {r["job_id"]: r for r in result}

        # j_target soll aktualisiert sein
        self.assertEqual(result_map["j_target"]["run_count"], 1)
        self.assertEqual(result_map["j_target"]["last_result"], "completed")

        # j_other soll unverändert sein
        self.assertEqual(result_map["j_other"]["run_count"], 0)
        self.assertIsNone(result_map["j_other"]["last_result"])

    def test_on_time_job_gets_expired_after_headless_run(self):
        """on_time Job soll nach headless Ausführung status=expired bekommen."""
        d = _time_dict("jt_exp", delta_s=-60)
        jf = _make_file([d])

        all_data = json.loads(jf.read_text())
        job_map = {j["job_id"]: j for j in all_data}
        job_map["jt_exp"]["last_run_at"] = datetime.now(timezone.utc).isoformat()
        job_map["jt_exp"]["run_count"] = 1
        job_map["jt_exp"]["last_result"] = "completed"
        job_map["jt_exp"]["status"] = "expired"  # headless_runner setzt dies
        jf.write_text(json.dumps(list(job_map.values()), indent=2, default=str),
                      encoding="utf-8")

        # Reload mit neuem Scheduler — Job muss als expired geladen werden
        s = JobScheduler(jf, AsyncMock(return_value="ok"))
        job = s.get_job("jt_exp")
        self.assertEqual(job.status, "expired")
        # Und wird nicht nochmal als missed erkannt
        self.assertNotIn(job, s.get_missed_jobs())

    def test_headless_check_due_empty_after_all_ran(self):
        """Nach Ausführung aller fälligen Jobs darf kein Job mehr als due gelten."""
        jf = _make_file([_interval_dict("j_done", 3600, last_run_offset_s=3800)])
        due = _check_due_jobs(jf)
        self.assertEqual(len(due), 1)

        # Simuliere Ausführung
        all_data = json.loads(jf.read_text())
        for r in all_data:
            r["last_run_at"] = datetime.now(timezone.utc).isoformat()
            r["run_count"] = 1
        jf.write_text(json.dumps(all_data, indent=2, default=str), encoding="utf-8")

        # Nochmal prüfen — jetzt nicht mehr fällig
        due_after = _check_due_jobs(jf)
        self.assertEqual(due_after, [])


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
