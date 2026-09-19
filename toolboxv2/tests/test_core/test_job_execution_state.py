# tests/test_core/test_job_execution_state.py
# Hypothesen-Tests: iCLI-Job-State-Bug ("Agent laeuft nach manuellem Job noch").
#
# H1 (Stick-Fall): Wenn _create_execution zwischen create_task und der
#    _tid_holder-Zuweisung wirft (z.B. ZenPlus-Registrierungsfehler), laeuft
#    _run_job als GEIST-TASK weiter: kein Execution-Eintrag, kein
#    _on_agent_task_done-Callback (Job-Pfad registriert nur das lokale _on_done),
#    Reparatur via _tid_holder[0]=None greift nicht.
# H2 (Mainline): Normalfall - Job laeuft durch, exc.status wird "completed".
# H3 (Fail-Fall): a_run wirft -> exc.status muss "failed" werden.
#
# Jeder Test diskriminiert: H2/H3 gruen + H1 rot beweist, dass der Stick-Fall
# NUR im Fehlerpfad von _create_execution entstehen kann -> Fix dort ansetzen.

import asyncio
import types
import unittest

from toolboxv2.flows.isaa.icli import ISAA_Host


class _FakeAgent:
    def __init__(self, behavior="ok", delay=0.0):
        self.behavior = behavior
        self.delay = delay
        self.calls = []

    async def a_run(self, query, **kwargs):
        self.calls.append(query)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.behavior == "raise":
            raise RuntimeError("boom")
        return f"done:{query}"


class _FakeIsaaTools:
    def __init__(self, agent):
        self._agent = agent

    async def get_agent(self, name):
        return self._agent


def _mk_host(agent):
    host = ISAA_Host(None)
    host.isaa_tools = _FakeIsaaTools(agent)
    host.job_scheduler = None  # _live-Branch ist None-sicher (host_ref.job_scheduler and ..._live)
    host.max_iteration = 5
    host.all_executions = {}
    host._task_views = {}
    host._task_counter = 0
    host._focused_task_id = None
    return host


def _mk_job(jid="j1"):
    return types.SimpleNamespace(
        job_id=jid, name="demo", agent_name="self", query="q",
        session_id="default", timeout_seconds=10,
    )


class JobExecutionStateTests(unittest.IsolatedAsyncioTestCase):

    async def test_h2_manual_job_marks_completed(self):
        """Mainline: nach manuellem Job-Lauf darf KEIN Task auf 'running' haengen."""
        agent = _FakeAgent("ok")
        host = _mk_host(agent)
        res = await host._fire_job_from_scheduler(_mk_job())
        self.assertTrue(str(res).startswith("done:"))
        statuses = [e.status for e in host.all_executions.values()]
        self.assertEqual(statuses, ["completed"])

    async def test_h3_manual_job_failure_marks_failed(self):
        agent = _FakeAgent("raise")
        host = _mk_host(agent)
        await host._fire_job_from_scheduler(_mk_job("j2"))
        statuses = [e.status for e in host.all_executions.values()]
        self.assertEqual(statuses, ["failed"])

    async def test_h1_create_execution_crash_leaves_ghost_run(self):
        """H1: _create_execution wirft -> _fire_job bricht nach oben ab, aber
        _run_job wurde schon via create_task gestartet und laeuft als Geist:
        Agent fuehrt aus, ALL_EXECUTIONS bleibt leer (kein Tracking, kein
        Done-Callback am Host) - exakt der 'State-Fehler'-Nährboden."""
        agent = _FakeAgent("ok", delay=0.05)
        host = _mk_host(agent)

        def broken_create_execution(*a, **k):
            raise RuntimeError("zen registry down")

        host._create_execution = broken_create_execution
        with self.assertRaises(RuntimeError):
            await host._fire_job_from_scheduler(_mk_job("j3"))
        await asyncio.sleep(0.2)  # Geist-Task laufen lassen
        self.assertEqual(len(agent.calls), 1, "Job lief als Geist-Task")
        self.assertEqual(host.all_executions, {}, "kein Tracking vorhanden")


if __name__ == "__main__":
    unittest.main()
