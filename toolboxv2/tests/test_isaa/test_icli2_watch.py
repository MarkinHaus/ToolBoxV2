# -*- coding: utf-8 -*-
"""Hintergrund-Befehle melden ihr Ende, Jobs bekommen eigene Sessions, /runs findet Haenger."""
import asyncio
from types import SimpleNamespace

from toolboxv2.mods.isaa.extras.icli2.watch import WatchMixin, bg_message

EVENT = {"agent": "self", "session": "default", "scope": "", "session_id": "B", "command": "pytest -q",
         "exit_code": 0, "success": True, "seconds": 12.0, "tail": "5 passed"}


class Engine:
    def __init__(self, ctx=None):
        self.ctx = ctx
        self._active_executions = {"r1": ctx} if ctx else {}
        self._session_last_run = {"default": "r1"} if ctx else {}

    def list_executions(self):
        return [{"run_id": "r1", "session_id": "default", "status": self.ctx.status, "iteration": 4,
                 "query": "tests"}] if self.ctx else []

    def get_execution(self, rid):
        return self._active_executions.get(rid)


class Agent:
    def __init__(self, engine):
        self.engine, self.cancelled = engine, []

    def _get_execution_engine(self):
        return self.engine

    async def cancel_execution(self, rid):
        self.cancelled.append(rid)
        return True


class B(WatchMixin):
    def __init__(self, ctx=None, live=()):
        self.isaa = SimpleNamespace(config={"agent-instance-self": Agent(Engine(ctx))})
        self._tasks, self.spawned, self.notes = set(), [], []
        self.attach_engine(lambda: list(live), lambda text, level="info": self.notes.append((level, text)))

        async def spawn(agent, session, query, kind="job", stream=None):
            self.spawned.append((agent, session, kind, query))
            return "ok"
        self._spawn = spawn


def ctx(status="running"):
    return SimpleNamespace(status=status, working_history=[{"role": "user", "content": "los"}], session_id="default")


def test_job_session_never_the_chat_session():
    b = B()
    assert b.job_session(SimpleNamespace(session_id="default", name="Morgen Routine!", job_id="j1")) == "job-Morgen-Routine"
    assert b.job_session(SimpleNamespace(session_id="", name="", job_id="j9")) == "job-j9"
    assert b.job_session(SimpleNamespace(session_id="eigene", name="x", job_id="j2")) == "eigene"


def test_shell_scope_only_for_job_sessions():
    from toolboxv2.mods.isaa.module import SHELL_CONTEXT
    b = B()
    b.job_session(SimpleNamespace(session_id="", name="nightly", job_id="j1"))

    async def run(session):
        b.shell_enter("self", session)
        return SHELL_CONTEXT.get()
    assert asyncio.run(run("job-nightly"))["scope"] == "job-nightly"
    assert asyncio.run(run("default")) == {"agent": "self", "session": "default", "scope": ""}


def test_finished_run_gets_a_followup():
    b = B(ctx("completed"), live=[])
    asyncio.run(b.bg_done(EVENT))
    assert b.spawned and b.spawned[0][:3] == ("self", "default", "shell")
    assert "exit_code=0" in b.spawned[0][3] and "5 passed" in b.spawned[0][3]
    assert b.notes and "Hintergrund-Befehl in B fertig" in b.notes[0][1]


def test_stale_running_ctx_without_live_run_gets_a_followup():
    c = ctx("running")                                  # haengt: kein Lauf in der icli2
    b = B(c, live=[])
    asyncio.run(b.bg_done(EVENT))
    assert b.spawned and len(c.working_history) == 1


def test_running_run_reads_it_no_followup():
    c = ctx("running")
    b = B(c, live=[("self", "default", "x1", "chat")])

    async def scenario():
        task = asyncio.ensure_future(b.bg_done(EVENT))
        await asyncio.sleep(0.05)
        assert c.working_history[-1]["content"] == bg_message(EVENT)
        c.working_history.append({"role": "assistant", "content": "Tests sind gruen"})
        c.status = "completed"
        await asyncio.wait_for(task, 3)
    asyncio.run(scenario())
    assert not b.spawned


def test_run_ended_before_reading_gets_a_followup():
    c = ctx("running")
    b = B(c, live=[("self", "default", "x1", "chat")])

    async def scenario():
        task = asyncio.ensure_future(b.bg_done(EVENT))
        await asyncio.sleep(0.05)
        c.status = "completed"                          # endete, ohne die Meldung zu lesen
        await asyncio.wait_for(task, 3)
    asyncio.run(scenario())
    assert b.spawned and b.spawned[0][2] == "shell"


def test_runs_marks_stuck_and_reset_stop():
    c = ctx("running")
    b = B(c, live=[])
    title, rows = asyncio.run(b._rows_runs())
    assert rows[0]["value"] == "self|r1" and "hängt" in rows[0]["label"]
    asyncio.run(b._do_run("self|r1", "reset"))
    assert c.status == "paused"
    asyncio.run(b._do_run("self|r1", "stop"))
    assert b.isaa.config["agent-instance-self"].cancelled == ["r1"]
    b2 = B(ctx("running"), live=[("self", "default", "x", "chat")])
    assert "läuft" in asyncio.run(b2._rows_runs())[1][0]["label"]
