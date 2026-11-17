"""
Regression: VFS-Tools muessen auf die VFS IHRER Session zugreifen.

init_session_tools laeuft einmal pro Session und registriert die VFS-Tools
(Closures ueber session.vfs) im ToolManager des Agenten. Der ToolManager ist
pro Agent - eine zweite Session desselben Agenten (Sub-Agent
"<sid>__sub__<id>", Job, Delegation ...) darf die Registrierung der ersten
nicht ueberschreiben.

Praxisfall (icli): Paste nach /userpaste/<ts>_paste.txt in Session "default",
danach Sub-Agent-Spawn -> vfs_view meldete "file not found".
"""

import asyncio
import types
import unittest
from unittest.mock import MagicMock

from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2

PASTE = "/userpaste/1794846315_paste.txt"


def _session(sid: str):
    s = MagicMock()
    s.session_id = sid
    s.agent_name = "self"
    s.vfs = VirtualFileSystemV2(sid, "self")
    s.tools_initialized = False
    s.get_docker_tools = MagicMock(return_value=([], {}))
    s.is_tool_allowed = MagicMock(return_value=True)
    return s


class _Sessions:
    def __init__(self, *sessions):
        self.sessions = {s.session_id: s for s in sessions}

    def get(self, sid):
        return self.sessions.get(sid)

    async def get_or_create(self, sid):
        return self.sessions[sid]


def _agent(*sessions):
    """MagicMock-Agent mit ECHTEM ToolManager und echtem add_tools/add_tool-Pfad."""
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
    from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolManager

    agent = MagicMock()
    agent.tool_manager = ToolManager()
    agent.amd.name = "self"
    agent.name = "self"
    agent.active_session = None
    agent.session_manager = _Sessions(*sessions)
    agent.add_tool = lambda **kw: FlowAgent.add_tool(agent, **kw)
    agent.add_tools = lambda tools: FlowAgent.add_tools(agent, tools)
    agent.init_session_tools = lambda s: FlowAgent.init_session_tools(agent, s)
    return agent


def _paste(session, text="zeile 1\nzeile 2"):
    # == icli._save_to_vfs_and_insert (icli.py:5103-5110)
    session.vfs.mkdir("/userpaste", parents=True)
    session.vfs.create(PASTE, text)


def _call(agent, name, **kw):
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent

    return asyncio.run(FlowAgent.arun_function(agent, name, **kw))


def _setup_context(agent, session_id):
    """ExecutionEngine._setup_context wie beim Hot-/Cold-Resume (ctx gegeben)."""
    from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

    async def _noop(*a, **k):
        return None

    agent.obs = None
    engine = types.SimpleNamespace(
        agent=agent, is_sub_agent=False, sub_agent_output_dir=None,
        _active_executions={}, _session_last_run={},
        _wait_for_pending_finalize=_noop, live=MagicMock(),
    )
    ctx = types.SimpleNamespace(run_id="r1", session_id=session_id, query="")
    return asyncio.run(ExecutionEngine._setup_context(engine, "q", session_id, 10, ctx))


class TestSecondSessionDoesNotOverrideFirst(unittest.TestCase):
    def setUp(self):
        self.main = _session("default")
        self.sub = _session("default__sub__ab12")
        self.agent = _agent(self.main, self.sub)
        self.agent.active_session = "default"
        self.agent.init_session_tools(self.main)  # a_stream("default")
        self.agent.init_session_tools(self.sub)   # sub_agent.py:483-484
        _paste(self.main)

    def test_vfs_view_reads_paste_of_own_session(self):
        res = _call(self.agent, "vfs_view", path=PASTE)
        self.assertTrue(res.get("success"), res)
        self.assertIn("zeile 1", res.get("content", ""))

    def test_vfs_shell_reads_paste_of_own_session(self):
        res = _call(self.agent, "vfs_shell", reason="t", command=f"cat {PASTE}")
        self.assertTrue(res.get("success"), res)
        self.assertIn("zeile 2", res.get("stdout", ""))

    def test_sub_session_keeps_its_own_vfs(self):
        self.sub.vfs.create("/sub_only.txt", "s")
        self.agent.active_session = "default__sub__ab12"
        self.assertTrue(_call(self.agent, "vfs_view", path="/sub_only.txt").get("success"))
        self.assertFalse(_call(self.agent, "vfs_view", path=PASTE).get("success"))

    def test_tool_kwarg_named_session_id_is_passed_through(self):
        async def echo(session_id: str = "") -> str:
            return session_id

        self.agent.add_tool(tool_func=echo, name="echo_sid", description="echo")
        self.assertEqual(_call(self.agent, "echo_sid", session_id="xyz"), "xyz")


class TestResumeUsesSessionOfRun(unittest.TestCase):
    """Resume umgeht a_run/a_stream; _setup_context muss die Session setzen."""

    def test_resume_after_other_session_run(self):
        main = _session("default")
        job = _session("job_42")
        agent = _agent(main, job)
        agent.active_session = "default"
        agent.init_session_tools(main)
        agent.active_session = "job_42"           # a_run(job.session_id) (icli.py:8748)
        agent.init_session_tools(job)
        _paste(main)
        _setup_context(agent, "default")          # icli-Direkt-Resume (icli.py:14524)
        self.assertEqual(agent.active_session, "default")
        res = _call(agent, "vfs_view", path=PASTE)
        self.assertTrue(res.get("success"), res)

    def test_replaced_session_object_gets_its_own_tools(self):
        # session_manager.from_checkpoint / close_session+get_or_create ersetzen das
        # Session-Objekt (neue VFS) bei gleicher session_id.
        old = _session("default")
        agent = _agent(old)
        agent.active_session = "default"
        agent.init_session_tools(old)
        new = _session("default")
        agent.session_manager.sessions["default"] = new
        _paste(new)
        _setup_context(agent, "default")
        res = _call(agent, "vfs_view", path=PASTE)
        self.assertTrue(res.get("success"), res)


class _Stop(Exception):
    pass


class TestExplicitDefaultSessionIsNotRemapped(unittest.TestCase):
    """icli uebergibt immer active_session_id (oft "default"). Nach einem Job
    (a_run(session_id="job_42") setzt agent.active_session) darf ein explizites
    "default" nicht in die Job-Session umgeleitet werden (flow_agent a_run/a_stream)."""

    def _agent(self):
        agent = MagicMock()
        agent.active_session = "job_42"
        seen = []

        async def get_or_create(sid):
            seen.append(sid)
            raise _Stop()

        agent.session_manager.get_or_create = get_or_create
        return agent, seen

    def test_a_run_explicit_default(self):
        from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent

        agent, seen = self._agent()
        with self.assertRaises(_Stop):
            asyncio.run(FlowAgent.a_run(agent, "q", session_id="default"))
        self.assertEqual(seen, ["default"])
        self.assertEqual(agent.active_session, "default")

    def test_a_stream_explicit_default(self):
        from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent

        agent, seen = self._agent()

        async def go():
            async for _ in FlowAgent.a_stream(agent, "q", session_id="default"):
                pass

        with self.assertRaises(_Stop):
            asyncio.run(go())
        self.assertEqual(seen, ["default"])

    def test_missing_session_id_follows_active_session(self):
        from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent

        agent, seen = self._agent()
        with self.assertRaises(_Stop):
            asyncio.run(FlowAgent.a_run(agent, "q"))
        self.assertEqual(seen, ["job_42"])


if __name__ == "__main__":
    unittest.main()
