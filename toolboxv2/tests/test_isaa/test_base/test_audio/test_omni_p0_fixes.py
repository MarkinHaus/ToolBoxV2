"""
Runnable checks for the P0 omni/icli fixes (T1, T2, T9, T8).

unittest only. No pytest, no frameworks. The Omni tested-core imports nothing
from toolboxv2, so we load omni.py in isolation by file path and drive it with
fakes — same contract the existing StubOmniBackend tests rely on.

Run:
    python -m unittest toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py
or from repo root:
    python toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import sys
import unittest
from pathlib import Path


# --- load omni.py in isolation (tested core; no toolboxv2 package import) -----
def _load_omni():
    here = Path(__file__).resolve()
    # repo_root/toolboxv2/mods/isaa/base/audio_io/omni.py
    root = here
    for _ in range(20):
        cand = root / "toolboxv2" / "mods" / "isaa" / "base" / "audio_io" / "omni.py"
        if cand.exists():
            break
        root = root.parent
    else:  # pragma: no cover
        raise FileNotFoundError("omni.py not found walking up from test file")
    spec = importlib.util.spec_from_file_location("omni_iso_p0", cand)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["omni_iso_p0"] = mod  # dataclasses/enums read __module__ at exec
    spec.loader.exec_module(mod)
    return mod


omni = _load_omni()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# --- T1: freeze fix — compress happens BEFORE the event task is cancelled -----
class TestT1RestartOrder(unittest.TestCase):
    def test_compress_runs_before_cancel_and_stop(self):
        src = inspect.getsource(omni.OmniSession._do_restart)
        i_comp = src.find("_compress(merge_old=True)")
        i_cancel = src.find("_event_task.cancel()")
        i_stop = src.find("self.backend.stop()")
        self.assertGreaterEqual(i_comp, 0, "compress call missing in _do_restart")
        self.assertGreaterEqual(i_cancel, 0, "event cancel missing in _do_restart")
        self.assertGreaterEqual(i_stop, 0, "backend.stop missing in _do_restart")
        # the whole point of T1: summarize while the old backend is still live
        self.assertLess(i_comp, i_cancel, "compress must run BEFORE event_task.cancel")
        self.assertLess(i_comp, i_stop, "compress must run BEFORE backend.stop")


# --- T2: text serialization — queue while speaking, flush once on turn end ----
class _FakeBackend:
    def __init__(self):
        self.sent: list[str] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(text)


def _bare_session(backend):
    """OmniSession.__init__ is heavy; build a bare object and set just the
    attributes the T2 methods touch. Mirrors how the unit-tested surface is
    driven without real recorders/players."""
    s = omni.OmniSession.__new__(omni.OmniSession)
    s.backend = backend
    s._agent_speaking = False
    s._pending_text = []
    return s


class TestT2TextSerialize(unittest.TestCase):
    def test_passes_through_when_idle(self):
        b = _FakeBackend()
        s = _bare_session(b)
        run(s.send_user_text("hello"))
        self.assertEqual(b.sent, ["hello"])
        self.assertEqual(s._pending_text, [])

    def test_queues_while_speaking_then_flushes_once(self):
        b = _FakeBackend()
        s = _bare_session(b)
        s._agent_speaking = True
        run(s.send_user_text("first"))
        run(s.send_user_text("second"))
        # nothing sent mid-speech
        self.assertEqual(b.sent, [])
        self.assertEqual(s._pending_text, ["first", "second"])
        # turn ends -> exactly one combined message, queue cleared
        run(s._flush_pending_text())
        self.assertEqual(b.sent, ["first\nsecond"])
        self.assertEqual(s._pending_text, [])

    def test_flush_noop_when_empty(self):
        b = _FakeBackend()
        s = _bare_session(b)
        run(s._flush_pending_text())
        self.assertEqual(b.sent, [])


# --- T9: agent_status reports each finished delegation exactly once -----------
class _FakeJobs:
    def __init__(self):
        self._state: list[dict] = []

    def set_state(self, state):
        self._state = state

    def live_state(self):
        return list(self._state)


def _status_tool(jobs):
    tools = omni.make_agent_tools(jobs, agent_provider=lambda name: object())
    for t in tools:
        if t["name"] == "agent_status":
            return t["tool_func"]
    raise AssertionError("agent_status tool not built")


class TestT9StatusReportsOnce(unittest.TestCase):
    def test_each_finished_job_new_once(self):
        jobs = _FakeJobs()
        status = _status_tool(jobs)

        # task1 running, task2 running
        jobs.set_state([
            {"job_id": "j1", "kind": "agent", "label": "t1", "status": "running"},
            {"job_id": "j2", "kind": "agent", "label": "t2", "status": "running"},
        ])
        out = json.loads(run(status()))
        self.assertTrue(all(j["status"] == "running" for j in out))

        # task1 finishes
        jobs.set_state([
            {"job_id": "j1", "kind": "agent", "label": "t1", "status": "finished"},
            {"job_id": "j2", "kind": "agent", "label": "t2", "status": "running"},
        ])
        out = json.loads(run(status()))
        st = {j["job_id"]: j["status"] for j in out}
        self.assertIn("(new)", st["j1"])
        self.assertEqual(st["j2"], "running")

        # task2 finishes -> j1 must NOT be re-announced as new, only j2 is new
        jobs.set_state([
            {"job_id": "j1", "kind": "agent", "label": "t1", "status": "finished"},
            {"job_id": "j2", "kind": "agent", "label": "t2", "status": "finished"},
        ])
        out = json.loads(run(status()))
        st = {j["job_id"]: j["status"] for j in out}
        self.assertIn("already reported", st["j1"])
        self.assertIn("(new)", st["j2"])


# --- T8: chat-view tools expose running executions + history (read-only) ------
class TestT8ChatViewTools(unittest.TestCase):
    def _tools(self, execs, hist):
        tools = omni.make_chat_view_tools(
            executions_provider=lambda: execs,
            history_provider=lambda n: hist[-n:],
        )
        return {t["name"]: t["tool_func"] for t in tools}

    def test_chat_executions_returns_running(self):
        execs = [{"run_id": "r1", "query": "do x", "status": "running"}]
        funcs = self._tools(execs, [])
        out = json.loads(run(funcs["chat_executions"]()))
        self.assertEqual(out[0]["run_id"], "r1")

    def test_chat_history_returns_last_n(self):
        hist = [{"role": "user", "content": str(i)} for i in range(20)]
        funcs = self._tools([], hist)
        out = json.loads(run(funcs["chat_history"](5)))
        self.assertEqual(len(out), 5)
        self.assertEqual(out[-1]["content"], "19")

    def test_provider_errors_are_caught(self):
        def boom():
            raise RuntimeError("provider down")
        tools = omni.make_chat_view_tools(
            executions_provider=boom, history_provider=lambda n: [],
        )
        func = {t["name"]: t["tool_func"] for t in tools}["chat_executions"]
        out = run(func())
        self.assertTrue(out.startswith("Error:"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
