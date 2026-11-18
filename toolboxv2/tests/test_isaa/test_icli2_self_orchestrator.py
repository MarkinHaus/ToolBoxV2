# -*- coding: utf-8 -*-
"""T9: Orchestrierung - der self-Agent steuert andere Agenten und liest deren OBS."""
import asyncio
import json
from pathlib import Path

import pytest

from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend


def _backend(tmp: Path, agent: str = "self"):
    class _App:
        data_dir = str(tmp / "data")
        appdata = str(tmp / "appdata")

        class logger:
            @staticmethod
            def warning(*a, **k):
                pass

    class FakeEngine:
        def list_executions(self):
            return [{"run_id": "r1", "session_id": "s1", "query": "q...", "status": "running",
                     "iteration": 3, "tools_used": 2, "dynamic_tools": []}]

    class FakeAgent:
        name = "tb_admin"
        tools: list = []

        def add_tools(self, tools):                 # icli2 bindet beim Holen seine Tools (widget_tools)
            self.tools = [t["name"] for t in tools]

        def _get_execution_engine(self):
            return FakeEngine()

        async def cancel_execution(self, run_id):
            return True

        async def pause_execution(self, run_id):
            return {"run_id": run_id}

        async def resume_execution(self, run_id, max_iterations=30, content="", stream=False):
            return "resumed-final"

    class _Isaa:
        agent_data = {}

        async def get_agent(self, name):
            return FakeAgent()

    class _B(IsaaBackend):
        def __init__(self):
            self.app = _App()
            self.isaa = _Isaa()
            self.agent = agent
            self.session = "default"
            self.max_iterations = 5
            self._settings = {}

    return _B()


def _obs(tmp: Path, name: str = "tb_admin") -> Path:
    obs = Path(tmp / "data" / "Agents" / name / "obs")
    obs.mkdir(parents=True, exist_ok=True)
    (obs / "_index.json").write_text(json.dumps({"runs": ["b13b03d9"]}), encoding="utf-8")
    (obs / "run_b13b03d9.json").write_text(json.dumps({
        "run_id": "b13b03d9", "agent_name": name, "query": "test query",
        "success": True, "total_iterations": 7, "tool_call_count": 12,
    }), encoding="utf-8")
    (obs / "live_b13b03d9.jsonl").write_text(
        "\n".join(json.dumps({"i": i, "ev": f"tool_{i}"}) for i in range(10)), encoding="utf-8")
    return obs


def test_7_guard_rejects_non_self(tmp_path):
    b = _backend(tmp_path, agent="worker")
    assert asyncio.run(b.self_agent_stop("tb_admin")) is None
    assert asyncio.run(b.self_agent_status("tb_admin")) is None
    assert b.self_agent_obs("tb_admin") is None


def test_8_agent_name_validation(tmp_path):
    b = _backend(tmp_path)
    import asyncio as _aio
    for bad in ("../etc", "a b", "a/b", ""):
        with pytest.raises(ValueError):
            _aio.run(b.self_agent_status(bad))


def test_9_stop_uses_cancel(tmp_path):
    b = _backend(tmp_path)
    res = asyncio.run(b.self_agent_stop("tb_admin"))
    assert res["stopped"] is True


def test_10_status_lists_runs(tmp_path):
    b = _backend(tmp_path)
    res = asyncio.run(b.self_agent_status("tb_admin"))
    assert res["agent"] == "tb_admin"
    assert res["runs"][0]["run_id"] == "r1"
    assert res["runs"][0]["status"] == "running"


def test_11_obs_list_and_detail(tmp_path):
    b = _backend(tmp_path)
    _obs(tmp_path)
    res = b.self_agent_obs("tb_admin")
    assert res["runs"] == ["b13b03d9"]
    det = b.self_agent_obs("tb_admin", run_id="b13b03d9", tail=3)
    assert det["summary"]["run_id"] == "b13b03d9"
    assert len(det["actions"]) == 3
    assert det["actions"][-1]["ev"] == "tool_9"


def test_12_obs_unknown_agent_warns(tmp_path):
    b = _backend(tmp_path)
    res = b.self_agent_obs("ghost")
    assert res.get("level") == "warn"


def test_13_pause_and_resume(tmp_path):
    b = _backend(tmp_path)
    res = asyncio.run(b.self_agent_pause("tb_admin"))
    assert res["paused"] is True
    res2 = asyncio.run(b.self_agent_resume("tb_admin", "r1"))
    assert "resumed-final" in res2["text"] or res2.get("final") == "resumed-final"

def test_14_start_runs_background_task(tmp_path):
    import asyncio as _aio
    b = _backend(tmp_path)

    class StartAgent:
        name = "tb_admin"

        def add_tools(self, tools):                 # icli2 bindet beim Holen seine Tools (widget_tools)
            self.tools = [t["name"] for t in tools]

        async def a_stream(self, query, session_id=None, **kw):
            for i in range(3):
                yield {"type": "content", "chunk": f"c{i}"}

    class StartIsaa:
        agent_data = {}

        async def get_agent(self, name):
            return StartAgent()

    b.isaa = StartIsaa()

    async def _main():
        res = await b.self_agent_start("tb_admin", "mach was")
        assert "gestartet" in res["text"]
        sess = res["session"]
        await b._orch_tasks[sess]
        assert b._orch_results[sess]["ok"] is True
        assert b._orch_results[sess]["chunks"] == 3

    _aio.run(_main())
