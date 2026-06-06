"""unittest suite for sub-agent-aware ObservabilityLayer (contextvar routing)."""
import asyncio
import json
import os
import sys
import types
import tempfile
import unittest

import toolboxv2.mods.isaa.base.Agent.observability as obs_mod
from toolboxv2.mods.isaa.base.Agent.observability import ObservabilityLayer, _active_run_id
from toolboxv2.mods.isaa.extras import obs_viewer


def _new_obs():
    d = tempfile.mkdtemp()
    return ObservabilityLayer(agent_name="tester", obs_dir=d, max_runs=10)


class SlotRoutingTest(unittest.TestCase):
    def test_single_run_records_into_its_slot(self):
        obs = _new_obs()
        obs.begin_run("r1", "do a thing")
        obs.begin_step(1)
        obs.record_tool_start("vfs_read", "x", call_id="c1")
        obs.record_tool_end("vfs_read", "ok", call_id="c1")
        obs.end_step()
        obs.end_run(success=True, final_answer="done")
        run = obs.get_run("r1")
        self.assertIsNotNone(run)
        self.assertEqual(run.tool_call_count, 1)
        self.assertEqual(run.steps[0].tool_calls[0].name, "vfs_read")
        # slot cleaned up
        self.assertNotIn("r1", obs._slots)
        self.assertIsNone(_active_run_id.get())

    def test_record_without_active_run_is_safe(self):
        obs = _new_obs()
        # no begin_run → no active context; must not raise
        self.assertEqual(obs.record_tool_start("x"), "")
        obs.record_tool_end("x")
        obs.begin_step(1)   # no-op
        obs.record_compression({"kept": 1})
        # nothing persisted, no crash
        self.assertEqual(obs._slots, {})


class ParallelSubAgentTest(unittest.IsolatedAsyncioTestCase):
    async def _sub(self, obs, parent_id, sub_id, tools):
        obs.begin_run(sub_id, f"task {sub_id}", session_id=f"s__sub__{sub_id}",
                      parent_run_id=parent_id)
        obs.begin_step(1)
        for i, t in enumerate(tools):
            await asyncio.sleep(0.001)  # force interleave
            cid = f"{sub_id}_c{i}"
            obs.record_tool_start(t, "", call_id=cid)
            await asyncio.sleep(0.001)
            obs.record_tool_end(t, "ok", call_id=cid)
        obs.end_step()
        # contextvar still ours after awaits
        self.assertEqual(_active_run_id.get(), sub_id)
        obs.end_run(success=True, final_answer=f"{sub_id} done")

    async def test_three_parallel_subagents_no_crosstalk(self):
        obs = _new_obs()
        obs.begin_run("parent", "spawn 3", session_id="s")
        obs.begin_step(1)
        obs.record_tool_start("spawn_sub_agent", call_id="p0")
        # parent now waits on 3 children running in parallel tasks
        await asyncio.gather(
            asyncio.create_task(self._sub(obs, "parent", "subA", ["vfs_read", "vfs_write"])),
            asyncio.create_task(self._sub(obs, "parent", "subB", ["think"])),
            asyncio.create_task(self._sub(obs, "parent", "subC", ["vfs_list", "vfs_read", "think"])),
        )
        # parent's context untouched by children
        self.assertEqual(_active_run_id.get(), "parent")
        obs.record_tool_end("spawn_sub_agent", "ok", call_id="p0")
        obs.end_step()
        obs.end_run(success=True, final_answer="all done")

        # each child's run persisted with ONLY its own tools
        a, b, c = obs.get_run("subA"), obs.get_run("subB"), obs.get_run("subC")
        self.assertEqual(a.tool_call_count, 2)
        self.assertEqual(b.tool_call_count, 1)
        self.assertEqual(c.tool_call_count, 3)
        # parent has exactly its own one tool call (spawn), no child tools
        p = obs.get_run("parent")
        self.assertEqual(p.tool_call_count, 1)
        self.assertEqual(p.steps[0].tool_calls[0].name, "spawn_sub_agent")

    async def test_lineage_captured_live_and_finalized(self):
        obs = _new_obs()
        obs.begin_run("parent", "spawn", session_id="s")
        # spawn child; lineage must appear in parent slot IMMEDIATELY
        obs.begin_run("kid", "subtask", session_id="s__sub__kid", parent_run_id="parent")
        parent_slot = obs._slots["parent"]
        self.assertEqual(len(parent_slot.run.sub_agent_runs), 1)
        self.assertEqual(parent_slot.run.sub_agent_runs[0]["run_id"], "kid")
        self.assertEqual(parent_slot.run.sub_agent_runs[0]["status"], "running")
        # finish child → parent entry flips to completed (live, no disk reload)
        _active_run_id.set("kid")
        obs.end_run(success=True, final_answer="kid done")
        self.assertEqual(parent_slot.run.sub_agent_runs[0]["status"], "completed")
        # parent run persists lineage to disk
        _active_run_id.set("parent")
        obs.end_run(success=True, final_answer="parent done")
        p = obs.get_run("parent")
        self.assertEqual(p.parent_run_id, "")
        self.assertEqual(len(p.sub_agent_runs), 1)
        self.assertEqual(p.sub_agent_runs[0]["status"], "completed")
        kid = obs.get_run("kid")
        self.assertEqual(kid.parent_run_id, "parent")

    async def test_failed_child_marks_failed_in_parent(self):
        obs = _new_obs()
        obs.begin_run("parent", "spawn", session_id="s")
        obs.begin_run("kid", "subtask", session_id="s__sub__kid", parent_run_id="parent")
        parent_slot = obs._slots["parent"]
        _active_run_id.set("kid")
        obs.end_run(success=False, final_answer="kid failed")
        self.assertEqual(parent_slot.run.sub_agent_runs[0]["status"], "failed")
        _active_run_id.set("parent")
        obs.end_run(success=True, final_answer="parent ok")


class RoundTripTest(unittest.TestCase):
    def test_lineage_survives_dict_roundtrip(self):
        from toolboxv2.mods.isaa.base.Agent.observability import RunRecord
        r = RunRecord(run_id="x", parent_run_id="p",
                      sub_agent_runs=[{"run_id": "y", "status": "completed"}])
        r2 = RunRecord.from_dict(r.to_dict())
        self.assertEqual(r2.parent_run_id, "p")
        self.assertEqual(r2.sub_agent_runs[0]["run_id"], "y")

    def test_old_run_file_without_lineage_loads(self):
        from toolboxv2.mods.isaa.base.Agent.observability import RunRecord
        # simulate a pre-patch run dict (no lineage keys)
        old = {"run_id": "old", "agent_name": "a", "query": "q", "steps": []}
        r = RunRecord.from_dict(old)
        self.assertEqual(r.parent_run_id, "")
        self.assertEqual(r.sub_agent_runs, [])


class WaitTruePathTest(unittest.IsolatedAsyncioTestCase):
    """Synchronous (wait=True) sub-agents share the parent's context."""

    async def test_parent_slot_survives_sync_child_and_records_after(self):
        obs = _new_obs()
        obs.begin_run("parent", "q", session_id="s")
        obs.begin_step(1)
        obs.record_tool_start("spawn", call_id="p0")
        # synchronous child (same context, no create_task)
        obs.begin_run("kid", "subq", session_id="s__sub__kid", parent_run_id="parent")
        obs.begin_step(1)
        obs.record_tool_start("vfs_read", call_id="k0")
        obs.record_tool_end("vfs_read", "ok", call_id="k0")
        obs.end_step()
        obs.end_run(success=True, final_answer="kid done")
        # parent slot still present, context restored to parent
        self.assertIn("parent", obs._slots)
        self.assertEqual(_active_run_id.get(), "parent")
        # parent can still record into its open step
        obs.record_tool_end("spawn", "ok", call_id="p0")
        obs.end_step()
        obs.end_run(success=True, final_answer="parent done")
        p = obs.get_run("parent")
        self.assertEqual(p.tool_call_count, 1)
        self.assertEqual(p.sub_agent_runs[0]["status"], "completed")

    async def test_two_sync_children_sequential(self):
        """Parent spawns two children one after another (both wait=True)."""
        obs = _new_obs()
        obs.begin_run("parent", "q", session_id="s")
        for kid in ("kidA", "kidB"):
            obs.begin_run(kid, "s", session_id=f"s__sub__{kid}", parent_run_id="parent")
            obs.begin_step(1)
            obs.record_tool_start("t", call_id=f"{kid}0")
            obs.record_tool_end("t", "ok", call_id=f"{kid}0")
            obs.end_step()
            obs.end_run(success=True, final_answer=f"{kid} done")
            # after each child, context is back to parent
            self.assertEqual(_active_run_id.get(), "parent")
        obs.end_run(success=True, final_answer="parent done")
        p = obs.get_run("parent")
        self.assertEqual(len(p.sub_agent_runs), 2)
        self.assertTrue(all(s["status"] == "completed" for s in p.sub_agent_runs))


class ObsViewerSubAgentTest(unittest.TestCase):
    def _runs(self):
        parent = {
            "run_id": "parent", "agent_name": "a", "query": "spawn kids",
            "success": True, "duration_s": 5.0, "tool_call_count": 1,
            "parent_run_id": "",
            "sub_agent_runs": [
                {"run_id": "kidA", "task": "research X", "status": "completed"},
                {"run_id": "kidB", "task": "research Y", "status": "failed"},
            ],
            "steps": [{
                "step_id": 1, "t_start": 0, "t_end": 5, "duration_s": 5,
                "tool_calls": [
                    {"name": "spawn_sub_agent", "duration_s": 2.0, "status": "ok"},
                    {"name": "spawn_sub_agent", "duration_s": 3.0, "status": "ok"},
                ],
            }],
        }
        kidA = {"run_id": "kidA", "agent_name": "a", "query": "research X",
                "success": True, "duration_s": 2.0, "tool_call_count": 1,
                "parent_run_id": "parent", "sub_agent_runs": [],
                "steps": [{"step_id": 1, "t_start": 0, "t_end": 2, "duration_s": 2,
                           "tool_calls": [{"name": "vfs_read", "duration_s": 0.5, "status": "ok"}]}]}
        kidB = {"run_id": "kidB", "agent_name": "a", "query": "research Y",
                "success": False, "duration_s": 1.0, "tool_call_count": 0,
                "parent_run_id": "parent", "sub_agent_runs": [], "steps": []}
        return [parent, kidA, kidB]

    def test_viewer_generates_with_nested_subagents(self):
        out = os.path.join(tempfile.mkdtemp(), "viewer.html")
        path = obs_viewer.generate_viewer(runs=self._runs(), agent_name="a", output=out)
        html = open(path, encoding="utf-8").read()
        # data embedded
        self.assertIn('"sub_agent_runs"', html)
        self.assertIn("kidA", html);
        self.assertIn("kidB", html)
        # new JS machinery present
        self.assertIn("makeSubMatcher", html)
        self.assertIn("renderSubAgentRun", html)
        self.assertIn("RUN_BY_ID", html)
        self.assertIn("SPAWN_TOOLS", html)
        # CSS present
        self.assertIn(".subagent-wrap", html)
        # payload roundtrip: extract embedded JSON, confirm structure
        marker = "const DATA = "
        start = html.index(marker) + len(marker)
        end = html.index(";\n", start)
        data = json.loads(html[start:end])
        rids = {r["run_id"] for r in data["runs"]}
        self.assertEqual(rids, {"parent", "kidA", "kidB"})
        p = next(r for r in data["runs"] if r["run_id"] == "parent")
        self.assertEqual(len(p["sub_agent_runs"]), 2)


class ObsViewerSubAgentTest2(unittest.TestCase):
    def _runs(self):
        parent = {
            "run_id": "parent", "agent_name": "a", "query": "spawn kids",
            "success": True, "duration_s": 5.0, "tool_call_count": 1,
            "parent_run_id": "",
            "sub_agent_runs": [
                {"run_id": "kidA", "task": "research X", "status": "completed"},
                {"run_id": "kidB", "task": "research Y", "status": "failed"},
            ],
            "steps": [{
                "step_id": 1, "t_start": 0, "t_end": 5, "duration_s": 5,
                "tool_calls": [
                    {"name": "spawn_sub_agent", "duration_s": 2.0, "status": "ok"},
                    {"name": "spawn_sub_agent", "duration_s": 3.0, "status": "ok"},
                ],
            }],
        }
        kidA = {"run_id": "kidA", "agent_name": "a", "query": "research X",
                "success": True, "duration_s": 2.0, "tool_call_count": 1,
                "parent_run_id": "parent", "sub_agent_runs": [],
                "steps": [{"step_id": 1, "t_start": 0, "t_end": 2, "duration_s": 2,
                           "tool_calls": [{"name": "vfs_read", "duration_s": 0.5, "status": "ok"}]}]}
        kidB = {"run_id": "kidB", "agent_name": "a", "query": "research Y",
                "success": False, "duration_s": 1.0, "tool_call_count": 0,
                "parent_run_id": "parent", "sub_agent_runs": [], "steps": []}
        return [parent, kidA, kidB]

    def test_viewer_generates_with_nested_subagents(self):
        out = os.path.join(tempfile.mkdtemp(), "viewer.html")
        path = obs_viewer.generate_viewer(runs=self._runs(), agent_name="a", output=out)
        html = open(path, encoding="utf-8").read()
        # data embedded
        self.assertIn('"sub_agent_runs"', html)
        self.assertIn("kidA", html);
        self.assertIn("kidB", html)
        # new JS machinery present
        self.assertIn("makeSubMatcher", html)
        self.assertIn("renderSubAgentRun", html)
        self.assertIn("RUN_BY_ID", html)
        self.assertIn("SPAWN_TOOLS", html)
        # CSS present
        self.assertIn(".subagent-wrap", html)
        # payload roundtrip: extract embedded JSON, confirm structure
        marker = "const DATA = "
        start = html.index(marker) + len(marker)
        end = html.index(";\n", start)
        data = json.loads(html[start:end])
        rids = {r["run_id"] for r in data["runs"]}
        self.assertEqual(rids, {"parent", "kidA", "kidB"})
        p = next(r for r in data["runs"] if r["run_id"] == "parent")
        self.assertEqual(len(p["sub_agent_runs"]), 2)


class LlmDecodeRenderTest(unittest.TestCase):
    """Screenshot bug: double-encoded \\n / \\" in LLM output rendered raw."""

    def test_viewer_embeds_decode_and_double_encoded_data(self):
        runs = [{
            "run_id": "r", "agent_name": "a", "query": "q", "success": True,
            "duration_s": 1.0, "tool_call_count": 0, "parent_run_id": "",
            "sub_agent_runs": [],
            "steps": [{
                "step_id": 1, "t_start": 0, "t_end": 1, "duration_s": 1,
                "llm": {
                    "model": "m", "t_start": 0, "t_end": 1, "duration_s": 1,
                    # double-encoded, exactly like the screenshot
                    "input_messages": [
                        {"role": "user", "content": "def f():\\n    return Result.html(x)\\n\\n\\\"\\\"\\\""}],
                    "output_text": "line1\\nline2\\nReturn \\\"ok\\\"",
                },
            }],
        }]
        out = os.path.join(tempfile.mkdtemp(), "v.html")
        path = obs_viewer.generate_viewer(runs=runs, agent_name="a", output=out)
        html = open(path, encoding="utf-8").read()
        # decode helper + msg-content box must be embedded
        self.assertIn("function decodeMaybe", html)
        self.assertIn(".msg-content", html)
        self.assertIn("decodeMaybe(content)", html)
        self.assertIn("decodeMaybe(out)", html)


def _new_obs():
    return ObservabilityLayer(agent_name="t", obs_dir=tempfile.mkdtemp(), max_runs=10)


class ObsAccessorTest(unittest.TestCase):
    def test_accessors_exist_and_work(self):
        obs = _new_obs()
        self.assertEqual(obs.active_runs(), [])
        self.assertIsNone(obs.active_run())
        self.assertIsNone(obs.top_level_active_run())
        obs.begin_run("p", "q", session_id="s")
        self.assertEqual(len(obs.active_runs()), 1)
        self.assertEqual(obs.active_run().run_id, "p")
        self.assertEqual(obs.top_level_active_run().run_id, "p")
        obs.end_run(success=True)
        self.assertEqual(obs.active_runs(), [])


class HubWrapLogicTest(unittest.IsolatedAsyncioTestCase):
    """Replicates LiveObsHub.register's begin/end wrap + snapshot, verifying
    they work against the patched obs (no _current_run crash)."""

    def _wrap(self, obs, fired):
        prev_begin = obs.begin_run

        def hub_begin(run_id, query, session_id="", persona="", skills=None,
                      is_resume=False, parent_run_id=""):
            prev_begin(run_id, query, session_id, persona, skills,
                       is_resume=is_resume, parent_run_id=parent_run_id)
            current = obs.active_run(run_id)
            fired.append({"type": "run_start", "run_id": run_id,
                          "parent_run_id": parent_run_id,
                          "is_sub_agent": bool(parent_run_id),
                          "t_start": current.t_start if current else None})

        obs.begin_run = hub_begin
        prev_end = obs.end_run

        def hub_end(success, final_answer=""):
            run = obs.active_run()
            rid = run.run_id if run else None
            parent_rid = run.parent_run_id if run else ""
            prev_end(success, final_answer)
            fired.append({"type": "run_end", "run_id": rid,
                          "parent_run_id": parent_rid,
                          "is_sub_agent": bool(parent_rid)})

        obs.end_run = hub_end

    def _snapshot(self, obs):
        actives = obs.active_runs()
        top = next((r for r in actives if not r.parent_run_id), None)
        return {"active_run": top.to_dict() if top else None,
                "active_runs": [r.to_dict() for r in actives]}

    async def _sub(self, obs, parent, sub):
        obs.begin_run(sub, f"t{sub}", session_id=f"s__sub__{sub}", parent_run_id=parent)
        await asyncio.sleep(0.001)
        obs.end_run(success=True, final_answer="ok")

    async def test_wrap_no_crash_and_lineage_events(self):
        obs = _new_obs();
        fired = []
        self._wrap(obs, fired)
        obs.begin_run("parent", "q", session_id="s")
        # snapshot mid-run with parallel sub-agents would need them active;
        # here test sequential to keep deterministic ordering of fired events
        await self._sub(obs, "parent", "kidA")
        await self._sub(obs, "parent", "kidB")
        # after subs, context restored to parent → snapshot sees parent active
        snap = self._snapshot(obs)
        self.assertIsNotNone(snap["active_run"])
        self.assertEqual(snap["active_run"]["run_id"], "parent")
        obs.end_run(success=True, final_answer="done")
        # events fired: parent start, kidA start/end, kidB start/end, parent end
        starts = [f for f in fired if f["type"] == "run_start"]
        ends = [f for f in fired if f["type"] == "run_end"]
        self.assertEqual({s["run_id"] for s in starts}, {"parent", "kidA", "kidB"})
        sub_starts = [s for s in starts if s["is_sub_agent"]]
        self.assertEqual({s["run_id"] for s in sub_starts}, {"kidA", "kidB"})
        self.assertTrue(all(s["parent_run_id"] == "parent" for s in sub_starts))
        # kid end events carry parent linkage
        kid_ends = [e for e in ends if e["is_sub_agent"]]
        self.assertEqual({e["run_id"] for e in kid_ends}, {"kidA", "kidB"})

    async def test_parallel_snapshot_sees_all_actives(self):
        obs = _new_obs()
        obs.begin_run("parent", "q", session_id="s")

        # start two subs WITHOUT ending them (parallel mid-flight) in their own contexts
        async def start_only(sub):
            obs.begin_run(sub, "t", session_id=f"s__sub__{sub}", parent_run_id="parent")

        await asyncio.gather(asyncio.create_task(start_only("kidA")),
                             asyncio.create_task(start_only("kidB")))
        snap = self._snapshot(obs)
        ids = {r["run_id"] for r in snap["active_runs"]}
        self.assertEqual(ids, {"parent", "kidA", "kidB"})
        # top-level resolves to parent only
        self.assertEqual(snap["active_run"]["run_id"], "parent")


class ConsumerContractTest(unittest.TestCase):
    def test_engine_finally_guard_pattern(self):
        """execution_engine.py: `if _obs and _obs.active_run(): _obs.end_run(...)`"""
        obs = _new_obs()
        # no run active → guard must be falsy, no crash, no end_run
        self.assertFalse(obs.active_run())
        # active run → guard truthy → end_run works
        obs.begin_run("r", "q", session_id="s")
        self.assertTrue(obs.active_run())
        if obs.active_run():
            obs.end_run(success=True, final_answer="ok")
        self.assertFalse(obs.active_run())  # ended

    def test_flow_agent_first_token_guard(self):
        """flow_agent.py: `if _obs and _obs.needs_first_token(): record_llm_first_token()`"""
        obs = _new_obs()
        # no run / no llm → False
        self.assertFalse(obs.needs_first_token())
        obs.begin_run("r", "q", session_id="s")
        obs.begin_step(1)
        self.assertFalse(obs.needs_first_token())  # no llm started yet
        obs.record_llm_start(model="m")
        self.assertTrue(obs.needs_first_token())  # llm in flight, no ttft
        # the guarded call
        if obs.needs_first_token():
            obs.record_llm_first_token()
        self.assertFalse(obs.needs_first_token())  # ttft now set → guard closes
        # idempotent: calling again stays False
        self.assertFalse(obs.needs_first_token())
        obs.end_run(success=True)

    def test_no_removed_attributes_accessed_externally(self):
        """The removed internals must not be needed by any documented accessor."""
        obs = _new_obs()
        obs.begin_run("r", "q", session_id="s")
        # all public accessors a consumer might use
        self.assertIsNotNone(obs.active_run())
        self.assertEqual(len(obs.active_runs()), 1)
        self.assertIsNotNone(obs.top_level_active_run())
        self.assertIn(obs.needs_first_token(), (True, False))
        obs.end_run(success=True)

    def test_crash_path_double_end_run_safe(self):
        """finally-block may fire end_run after normal end_run already ran."""
        obs = _new_obs()
        obs.begin_run("r", "q", session_id="s")
        obs.end_run(success=True, final_answer="ok")
        # simulated finally guard: active_run() is now None → no double-end
        if obs.active_run():
            obs.end_run(success=False, final_answer="CRASH")  # must NOT run
        # still consistent
        self.assertFalse(obs.active_run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
