"""Unittests for dreamer/run_aggregator.py — no agent, no real VFS."""
import asyncio
import json
import unittest

from run_aggregator import (
    RunAggregator, RunMetrics, extract_metrics,
    fuzzy_preselect, parse_classify_guide, update_classify_guide,
    default_classify_guide, TASKMAP_ROOT, CLASSIFY_GUIDE_PATH, NEW_TYPE,
    _sanitize_class,
)


class MockVFS:
    def __init__(self):
        self.files = {}
        self.dirs = set()

    def read(self, path, max_chars=25000):
        if path in self.files:
            return {"success": True, "content": self.files[path]}
        return {"success": False, "error": "not found"}

    def write(self, path, content):
        self.files[path] = content
        return {"success": True}

    def mkdir(self, path, parents=False):
        self.dirs.add(path)
        return {"success": True}


def make_run_record(run_id="r1", success=True, iters=4, tools=None, resume=False,
                    user_content=""):
    tools = tools if tools is not None else ["vfs_read", "vfs_write"]
    steps = [{
        "tool_calls": [{"name": t, "duration_s": 0.5, "status": "ok"} for t in tools],
        "is_resume_point": resume,
        "user_provided_content": user_content,
    }]
    return {
        "run_id": run_id, "success": success, "duration_s": 12.5,
        "total_iterations": iters, "files_modified": ["/a.py"],
        "skills_matched": ["s1"], "parent_run_id": "",
        "sub_agent_runs": [{"run_id": "sub1", "status": "completed"}],
        "steps": steps,
    }


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestExtractMetrics(unittest.TestCase):
    def test_basic_extraction(self):
        m = extract_metrics(make_run_record(), query="fix bug",
                            narrator_snapshot={"drift": True, "repeat": False,
                                               "plan_summary": "plan x"})
        self.assertEqual(m.run_id, "r1")
        self.assertEqual(m.query, "fix bug")
        self.assertEqual(m.total_iterations, 4)
        self.assertEqual([t["name"] for t in m.tool_call_sequence],
                         ["vfs_read", "vfs_write"])
        self.assertEqual(m.sub_agent_run_ids, ["sub1"])
        self.assertEqual(m.topic_drift, 1.0)
        self.assertEqual(m.plan_summary, "plan x")

    def test_resume_facts_not_judgement(self):
        m = extract_metrics(make_run_record(resume=True, user_content="nein, anders"))
        self.assertEqual(m.resume_count, 1)
        self.assertEqual(m.resume_type, "user_content")
        self.assertEqual(m.user_provided_content, "nein, anders")

    def test_error_tools_collected(self):
        rec = make_run_record()
        rec["steps"][0]["tool_calls"].append(
            {"name": "broken_tool", "duration_s": 1.0, "status": "error",
             "error": "boom"})
        m = extract_metrics(rec)
        self.assertIn("broken_tool", m.error_tools)


class TestClassifyGuide(unittest.TestCase):
    def test_parse_and_fuzzy(self):
        guide = default_classify_guide()
        self.assertTrue(parse_classify_guide(guide))
        res = fuzzy_preselect("fix toolbox mod export bug", guide)
        self.assertEqual((res[0][0], res[0][1]), ("coding", "toolbox"))

    def test_fuzzy_no_match(self):
        self.assertEqual(fuzzy_preselect("xyzzy quux", default_classify_guide()), [])

    def test_update_appends_keywords_bounded(self):
        guide = default_classify_guide()
        g2 = update_classify_guide(guide, "coding", "toolbox", "implement websocket streaming bridge")
        entries = {(t, s): k for t, s, k in parse_classify_guide(g2)}
        self.assertIn("websocket", entries[("coding", "toolbox")])

    def test_update_creates_new_class_line(self):
        g2 = update_classify_guide(default_classify_guide(), "research", "papers",
                                   "summarize arxiv paper embeddings")
        self.assertIn(("research", "papers"),
                      [(t, s) for t, s, _ in parse_classify_guide(g2)])

    def test_new_type_never_accumulates(self):
        guide = default_classify_guide()
        self.assertEqual(update_classify_guide(guide, NEW_TYPE, "general", "whatever stuff"), guide)

    def test_sanitize_splits_slash(self):
        self.assertEqual(_sanitize_class("coding/toolbox", "general"), ("coding", "toolbox"))
        self.assertEqual(_sanitize_class("coding/isaa", "general"), ("coding", "isaa"))

    def test_sanitize_slash_preserves_explicit_subtype(self):
        self.assertEqual(_sanitize_class("coding/toolbox", "isaa"), ("coding", "isaa"))

    def test_sanitize_no_slash(self):
        self.assertEqual(_sanitize_class("coding", "toolbox"), ("coding", "toolbox"))

    def test_sanitize_empty(self):
        self.assertEqual(_sanitize_class("", ""), (NEW_TYPE, "general"))


class TestAggregator(unittest.TestCase):
    def setUp(self):
        self.vfs = MockVFS()

    def _agg(self, llm=None):
        return RunAggregator(self.vfs, llm_completion_func=llm)

    def test_fuzzy_only_classification_and_persist(self):
        agg = self._agg(llm=None)
        m = run(agg.aggregate(make_run_record(), query="fix toolbox mod export bug"))
        self.assertEqual((m.task_type, m.subtype), ("coding", "toolbox"))
        self.assertTrue(m.is_new_task_type)
        self.assertTrue(m.is_new_subtype)
        base = f"{TASKMAP_ROOT}/coding/toolbox"
        self.assertIn(f"{base}/tracks.jsonl", self.vfs.files)
        self.assertIn(f"{base}/formatted_row.jsonl", self.vfs.files)
        self.assertIn(f"{base}/_index.json", self.vfs.files)
        top = json.loads(self.vfs.files[f"{TASKMAP_ROOT}/_index.json"])
        self.assertIn("coding", top["task_types"])
        self.assertIn("toolbox", top["task_types"]["coding"]["subtypes"])

    def test_unclear_goes_to_new_sink(self):
        agg = self._agg(llm=None)
        m = run(agg.aggregate(make_run_record(), query="xyzzy quux"))
        self.assertEqual((m.task_type, m.subtype), (NEW_TYPE, "general"))
        # NEW_TYPE never writes a happypath
        self.assertNotIn(f"{TASKMAP_ROOT}/{NEW_TYPE}/general/happypath.md", self.vfs.files)

    def test_llm_classification_used(self):
        async def llm(messages, **kw):
            return '{"task_type": "coding", "subtype": "isaa"}'
        m = run(self._agg(llm=llm).aggregate(make_run_record(), query="whatever"))
        self.assertEqual((m.task_type, m.subtype), ("coding", "isaa"))

    def test_llm_slash_in_task_type_splits_correctly(self):
        async def llm(messages, **kw):
            return '{"task_type": "coding/toolbox", "subtype": "general"}'
        m = run(self._agg(llm=llm).aggregate(make_run_record(), query="whatever"))
        self.assertEqual((m.task_type, m.subtype), ("coding", "toolbox"))

    def test_llm_failure_falls_back_to_fuzzy_or_new(self):
        async def llm(messages, **kw):
            raise RuntimeError("down")
        m = run(self._agg(llm=llm).aggregate(make_run_record(), query="zz yy"))
        self.assertEqual(m.task_type, NEW_TYPE)

    def test_happypath_only_on_improvement(self):
        agg = self._agg(llm=None)
        q = "fix toolbox mod export bug"
        run(agg.aggregate(make_run_record(run_id="a", iters=6), query=q))
        hp1 = self.vfs.files[f"{TASKMAP_ROOT}/coding/toolbox/happypath.md"]
        run(agg.aggregate(make_run_record(run_id="b", iters=9), query=q))
        self.assertEqual(self.vfs.files[f"{TASKMAP_ROOT}/coding/toolbox/happypath.md"], hp1)
        run(agg.aggregate(make_run_record(run_id="c", iters=3), query=q))
        self.assertIn("run c", self.vfs.files[f"{TASKMAP_ROOT}/coding/toolbox/happypath.md"])

    def test_effort_ratio_gated_until_three_entries(self):
        agg = self._agg(llm=None)
        q = "fix toolbox mod export bug"
        m1 = run(agg.aggregate(make_run_record(run_id="a"), query=q))
        m2 = run(agg.aggregate(make_run_record(run_id="b"), query=q))
        self.assertEqual(m1.effort_ratio, -1.0)
        self.assertEqual(m2.effort_ratio, -1.0)
        run(agg.aggregate(make_run_record(run_id="c"), query=q))
        m4 = run(agg.aggregate(make_run_record(run_id="d", iters=8), query=q))
        self.assertGreater(m4.effort_ratio, 0)

    def test_index_stats_accumulate(self):
        agg = self._agg(llm=None)
        q = "fix toolbox mod export bug"
        run(agg.aggregate(make_run_record(run_id="a", success=True, iters=4), query=q))
        run(agg.aggregate(make_run_record(run_id="b", success=False, iters=6), query=q))
        idx = json.loads(self.vfs.files[f"{TASKMAP_ROOT}/coding/toolbox/_index.json"])
        self.assertEqual(idx["entry_count"], 2)
        self.assertAlmostEqual(idx["performance"], 0.5)
        self.assertAlmostEqual(idx["avg_trace_length"], 5.0)

    def test_guide_written_and_extended(self):
        agg = self._agg(llm=None)
        run(agg.aggregate(make_run_record(), query="fix toolbox mod export websocket"))
        guide = self.vfs.files[CLASSIFY_GUIDE_PATH]
        self.assertIn("websocket", guide)

    def test_guid_md_never_touched(self):
        self.vfs.files[f"{TASKMAP_ROOT}/coding/toolbox/guid.md"] = "DREAMER OWNED"
        agg = self._agg(llm=None)
        run(agg.aggregate(make_run_record(), query="fix toolbox mod export bug"))
        self.assertEqual(self.vfs.files[f"{TASKMAP_ROOT}/coding/toolbox/guid.md"], "DREAMER OWNED")

    def test_aggregate_never_raises(self):
        class BrokenVFS:
            def read(self, *a, **k): raise RuntimeError("io")
            def write(self, *a, **k): raise RuntimeError("io")
            def mkdir(self, *a, **k): raise RuntimeError("io")
        m = run(RunAggregator(BrokenVFS()).aggregate(make_run_record(), query="x"))
        self.assertIsInstance(m, RunMetrics)


if __name__ == "__main__":
    unittest.main()


class TestPreInjection(unittest.TestCase):
    """Flag-aktivierbarer Run-Start-Hook: fuzzy + 1 Narrator-Call."""

    def setUp(self):
        from run_aggregator import build_preinjection, classify_for_injection
        self.build = build_preinjection
        self.classify = classify_for_injection
        self.vfs = MockVFS()
        self.vfs.files[CLASSIFY_GUIDE_PATH] = default_classify_guide()
        base = f"{TASKMAP_ROOT}/coding/toolbox"
        self.vfs.files[f"{base}/happypath.md"] = "# Happy Path\n-> vfs_read\n-> vfs_write\n"
        self.vfs.files[f"{base}/guid.md"] = "# Guide\nUse tb flows export.\n"

    def test_dominant_fuzzy_skips_llm(self):
        calls = []
        async def narr(system, q):
            calls.append(1)
            return {"task_type": "coding", "subtype": "toolbox"}
        tt, st = run(self.classify("fix toolbox mod flows export bug",
                                   default_classify_guide(), narr))
        self.assertEqual((tt, st), ("coding", "toolbox"))
        self.assertEqual(calls, [])  # kein LLM-Call bei dominantem Match

    def test_narrator_call_used_when_ambiguous(self):
        async def narr(system, q):
            return {"task_type": "coding", "subtype": "isaa"}
        tt, st = run(self.classify("agent", default_classify_guide(), narr))
        self.assertEqual((tt, st), ("coding", "isaa"))

    def test_uncertain_returns_new(self):
        async def narr(system, q):
            return None
        tt, st = run(self.classify("xyzzy quux", default_classify_guide(), narr))
        self.assertEqual(tt, NEW_TYPE)

    def test_injection_block_contains_guide_happypath_and_swap_notice(self):
        block = run(self.build(self.vfs, "fix toolbox mod flows export bug"))
        self.assertIn("coding/toolbox", block)
        self.assertIn("Task Guide", block)
        self.assertIn("Known Happy Path", block)
        self.assertIn("IGNORE", block)  # Austausch/Schließen-Hinweis an den Agent

    def test_new_type_injects_nothing(self):
        block = run(self.build(self.vfs, "xyzzy quux"))
        self.assertEqual(block, "")

    def test_no_data_injects_nothing(self):
        vfs = MockVFS()
        vfs.files[CLASSIFY_GUIDE_PATH] = default_classify_guide()
        block = run(vfs and self.build(vfs, "fix toolbox mod flows export bug"))
        self.assertEqual(block, "")  # Klasse existiert im Guide, aber kein happypath/guid

    def test_never_raises(self):
        class BrokenVFS:
            def read(self, *a, **k): raise RuntimeError("io")
        self.assertEqual(run(self.build(BrokenVFS(), "x")), "")


class TestDreamReport(unittest.TestCase):
    def setUp(self):
        import report
        self.report = report
        self.vfs = MockVFS()

    def _skill(self, name, conf, usage=3, active=True):
        class S:
            pass
        s = S(); s.name = name; s.confidence = conf; s.usage_count = usage
        s.triggers = ["a", "b"]; s.is_active = lambda: active
        return s

    def _sm(self, skills):
        class SM:
            pass
        sm = SM(); sm.skills = {s.name: s for s in skills}
        return sm

    def test_snapshot_counts(self):
        sm = self._sm([self._skill("s1", 0.8), self._skill("s2", 0.4, active=False)])
        snap = self.report.collect_system_snapshot(sm, None, {"p1": {}})
        self.assertEqual(snap["skill_count"], 2)
        self.assertEqual(snap["active_skills"], 1)
        self.assertAlmostEqual(snap["avg_confidence"], 0.6)
        self.assertEqual(snap["persona_count"], 1)

    def test_taskmap_overview(self):
        self.vfs.files[f"{TASKMAP_ROOT}/_index.json"] = json.dumps(
            {"task_types": {"coding": {"subtypes": ["toolbox"], "entry_count": 5,
                                       "success_rate": 0.8}}})
        self.vfs.files[f"{TASKMAP_ROOT}/coding/toolbox/_index.json"] = json.dumps(
            {"performance": 0.8, "entry_count": 5, "avg_trace_length": 4.0})
        self.vfs.files[f"{TASKMAP_ROOT}/coding/toolbox/guid.md"] = "g"
        ov = self.report.collect_taskmap_overview(self.vfs)
        sub = ov["task_types"]["coding"]["subtypes"]["toolbox"]
        self.assertTrue(sub["has_guid"])
        self.assertFalse(sub["has_happypath"])

    def test_html_report_complete_and_escaped(self):
        before = {"skill_count": 3, "active_skills": 2, "avg_confidence": 0.5,
                  "rule_count": 1, "pattern_count": 0, "persona_count": 0, "skills": []}
        after = dict(before, skill_count=5, avg_confidence=0.62,
                     skills=[{"name": "<b>x</b>", "confidence": 0.9, "usage": 4,
                              "active": True, "triggers": ["t"]}])
        actions = {"skills_created": ["alpha", "beta"], "skills_deleted": ["old"],
                   "taskmap_guides_written": ["coding/toolbox"]}
        tm = {"task_types": {"coding": {"subtypes": {"toolbox": {
            "performance": 0.8, "entry_count": 5, "avg_trace_length": 4.0,
            "improvement_trend": 0.1, "has_guid": True, "has_happypath": True}}}}}
        html_out = self.report.build_dream_report_html(
            {"agent": "isaa", "duration_s": 42.0}, before, after, actions, tm,
            final_answer="Report <script>alert(1)</script>")
        self.assertIn("<!DOCTYPE html>", html_out)
        self.assertIn("Nachtprotokoll", html_out)
        self.assertIn("Skills erstellt", html_out)
        self.assertIn("coding/toolbox", html_out)
        self.assertNotIn("<script>alert", html_out)       # escaped
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", html_out)   # skill name escaped
        self.assertIn("+2", html_out)                     # delta skill_count

    def test_write_report_to_vfs(self):
        path = self.report.write_dream_report(self.vfs, "<html></html>", "isaa")
        self.assertTrue(path.startswith(self.report.REPORT_DIR))
        self.assertIn(path, self.vfs.files)
