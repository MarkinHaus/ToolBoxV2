"""Unit tests for bench.core — dataclasses, scoring, serialization."""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from toolboxv2.mods.isaa.base.bench.core import (
    Attachment,
    Check,
    CheckResult,
    JudgeProfile,
    Report,
    Suite,
    Task,
    TaskContext,
    TaskResult,
)


def make_check_result(name="contains", passed=True, detail="ok"):
    return CheckResult(validator_name=name, passed=passed, detail=detail)


def make_task_result(task_id="t1", complexity="tutorial", tags=None, checks=None, **kw):
    return TaskResult(
        task_id=task_id,
        complexity=complexity,
        tags=tags or ["logic"],
        checks=checks or [],
        **kw,
    )


def make_task(task_id="test-001", **overrides):
    defaults = {
        "id": task_id,
        "complexity": "tutorial",
        "modality": ["text"],
        "prompt": "What is 2+2?",
        "checks": [],
        "tags": ["logic"],
    }
    defaults.update(overrides)
    checks = defaults.pop("checks")
    return Task(checks=[Check(**c) if isinstance(c, dict) else c for c in checks], **defaults)


class TestTask(unittest.TestCase):

    def test_is_multimodal_text_only_returns_false(self):
        t = make_task(modality=["text"])
        self.assertFalse(t.is_multimodal)

    def test_is_multimodal_with_image_returns_true(self):
        t = make_task(modality=["text", "image"])
        self.assertTrue(t.is_multimodal)

    def test_is_multimodal_image_only_returns_true(self):
        t = make_task(modality=["image"])
        self.assertTrue(t.is_multimodal)

    def test_default_fields(self):
        t = make_task()
        self.assertEqual(t.attachments, [])
        self.assertIsNone(t.ground_truth)


class TestTaskResult(unittest.TestCase):

    def test_score_no_checks_returns_zero(self):
        r = make_task_result(checks=[])
        self.assertEqual(r.score, 0.0)

    def test_score_all_passed(self):
        checks = [make_check_result(passed=True), make_check_result(passed=True)]
        r = make_task_result(checks=checks)
        self.assertEqual(r.score, 1.0)

    def test_score_all_failed(self):
        checks = [make_check_result(passed=False), make_check_result(passed=False)]
        r = make_task_result(checks=checks)
        self.assertEqual(r.score, 0.0)

    def test_score_mixed(self):
        checks = [
            make_check_result(passed=True),
            make_check_result(passed=False),
            make_check_result(passed=True),
        ]
        r = make_task_result(checks=checks)
        self.assertAlmostEqual(r.score, 2 / 3)

    def test_passed_count(self):
        checks = [make_check_result(passed=True), make_check_result(passed=False)]
        r = make_task_result(checks=checks)
        self.assertEqual(r.passed, 1)

    def test_total_checks(self):
        checks = [make_check_result(), make_check_result(), make_check_result()]
        r = make_task_result(checks=checks)
        self.assertEqual(r.total_checks, 3)

    def test_to_dict_contains_required_keys(self):
        r = make_task_result(
            checks=[make_check_result(passed=True)],
            response="42",
            latency_ms=150,
            tokens_in=10,
            tokens_out=5,
            cost=0.001,
        )
        d = r.to_dict()
        self.assertEqual(d["task_id"], "t1")
        self.assertEqual(d["score"], 1.0)
        self.assertEqual(d["passed"], 1)
        self.assertEqual(d["total_checks"], 1)
        self.assertEqual(d["response"], "42")
        self.assertEqual(d["latency_ms"], 150)
        self.assertEqual(len(d["checks"]), 1)
        self.assertTrue(d["checks"][0]["passed"])


class TestReport(unittest.TestCase):

    def test_total_score_empty_results(self):
        r = Report(model_id="m", suite_id="s")
        self.assertEqual(r.total_score, 0.0)

    def test_total_score_single_perfect(self):
        r = Report(model_id="m", suite_id="s")
        r.results = [make_task_result(checks=[make_check_result(passed=True)])]
        self.assertEqual(r.total_score, 1.0)

    def test_total_score_averages_across_tasks(self):
        r = Report(model_id="m", suite_id="s")
        r.results = [
            make_task_result(checks=[make_check_result(passed=True)]),  # 1.0
            make_task_result(checks=[make_check_result(passed=False)]),  # 0.0
        ]
        self.assertAlmostEqual(r.total_score, 0.5)

    def test_total_tokens(self):
        r = Report(model_id="m", suite_id="s", total_tokens_in=100, total_tokens_out=50)
        self.assertEqual(r.total_tokens, 150)

    def test_scores_by_tag(self):
        r = Report(model_id="m", suite_id="s")
        r.results = [
            make_task_result(tags=["logic"], checks=[make_check_result(passed=True)]),
            make_task_result(tags=["logic"], checks=[make_check_result(passed=False)]),
            make_task_result(tags=["extract"], checks=[make_check_result(passed=True)]),
        ]
        by_tag = r.scores_by_tag()
        self.assertAlmostEqual(by_tag["logic"], 0.5)
        self.assertAlmostEqual(by_tag["extract"], 1.0)

    def test_scores_by_complexity(self):
        r = Report(model_id="m", suite_id="s")
        r.results = [
            make_task_result(complexity="tutorial", checks=[make_check_result(passed=True)]),
            make_task_result(complexity="phd", checks=[make_check_result(passed=False)]),
        ]
        by_c = r.scores_by_complexity()
        self.assertAlmostEqual(by_c["tutorial"], 1.0)
        self.assertAlmostEqual(by_c["phd"], 0.0)

    def test_to_dict_dashboard_compat(self):
        r = Report(model_id="test-model", suite_id="quick", mode="standard")
        r.results = [
            make_task_result(
                tags=["logic"],
                checks=[make_check_result(passed=True), make_check_result(passed=False)],
                response="some response",
            ),
        ]
        r.total_cost = 0.05
        r.total_tokens_in = 100
        r.total_tokens_out = 50
        r.total_time_s = 3.0

        d = r.to_dict()

        # Dashboard required keys
        self.assertEqual(d["model"], "test-model")
        self.assertEqual(d["mode"], "standard")
        self.assertIn("total", d)
        self.assertIn("dimensions", d)
        self.assertIn("flags", d)
        self.assertIn("cost", d)
        self.assertIn("results", d)
        self.assertIn("probes", d)

        # Total is 0-100 scale
        self.assertAlmostEqual(d["total"], 50.0)

        # Dimensions mapped to 0-100
        self.assertIn("logic", d["dimensions"])
        self.assertAlmostEqual(d["dimensions"]["logic"], 50.0)

        # Cost structure
        self.assertEqual(d["cost"]["total_cost"], 0.05)
        self.assertEqual(d["cost"]["total_tokens"], 150)

    def test_to_dict_empty_report_no_division_by_zero(self):
        r = Report(model_id="m", suite_id="s")
        d = r.to_dict()
        self.assertEqual(d["total"], 0.0)
        self.assertEqual(d["probes"], 0)
        # cost_per_probe should not crash
        self.assertIsNotNone(d["cost"]["cost_per_probe"])

    def test_save_and_load_roundtrip(self):
        r = Report(model_id="roundtrip-model", suite_id="test-suite")
        r.results = [make_task_result(checks=[make_check_result(passed=True)])]
        r.total_cost = 0.01
        r.total_time_s = 2.5

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            r.save(path)

            # Verify JSON is valid
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["model"], "roundtrip-model")

            # Load back
            loaded = Report.load(path)
            self.assertEqual(loaded.model_id, "roundtrip-model")
            self.assertEqual(loaded.suite_id, "test-suite")
            self.assertAlmostEqual(loaded.total_cost, 0.01)
        finally:
            Path(path).unlink(missing_ok=True)


class TestJudgeProfile(unittest.TestCase):

    def test_to_dict_and_from_dict_roundtrip(self):
        p = JudgeProfile(
            model="gpt-4o",
            disqualified=False,
            batch_sizes={"tutorial": 32, "extended": 8, "phd": 2},
            accuracy={"tutorial": 1.0, "extended": 0.97, "phd": 0.95},
        )
        d = p.to_dict()
        p2 = JudgeProfile.from_dict(d)
        self.assertEqual(p2.model, "gpt-4o")
        self.assertFalse(p2.disqualified)
        self.assertEqual(p2.batch_sizes["tutorial"], 32)
        self.assertAlmostEqual(p2.accuracy["phd"], 0.95)

    def test_from_dict_missing_keys_uses_defaults(self):
        p = JudgeProfile.from_dict({})
        self.assertEqual(p.model, "")
        self.assertFalse(p.disqualified)
        self.assertEqual(p.batch_sizes, {})


class TestTaskContext(unittest.TestCase):

    def test_default_fields(self):
        t = make_task()
        ctx = TaskContext(task=t, prompt="test", response="42")
        self.assertEqual(ctx.tool_calls, [])
        self.assertEqual(ctx.execution_time, 0.0)
        self.assertIsNone(ctx.sandbox_state)
        self.assertEqual(ctx.token_usage, {})


if __name__ == "__main__":
    unittest.main()
