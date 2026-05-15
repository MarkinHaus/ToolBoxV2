"""Unit tests for bench.loader — task loading, suite resolution, modality filtering."""

import json
import tempfile
import unittest
from pathlib import Path

from toolboxv2.mods.isaa.base.bench.core import Task
from toolboxv2.mods.isaa.base.bench.loader import (
    filter_by_modality,
    load_suite,
    load_task,
    load_tasks_from_dir,
    resolve_suite,
)


def _write_yaml(dir_path: Path, filename: str, content: str) -> Path:
    """Write content to a YAML file in the given directory."""
    p = dir_path / filename
    p.write_text(content, encoding="utf-8")
    return p


class TestLoadTask(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_minimal_task(self):
        _write_yaml(self.tmpdir, "t.yaml", """
id: test-001
prompt: "What is 1+1?"
checks:
  - type: contains
    value: "2"
""")
        t = load_task(self.tmpdir / "t.yaml")
        self.assertEqual(t.id, "test-001")
        self.assertEqual(t.prompt, "What is 1+1?")
        self.assertEqual(t.complexity, "tutorial")
        self.assertEqual(t.modality, ["text"])
        self.assertEqual(len(t.checks), 1)
        self.assertEqual(t.checks[0].type, "contains")
        self.assertEqual(t.checks[0].params["value"], "2")

    def test_load_full_task(self):
        _write_yaml(self.tmpdir, "t.yaml", """
id: multi-001
complexity: phd
modality: [text, image]
prompt: "Describe the image."
tags: [vision, extraction]
ground_truth: "A red car"
attachments:
  - type: image
    path: test.png
checks:
  - type: contains
    value: "car"
  - type: judge
    question: "Does it mention a vehicle?"
""")
        t = load_task(self.tmpdir / "t.yaml")
        self.assertEqual(t.complexity, "phd")
        self.assertEqual(t.modality, ["text", "image"])
        self.assertTrue(t.is_multimodal)
        self.assertEqual(len(t.tags), 2)
        self.assertEqual(t.ground_truth, "A red car")
        self.assertEqual(len(t.attachments), 1)
        self.assertEqual(t.attachments[0].type, "image")
        self.assertEqual(len(t.checks), 2)

    def test_load_task_missing_id_raises(self):
        _write_yaml(self.tmpdir, "bad.yaml", """
prompt: "no id here"
checks: []
""")
        with self.assertRaises(KeyError):
            load_task(self.tmpdir / "bad.yaml")

    def test_load_json_task(self):
        p = self.tmpdir / "t.json"
        p.write_text(json.dumps({
            "id": "json-001",
            "prompt": "Test?",
            "checks": [{"type": "contains", "value": "yes"}],
        }))
        t = load_task(p)
        self.assertEqual(t.id, "json-001")


class TestLoadTasksFromDir(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_loads_multiple_files(self):
        for i in range(3):
            _write_yaml(self.tmpdir, f"t{i}.yaml", f"""
id: task-{i}
prompt: "Q{i}"
checks: []
""")
        tasks = load_tasks_from_dir(self.tmpdir)
        self.assertEqual(len(tasks), 3)

    def test_skips_non_task_files(self):
        _write_yaml(self.tmpdir, "task.yaml", """
id: good
prompt: "Q"
checks: []
""")
        # Suite file — no 'prompt' key
        _write_yaml(self.tmpdir, "suite.yaml", """
id: my-suite
name: Test Suite
tasks: [good]
""")
        tasks = load_tasks_from_dir(self.tmpdir)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].id, "good")

    def test_loads_from_subdirectories(self):
        sub = self.tmpdir / "logic"
        sub.mkdir()
        _write_yaml(sub, "t.yaml", """
id: sub-task
prompt: "Q"
checks: []
""")
        tasks = load_tasks_from_dir(self.tmpdir)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].id, "sub-task")

    def test_empty_dir_returns_empty_list(self):
        tasks = load_tasks_from_dir(self.tmpdir)
        self.assertEqual(tasks, [])

    def test_nonexistent_dir_returns_empty_list(self):
        tasks = load_tasks_from_dir(Path("/nonexistent/path/xyz"))
        self.assertEqual(tasks, [])


class TestSuiteResolution(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.tasks = [
            Task(id="logic-001", complexity="tutorial", modality=["text"],
                 prompt="Q1", checks=[], tags=["logic"]),
            Task(id="logic-002", complexity="extended", modality=["text"],
                 prompt="Q2", checks=[], tags=["logic", "math"]),
            Task(id="extract-001", complexity="tutorial", modality=["text"],
                 prompt="Q3", checks=[], tags=["extraction"]),
        ]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resolve_by_explicit_ids(self):
        _write_yaml(self.tmpdir, "s.yaml", """
id: test-suite
name: Test
tasks: [logic-001, extract-001]
""")
        suite = load_suite(self.tmpdir / "s.yaml")
        resolved = resolve_suite(suite, self.tasks)
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0].id, "logic-001")
        self.assertEqual(resolved[1].id, "extract-001")

    def test_resolve_by_pattern(self):
        _write_yaml(self.tmpdir, "s.yaml", """
id: logic-all
name: Logic
task_pattern: "logic-*"
""")
        suite = load_suite(self.tmpdir / "s.yaml")
        resolved = resolve_suite(suite, self.tasks)
        self.assertEqual(len(resolved), 2)
        ids = [t.id for t in resolved]
        self.assertIn("logic-001", ids)
        self.assertIn("logic-002", ids)

    def test_resolve_by_tags(self):
        _write_yaml(self.tmpdir, "s.yaml", """
id: math
name: Math
tags_filter: [math]
""")
        suite = load_suite(self.tmpdir / "s.yaml")
        resolved = resolve_suite(suite, self.tasks)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].id, "logic-002")

    def test_resolve_empty_suite_returns_all(self):
        _write_yaml(self.tmpdir, "s.yaml", """
id: all
name: All
""")
        suite = load_suite(self.tmpdir / "s.yaml")
        resolved = resolve_suite(suite, self.tasks)
        self.assertEqual(len(resolved), 3)

    def test_resolve_unknown_id_skipped(self):
        _write_yaml(self.tmpdir, "s.yaml", """
id: s
name: S
tasks: [logic-001, nonexistent-999]
""")
        suite = load_suite(self.tmpdir / "s.yaml")
        resolved = resolve_suite(suite, self.tasks)
        self.assertEqual(len(resolved), 1)


class TestFilterByModality(unittest.TestCase):

    def test_text_only_model_skips_image_tasks(self):
        tasks = [
            Task(id="t1", complexity="tutorial", modality=["text"], prompt="Q", checks=[]),
            Task(id="t2", complexity="tutorial", modality=["text", "image"], prompt="Q", checks=[]),
        ]
        filtered = filter_by_modality(tasks, ["text"])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].id, "t1")

    def test_multimodal_model_keeps_all(self):
        tasks = [
            Task(id="t1", complexity="tutorial", modality=["text"], prompt="Q", checks=[]),
            Task(id="t2", complexity="tutorial", modality=["text", "image"], prompt="Q", checks=[]),
        ]
        filtered = filter_by_modality(tasks, ["text", "image"])
        self.assertEqual(len(filtered), 2)

    def test_empty_tasks_returns_empty(self):
        self.assertEqual(filter_by_modality([], ["text"]), [])


if __name__ == "__main__":
    unittest.main()
