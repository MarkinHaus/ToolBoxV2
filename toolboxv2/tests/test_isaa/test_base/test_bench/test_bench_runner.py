"""Unit tests for bench.runner — task execution, circuit breaker, batching."""

import unittest

from toolboxv2.mods.isaa.base.bench.core import Check, Task
from toolboxv2.mods.isaa.base.bench.runner import run_task, run_tasks


def make_task(task_id="t1", checks=None, **kw):
    defaults = {
        "id": task_id,
        "complexity": "tutorial",
        "modality": ["text"],
        "prompt": "What is 2+2?",
        "checks": checks or [Check(type="contains", params={"value": "4"})],
        "tags": ["logic"],
    }
    defaults.update(kw)
    return Task(**defaults)


class TestRunTask(unittest.IsolatedAsyncioTestCase):

    async def test_correct_answer_passes_check(self):
        async def model_fn(p):
            return "The answer is 4.", {"tokens_in": 5, "tokens_out": 5, "total_cost": 0.001}

        result = await run_task(make_task(), model_fn)
        self.assertEqual(result.task_id, "t1")
        self.assertEqual(len(result.checks), 1)
        self.assertTrue(result.checks[0].passed)
        self.assertAlmostEqual(result.score, 1.0)
        self.assertEqual(result.tokens_in, 5)
        self.assertEqual(result.cost, 0.001)

    async def test_wrong_answer_fails_check(self):
        async def model_fn(p):
            return "The answer is 5."

        result = await run_task(make_task(), model_fn)
        self.assertFalse(result.checks[0].passed)
        self.assertAlmostEqual(result.score, 0.0)

    async def test_model_returns_tuple_without_cost(self):
        async def model_fn(p):
            return "4"

        result = await run_task(make_task(), model_fn)
        self.assertTrue(result.checks[0].passed)
        self.assertEqual(result.tokens_in, 0)

    async def test_model_exception_returns_error_string(self):
        async def model_fn(p):
            raise ValueError("boom")

        result = await run_task(make_task(), model_fn)
        self.assertIn("Error:", result.response)
        self.assertIn("ValueError", result.response)

    async def test_model_timeout_returns_error(self):
        import asyncio

        async def model_fn(p):
            await asyncio.sleep(10)
            return "never"

        result = await run_task(make_task(), model_fn, timeout=0.1)
        self.assertIn("Error:", result.response)
        self.assertIn("timeout", result.response)

    async def test_sync_model_function(self):
        def model_fn(p):
            return "4", {"tokens_in": 1, "tokens_out": 1, "total_cost": 0}

        result = await run_task(make_task(), model_fn)
        self.assertTrue(result.checks[0].passed)

    async def test_multiple_checks(self):
        task = make_task(checks=[
            Check(type="contains", params={"value": "4"}),
            Check(type="not_contains", params={"value": "5"}),
            Check(type="char_count_lte", params={"value": 50}),
        ])

        async def model_fn(p):
            return "4"

        result = await run_task(task, model_fn)
        self.assertEqual(len(result.checks), 3)
        self.assertTrue(all(c.passed for c in result.checks))
        self.assertAlmostEqual(result.score, 1.0)

    async def test_latency_recorded(self):
        import asyncio

        async def model_fn(p):
            await asyncio.sleep(0.01)  # ensure measurable latency
            return "4"

        result = await run_task(make_task(), model_fn)
        self.assertGreaterEqual(result.latency_ms, 1)

    async def test_attachments_injected_into_prompt(self):
        from toolboxv2.mods.isaa.base.bench.core import Attachment

        task = make_task(
            prompt="Describe this:",
            checks=[Check(type="contains", params={"value": "ok"})],
        )
        task.attachments = [Attachment(type="image", path="/tmp/test.png")]

        received_prompt = None

        async def model_fn(p):
            nonlocal received_prompt
            received_prompt = p
            return "ok"

        await run_task(task, model_fn)
        self.assertIn("[media:/tmp/test.png]", received_prompt)


class TestRunTasks(unittest.IsolatedAsyncioTestCase):

    async def test_sequential_execution(self):
        tasks = [make_task(task_id=f"t{i}") for i in range(3)]
        call_count = 0

        async def model_fn(p):
            nonlocal call_count
            call_count += 1
            return "4"

        report = await run_tasks(tasks, model_fn, model_id="test", suite_id="s")
        self.assertEqual(len(report.results), 3)
        self.assertEqual(call_count, 3)
        self.assertEqual(report.model_id, "test")
        self.assertEqual(report.suite_id, "s")

    async def test_circuit_breaker_aborts_after_consecutive_errors(self):
        tasks = [make_task(task_id=f"t{i}") for i in range(10)]

        async def model_fn(p):
            raise RuntimeError("dead")

        report = await run_tasks(
            tasks, model_fn, max_consecutive_errors=3
        )
        # Should abort after 3 consecutive errors, not run all 10
        self.assertLess(len(report.results), 10)
        self.assertTrue(report.metadata.get("aborted", False))

    async def test_circuit_breaker_resets_on_success(self):
        tasks = [make_task(task_id=f"t{i}") for i in range(6)]
        call_idx = 0

        async def model_fn(p):
            nonlocal call_idx
            call_idx += 1
            # Fail twice, succeed, fail twice, succeed — never hit 3 consecutive
            if call_idx in (1, 2, 4, 5):
                return "Error: simulated", {}
            return "4", {}

        report = await run_tasks(
            tasks, model_fn, max_consecutive_errors=3
        )
        self.assertEqual(len(report.results), 6)
        self.assertFalse(report.metadata.get("aborted", False))

    async def test_cost_accumulation(self):
        tasks = [make_task(task_id=f"t{i}") for i in range(3)]

        async def model_fn(p):
            return "4", {"tokens_in": 10, "tokens_out": 5, "total_cost": 0.01}

        report = await run_tasks(tasks, model_fn)
        self.assertEqual(report.total_tokens_in, 30)
        self.assertEqual(report.total_tokens_out, 15)
        self.assertAlmostEqual(report.total_cost, 0.03)

    async def test_total_time_recorded(self):
        tasks = [make_task()]

        async def model_fn(p):
            return "4"

        report = await run_tasks(tasks, model_fn)
        self.assertGreater(report.total_time_s, 0)

    async def test_batch_execution(self):
        tasks = [make_task(task_id=f"t{i}") for i in range(4)]

        async def model_fn(p):
            return "4"

        report = await run_tasks(tasks, model_fn, batch_size=2)
        self.assertEqual(len(report.results), 4)

    async def test_batch_circuit_breaker(self):
        tasks = [make_task(task_id=f"t{i}") for i in range(10)]

        async def model_fn(p):
            raise RuntimeError("dead")

        report = await run_tasks(
            tasks, model_fn, batch_size=2, max_consecutive_errors=3
        )
        self.assertLess(len(report.results), 10)
        self.assertTrue(report.metadata.get("aborted", False))


if __name__ == "__main__":
    unittest.main()
