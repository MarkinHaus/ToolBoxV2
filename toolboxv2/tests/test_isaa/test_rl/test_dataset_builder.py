"""
Unit tests for dataset_builder module
"""

import unittest
import tempfile
import shutil
from pathlib import Path


class TestKTOExample(unittest.TestCase):
    """Tests for KTOExample dataclass"""

    def test_creation(self):
        """Test KTOExample creation"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import KTOExample

        example = KTOExample(
            prompt="What is 2+2?",
            completion="The answer is 4.",
            label=True
        )

        self.assertEqual(example.prompt, "What is 2+2?")
        self.assertEqual(example.completion, "The answer is 4.")
        self.assertTrue(example.label)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import KTOExample

        example = KTOExample(prompt="test", completion="response", label=False)
        data = example.to_dict()

        self.assertEqual(data["prompt"], "test")
        self.assertEqual(data["completion"], "response")
        self.assertFalse(data["label"])


class TestGRPOExample(unittest.TestCase):
    """Tests for GRPOExample dataclass"""

    def test_creation(self):
        """Test GRPOExample creation"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import GRPOExample

        example = GRPOExample(
            prompt="What is 2+2?",
            completions=["4", "Four", "2+2=4"],
            rewards=[0.9, 0.8, 0.7]
        )

        self.assertEqual(len(example.completions), 3)
        self.assertEqual(len(example.rewards), 3)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import GRPOExample

        example = GRPOExample(
            prompt="test",
            completions=["a", "b"],
            rewards=[0.5, 0.5]
        )
        data = example.to_dict()

        self.assertEqual(data["prompt"], "test")
        self.assertEqual(len(data["completions"]), 2)


class TestKTODatasetBuilder(unittest.TestCase):
    """Tests for KTODatasetBuilder"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trace_to_example(self):
        """Test converting trace to KTO example"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import KTODatasetBuilder
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        builder = KTODatasetBuilder()
        trace = ExecutionTrace(
            user_query="Hello",
            final_response="Hi there!",
            label=True
        )

        example = builder.trace_to_example(trace)

        self.assertIn("Hello", example.prompt)
        self.assertEqual(example.completion, "Hi there!")
        self.assertTrue(example.label)

    def test_build_dataset(self):
        """Test building dataset from traces"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import KTODatasetBuilder
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        builder = KTODatasetBuilder()
        traces = [
            ExecutionTrace(user_query=f"Query {i}", final_response=f"Response {i}", label=i % 2 == 0)
            for i in range(10)
        ]

        examples = builder.build_dataset(traces, balance=False)

        self.assertEqual(len(examples), 10)

    def test_save_and_load_dataset(self):
        """Test saving and loading dataset"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import KTODatasetBuilder, KTOExample

        builder = KTODatasetBuilder()
        examples = [
            KTOExample(prompt=f"P{i}", completion=f"C{i}", label=True)
            for i in range(5)
        ]

        output_path = Path(self.temp_dir) / "test.jsonl"
        builder.save_dataset(examples, str(output_path))

        loaded = builder.load_dataset(str(output_path))

        self.assertEqual(len(loaded), 5)
        self.assertEqual(loaded[0].prompt, "P0")

    def test_get_statistics(self):
        """Test getting dataset statistics"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import KTODatasetBuilder, KTOExample

        builder = KTODatasetBuilder()


class TestGRPODatasetBuilder(unittest.TestCase):
    """Tests for GRPODatasetBuilder"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_group_traces_by_query(self):
        """Test grouping traces by query"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import GRPODatasetBuilder
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        builder = GRPODatasetBuilder()
        traces = [
            ExecutionTrace(user_query="Hello", final_response="Hi 1"),
            ExecutionTrace(user_query="Hello", final_response="Hi 2"),
            ExecutionTrace(user_query="Goodbye", final_response="Bye"),
        ]

        groups = builder.group_traces_by_query(traces)

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups["hello"]), 2)

    def test_build_example_from_group(self):
        """Test building GRPO example from group"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import GRPODatasetBuilder
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        builder = GRPODatasetBuilder()
        traces = [
            ExecutionTrace(user_query="Hello", final_response="Hi 1"),
            ExecutionTrace(user_query="Hello", final_response="Hi 2"),
        ]

        example = builder.build_example_from_group("Hello", traces)

        self.assertIsNotNone(example)
        self.assertEqual(len(example.completions), 2)
        self.assertEqual(len(example.rewards), 2)

    def test_save_and_load_dataset(self):
        """Test saving and loading GRPO dataset"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import GRPODatasetBuilder, GRPOExample

        builder = GRPODatasetBuilder()
        examples = [
            GRPOExample(prompt="P1", completions=["C1", "C2"], rewards=[0.5, 0.5]),
        ]

        output_path = Path(self.temp_dir) / "grpo.jsonl"
        builder.save_dataset(examples, str(output_path))

        loaded = builder.load_dataset(str(output_path))

        self.assertEqual(len(loaded), 1)
        self.assertEqual(len(loaded[0].completions), 2)

    def test_get_statistics(self):
        """Test getting GRPO dataset statistics"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import GRPODatasetBuilder, GRPOExample

        builder = GRPODatasetBuilder()
        examples = [
            GRPOExample(prompt="P1", completions=["C1", "C2", "C3"], rewards=[0.3, 0.5, 0.7]),
            GRPOExample(prompt="P2", completions=["C1", "C2"], rewards=[0.4, 0.6]),
        ]

        stats = builder.get_statistics(examples)

        self.assertEqual(stats["total_examples"], 2)
        self.assertEqual(stats["total_completions"], 5)


class TestDatasetPipeline(unittest.TestCase):
    """Tests for DatasetPipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test pipeline initialization"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import DatasetPipeline

        pipeline = DatasetPipeline(
            agent_name="test_agent",
            storage_path=self.temp_dir
        )

        self.assertIsNotNone(pipeline.trace_collector)
        self.assertIsNotNone(pipeline.kto_builder)
        self.assertIsNotNone(pipeline.grpo_builder)

    def test_get_pipeline_statistics(self):
        """Test getting pipeline statistics"""
        from toolboxv2.mods.isaa.base.rl.dataset_builder import DatasetPipeline

        pipeline = DatasetPipeline(
            agent_name="test_agent",
            storage_path=self.temp_dir
        )

        stats = pipeline.get_pipeline_statistics()

        self.assertIn("collector", stats)
        self.assertIn("checkpoints", stats)


if __name__ == "__main__":
    unittest.main()
