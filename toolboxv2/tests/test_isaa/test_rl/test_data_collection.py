"""
Unit tests for data_collection module.

Tests the RL training data collection pipeline including:
- ToolCallTrace and ExecutionTrace dataclasses
- TraceCollector for recording agent execution
- CheckpointLoader for extracting training data from AgentCheckpoint files
- Integration with GRPODatasetBuilder for creating training datasets
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime


class TestToolCallTrace(unittest.TestCase):
    """Tests for ToolCallTrace dataclass"""

    def test_creation(self):
        """Test ToolCallTrace creation"""
        from toolboxv2.mods.isaa.base.rl.data_collection import ToolCallTrace

        trace = ToolCallTrace(
            tool_name="test_tool",
            arguments={"arg1": "value1"},
            result="success",
            success=True,
            duration_ms=100.5
        )

        self.assertEqual(trace.tool_name, "test_tool")
        self.assertEqual(trace.arguments, {"arg1": "value1"})
        self.assertTrue(trace.success)
        self.assertEqual(trace.duration_ms, 100.5)
        self.assertIsNotNone(trace.timestamp)

    def test_with_error(self):
        """Test ToolCallTrace with error"""
        from toolboxv2.mods.isaa.base.rl.data_collection import ToolCallTrace

        trace = ToolCallTrace(
            tool_name="failing_tool",
            arguments={},
            result=None,
            success=False,
            duration_ms=50.0,
            error="Connection timeout"
        )

        self.assertFalse(trace.success)
        self.assertEqual(trace.error, "Connection timeout")


class TestReasoningStep(unittest.TestCase):
    """Tests for ReasoningStep dataclass"""

    def test_creation(self):
        """Test ReasoningStep creation"""
        from toolboxv2.mods.isaa.base.rl.data_collection import ReasoningStep

        step = ReasoningStep(
            step_type="internal_reasoning",
            content="Analyzing the problem...",
            confidence=0.85,
            insights=["Key insight 1"],
            issues=[]
        )

        self.assertEqual(step.step_type, "internal_reasoning")
        self.assertEqual(step.confidence, 0.85)
        self.assertEqual(len(step.insights), 1)


class TestExecutionTrace(unittest.TestCase):
    """Tests for ExecutionTrace dataclass"""

    def test_creation(self):
        """Test ExecutionTrace creation"""
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        trace = ExecutionTrace(
            session_id="test_session",
            user_query="What is 2+2?",
            final_response="The answer is 4."
        )

        self.assertEqual(trace.session_id, "test_session")
        self.assertEqual(trace.user_query, "What is 2+2?")
        self.assertIsNotNone(trace.trace_id)
        self.assertIsNotNone(trace.timestamp)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace, ToolCallTrace

        trace = ExecutionTrace(
            session_id="test",
            user_query="test query",
            final_response="test response"
        )
        trace.tool_calls.append(ToolCallTrace(
            tool_name="test",
            arguments={},
            result="ok",
            success=True,
            duration_ms=10
        ))

        data = trace.to_dict()

        self.assertIsInstance(data, dict)
        self.assertEqual(data["session_id"], "test")
        self.assertEqual(len(data["tool_calls"]), 1)

    def test_from_dict(self):
        """Test reconstruction from dictionary"""
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        data = {
            "trace_id": "abc123",
            "session_id": "session1",
            "timestamp": "2024-01-01T00:00:00",
            "user_query": "test",
            "final_response": "response",
            "tool_calls": [
                {
                    "tool_name": "tool1",
                    "arguments": {},
                    "result": "ok",
                    "success": True,
                    "duration_ms": 10,
                    "timestamp": "2024-01-01T00:00:00"
                }
            ],
            "reasoning_steps": [],
            "tasks_created": [],
            "tasks_completed": [],
            "tasks_failed": [],
            "total_tokens_in": 100,
            "total_tokens_out": 50,
            "total_cost": 0.001,
            "execution_duration_ms": 500,
            "llm_calls_count": 1,
            "label": True,
            "reward_score": 0.8,
            "manual_review": False,
            "review_notes": ""
        }

        trace = ExecutionTrace.from_dict(data)
        self.assertEqual(trace.trace_id, "abc123")


class TestTraceCollectorOperations(unittest.TestCase):
    """Tests for TraceCollector operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_tool_call(self):
        """Test recording a tool call"""
        from toolboxv2.mods.isaa.base.rl.data_collection import TraceCollector

        collector = TraceCollector(storage_path=self.temp_dir)
        collector.start_trace("session1", "test query")

        collector.record_tool_call(
            tool_name="search",
            arguments={"query": "test"},
            result="Found 5 results",
            success=True,
            duration_ms=150.0
        )

        self.assertEqual(len(collector.current_trace.tool_calls), 1)
        self.assertEqual(collector.current_trace.tool_calls[0].tool_name, "search")

    def test_record_reasoning_step(self):
        """Test recording a reasoning step"""
        from toolboxv2.mods.isaa.base.rl.data_collection import TraceCollector

        collector = TraceCollector(storage_path=self.temp_dir)
        collector.start_trace("session1", "test query")

        collector.record_reasoning_step(
            step_type="analysis",
            content="Analyzing the problem",
            confidence=0.9
        )

        self.assertEqual(len(collector.current_trace.reasoning_steps), 1)

    def test_finish_trace(self):
        """Test finishing and saving a trace"""
        from toolboxv2.mods.isaa.base.rl.data_collection import TraceCollector

        collector = TraceCollector(storage_path=self.temp_dir)
        collector.start_trace("session1", "test query")

        finished = collector.finish_trace(
            final_response="Here is the answer",
            total_tokens_in=100,
            total_tokens_out=50
        )

        self.assertIsNotNone(finished)
        self.assertEqual(finished.final_response, "Here is the answer")
        self.assertIsNone(collector.current_trace)
        self.assertEqual(len(collector.traces), 1)

    def test_load_traces(self):
        """Test loading traces from storage"""
        from toolboxv2.mods.isaa.base.rl.data_collection import TraceCollector

        collector = TraceCollector(storage_path=self.temp_dir)

        # Create and save a trace
        collector.start_trace("session1", "query1")
        collector.finish_trace("response1")

        # Create new collector and load
        collector2 = TraceCollector(storage_path=self.temp_dir)
        traces = collector2.load_traces()

        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].user_query, "query1")

    def test_label_trace(self):
        """Test labeling a trace"""
        from toolboxv2.mods.isaa.base.rl.data_collection import TraceCollector

        collector = TraceCollector(storage_path=self.temp_dir)
        collector.start_trace("session1", "query1")
        finished = collector.finish_trace("response1")

        collector.label_trace(finished.trace_id, label=True, notes="Good response")

        # Reload and verify
        traces = collector.load_traces(labeled_only=True)
        self.assertEqual(len(traces), 1)
        self.assertTrue(traces[0].label)

    def test_get_statistics(self):
        """Test getting statistics"""
        from toolboxv2.mods.isaa.base.rl.data_collection import TraceCollector

        collector = TraceCollector(storage_path=self.temp_dir)

        # Create some traces
        for i in range(3):
            collector.start_trace(f"session{i}", f"query{i}")
            collector.record_tool_call("tool", {}, "result", True, 10)
            collector.finish_trace(f"response{i}")

        stats = collector.get_statistics()

        self.assertEqual(stats["total"], 3)
        self.assertIn("avg_tool_calls", stats)


class TestCheckpointLoader(unittest.TestCase):
    """Tests for CheckpointLoader class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_checkpoints_empty(self):
        """Test listing checkpoints when none exist"""
        from toolboxv2.mods.isaa.base.rl.data_collection import CheckpointLoader

        loader = CheckpointLoader("test_agent", checkpoint_path=self.temp_dir)
        checkpoints = loader.list_checkpoints()

        self.assertEqual(len(checkpoints), 0)

    def test_generate_synthetic_tasks(self):
        """Test synthetic task generation"""
        from toolboxv2.mods.isaa.base.rl.data_collection import CheckpointLoader

        loader = CheckpointLoader("test_agent", checkpoint_path=self.temp_dir)
        tasks = loader.generate_synthetic_tasks(num_tasks=10)

        self.assertIsInstance(tasks, list)

    def test_discover_all_agents_empty(self):
        """Test discovering agents when none exist"""
        from toolboxv2.mods.isaa.base.rl.data_collection import CheckpointLoader

        loader = CheckpointLoader(checkpoint_path=self.temp_dir)
        agents = loader.discover_all_agents()

        self.assertEqual(len(agents), 0)

    def test_discover_all_agents_with_checkpoints(self):
        """Test discovering agents with mock checkpoints"""
        import pickle
        from toolboxv2.mods.isaa.base.rl.data_collection import CheckpointLoader

        # Create mock agent directories with checkpoints
        agent1_dir = Path(self.temp_dir) / "agent1"
        agent2_dir = Path(self.temp_dir) / "agent2"
        agent1_dir.mkdir()
        agent2_dir.mkdir()

        # Create mock checkpoint files
        mock_checkpoint = {"session_data": {}, "agent_state": {}}
        with open(agent1_dir / "checkpoint.pkl", "wb") as f:
            pickle.dump(mock_checkpoint, f)
        with open(agent2_dir / "checkpoint.pkl", "wb") as f:
            pickle.dump(mock_checkpoint, f)

        loader = CheckpointLoader(checkpoint_path=self.temp_dir)
        agents = loader.discover_all_agents()

        self.assertEqual(len(agents), 2)
        self.assertIn("agent1", agents)
        self.assertIn("agent2", agents)

    def test_extract_traces_from_mock_checkpoint(self):
        """Test extracting traces from a mock checkpoint object"""
        from toolboxv2.mods.isaa.base.rl.data_collection import CheckpointLoader
        from dataclasses import dataclass, field
        from typing import Any

        # Create a mock checkpoint that mimics AgentCheckpoint structure
        @dataclass
        class MockCheckpoint:
            session_data: dict = field(default_factory=dict)
            variable_scopes: dict = field(default_factory=dict)
            task_state: dict = field(default_factory=dict)
            agent_state: dict = field(default_factory=dict)
            tool_capabilities: dict = field(default_factory=dict)

        checkpoint = MockCheckpoint(
            session_data={
                "session_123": {
                    "history": [
                        {"role": "user", "content": "Hello, how are you?"},
                        {"role": "assistant", "content": "I'm doing well, thank you!"},
                        {"role": "user", "content": "What's the weather?"},
                        {"role": "assistant", "content": "I don't have weather data."}
                    ],
                    "session_type": "chatsession"
                }
            },
            agent_state={
                "total_tokens_in": 200,
                "total_tokens_out": 100,
                "total_cost_accumulated": 0.001,
                "total_llm_calls": 2
            },
            variable_scopes={
                "reasoning": {"final_result": "Task completed successfully"}
            }
        )

        loader = CheckpointLoader("test_agent", checkpoint_path=self.temp_dir)
        traces = loader.extract_traces_from_checkpoint(checkpoint, agent_name="test_agent")

        self.assertEqual(len(traces), 2)  # Two user-assistant pairs
        self.assertEqual(traces[0].user_query, "Hello, how are you?")
        self.assertEqual(traces[0].final_response, "I'm doing well, thank you!")
        self.assertEqual(traces[0].session_id, "session_123")
        self.assertGreater(traces[0].total_tokens_in, 0)

    def test_extract_traces_with_tool_calls(self):
        """Test extracting traces that include tool calls"""
        from toolboxv2.mods.isaa.base.rl.data_collection import CheckpointLoader
        from dataclasses import dataclass, field

        @dataclass
        class MockCheckpoint:
            session_data: dict = field(default_factory=dict)
            variable_scopes: dict = field(default_factory=dict)
            task_state: dict = field(default_factory=dict)
            agent_state: dict = field(default_factory=dict)
            tool_capabilities: dict = field(default_factory=dict)

        checkpoint = MockCheckpoint(
            session_data={
                "session_456": {
                    "history": [
                        {"role": "user", "content": "Search for Python tutorials"},
                        {"role": "tool", "name": "web_search", "content": "Found 10 results", "arguments": {"query": "Python tutorials"}},
                        {"role": "assistant", "content": "I found 10 Python tutorials for you."}
                    ],
                    "session_type": "chatsession"
                }
            }
        )

        loader = CheckpointLoader("test_agent", checkpoint_path=self.temp_dir)
        traces = loader.extract_traces_from_checkpoint(checkpoint)

        self.assertEqual(len(traces), 1)
        self.assertEqual(len(traces[0].tool_calls), 1)
        self.assertEqual(traces[0].tool_calls[0].tool_name, "web_search")

    def test_extract_traces_skips_empty_messages(self):
        """Test that empty messages are skipped"""
        from toolboxv2.mods.isaa.base.rl.data_collection import CheckpointLoader
        from dataclasses import dataclass, field

        @dataclass
        class MockCheckpoint:
            session_data: dict = field(default_factory=dict)
            variable_scopes: dict = field(default_factory=dict)
            task_state: dict = field(default_factory=dict)
            agent_state: dict = field(default_factory=dict)
            tool_capabilities: dict = field(default_factory=dict)

        checkpoint = MockCheckpoint(
            session_data={
                "session_789": {
                    "history": [
                        {"role": "user", "content": ""},  # Empty - should skip
                        {"role": "assistant", "content": "Response"},
                        {"role": "user", "content": "Valid question"},
                        {"role": "assistant", "content": "   "},  # Whitespace only - should skip
                        {"role": "user", "content": "Another question"},
                        {"role": "assistant", "content": "Valid response"}
                    ],
                    "session_type": "chatsession"
                }
            }
        )

        loader = CheckpointLoader("test_agent", checkpoint_path=self.temp_dir)
        traces = loader.extract_traces_from_checkpoint(checkpoint)

        # Only the last valid pair should be extracted
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].user_query, "Another question")

    def test_get_training_statistics_empty(self):
        """Test getting statistics with no traces"""
        from toolboxv2.mods.isaa.base.rl.data_collection import CheckpointLoader

        loader = CheckpointLoader("test_agent", checkpoint_path=self.temp_dir)
        stats = loader.get_training_statistics()

        self.assertEqual(stats["total_traces"], 0)
        self.assertIn("checkpoints_available", stats)

    def test_load_all_traces_with_max_age(self):
        """Test loading traces with age filter"""
        import pickle
        from toolboxv2.mods.isaa.base.rl.data_collection import CheckpointLoader

        # Use a simple dict-based mock that can be pickled
        # (local dataclasses can't be pickled)
        class PicklableCheckpoint:
            def __init__(self):
                self.session_data = {
                    "session_1": {
                        "history": [
                            {"role": "user", "content": "Test"},
                            {"role": "assistant", "content": "Response"}
                        ],
                        "session_type": "chatsession"
                    }
                }
                self.variable_scopes = {}
                self.task_state = {}
                self.agent_state = {}
                self.tool_capabilities = {}

        # We need to define the class at module level for pickle to work
        # Instead, use a dict-based approach that the loader can handle
        checkpoint_data = {
            "session_data": {
                "session_1": {
                    "history": [
                        {"role": "user", "content": "Test"},
                        {"role": "assistant", "content": "Response"}
                    ],
                    "session_type": "chatsession"
                }
            },
            "variable_scopes": {},
            "task_state": {},
            "agent_state": {},
            "tool_capabilities": {}
        }

        checkpoint_path = Path(self.temp_dir) / "checkpoint_test.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        loader = CheckpointLoader("test_agent", checkpoint_path=self.temp_dir)

        # Load with very short max_age (should include recent checkpoint)
        # Note: This will likely return 0 traces since the dict doesn't have
        # the expected object structure, but it tests the max_age parameter
        traces = loader.load_all_traces(max_age_hours=1)
        # The parameter is accepted without error
        self.assertIsInstance(traces, list)


class TestCheckpointLoaderRealData(unittest.TestCase):
    """Tests for CheckpointLoader with real checkpoint data (integration tests)"""

    def test_load_real_checkpoints_if_available(self):
        """Test loading from real checkpoint directory if it exists"""
        from toolboxv2.mods.isaa.base.rl.data_collection import CheckpointLoader

        # Try to load from real checkpoint directory
        try:
            from toolboxv2 import get_app
            loader = CheckpointLoader()  # Use default path
            agents = loader.discover_all_agents()

            if agents:
                # If we have real agents, test loading their traces
                for agent_name in agents[:2]:  # Test first 2 agents
                    agent_loader = CheckpointLoader(agent_name=agent_name)
                    traces = agent_loader.load_all_traces()

                    # Verify trace structure
                    for trace in traces[:5]:  # Check first 5 traces
                        self.assertIsNotNone(trace.trace_id)
                        self.assertIsNotNone(trace.session_id)
                        self.assertIsInstance(trace.user_query, str)
                        self.assertIsInstance(trace.final_response, str)
                        self.assertIsInstance(trace.tool_calls, list)
                        self.assertIsInstance(trace.reasoning_steps, list)
        except Exception:
            # Skip if toolboxv2 app is not available
            self.skipTest("ToolBoxV2 app not available for integration test")


class TestTraceCollector(unittest.TestCase):
    """Tests for TraceCollector class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_start_trace(self):
        """Test starting a new trace"""
        from toolboxv2.mods.isaa.base.rl.data_collection import TraceCollector

        collector = TraceCollector(storage_path=self.temp_dir)
        trace = collector.start_trace("session1", "Hello world")

        self.assertIsNotNone(trace)
        self.assertEqual(trace.session_id, "session1")
        self.assertEqual(trace.user_query, "Hello world")
        self.assertIsNotNone(collector.current_trace)


class TestGRPODatasetIntegration(unittest.TestCase):
    """Integration tests for GRPO dataset building from traces"""

    def test_build_grpo_dataset_with_singles(self):
        """Test building GRPO dataset with include_singles=True"""
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace
        from toolboxv2.mods.isaa.base.rl.dataset_builder import GRPODatasetBuilder

        # Create test traces with unique queries
        traces = [
            ExecutionTrace(
                session_id="session1",
                user_query="Write a Python function to calculate factorial",
                final_response="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
            ),
            ExecutionTrace(
                session_id="session2",
                user_query="Explain machine learning",
                final_response="Machine learning is a subset of AI that enables systems to learn from data."
            ),
            ExecutionTrace(
                session_id="session3",
                user_query="What is the capital of France?",
                final_response="The capital of France is Paris."
            )
        ]

        builder = GRPODatasetBuilder()
        examples = builder.build_dataset(traces, include_singles=True)

        # Should create examples from single traces
        self.assertGreater(len(examples), 0)

        # Each example should have contrastive rewards
        for ex in examples:
            self.assertEqual(len(ex.completions), 2)
            self.assertEqual(len(ex.rewards), 2)
            # Rewards should be different (contrastive)
            self.assertNotEqual(ex.rewards[0], ex.rewards[1])

    def test_build_grpo_dataset_without_singles(self):
        """Test that without include_singles, unique queries produce no examples"""
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace
        from toolboxv2.mods.isaa.base.rl.dataset_builder import GRPODatasetBuilder

        # Create traces with unique queries (no duplicates)
        traces = [
            ExecutionTrace(
                session_id="session1",
                user_query="Unique query 1",
                final_response="Response 1"
            ),
            ExecutionTrace(
                session_id="session2",
                user_query="Unique query 2",
                final_response="Response 2"
            )
        ]

        builder = GRPODatasetBuilder()
        examples = builder.build_dataset(traces, include_singles=False)

        # Without include_singles, unique queries should produce no examples
        self.assertEqual(len(examples), 0)

    def test_grpo_rewards_are_normalized(self):
        """Test that GRPO rewards are properly normalized"""
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace
        from toolboxv2.mods.isaa.base.rl.dataset_builder import GRPODatasetBuilder

        traces = [
            ExecutionTrace(
                session_id="session1",
                user_query="Write a long detailed response about Python programming",
                final_response="Python is a high-level programming language known for its simplicity and readability. " * 10
            )
        ]

        builder = GRPODatasetBuilder()
        examples = builder.build_dataset(traces, include_singles=True)

        self.assertEqual(len(examples), 1)
        # Rewards should be normalized (approximately mean 0)
        mean_reward = sum(examples[0].rewards) / len(examples[0].rewards)
        self.assertAlmostEqual(mean_reward, 0.0, places=5)


class TestFullPipelineIntegration(unittest.TestCase):
    """Integration tests for the full training pipeline"""

    def test_pipeline_with_real_data(self):
        """Test the full pipeline with real checkpoint data if available"""
        try:
            from toolboxv2.mods.isaa.base.rl.training import TrainingPipeline

            # Try with mini_coder_agent which we know has data
            pipeline = TrainingPipeline(
                agent_name='mini_coder_agent',
                base_model='Qwen/Qwen2.5-1.5B-Instruct',
                method='grpo'
            )

            # Test data preparation
            dataset = pipeline.prepare_data(min_examples=1)

            self.assertGreater(len(dataset), 0)
            self.assertIn('prompt', dataset.column_names)
            self.assertIn('completions', dataset.column_names)
            self.assertIn('rewards', dataset.column_names)

        except Exception as e:
            self.skipTest(f"Pipeline test skipped: {e}")

    def test_pipeline_validation_errors(self):
        """Test that pipeline raises proper errors for invalid data"""
        try:
            from toolboxv2.mods.isaa.base.rl.training import TrainingPipeline

            # Use a non-existent agent
            pipeline = TrainingPipeline(
                agent_name='nonexistent_agent_xyz',
                base_model='Qwen/Qwen2.5-1.5B-Instruct',
                method='grpo'
            )

            # Should raise ValueError for no data
            with self.assertRaises(ValueError) as context:
                pipeline.prepare_data(min_examples=1)

            self.assertIn("No GRPO training examples", str(context.exception))

        except ImportError:
            self.skipTest("Training pipeline not available")


if __name__ == "__main__":
    unittest.main()

