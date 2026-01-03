"""
Unit tests for data_collection module
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


if __name__ == "__main__":
    unittest.main()


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

