"""
Unit tests for reward_functions module
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os


class TestRewardResult(unittest.TestCase):
    """Tests for RewardResult dataclass"""

    def test_creation(self):
        """Test RewardResult creation"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import RewardResult

        result = RewardResult(score=0.8, is_binary=True, details={"key": "value"})

        self.assertEqual(result.score, 0.8)
        self.assertTrue(result.is_binary)
        self.assertEqual(result.details["key"], "value")

    def test_to_binary(self):
        """Test binary conversion"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import RewardResult

        high_score = RewardResult(score=0.7, is_binary=False)
        low_score = RewardResult(score=0.3, is_binary=False)

        self.assertEqual(high_score.to_binary(threshold=0.5), 1)
        self.assertEqual(low_score.to_binary(threshold=0.5), 0)


class TestSyntaxValidationReward(unittest.TestCase):
    """Tests for SyntaxValidationReward"""

    def test_valid_python_code(self):
        """Test reward for valid Python code"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import SyntaxValidationReward
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        reward = SyntaxValidationReward()
        trace = ExecutionTrace(
            user_query="Write a function",
            final_response='```python\ndef hello():\n    return "world"\n```'
        )

        result = reward.compute(trace)

        self.assertEqual(result.score, 1.0)

    def test_invalid_python_code(self):
        """Test reward for invalid Python code"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import SyntaxValidationReward
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        reward = SyntaxValidationReward()
        trace = ExecutionTrace(
            user_query="Write a function",
            final_response='```python\ndef hello(\n    return "world"\n```'
        )

        result = reward.compute(trace)

        self.assertLess(result.score, 1.0)

    def test_no_code(self):
        """Test reward when no code is present"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import SyntaxValidationReward
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        reward = SyntaxValidationReward()
        trace = ExecutionTrace(
            user_query="Hello",
            final_response="Hello! How can I help you?"
        )

        result = reward.compute(trace)

        self.assertEqual(result.score, 0.5)  # Neutral score


class TestToolSuccessReward(unittest.TestCase):
    """Tests for ToolSuccessReward"""

    def test_all_tools_successful(self):
        """Test reward when all tools succeed"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import ToolSuccessReward
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace, ToolCallTrace

        reward = ToolSuccessReward()
        trace = ExecutionTrace(user_query="Search for something")
        trace.tool_calls = [
            ToolCallTrace("search", {}, "result1", True, 100),
            ToolCallTrace("search", {}, "result2", True, 100),
        ]

        result = reward.compute(trace)

        self.assertGreaterEqual(result.score, 1.0)

    def test_some_tools_failed(self):
        """Test reward when some tools fail"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import ToolSuccessReward
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace, ToolCallTrace

        reward = ToolSuccessReward()
        trace = ExecutionTrace(user_query="Search for something")
        trace.tool_calls = [
            ToolCallTrace("search", {}, "result1", True, 100),
            ToolCallTrace("search", {}, None, False, 100, error="Failed"),
        ]

        result = reward.compute(trace)

        self.assertLess(result.score, 1.0)
        self.assertGreater(result.score, 0.0)

    def test_no_tools_used(self):
        """Test reward when no tools are used"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import ToolSuccessReward
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        reward = ToolSuccessReward()
        trace = ExecutionTrace(user_query="Hello")

        result = reward.compute(trace)

        self.assertEqual(result.score, 0.5)  # Neutral


class TestTaskCompletionReward(unittest.TestCase):
    """Tests for TaskCompletionReward"""

    def test_all_tasks_completed(self):
        """Test reward when all tasks are completed"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import TaskCompletionReward
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        reward = TaskCompletionReward()
        trace = ExecutionTrace(user_query="Do something")
        trace.tasks_created = [{"id": "1"}, {"id": "2"}]
        trace.tasks_completed = [{"id": "1"}, {"id": "2"}]

        result = reward.compute(trace)



class TestEfficiencyReward(unittest.TestCase):
    """Tests for EfficiencyReward"""

    def test_efficient_response(self):
        """Test reward for efficient response"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import EfficiencyReward
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        reward = EfficiencyReward(max_tokens=2000, max_tool_calls=10)
        trace = ExecutionTrace(user_query="Quick question")
        trace.total_tokens_in = 100
        trace.total_tokens_out = 50

        result = reward.compute(trace)

        self.assertGreater(result.score, 0.5)

    def test_inefficient_response(self):
        """Test reward for inefficient response"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import EfficiencyReward
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace, ToolCallTrace

        reward = EfficiencyReward(max_tokens=2000, max_tool_calls=10)
        trace = ExecutionTrace(user_query="Question")
        trace.total_tokens_in = 1500
        trace.total_tokens_out = 1000
        trace.tool_calls = [ToolCallTrace("t", {}, "r", True, 10) for _ in range(8)]

        result = reward.compute(trace)

        self.assertLess(result.score, 0.5)


class TestFormatComplianceReward(unittest.TestCase):
    """Tests for FormatComplianceReward"""

    def test_clean_response(self):
        """Test reward for clean response without XML"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import FormatComplianceReward
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        reward = FormatComplianceReward()
        trace = ExecutionTrace(
            user_query="Hello",
            final_response="Hello! How can I help you today?"
        )

        result = reward.compute(trace)

        self.assertEqual(result.score, 1.0)

    def test_response_with_xml(self):
        """Test reward for response with XML tags"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import FormatComplianceReward
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        reward = FormatComplianceReward()
        trace = ExecutionTrace(
            user_query="Hello",
            final_response="<response>Hello!</response>"
        )

        result = reward.compute(trace)

        self.assertLess(result.score, 1.0)


class TestRewardEngine(unittest.TestCase):
    """Tests for RewardEngine"""

    def test_default_rewards(self):
        """Test engine with default rewards"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import RewardEngine

        engine = RewardEngine()

        self.assertGreater(len(engine.rewards), 0)

    def test_compute_all(self):
        """Test computing all rewards"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import RewardEngine
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        engine = RewardEngine()
        trace = ExecutionTrace(
            user_query="Hello",
            final_response="Hello! How can I help?"
        )

        results = engine.compute_all(trace)

        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)

    def test_compute_combined(self):
        """Test computing combined reward"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import RewardEngine
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        engine = RewardEngine()
        trace = ExecutionTrace(
            user_query="Hello",
            final_response="Hello! How can I help?"
        )

        combined = engine.compute_combined(trace)

        self.assertIsInstance(combined, float)
        self.assertGreaterEqual(combined, 0.0)
        self.assertLessEqual(combined, 1.0)

    def test_get_binary_label(self):
        """Test getting binary label"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import RewardEngine
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        engine = RewardEngine()
        trace = ExecutionTrace(
            user_query="Hello",
            final_response="Hello! How can I help?"
        )

        label = engine.get_binary_label(trace, threshold=0.3)

        self.assertIsInstance(label, bool)

    def test_summary(self):
        """Test summary generation"""
        from toolboxv2.mods.isaa.base.rl.reward_functions import RewardEngine
        from toolboxv2.mods.isaa.base.rl.data_collection import ExecutionTrace

        engine = RewardEngine()
        trace = ExecutionTrace(
            user_query="Hello",
            final_response="Hello!"
        )

        summary = engine.summary(trace)

        self.assertIn("Reward Summary", summary)
        self.assertIn("Combined", summary)


if __name__ == "__main__":
    unittest.main()

