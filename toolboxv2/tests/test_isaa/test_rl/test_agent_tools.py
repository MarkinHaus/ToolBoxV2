"""
Unit tests for agent_tools module
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import shutil
import asyncio
from pathlib import Path


class TestTrainingState(unittest.TestCase):
    """Tests for TrainingState enum"""

    def test_state_values(self):
        """Test that all expected states exist"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import TrainingState

        self.assertEqual(TrainingState.IDLE.value, "idle")
        self.assertEqual(TrainingState.STARTING.value, "starting")
        self.assertEqual(TrainingState.RUNNING.value, "running")
        self.assertEqual(TrainingState.STOPPING.value, "stopping")
        self.assertEqual(TrainingState.COMPLETED.value, "completed")
        self.assertEqual(TrainingState.FAILED.value, "failed")


class TestTrainingSession(unittest.TestCase):
    """Tests for TrainingSession dataclass"""

    def test_creation(self):
        """Test TrainingSession creation"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import TrainingSession, TrainingState

        session = TrainingSession(
            session_id="test_123",
            model_name="my-model",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            method="grpo"
        )

        self.assertEqual(session.session_id, "test_123")
        self.assertEqual(session.model_name, "my-model")
        self.assertEqual(session.state, TrainingState.IDLE)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import TrainingSession, TrainingState

        session = TrainingSession(
            session_id="test_123",
            model_name="my-model",
            base_model="test",
            method="kto",
            state=TrainingState.RUNNING
        )

        data = session.to_dict()

        self.assertEqual(data["session_id"], "test_123")
        self.assertEqual(data["state"], "running")


class TestRLTrainingManager(unittest.TestCase):
    """Tests for RLTrainingManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        # Reset singleton
        from toolboxv2.mods.isaa.base.rl.agent_tools import RLTrainingManager
        RLTrainingManager._instance = None

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        from toolboxv2.mods.isaa.base.rl.agent_tools import RLTrainingManager
        RLTrainingManager._instance = None

    def test_singleton(self):
        """Test singleton pattern"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import RLTrainingManager

        manager1 = RLTrainingManager()
        manager2 = RLTrainingManager()

        self.assertIs(manager1, manager2)

    def test_generate_session_id(self):
        """Test session ID generation"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import RLTrainingManager

        manager = RLTrainingManager()
        manager.storage_path = Path(self.temp_dir)

        session_id = manager._generate_session_id()

        self.assertTrue(session_id.startswith("train_"))
        self.assertGreater(len(session_id), 20)

    def test_check_training_status_no_session(self):
        """Test status check with no session"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import RLTrainingManager

        manager = RLTrainingManager()
        manager.storage_path = Path(self.temp_dir)

        status = manager.check_training_status()

        self.assertFalse(status["has_active_session"])

    def test_list_models_empty(self):
        """Test listing models when none exist"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import RLTrainingManager

        manager = RLTrainingManager()
        manager.storage_path = Path(self.temp_dir)

        result = manager.list_models()

        self.assertEqual(result["total"], 0)
        self.assertEqual(len(result["models"]), 0)

    def test_switch_model_not_found(self):
        """Test switching to non-existent model"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import RLTrainingManager

        manager = RLTrainingManager()
        manager.storage_path = Path(self.temp_dir)

        result = manager.switch_model("nonexistent")

        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])

    @patch('toolboxv2.mods.isaa.base.rl.agent_tools.RLTrainingManager._run_training')
    def test_start_training(self, mock_run):
        """Test starting training"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import RLTrainingManager

        manager = RLTrainingManager()
        manager.storage_path = Path(self.temp_dir)

        result = manager.start_training(
            model_name="test-model",
            base_model="test-base",
            method="grpo"
        )

        self.assertTrue(result["success"])
        self.assertIn("session_id", result)

    @patch('toolboxv2.mods.isaa.base.rl.agent_tools.RLTrainingManager._run_training')
    def test_start_training_already_running(self, mock_run):
        """Test starting training when already running"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import RLTrainingManager, TrainingState

        manager = RLTrainingManager()
        manager.storage_path = Path(self.temp_dir)

        # Start first training
        manager.start_training(model_name="model1", base_model="test", method="grpo")
        manager.current_session.state = TrainingState.RUNNING

        # Try to start second
        result = manager.start_training(model_name="model2", base_model="test", method="grpo")

        self.assertFalse(result["success"])
        self.assertIn("already in progress", result["error"])


class TestAgentToolFunctions(unittest.TestCase):
    """Tests for agent tool functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        from toolboxv2.mods.isaa.base.rl.agent_tools import RLTrainingManager
        RLTrainingManager._instance = None

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        from toolboxv2.mods.isaa.base.rl.agent_tools import RLTrainingManager, _manager
        RLTrainingManager._instance = None

    def test_check_training_status_async(self):
        """Test async check_training_status function"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import check_training_status

        result = asyncio.run(check_training_status())

        self.assertIsInstance(result, str)
        self.assertIn("No training sessions", result)

    def test_list_rl_models_async(self):
        """Test async list_rl_models function"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import list_rl_models

        result = asyncio.run(list_rl_models())

        self.assertIsInstance(result, str)
        self.assertIn("No trained models", result)

    @patch('toolboxv2.mods.isaa.base.rl.agent_tools.RLTrainingManager._run_training')
    def test_start_rl_training_async(self, mock_run):
        """Test async start_rl_training function"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import start_rl_training

        result = asyncio.run(start_rl_training(
            model_name="test-model",
            base_model="test-base"
        ))

        self.assertIsInstance(result, str)
        self.assertIn("Training started", result)

    def test_switch_rl_model_async(self):
        """Test async switch_rl_model function"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import switch_rl_model

        result = asyncio.run(switch_rl_model("nonexistent"))

        self.assertIsInstance(result, str)
        self.assertIn("not found", result)


class TestGetRLTrainingTools(unittest.TestCase):
    """Tests for get_rl_training_tools function"""

    def test_returns_list(self):
        """Test that function returns a list of tools"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import get_rl_training_tools

        tools = get_rl_training_tools()

        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 5)

    def test_tool_structure(self):
        """Test that each tool has correct structure"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import get_rl_training_tools

        tools = get_rl_training_tools()

        for func, name, description in tools:
            self.assertTrue(callable(func))
            self.assertIsInstance(name, str)
            self.assertIsInstance(description, str)

    def test_tool_names(self):
        """Test that expected tool names are present"""
        from toolboxv2.mods.isaa.base.rl.agent_tools import get_rl_training_tools

        tools = get_rl_training_tools()
        names = [name for _, name, _ in tools]

        self.assertIn("start_rl_training", names)
        self.assertIn("stop_rl_training", names)
        self.assertIn("check_training_status", names)
        self.assertIn("switch_rl_model", names)
        self.assertIn("list_rl_models", names)


if __name__ == "__main__":
    unittest.main()
