"""
Unit tests for training module
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path


class TestTrainingConfig(unittest.TestCase):
    """Tests for TrainingConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        from toolboxv2.mods.isaa.base.rl.training import TrainingConfig

        config = TrainingConfig()

        self.assertEqual(config.num_epochs, 3)
        self.assertEqual(config.learning_rate, 5e-5)
        self.assertEqual(config.per_device_batch_size, 1)
        self.assertTrue(config.gradient_checkpointing)

    def test_custom_values(self):
        """Test custom configuration values"""
        from toolboxv2.mods.isaa.base.rl.training import TrainingConfig

        config = TrainingConfig(
            num_epochs=5,
            learning_rate=1e-4,
            per_device_batch_size=4,
            lora_r=32
        )

        self.assertEqual(config.num_epochs, 5)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.per_device_batch_size, 4)
        self.assertEqual(config.lora_r, 32)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        from toolboxv2.mods.isaa.base.rl.training import TrainingConfig

        config = TrainingConfig(
            num_epochs=3,
            per_device_batch_size=2,
            learning_rate=1e-4
        )

        data = config.to_dict()

        self.assertEqual(data["num_epochs"], 3)
        self.assertEqual(data["per_device_batch_size"], 2)
        self.assertEqual(data["learning_rate"], 1e-4)

    def test_lora_settings(self):
        """Test LoRA settings"""
        from toolboxv2.mods.isaa.base.rl.training import TrainingConfig

        config = TrainingConfig(lora_r=16, lora_alpha=32)
        data = config.to_dict()

        self.assertEqual(data["lora_r"], 16)
        self.assertEqual(data["lora_alpha"], 32)


class TestRLTrainer(unittest.TestCase):
    """Tests for RLTrainer"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test trainer initialization"""
        from toolboxv2.mods.isaa.base.rl.training import RLTrainer, TrainingConfig

        config = TrainingConfig(output_dir=self.temp_dir)
        trainer = RLTrainer(config)

        self.assertEqual(trainer.config, config)
        self.assertIsNone(trainer.model)
        self.assertIsNone(trainer.tokenizer)

    def test_config_saved(self):
        """Test that config is saved on initialization"""
        from toolboxv2.mods.isaa.base.rl.training import RLTrainer, TrainingConfig
        import os

        config = TrainingConfig(output_dir=self.temp_dir)
        trainer = RLTrainer(config)

        config_path = os.path.join(self.temp_dir, "training_config.json")
        self.assertTrue(os.path.exists(config_path))


class TestTrainingPipeline(unittest.TestCase):
    """Tests for TrainingPipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test pipeline initialization"""
        from toolboxv2.mods.isaa.base.rl.training import TrainingPipeline

        pipeline = TrainingPipeline(
            agent_name="test_agent",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            output_dir=self.temp_dir,
            method="grpo"
        )

        self.assertEqual(pipeline.agent_name, "test_agent")
        self.assertEqual(pipeline.base_model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(pipeline.method, "grpo")

    def test_initialization_kto(self):
        """Test pipeline initialization with KTO method"""
        from toolboxv2.mods.isaa.base.rl.training import TrainingPipeline

        pipeline = TrainingPipeline(
            agent_name="test_agent",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            output_dir=self.temp_dir,
            method="kto"
        )

        self.assertEqual(pipeline.method, "kto")

    def test_output_dir_created(self):
        """Test that output directory is created"""
        from toolboxv2.mods.isaa.base.rl.training import TrainingPipeline

        pipeline = TrainingPipeline(
            agent_name="test_agent",
            base_model="test",
            output_dir=self.temp_dir,
            method="grpo"
        )

        self.assertTrue(pipeline.output_dir.exists())

    def test_training_config_created(self):
        """Test that training config is created"""
        from toolboxv2.mods.isaa.base.rl.training import TrainingPipeline

        pipeline = TrainingPipeline(
            agent_name="test_agent",
            base_model="test",
            output_dir=self.temp_dir,
            method="grpo"
        )

        self.assertIsNotNone(pipeline.training_config)
        self.assertEqual(pipeline.training_config.method, "grpo")


if __name__ == "__main__":
    unittest.main()
