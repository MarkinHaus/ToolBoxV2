"""
Unit tests for hardware_config module
"""

import unittest
from unittest.mock import patch, MagicMock
import os


class TestHardwareProfile(unittest.TestCase):
    """Tests for HardwareProfile enum"""

    def test_hardware_profile_values(self):
        """Test that all expected profiles exist"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareProfile

        self.assertEqual(HardwareProfile.RYZEN_OPTIMIZED.value, "ryzen_optimized")
        self.assertEqual(HardwareProfile.AUTO_DETECT.value, "auto_detect")
        self.assertEqual(HardwareProfile.GPU_ENABLED.value, "gpu_enabled")
        self.assertEqual(HardwareProfile.CPU_ONLY.value, "cpu_only")


class TestHardwareConfig(unittest.TestCase):
    """Tests for HardwareConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareConfig, HardwareProfile

        config = HardwareConfig()

        self.assertEqual(config.cpu_cores, 1)
        self.assertEqual(config.cpu_threads, 1)
        self.assertFalse(config.has_gpu)
        self.assertEqual(config.ram_gb, 8.0)
        self.assertEqual(config.profile, HardwareProfile.AUTO_DETECT)

    def test_optimization_for_high_ram(self):
        """Test optimization for high RAM systems"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareConfig

        config = HardwareConfig(ram_gb=64.0, available_ram_gb=60.0)

        self.assertEqual(config.recommended_batch_size, 4)
        self.assertEqual(config.recommended_model_size, "3B")
        self.assertEqual(config.lora_r, 16)
        self.assertEqual(config.num_generations, 8)

    def test_optimization_for_medium_ram(self):
        """Test optimization for medium RAM systems"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareConfig

        config = HardwareConfig(ram_gb=32.0, available_ram_gb=28.0)

        self.assertEqual(config.recommended_batch_size, 2)
        self.assertEqual(config.recommended_model_size, "1.5B")

    def test_optimization_for_low_ram(self):
        """Test optimization for low RAM systems"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareConfig

        config = HardwareConfig(ram_gb=8.0, available_ram_gb=6.0)

        self.assertEqual(config.recommended_batch_size, 1)
        self.assertEqual(config.lora_r, 4)

    def test_ryzen_optimization(self):
        """Test Ryzen-specific optimizations"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareConfig, HardwareProfile

        config = HardwareConfig(
            cpu_name="AMD Ryzen 9 5950X",
            cpu_cores=16,
            cpu_threads=32,
            ram_gb=40.0
        )

        self.assertEqual(config.profile, HardwareProfile.RYZEN_OPTIMIZED)
        self.assertEqual(config.num_workers, 8)

    def test_get_training_device_cpu(self):
        """Test training device selection for CPU"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareConfig

        config = HardwareConfig(has_gpu=False)
        self.assertEqual(config.get_training_device(), "cpu")

    def test_get_training_device_gpu(self):
        """Test training device selection for GPU"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareConfig

        config = HardwareConfig(has_gpu=True, cuda_available=True)
        self.assertEqual(config.get_training_device(), "cuda")

    def test_get_torch_dtype(self):
        """Test torch dtype selection"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareConfig

        config_fp32 = HardwareConfig()
        self.assertEqual(config_fp32.get_torch_dtype(), "float32")

        config_fp16 = HardwareConfig(use_fp16=True)
        self.assertEqual(config_fp16.get_torch_dtype(), "float16")

        config_bf16 = HardwareConfig(use_bf16=True)
        self.assertEqual(config_bf16.get_torch_dtype(), "bfloat16")

    def test_to_training_args(self):
        """Test conversion to training arguments"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareConfig

        config = HardwareConfig(
            recommended_batch_size=2,
            gradient_checkpointing=True,
            num_workers=4
        )

        args = config.to_training_args()

        # Check that the method returns a dict with expected keys
        self.assertIn("per_device_train_batch_size", args)
        self.assertIn("gradient_checkpointing", args)
        self.assertIn("dataloader_num_workers", args)

    def test_to_lora_config(self):
        """Test conversion to LoRA config"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareConfig

        config = HardwareConfig(lora_r=16, lora_alpha=32)
        lora = config.to_lora_config()

        # Check that the method returns a dict with expected keys
        self.assertIn("r", lora)
        self.assertIn("lora_alpha", lora)
        self.assertEqual(lora["task_type"], "CAUSAL_LM")

    def test_summary(self):
        """Test summary generation"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import HardwareConfig

        config = HardwareConfig(cpu_name="Test CPU", ram_gb=16.0)
        summary = config.summary()

        self.assertIn("Hardware Configuration Summary", summary)
        self.assertIn("Test CPU", summary)
        self.assertIn("16.0 GB", summary)


class TestDetectHardware(unittest.TestCase):
    """Tests for detect_hardware function"""

    def test_detect_hardware_returns_config(self):
        """Test hardware detection returns a HardwareConfig"""
        from toolboxv2.mods.isaa.base.rl.hardware_config import detect_hardware, HardwareConfig

        config = detect_hardware()

        self.assertIsNotNone(config)
        self.assertIsInstance(config, HardwareConfig)
        self.assertIsInstance(config.cpu_cores, int)
        self.assertGreater(config.cpu_cores, 0)


if __name__ == "__main__":
    unittest.main()

