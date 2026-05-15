"""
Unit tests for export module
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path


class TestGGUFQuantization(unittest.TestCase):
    """Tests for GGUFQuantization dataclass"""

    def test_quantization_values(self):
        """Test that all expected quantization levels exist"""
        from toolboxv2.mods.isaa.base.rl.export import GGUFQuantization

        available = GGUFQuantization.available()

        self.assertIn("Q4_K_M", available)
        self.assertIn("Q5_K_M", available)
        self.assertIn("Q8_0", available)
        self.assertIn("F16", available)

        # Test dataclass structure
        q4 = available["Q4_K_M"]
        self.assertEqual(q4.name, "Q4_K_M")
        self.assertIsInstance(q4.bits, float)


class TestOllamaHostingProfile(unittest.TestCase):
    """Tests for OllamaHostingProfile dataclass"""

    def test_profile_creation(self):
        """Test profile creation"""
        from toolboxv2.mods.isaa.base.rl.export import OllamaHostingProfile

        profile = OllamaHostingProfile(name="test", num_parallel=2, num_ctx=8192)

        self.assertEqual(profile.name, "test")
        self.assertEqual(profile.num_parallel, 2)
        self.assertEqual(profile.num_ctx, 8192)

    def test_to_env(self):
        """Test conversion to environment variables"""
        from toolboxv2.mods.isaa.base.rl.export import OllamaHostingProfile

        profile = OllamaHostingProfile(name="test", num_parallel=4, num_thread=8)
        env = profile.to_env()

        self.assertEqual(env["OLLAMA_NUM_PARALLEL"], "4")
        self.assertEqual(env["OLLAMA_NUM_THREAD"], "8")


class TestGGUFExporter(unittest.TestCase):
    """Tests for GGUFExporter"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('toolboxv2.mods.isaa.base.rl.export.GGUFExporter._find_or_install_llama_cpp')
    def test_initialization(self, mock_find):
        """Test exporter initialization"""
        from toolboxv2.mods.isaa.base.rl.export import GGUFExporter

        mock_find.return_value = Path("/mock/llama.cpp")

        exporter = GGUFExporter(model_path=self.temp_dir)

        self.assertEqual(exporter.model_path, Path(self.temp_dir))

    @patch('toolboxv2.mods.isaa.base.rl.export.GGUFExporter._find_or_install_llama_cpp')
    def test_output_dir_creation(self, mock_find):
        """Test output directory is created"""
        from toolboxv2.mods.isaa.base.rl.export import GGUFExporter

        mock_find.return_value = Path("/mock/llama.cpp")

        exporter = GGUFExporter(model_path=self.temp_dir)

        self.assertTrue(exporter.output_dir.exists())

    def test_quantization_available(self):
        """Test quantization options are available"""
        from toolboxv2.mods.isaa.base.rl.export import GGUFQuantization

        available = GGUFQuantization.available()

        self.assertGreater(len(available), 0)
        self.assertIn("Q4_K_M", available)


class TestOllamaDeployer(unittest.TestCase):
    """Tests for OllamaDeployer"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('toolboxv2.mods.isaa.base.rl.export.subprocess.run')
    def test_initialization(self, mock_run):
        """Test deployer initialization"""
        from toolboxv2.mods.isaa.base.rl.export import OllamaDeployer

        mock_run.return_value = MagicMock(returncode=0, stdout="0.1.0")

        deployer = OllamaDeployer()

        self.assertEqual(deployer.ollama_path, "ollama")

    @patch('toolboxv2.mods.isaa.base.rl.export.subprocess.run')
    def test_create_modelfile(self, mock_run):
        """Test Modelfile creation"""
        from toolboxv2.mods.isaa.base.rl.export import OllamaDeployer

        mock_run.return_value = MagicMock(returncode=0, stdout="0.1.0")

        deployer = OllamaDeployer()
        modelfile = deployer.create_modelfile(
            gguf_path="/path/to/model.gguf",
            system_prompt="You are a helpful assistant."
        )

        self.assertIn("FROM /path/to/model.gguf", modelfile)
        self.assertIn("SYSTEM", modelfile)

    def test_hosting_profile(self):
        """Test OllamaHostingProfile"""
        from toolboxv2.mods.isaa.base.rl.export import OllamaHostingProfile

        profile = OllamaHostingProfile(
            name="test",
            num_parallel=2,
            num_ctx=8192
        )

        self.assertEqual(profile.name, "test")
        env = profile.to_env()
        self.assertIn("OLLAMA_NUM_PARALLEL", env)


class TestQuickExport(unittest.TestCase):
    """Tests for quick_export function"""

    @patch('toolboxv2.mods.isaa.base.rl.export.ExportPipeline')
    def test_quick_export(self, mock_pipeline_class):
        """Test quick_export function"""
        from toolboxv2.mods.isaa.base.rl.export import quick_export

        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run.return_value = {"success": True, "ollama_model": "test-model"}

        result = quick_export(
            model_path="/path/to/model",
            model_name="test-model"
        )

        mock_pipeline.run.assert_called_once()
        self.assertEqual(result, "test-model")


if __name__ == "__main__":
    unittest.main()
