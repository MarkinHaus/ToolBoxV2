"""
Unit Tests for ModelManager - Ollama Backend

Uses unittest (NOT pytest)
"""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_manager import ModelManager, detect_model_type, MODEL_CAPABILITIES


class TestDetectModelType(unittest.TestCase):
    """Test detect_model_type function"""

    def test_text_models(self):
        """Standard text models detected as 'text'"""
        self.assertEqual(detect_model_type("llama3.2"), "text")
        self.assertEqual(detect_model_type("mistral"), "text")
        self.assertEqual(detect_model_type("phi-3"), "text")
        self.assertEqual(detect_model_type("gemma:7b"), "text")

    def test_vision_models(self):
        """Vision models detected correctly"""
        self.assertEqual(detect_model_type("llava"), "vision")
        self.assertEqual(detect_model_type("llava:13b"), "vision")
        self.assertEqual(detect_model_type("bakllava"), "vision")
        self.assertEqual(detect_model_type("qwen2-vl"), "vision")
        self.assertEqual(detect_model_type("minicpm-v"), "vision")
        self.assertEqual(detect_model_type("model-vision"), "vision")

    def test_embedding_models(self):
        """Embedding models detected correctly"""
        self.assertEqual(detect_model_type("nomic-embed-text"), "embedding")
        self.assertEqual(detect_model_type("mxbai-embed-large"), "embedding")
        self.assertEqual(detect_model_type("all-minilm-embed"), "embedding")

    def test_vision_embedding_models(self):
        """Vision embedding models detected correctly"""
        self.assertEqual(detect_model_type("nomic-embed-vision"), "vision-embedding")
        self.assertEqual(detect_model_type("clip-vl-embed"), "vision-embedding")

    def test_audio_models(self):
        """Audio models detected correctly"""
        self.assertEqual(detect_model_type("whisper"), "audio")
        self.assertEqual(detect_model_type("whisper-large"), "audio")

    def test_tts_models(self):
        """TTS models detected correctly"""
        self.assertEqual(detect_model_type("kokoro-tts"), "tts")
        self.assertEqual(detect_model_type("f5-tts"), "tts")
        self.assertEqual(detect_model_type("f5tts"), "tts")
        self.assertEqual(detect_model_type("outetts"), "tts")
        self.assertEqual(detect_model_type("parler-tts"), "tts")
        self.assertEqual(detect_model_type("coqui"), "tts")
        self.assertEqual(detect_model_type("xtts"), "tts")
        self.assertEqual(detect_model_type("bark-tts"), "tts")
        self.assertEqual(detect_model_type("vits"), "tts")
        self.assertEqual(detect_model_type("piper"), "tts")
        self.assertEqual(detect_model_type("silero"), "tts")
        self.assertEqual(detect_model_type("speecht5"), "tts")
        self.assertEqual(detect_model_type("tortoise"), "tts")

    def test_omni_models(self):
        """Omni models detected correctly"""
        self.assertEqual(detect_model_type("gpt-4-omni"), "omni")
        self.assertEqual(detect_model_type("model-omni"), "omni")

    def test_case_insensitive(self):
        """Detection is case insensitive"""
        self.assertEqual(detect_model_type("LLAVA"), "vision")
        self.assertEqual(detect_model_type("Whisper"), "audio")
        self.assertEqual(detect_model_type("KOKORO-TTS"), "tts")


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def run_async(self, coro):
        return self.loop.run_until_complete(coro)


class TestModelManagerInit(AsyncTestCase):
    """Test ModelManager initialization"""

    def test_init_default_config(self):
        """ModelManager initializes with default config"""
        config = {"ollama_url": "http://localhost:11434"}
        manager = ModelManager(config)

        self.assertEqual(manager.ollama_url, "http://localhost:11434")
        self.assertEqual(len(manager.loaded_models), 0)
        self.assertIsInstance(manager.models_dir, Path)

    def test_init_custom_config(self):
        """ModelManager initializes with custom config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "ollama_url": "http://custom:8080",
                "models_dir": tmpdir,
                "loaded_models": {"test-model": {"model_type": "text"}},
            }
            manager = ModelManager(config)

            self.assertEqual(manager.ollama_url, "http://custom:8080")
            self.assertEqual(str(manager.models_dir), tmpdir)

    def test_models_dir_created(self):
        """Models directory is created if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_path = Path(tmpdir) / "models"
            config = {"models_dir": str(models_path)}
            manager = ModelManager(config)

            self.assertTrue(models_path.exists())
            self.assertTrue(models_path.is_dir())


class TestModelManagerFindModel(AsyncTestCase):
    """Test find_model methods"""

    def setUp(self):
        super().setUp()
        self.config = {"ollama_url": "http://localhost:11434"}
        self.manager = ModelManager(self.config)

    def test_find_model_exact_match(self):
        """find_model with exact name match"""
        self.manager.loaded_models = {
            "llama3.2": {
                "name": "llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            }
        }

        result = self.manager.find_model("llama3.2")

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "llama3.2")

    def test_find_model_partial_match(self):
        """find_model with partial name match"""
        self.manager.loaded_models = {
            "llama3.2:7b-instruct": {
                "name": "llama3.2:7b-instruct",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            }
        }

        result = self.manager.find_model("llama3.2")

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "llama3.2:7b-instruct")

    def test_find_model_not_found(self):
        """find_model returns None when not found"""
        self.manager.loaded_models = {
            "llama3.2": {
                "name": "llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            }
        }

        result = self.manager.find_model("mistral")

        self.assertIsNone(result)

    def test_find_model_case_insensitive(self):
        """find_model is case insensitive"""
        self.manager.loaded_models = {
            "Llama3.2": {
                "name": "Llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            }
        }

        result = self.manager.find_model("LLAMA3.2")

        self.assertIsNotNone(result)


class TestModelManagerFindByCapability(AsyncTestCase):
    """Test find_*_model methods"""

    def setUp(self):
        super().setUp()
        self.config = {"ollama_url": "http://localhost:11434"}
        self.manager = ModelManager(self.config)

    def test_find_text_model(self):
        """find_text_model returns text-capable model"""
        self.manager.loaded_models = {
            "llama3.2": {
                "name": "llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            }
        }

        result = self.manager.find_text_model()

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "llama3.2")

    def test_find_vision_model(self):
        """find_vision_model returns vision-capable model"""
        self.manager.loaded_models = {
            "llava": {
                "name": "llava",
                "model_type": "vision",
                "capabilities": MODEL_CAPABILITIES["vision"],
            }
        }

        result = self.manager.find_vision_model()

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "llava")
        self.assertTrue(result["capabilities"]["vision"])

    def test_find_embedding_model(self):
        """find_embedding_model returns embedding-capable model"""
        self.manager.loaded_models = {
            "nomic-embed-text": {
                "name": "nomic-embed-text",
                "model_type": "embedding",
                "capabilities": MODEL_CAPABILITIES["embedding"],
            }
        }

        result = self.manager.find_embedding_model()

        self.assertIsNotNone(result)
        self.assertTrue(result["capabilities"]["embedding"])

    def test_find_tts_model(self):
        """find_tts_model returns TTS-capable model"""
        self.manager.loaded_models = {
            "kokoro-tts": {
                "name": "kokoro-tts",
                "model_type": "tts",
                "capabilities": MODEL_CAPABILITIES["tts"],
            }
        }

        result = self.manager.find_tts_model()

        self.assertIsNotNone(result)
        self.assertTrue(result["capabilities"]["tts"])

    def test_find_audio_model(self):
        """find_audio_model returns audio-capable model"""
        self.manager.loaded_models = {
            "whisper": {
                "name": "whisper",
                "model_type": "audio",
                "capabilities": MODEL_CAPABILITIES["audio"],
            }
        }

        result = self.manager.find_audio_model()

        self.assertIsNotNone(result)
        self.assertTrue(result["capabilities"]["audio"])

    def test_find_model_returns_none_when_empty(self):
        """find_*_model returns None when no models loaded"""
        self.manager.loaded_models = {}

        self.assertIsNone(self.manager.find_text_model())
        self.assertIsNone(self.manager.find_vision_model())
        self.assertIsNone(self.manager.find_embedding_model())
        self.assertIsNone(self.manager.find_tts_model())


class TestModelManagerFindModelForRequest(AsyncTestCase):
    """Test find_model_for_request"""

    def setUp(self):
        super().setUp()
        self.config = {"ollama_url": "http://localhost:11434"}
        self.manager = ModelManager(self.config)

    def test_find_by_name(self):
        """find_model_for_request with exact model name"""
        self.manager.loaded_models = {
            "llama3.2": {
                "name": "llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            }
        }

        result = self.manager.find_model_for_request(model_name="llama3.2")

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "llama3.2")

    def test_find_vision_requirement(self):
        """find_model_for_request with vision requirement"""
        self.manager.loaded_models = {
            "llava": {
                "name": "llava",
                "model_type": "vision",
                "capabilities": MODEL_CAPABILITIES["vision"],
            },
            "llama3.2": {
                "name": "llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            },
        }

        result = self.manager.find_model_for_request(needs_vision=True)

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "llava")
        self.assertTrue(result["capabilities"]["vision"])

    def test_find_audio_requirement(self):
        """find_model_for_request with audio requirement"""
        self.manager.loaded_models = {
            "whisper": {
                "name": "whisper",
                "model_type": "audio",
                "capabilities": MODEL_CAPABILITIES["audio"],
            },
            "llama3.2": {
                "name": "llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            },
        }

        result = self.manager.find_model_for_request(needs_audio=True)

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "whisper")

    def test_find_embedding_requirement(self):
        """find_model_for_request with embedding requirement"""
        self.manager.loaded_models = {
            "nomic-embed-text": {
                "name": "nomic-embed-text",
                "model_type": "embedding",
                "capabilities": MODEL_CAPABILITIES["embedding"],
            }
        }

        result = self.manager.find_model_for_request(needs_embedding=True)

        self.assertIsNotNone(result)
        self.assertTrue(result["capabilities"]["embedding"])

    def test_model_name_fails_capability_check(self):
        """find_model_for_request returns None if named model lacks capability"""
        self.manager.loaded_models = {
            "llama3.2": {
                "name": "llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            },
            "llava": {
                "name": "llava",
                "model_type": "vision",
                "capabilities": MODEL_CAPABILITIES["vision"],
            },
        }

        # Request llama3.2 but need vision - should fall back to llava
        result = self.manager.find_model_for_request(
            model_name="llama3.2", needs_vision=True
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "llava")

    def test_omni_model_satisfies_multiple_requirements(self):
        """Omni model satisfies text, vision, and audio"""
        self.manager.loaded_models = {
            "gpt-4-omni": {
                "name": "gpt-4-omni",
                "model_type": "omni",
                "capabilities": MODEL_CAPABILITIES["omni"],
            }
        }

        result = self.manager.find_model_for_request(needs_vision=True, needs_audio=True)

        self.assertIsNotNone(result)
        self.assertTrue(result["capabilities"]["vision"])
        self.assertTrue(result["capabilities"]["audio"])


class TestModelManagerStatus(AsyncTestCase):
    """Test status methods"""

    def setUp(self):
        super().setUp()
        self.config = {"ollama_url": "http://localhost:11434"}
        self.manager = ModelManager(self.config)

    def test_get_active_models_empty(self):
        """get_active_models returns empty list when no models"""
        result = self.manager.get_active_models()

        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, list)

    def test_get_active_models(self):
        """get_active_models returns list of loaded models"""
        self.manager.loaded_models = {
            "llama3.2": {
                "name": "llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            },
            "llava": {
                "name": "llava",
                "model_type": "vision",
                "capabilities": MODEL_CAPABILITIES["vision"],
            },
        }

        result = self.manager.get_active_models()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "llama3.2")
        self.assertEqual(result[1]["name"], "llava")

    def test_get_models_status(self):
        """get_models_status returns detailed status"""
        self.manager.loaded_models = {
            "llama3.2": {
                "name": "llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
                "status": "running",
                "keep_alive": "-1",
            }
        }

        result = self.manager.get_models_status()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["model_name"], "llama3.2")
        self.assertEqual(result[0]["model_type"], "text")
        self.assertEqual(result[0]["status"], "running")
        self.assertEqual(result[0]["keep_alive"], "-1")


class TestModelManagerLoadUnload(AsyncTestCase):
    """Test load_model and unload_model with mocked HTTP"""

    def setUp(self):
        super().setUp()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            self.config = {
                "ollama_url": "http://localhost:11434",
                "_config_path": str(config_path),
            }
        self.manager = ModelManager(self.config)

    @patch("model_manager.httpx.AsyncClient")
    def test_load_model_success(self, mock_client_class):
        """load_model successfully loads a model"""
        # Mock HTTP responses
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Mock health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200

        # Mock list models response
        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "models": [{"name": "llama3.2:latest"}]
        }
        mock_list_response.raise_for_status = Mock()

        # Mock generate response for warmup
        mock_generate_response = Mock()
        mock_generate_response.raise_for_status = Mock()

        mock_client.get.side_effect = [mock_health_response, mock_list_response]
        mock_client.post.return_value = mock_generate_response

        # Mock _save_loaded_config
        with patch.object(self.manager, "_save_loaded_config"):
            result = self.run_async(self.manager.load_model("llama3.2"))

        self.assertEqual(result["status"], "loaded")
        self.assertEqual(result["model"], "llama3.2")
        self.assertIn("llama3.2", self.manager.loaded_models)

    @patch("model_manager.httpx.AsyncClient")
    def test_load_embedding_model(self, mock_client_class):
        """load_model uses embed endpoint for embedding models"""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Mock responses
        mock_health_response = Mock()
        mock_health_response.status_code = 200

        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_list_response.raise_for_status = Mock()

        mock_embed_response = Mock()
        mock_embed_response.raise_for_status = Mock()

        mock_client.get.side_effect = [mock_health_response, mock_list_response]
        mock_client.post.return_value = mock_embed_response

        with patch.object(self.manager, "_save_loaded_config"):
            result = self.run_async(self.manager.load_model("nomic-embed-text"))

        self.assertEqual(result["model_type"], "embedding")
        # Verify embed endpoint was called
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        self.assertEqual(call_args[0][0], "/api/embed")

    @patch("model_manager.httpx.AsyncClient")
    def test_load_model_auto_detect_type(self, mock_client_class):
        """load_model auto-detects model type"""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_health_response = Mock()
        mock_health_response.status_code = 200

        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "models": [{"name": "llava:latest"}]
        }
        mock_list_response.raise_for_status = Mock()

        mock_generate_response = Mock()
        mock_generate_response.raise_for_status = Mock()

        mock_client.get.side_effect = [mock_health_response, mock_list_response]
        mock_client.post.return_value = mock_generate_response

        with patch.object(self.manager, "_save_loaded_config"):
            result = self.run_async(self.manager.load_model("llava", model_type="auto"))

        self.assertEqual(result["model_type"], "vision")

    @patch("model_manager.httpx.AsyncClient")
    def test_load_model_invalid_type(self, mock_client_class):
        """load_model raises error for invalid model_type"""
        with self.assertRaises(ValueError) as context:
            self.run_async(self.manager.load_model("test", model_type="invalid"))

        self.assertIn("Invalid model_type", str(context.exception))

    @patch("model_manager.httpx.AsyncClient")
    def test_load_model_ollama_unreachable(self, mock_client_class):
        """load_model raises error when Ollama unreachable"""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Mock health check failure
        mock_health_response = Mock()
        mock_health_response.status_code = 500
        mock_client.get.return_value = mock_health_response

        with self.assertRaises(RuntimeError) as context:
            self.run_async(self.manager.load_model("llama3.2"))

        self.assertIn("Ollama not reachable", str(context.exception))

    @patch("model_manager.httpx.AsyncClient")
    def test_unload_model(self, mock_client_class):
        """unload_model removes model from loaded_models"""
        # Setup loaded model
        self.manager.loaded_models = {
            "llama3.2": {
                "name": "llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            }
        }

        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock()

        with patch.object(self.manager, "_save_loaded_config"):
            result = self.run_async(self.manager.unload_model("llama3.2"))

        self.assertEqual(result["status"], "unloaded")
        self.assertEqual(result["model"], "llama3.2")
        self.assertNotIn("llama3.2", self.manager.loaded_models)

    @patch("model_manager.httpx.AsyncClient")
    def test_unload_model_handles_errors(self, mock_client_class):
        """unload_model handles HTTP errors gracefully"""
        self.manager.loaded_models = {
            "llama3.2": {
                "name": "llama3.2",
                "model_type": "text",
                "capabilities": MODEL_CAPABILITIES["text"],
            }
        }

        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post.side_effect = Exception("Network error")

        with patch.object(self.manager, "_save_loaded_config"):
            result = self.run_async(self.manager.unload_model("llama3.2"))

        # Should still remove from loaded_models despite error
        self.assertEqual(result["status"], "unloaded")
        self.assertNotIn("llama3.2", self.manager.loaded_models)


class TestModelManagerOllamaAPI(AsyncTestCase):
    """Test Ollama API interaction methods"""

    def setUp(self):
        super().setUp()
        self.config = {"ollama_url": "http://localhost:11434"}
        self.manager = ModelManager(self.config)

    @patch("model_manager.httpx.AsyncClient")
    def test_list_ollama_models(self, mock_client_class):
        """list_ollama_models returns formatted model list"""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3.2:latest",
                    "size": 4_800_000_000,
                    "modified_at": "2024-01-01T00:00:00Z",
                    "details": {
                        "family": "llama",
                        "parameter_size": "7B",
                        "quantization_level": "Q4_0",
                        "format": "gguf",
                    },
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response

        result = self.run_async(self.manager.list_ollama_models())

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "llama3.2:latest")
        self.assertAlmostEqual(result[0]["size_gb"], 4.47, places=1)

    @patch("model_manager.httpx.AsyncClient")
    def test_list_running_models(self, mock_client_class):
        """list_running_models returns currently running models"""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:latest", "size": 4800000000}]
        }
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response

        result = self.run_async(self.manager.list_running_models())

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "llama3.2:latest")

    @patch("model_manager.httpx.AsyncClient")
    def test_list_running_models_handles_errors(self, mock_client_class):
        """list_running_models returns empty list on error"""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.get.side_effect = Exception("Network error")

        result = self.run_async(self.manager.list_running_models())

        self.assertEqual(result, [])


class TestModelManagerLocalGGUF(AsyncTestCase):
    """Test local GGUF file management"""

    def test_list_local_gguf(self):
        """list_local_gguf finds GGUF files in models directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"models_dir": tmpdir}
            manager = ModelManager(config)

            # Create test GGUF files
            test_file = Path(tmpdir) / "model.gguf"
            test_file.write_bytes(b"x" * 1024 * 1024)  # 1 MB

            result = manager.list_local_gguf()

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["name"], "model.gguf")
            self.assertGreater(result[0]["size_mb"], 0)

    def test_list_local_gguf_recursive(self):
        """list_local_gguf finds GGUF files recursively"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"models_dir": tmpdir}
            manager = ModelManager(config)

            # Create nested structure
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.gguf").write_bytes(b"test")

            result = manager.list_local_gguf()

            self.assertEqual(len(result), 1)
            self.assertIn("subdir", result[0]["path"])

    def test_list_local_gguf_detects_type(self):
        """list_local_gguf detects model type from filename"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"models_dir": tmpdir}
            manager = ModelManager(config)

            (Path(tmpdir) / "llava-model.gguf").write_bytes(b"test")

            result = manager.list_local_gguf()

            self.assertEqual(result[0]["detected_type"], "vision")


if __name__ == "__main__":
    unittest.main(verbosity=2)
