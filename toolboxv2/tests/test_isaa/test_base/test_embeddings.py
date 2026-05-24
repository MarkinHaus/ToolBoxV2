"""Tests for embedding migration — mock router, verify numpy conversion."""
import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from toolboxv2.mods.isaa.base.llm_router.types import EmbedResult, UsageData


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestEmbedFunction(unittest.TestCase):
    """Test embed() with mocked router."""

    def _mock_router(self, embeddings_data):
        """Create a mock router that returns given embeddings."""
        mock_result = EmbedResult(
            embeddings=embeddings_data,
            usage=UsageData(5, 0, 5),
            model="test-model",
        )
        router = MagicMock()
        router.embed = AsyncMock(return_value=mock_result)
        return router

    @patch("llm_router.embeddings._get_router")
    def test_basic_embed(self, mock_get_router):
        embeddings_data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_get_router.return_value = self._mock_router(embeddings_data)

        from toolboxv2.mods.isaa.base.llm_router.embeddings import embed
        result = _run(embed(["hello", "world"], model="ollama/nomic-embed-text"))

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])

    @patch("llm_router.embeddings._get_router")
    def test_dimension_truncation(self, mock_get_router):
        """If provider returns more dims than requested, truncate."""
        embeddings_data = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_get_router.return_value = self._mock_router(embeddings_data)

        from toolboxv2.mods.isaa.base.llm_router.embeddings import embed
        result = _run(embed(["hello"], model="test/model", dimensions=3))

        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])

    @patch("llm_router.embeddings._get_router")
    def test_dimensions_passed_to_router(self, mock_get_router):
        embeddings_data = [[0.1, 0.2]]
        router = self._mock_router(embeddings_data)
        mock_get_router.return_value = router

        from toolboxv2.mods.isaa.base.llm_router.embeddings import embed
        _run(embed(["hello"], model="test/model", dimensions=256))

        # Verify dimensions was forwarded
        _, kwargs = router.embed.call_args
        self.assertEqual(kwargs.get("dimensions"), 256)

    @patch("llm_router.embeddings._get_router")
    def test_input_type_passed(self, mock_get_router):
        embeddings_data = [[0.1, 0.2]]
        router = self._mock_router(embeddings_data)
        mock_get_router.return_value = router

        from toolboxv2.mods.isaa.base.llm_router.embeddings import embed
        _run(embed(["hello"], model="cohere/embed-english-v3.0",
                    input_type="search_query"))

        _, kwargs = router.embed.call_args
        self.assertEqual(kwargs.get("input_type"), "search_query")

    @patch("llm_router.embeddings._get_router")
    def test_litellm_embed_compat(self, mock_get_router):
        """litellm_embed() is a thin wrapper, verify it delegates correctly."""
        embeddings_data = [[0.1, 0.2, 0.3]]
        router = self._mock_router(embeddings_data)
        mock_get_router.return_value = router

        from toolboxv2.mods.isaa.base.llm_router.embeddings import litellm_embed
        result = _run(litellm_embed(["hello"], model="openrouter/qwen/qwen3-embedding-8b",
                                     dimensions=256))

        self.assertIsInstance(result, np.ndarray)
        router.embed.assert_called_once()

    @patch("llm_router.embeddings._get_router")
    def test_smart_embed_delegates(self, mock_get_router):
        embeddings_data = [[0.1, 0.2]]
        router = self._mock_router(embeddings_data)
        mock_get_router.return_value = router

        from toolboxv2.mods.isaa.base.llm_router.embeddings import smart_embed
        result = _run(smart_embed(["hello"], model="openrouter/qwen/qwen3-embedding-8b"))

        self.assertIsInstance(result, np.ndarray)
        router.embed.assert_called_once()


class TestApiKeyResolution(unittest.TestCase):
    def test_openrouter_key(self):
        from toolboxv2.mods.isaa.base.llm_router.embeddings import _resolve_api_key
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}):
            self.assertEqual(_resolve_api_key("openrouter/model"), "or-key")

    def test_explicit_key_overrides(self):
        from toolboxv2.mods.isaa.base.llm_router.embeddings import _resolve_api_key
        result = _resolve_api_key("openrouter/model", api_key="explicit")
        self.assertEqual(result, "explicit")

    def test_ollama_no_key(self):
        from toolboxv2.mods.isaa.base.llm_router.embeddings import _resolve_api_key
        result = _resolve_api_key("ollama/nomic-embed-text")
        self.assertEqual(result, "")


class TestCosine(unittest.TestCase):
    def test_identical_vectors(self):
        from toolboxv2.mods.isaa.base.llm_router.embeddings import cosine_similarity
        a = np.array([[1.0, 0.0, 0.0]])
        sim = cosine_similarity(a, a)
        self.assertAlmostEqual(sim[0, 0], 1.0, places=5)

    def test_orthogonal_vectors(self):
        from toolboxv2.mods.isaa.base.llm_router.embeddings import cosine_similarity
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        sim = cosine_similarity(a, b)
        self.assertAlmostEqual(sim[0, 0], 0.0, places=5)


class TestAdapterEmbedDimensions(unittest.TestCase):
    """Verify OpenAICompatAdapter.embed() forwards dimensions in payload."""

    def test_embed_payload_includes_dimensions(self):
        from toolboxv2.mods.isaa.base.llm_router.adapters.openai_compat import OpenAICompatAdapter

        adapter = OpenAICompatAdapter("https://api.example.com/v1")
        session = MagicMock()

        embed_resp = {
            "data": [{"embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
            "model": "test",
        }
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value=embed_resp)
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        session.post = MagicMock(return_value=resp)

        _run(adapter.embed(session, "sk-x", "model", ["hi"], dimensions=128))

        # Check the payload that was sent
        call_args = session.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        self.assertEqual(payload["dimensions"], 128)

    def test_embed_payload_without_dimensions(self):
        from toolboxv2.mods.isaa.base.llm_router.adapters.openai_compat import OpenAICompatAdapter

        adapter = OpenAICompatAdapter("https://api.example.com/v1")
        session = MagicMock()

        embed_resp = {
            "data": [{"embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
            "model": "test",
        }
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value=embed_resp)
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        session.post = MagicMock(return_value=resp)

        _run(adapter.embed(session, "sk-x", "model", ["hi"]))

        call_args = session.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        self.assertNotIn("dimensions", payload)


if __name__ == "__main__":
    unittest.main()
