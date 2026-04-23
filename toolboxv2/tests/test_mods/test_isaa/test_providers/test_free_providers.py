# tests/test_free_providers.py
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, AsyncMock, MagicMock

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.free_providers.registry import REGISTRY_BY_ID
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.onboarding import (
     _next_slot, OnboardingResult,_persist_key, _env_var_names_for
)

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.build_config import build_config, _derive_fallbacks
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.free_providers.fetch_models import fetch_available_models, _parse_models


class TestAppendEnv(unittest.TestCase):
    def test_append_only(self):
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            _persist_key(p, "A", "1")
            _persist_key(p, "B", "2")
            content = p.read_text()
            self.assertIn("A=1", content)
            self.assertIn("B=2", content)
            self.assertEqual(content.count("A=1"), 1)

    def test_handles_missing_newline(self):
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            p.write_text("X=0")  # no trailing newline
            _persist_key(p, "Y", "1")
            self.assertEqual(p.read_text(), "X=0\nY=1\n")


class TestParseModels(unittest.TestCase):
    def test_openai_format(self):
        spec = REGISTRY_BY_ID["groq"]
        out = _parse_models(spec, {"data": [{"id": "llama-3.3"}, {"id": "mixtral"}]})
        self.assertEqual(out, ["llama-3.3", "mixtral"])

    def test_gemini_format(self):
        spec = REGISTRY_BY_ID["gemini"]
        out = _parse_models(spec, {"models": [{"name": "models/gemini-2.5-pro"}]})
        self.assertEqual(out, ["gemini-2.5-pro"])

class TestFetchModelsFallback(unittest.IsolatedAsyncioTestCase):
    async def test_warm_fallback_on_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("boom")

        async def mock_get(*a, **kw):
            return mock_response

        with patch("httpx.AsyncClient") as mc:
            instance = mc.return_value.__aenter__.return_value
            instance.get = AsyncMock(side_effect=Exception("boom"))
            out = await fetch_available_models({"groq": ["fake"]})
        self.assertEqual(out["groq"], list(REGISTRY_BY_ID["groq"].warm_models))

    async def test_custom_provider_uses_warm(self):
        out = await fetch_available_models({"zai": ["fake"]})
        self.assertEqual(out["zai"], list(REGISTRY_BY_ID["zai"].warm_models))

if __name__ == "__main__":
    unittest.main()
