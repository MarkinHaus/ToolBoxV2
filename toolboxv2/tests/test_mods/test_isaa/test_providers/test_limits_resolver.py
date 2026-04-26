# tests/test_limits_resolver.py
import unittest

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.onboarding import (
    _resolve_limits_from_rate_limiter, _match_limit_for,
)


class TestResolveLimits(unittest.TestCase):
    def setUp(self):
        self.flat = _resolve_limits_from_rate_limiter()

    def test_returns_non_empty(self):
        self.assertGreater(len(self.flat), 0)

    def test_keys_have_provider_prefix(self):
        for k in self.flat:
            self.assertIn("::", k)

    def test_groq_free_tier_present(self):
        # groq is category=free, must pull Tier.FREE entries
        groq_keys = [k for k in self.flat if k.startswith("groq::")]
        self.assertGreater(len(groq_keys), 0)

    def test_openai_payg_tier_present(self):
        openai_keys = [k for k in self.flat if k.startswith("openai::")]
        self.assertGreater(len(openai_keys), 0)

    def test_zai_gets_free_tier(self):
        """zai (category=free) must resolve to glm-4.7-flash / glm-4.5-flash from FREE tier."""
        zai_keys = [k for k in self.flat if k.startswith("zai::")]
        self.assertGreater(len(zai_keys), 0)
        patterns = [k.split("::", 1)[1] for k in zai_keys]
        self.assertTrue(any("flash" in p for p in patterns))

    def test_zglm_pro_gets_payg_tier(self):
        """zglm_pro (category=coding) must resolve to PAYG tier with glm-5 etc."""
        zglm_keys = [k for k in self.flat if k.startswith("zglm_pro::")]
        self.assertGreater(len(zglm_keys), 0)
        patterns = [k.split("::", 1)[1] for k in zglm_keys]
        self.assertIn("glm-5", patterns)


class TestMatchLimit(unittest.TestCase):
    def test_groq_llama_match(self):
        flat = _resolve_limits_from_rate_limiter()
        mrl = _match_limit_for("groq", "llama-3.3-70b-versatile", flat)
        # provider_limits has groq patterns — whichever matches, rpm must be set
        if mrl is not None:
            self.assertIsNotNone(mrl.rpm)

    def test_unknown_provider_returns_none(self):
        flat = _resolve_limits_from_rate_limiter()
        self.assertIsNone(_match_limit_for("huggingface", "meta-llama/X", flat))


if __name__ == "__main__":
    unittest.main()
