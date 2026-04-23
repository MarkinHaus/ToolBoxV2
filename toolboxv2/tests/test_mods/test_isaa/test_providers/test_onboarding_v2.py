# tests/test_onboarding_v2.py
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dotenv import dotenv_values

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.onboarding import (
    scan_env, _persist_key, _next_slot, _env_var_names_for,
    build_config, cost_for_tokens, cost_table, _parse_price_notes,
    OnboardingResult, REGISTRY_BY_ID,
)


class TestScanReturnsEnvNames(unittest.TestCase):
    def test_returns_names_not_values(self):
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            _persist_key(p, "GROQ_API_KEY", "gsk_1")
            _persist_key(p, "GROQ_API_KEY_2", "gsk_2")
            out = scan_env(p)
            self.assertEqual(out["groq"], ["GROQ_API_KEY", "GROQ_API_KEY_2"])

    def test_shared_env_key_not_duplicated(self):
        """zglm_lite and zglm_pro share ZGLM_API_KEY — only one spec should win."""
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            _persist_key(p, "ZGLM_API_KEY", "zglm")
            out = scan_env(p)
            zglm_entries = [pid for pid in out if pid.startswith("zglm")]
            self.assertEqual(len(zglm_entries), 1)


class TestBuildConfigSchema(unittest.TestCase):
    def test_top_level_is_rate_limiter(self):
        r = OnboardingResult(
            active_env_names={"groq": ["GROQ_API_KEY"]},
            selected_models={"groq": ["llama-3.3-70b-versatile"]},
        )
        cfg = build_config(r, models_raw={"groq": ["llama-3.3-70b-versatile"]})
        self.assertIn("rate_limiter", cfg)
        rl = cfg["rate_limiter"]
        self.assertIn("enable_rate_limiting", rl)
        self.assertIn("enable_model_fallback", rl)
        self.assertIn("enable_key_rotation", rl)
        self.assertIn("key_rotation_mode", rl)
        self.assertIn("api_keys", rl)
        self.assertIn("fallback_chains", rl)
        self.assertIn("custom_limits", rl)
        self.assertIn("max_retries", rl)
        self.assertIn("wait_if_all_exhausted", rl)

    def test_api_keys_are_env_names_not_values(self):
        r = OnboardingResult(
            active_env_names={"groq": ["GROQ_API_KEY", "GROQ_API_KEY_2"]},
            selected_models={"groq": ["llama-3.3-70b-versatile"]},
        )
        cfg = build_config(r, models_raw={"groq": ["llama-3.3-70b-versatile"]})
        self.assertEqual(cfg["rate_limiter"]["api_keys"]["groq"],
                         ["GROQ_API_KEY", "GROQ_API_KEY_2"])

    def test_custom_limits_has_is_free_tier(self):
        r = OnboardingResult(
            active_env_names={"groq": ["GROQ_API_KEY"]},
            selected_models={"groq": ["llama-3.3-70b-versatile"]},
        )
        cfg = build_config(r, models_raw={"groq": ["llama-3.3-70b-versatile"]})
        entry = cfg["rate_limiter"]["custom_limits"]["groq/llama-3.3-70b-versatile"]
        self.assertTrue(entry["is_free_tier"])

    def test_payg_has_price_notes(self):
        r = OnboardingResult(
            active_env_names={"openai": ["OPENAI_API_KEY"]},
            selected_models={"openai": ["gpt-5.4"]},
        )
        cfg = build_config(r, models_raw={"openai": ["gpt-5.4"]})
        entry = cfg["rate_limiter"]["custom_limits"]["openai/gpt-5.4"]
        self.assertFalse(entry["is_free_tier"])
        self.assertIn("price_in=2.5", entry["notes"])
        self.assertIn("price_out=15.0", entry["notes"])

    def test_enable_key_rotation_default_true(self):
        r = OnboardingResult(active_env_names={"groq": ["GROQ_API_KEY"]})
        cfg = build_config(r, models_raw={})
        self.assertTrue(cfg["rate_limiter"]["enable_key_rotation"])

    def test_key_rotation_mode_balance_default(self):
        r = OnboardingResult(active_env_names={"groq": ["GROQ_API_KEY"]})
        cfg = build_config(r, models_raw={})
        self.assertEqual(cfg["rate_limiter"]["key_rotation_mode"], "balance")


class TestCostCalculatorV2(unittest.TestCase):
    def _payg_cfg(self):
        r = OnboardingResult(
            active_env_names={"openai": ["OPENAI_API_KEY"]},
            selected_models={"openai": ["gpt-4.1-nano", "gpt-5.4"]},
        )
        return build_config(r, models_raw={"openai": ["gpt-4.1-nano", "gpt-5.4"]})

    def test_parse_price_notes(self):
        self.assertEqual(
            _parse_price_notes("price_in=2.5,price_out=15.0,cache_discount=0.9"),
            (2.5, 15.0, 0.9),
        )

    def test_parse_price_notes_returns_none_for_free(self):
        self.assertIsNone(_parse_price_notes(""))
        self.assertIsNone(_parse_price_notes("plan_tier=pro,monthly_flat_usd=30.0"))

    def test_cost_ratio_3_1(self):
        cfg = self._payg_cfg()
        r = cost_for_tokens("openai/gpt-4.1-nano", 4_000_000, cfg)
        self.assertEqual(r["input_tokens"], 3_000_000)
        self.assertEqual(r["output_tokens"], 1_000_000)
        self.assertAlmostEqual(r["total_cost_usd"], 0.70, places=4)

    def test_cost_table_skips_free(self):
        # mixed free + payg cfg
        r = OnboardingResult(
            active_env_names={
                "groq": ["GROQ_API_KEY"],
                "openai": ["OPENAI_API_KEY"],
            },
            selected_models={
                "groq": ["llama-3.3-70b-versatile"],
                "openai": ["gpt-4.1-nano"],
            },
        )
        cfg = build_config(r, models_raw={
            "groq": ["llama-3.3-70b-versatile"],
            "openai": ["gpt-4.1-nano"],
        })
        table = cost_table(cfg, max_tokens=1_000_000, step=500_000)
        self.assertIn("openai/gpt-4.1-nano", table)
        self.assertNotIn("groq/llama-3.3-70b-versatile", table)


if __name__ == "__main__":
    unittest.main()
