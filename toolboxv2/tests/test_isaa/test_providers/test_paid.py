# tests/test_paid.py
import unittest
from unittest.mock import patch

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.onboarding import (
    OnboardingResult,
)
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.build_config import build_config
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.paid_providers.paid_payg_registry import (
    cost_for_tokens, cost_table_for_config, _prefix_to_id, PAYG_BY_ID,
)


class TestCostCalculator(unittest.TestCase):
    def test_ratio_split(self):
        r = cost_for_tokens("openai", "gpt-4.1-nano", 4_000_000)
        # 3:1 ratio of 4M = 3M in, 1M out
        self.assertEqual(r["input_tokens"], 3_000_000)
        self.assertEqual(r["output_tokens"], 1_000_000)
        # 3M * 0.10 + 1M * 0.40 = 0.30 + 0.40 = 0.70
        self.assertAlmostEqual(r["total_cost_usd"], 0.70, places=4)

    def test_cache_discount(self):
        # fresh
        r0 = cost_for_tokens("openai", "gpt-5.4", 4_000_000, cache_hit_fraction=0.0)
        # 3M in * $2.50 = $7.50, 1M out * $15 = $15, total $22.50
        self.assertAlmostEqual(r0["total_cost_usd"], 22.50, places=2)
        # full cache hit (90% off on input)
        r1 = cost_for_tokens("openai", "gpt-5.4", 4_000_000, cache_hit_fraction=1.0)
        # 3M cached * 2.50 * 0.10 = 0.75, + 1M out * 15 = 15 => 15.75
        self.assertAlmostEqual(r1["total_cost_usd"], 15.75, places=2)

    def test_unknown_model_raises(self):
        with self.assertRaises(KeyError):
            cost_for_tokens("openai", "gpt-does-not-exist", 1_000_000)

    def test_prefix_mapping(self):
        self.assertEqual(_prefix_to_id("openai"), "openai")
        self.assertEqual(_prefix_to_id("gemini"), "gemini_payg")
        self.assertEqual(_prefix_to_id("nope"), None)



class TestBuildConfigPaid(unittest.TestCase):
    def test_payg_limits_shape(self):
        r = OnboardingResult(
            payg_providers={"openai": ["sk-1"]},
        )
        cfg = build_config(r, models_raw={"openai": ["gpt-5.4", "gpt-4.1-nano"]})
        self.assertIn("openai/gpt-5.4", cfg["limits"])
        limit = cfg["limits"]["openai/gpt-5.4"]
        self.assertEqual(limit["plan_type"], "payg")
        self.assertEqual(limit["price_input_per_mtok"], 2.50)
        self.assertEqual(limit["price_output_per_mtok"], 15.00)

    def test_coding_plan_limits(self):
        r = OnboardingResult(coding_providers={"zglm_pro": ["zglm-x"]})
        cfg = build_config(r, models_raw={})
        keys = [k for k in cfg["limits"] if k.startswith("zglm/")]
        self.assertTrue(keys)
        self.assertEqual(cfg["limits"][keys[0]]["plan_type"], "coding_plan")
        self.assertEqual(cfg["limits"][keys[0]]["monthly_flat_usd"], 30.0)

    def test_cost_table_from_config(self):
        r = OnboardingResult(payg_providers={"openai": ["sk-1"]})
        cfg = build_config(r, models_raw={"openai": ["gpt-4.1-nano"]})
        table = cost_table_for_config(cfg, max_tokens=1_500_000, step=500_000)
        self.assertIn("openai/gpt-4.1-nano", table)
        rows = table["openai/gpt-4.1-nano"]
        self.assertEqual(len(rows), 3)  # 500k, 1M, 1.5M
        self.assertEqual(rows[0][0], 500_000)
        # 500k * (3/4 * 0.10 + 1/4 * 0.40) / 1M = (0.0375 + 0.05) = 0.0875
        self.assertAlmostEqual(rows[0][1], 0.0875, places=3)


if __name__ == "__main__":
    unittest.main()
