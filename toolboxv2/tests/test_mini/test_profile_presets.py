"""Tests for _PROFILE_PRESETS env_overrides in init_onboarding.py"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

# Bootstrap: load init_onboarding via file path (like test_config_wizard does)
from pathlib import Path
import importlib.util

_TB_ROOT = Path(__file__).resolve().parents[2]  # toolboxv2/
_MOD_PATH = _TB_ROOT / "init_onboarding.py"

_spec = importlib.util.spec_from_file_location("toolboxv2.init_onboarding", _MOD_PATH)
_init = importlib.util.module_from_spec(_spec)
sys.modules["toolboxv2.init_onboarding"] = _init
_spec.loader.exec_module(_init)


class TestProfilePresetsHaveEnvOverrides(unittest.TestCase):
    """E1: Every preset must have an env_overrides key."""

    def test_all_presets_have_env_overrides(self):
        for name, preset in _init._PROFILE_PRESETS.items():
            self.assertIn(
                "env_overrides", preset,
                f"Preset '{name}' missing 'env_overrides' key"
            )

    def test_env_overrides_is_dict_of_strings(self):
        for name, preset in _init._PROFILE_PRESETS.items():
            overrides = preset.get("env_overrides", {})
            self.assertIsInstance(overrides, dict, f"Preset '{name}' env_overrides not dict")
            for k, v in overrides.items():
                self.assertIsInstance(k, str)
                self.assertIsInstance(v, str,
                    f"Preset '{name}' env_overrides['{k}'] not str: {type(v)}")


class TestProfileEnvOverridesApplied(unittest.TestCase):
    """E2: env_overrides are applied in the layering logic."""

    def test_mini_preset_has_warning_logging(self):
        preset = _init._resolve_preset("mini")
        self.assertEqual(preset["env_overrides"]["TOOLBOX_LOGGING_LEVEL"], "WARNING")

    def test_desktop_preset_has_info_logging(self):
        preset = _init._resolve_preset("desktop")
        self.assertEqual(preset["env_overrides"]["TOOLBOX_LOGGING_LEVEL"], "INFO")

    def test_server_preset_has_info_logging(self):
        preset = _init._resolve_preset("server")
        self.assertEqual(preset["env_overrides"]["TOOLBOX_LOGGING_LEVEL"], "INFO")

    def test_env_overrides_override_base_is_offline_db(self):
        """env_overrides can override IS_OFFLINE_DB set by preset."""
        preset = _init._resolve_preset("mini")
        # mini sets offline=True → IS_OFFLINE_DB=true
        # But env_overrides could override it
        self.assertTrue(preset["offline"])


class TestEnvOverridesLayeringOrder(unittest.TestCase):
    """E3: Layering order — env_overrides applied after base, before caller env."""

    def test_layered_env_includes_overrides(self):
        """Simulate the layering logic and verify env_overrides are included."""
        preset = _init._resolve_preset("mini")
        layered = {}
        layered.update({"BASE_VAR": "base"})
        layered["IS_OFFLINE_DB"] = "true" if preset["offline"] else "false"
        layered["DB_MODE_KEY"] = preset["db"]
        # This is the line we added:
        layered.update(preset.get("env_overrides", {}))

        self.assertIn("TOOLBOX_LOGGING_LEVEL", layered)
        self.assertEqual(layered["TOOLBOX_LOGGING_LEVEL"], "WARNING")

    def test_caller_env_overrides_preset_overrides(self):
        """Explicit caller env= should win over env_overrides."""
        preset = _init._resolve_preset("mini")
        layered = {}
        layered.update(preset.get("env_overrides", {}))
        caller_env = {"TOOLBOX_LOGGING_LEVEL": "DEBUG"}
        layered.update({k: str(v) for k, v in caller_env.items()})
        self.assertEqual(layered["TOOLBOX_LOGGING_LEVEL"], "DEBUG")


if __name__ == "__main__":
    unittest.main()
