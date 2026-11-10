"""First-run onboarding: 3 profiles, legacy aliases, headless apply, fail-safe.

Run: python -m unittest toolboxv2.tests.test_core.test_first_run_onboarding -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


class TestProfileSchema(unittest.TestCase):
    def test_only_three_profiles(self):
        from toolboxv2.utils.manifest.schema import ProfileType
        self.assertEqual({p.value for p in ProfileType}, {"local", "server", "developer"})

    def test_legacy_values_map_to_current(self):
        from toolboxv2.utils.manifest.schema import ProfileType
        self.assertIs(ProfileType("consumer"), ProfileType.LOCAL)
        self.assertIs(ProfileType("homelab"), ProfileType.LOCAL)
        self.assertIs(ProfileType("business"), ProfileType.SERVER)
        with self.assertRaises(ValueError):
            ProfileType("nope")

    def test_legacy_manifest_still_validates(self):
        from toolboxv2.utils.manifest.schema import TBManifest
        m = TBManifest.model_validate({"app": {"profile": "consumer"}})
        self.assertEqual(m.app.profile.value, "local")

    def test_every_profile_has_defaults_and_ui_label(self):
        from toolboxv2.utils.manifest.schema import ProfileType, PROFILE_DEFAULTS
        from toolboxv2.utils.clis.first_run import PROFILES
        for p in ProfileType:
            self.assertIn(p.value, PROFILE_DEFAULTS)
            self.assertIn(p.value, PROFILES)


class TestApplyProfile(unittest.TestCase):
    """apply_profile writes a valid, typed manifest into tb_root_dir."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tb_onboard_"))
        self.patch = mock.patch("toolboxv2.tb_root_dir", self.tmp)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def _manifest(self):
        return yaml.safe_load((self.tmp / "tb-manifest.yaml").read_text(encoding="utf-8"))

    def test_creates_manifest_with_profile(self):
        from toolboxv2.utils.clis.first_run import apply_profile
        res = apply_profile("developer")
        self.assertTrue(res["ok"], res["errors"])
        data = self._manifest()
        self.assertEqual(data["app"]["profile"], "developer")
        self.assertTrue(data["app"]["debug"])

    def test_lists_stay_lists(self):
        from toolboxv2.utils.clis.first_run import apply_profile
        apply_profile("server")
        services = self._manifest()["autostart"]["services"]
        self.assertIsInstance(services, list)
        self.assertIn("daemon", services)

    def test_alias_and_idempotent(self):
        from toolboxv2.utils.clis.first_run import apply_profile
        self.assertEqual(apply_profile("homelab")["profile"], "local")
        self.assertEqual(apply_profile("local")["profile"], "local")
        self.assertEqual(self._manifest()["app"]["profile"], "local")

    def test_unknown_profile_rejected(self):
        from toolboxv2.utils.clis.first_run import apply_profile
        with self.assertRaises(ValueError):
            apply_profile("nope")

    def test_onboarding_state_follows_manifest_profile(self):
        from toolboxv2.utils.clis.first_run import apply_profile, onboarding_state
        self.assertTrue(onboarding_state()["first_run"])  # no manifest yet
        apply_profile("server")
        self.assertEqual(onboarding_state(), {"installed": True, "first_run": False, "profile": "server"})

    def test_local_ui_and_wizard_use_same_state(self):
        from toolboxv2.utils.clis.first_run import apply_profile
        from toolboxv2.utils.workers.fast.local_ui import get_installation_state
        self.assertTrue(get_installation_state()["first_run"])
        apply_profile("local")
        self.assertFalse(get_installation_state()["first_run"])

    def test_non_interactive_first_run_defaults_to_local(self):
        from toolboxv2.utils.clis.first_run import run_first_run
        res = run_first_run(interactive=False)
        self.assertEqual(res["profile"], "local")
        self.assertFalse(res["login"])


class TestFailsafeKeepsOnboardingOpen(unittest.TestCase):
    def test_unonboarded_manifest_has_no_profile(self):
        from toolboxv2.init_onboarding import build_manifest
        self.assertIsNone(build_manifest("mini", save=False, onboarded=False).app.profile)
        self.assertEqual(build_manifest("mini", save=False).app.profile.value, "local")

    def test_presets_point_to_valid_profiles(self):
        from toolboxv2.init_onboarding import _PROFILE_PRESETS
        from toolboxv2.utils.manifest.schema import ProfileType
        for name, preset in _PROFILE_PRESETS.items():
            ProfileType(preset["profile"])  # raises if invalid


if __name__ == "__main__":
    unittest.main()
