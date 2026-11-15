"""T3/F6/F7: Nutzerzustand liegt bei Wheel-Installationen ausserhalb von site-packages.

Quelle: docs/devdocs/taskmap_stable_tauri_2026-11.md Abschnitt T3.
Gate: python -m unittest toolboxv2.tests.test_core.test_state_dir -> OK
"""
import os
import tempfile
import unittest
from pathlib import Path

REPO_MANIFEST = Path(__file__).resolve().parents[2] / "tb-manifest.yaml"


class TestStateDir(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old = {
            k: os.environ.get(k)
            for k in ("TB_MANIFEST_DIR", "TB_INSTALL_DIR", "TB_DATA_DIR")
        }
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_first_run_true_with_tmp_manifest_dir(self):
        os.environ["TB_MANIFEST_DIR"] = self._tmp.name
        from toolboxv2.utils.clis.first_run import onboarding_state
        self.assertTrue(onboarding_state()["first_run"])

    def test_apply_profile_writes_to_tmp_dir(self):
        os.environ["TB_MANIFEST_DIR"] = self._tmp.name
        from toolboxv2.utils.clis.first_run import apply_profile, onboarding_state
        result = apply_profile("local")
        self.assertTrue(result.get("ok", True), f"apply_profile: {result}")
        manifest = Path(self._tmp.name) / "tb-manifest.yaml"
        self.assertTrue(manifest.exists(), "tb-manifest.yaml fehlt in TB_MANIFEST_DIR")
        state = onboarding_state()
        self.assertFalse(state["first_run"])
        self.assertEqual(state["profile"], "local")

    def test_repo_manifest_untouched(self):
        before = REPO_MANIFEST.read_bytes()
        os.environ["TB_MANIFEST_DIR"] = self._tmp.name
        from toolboxv2.utils.clis.first_run import apply_profile
        apply_profile("server")
        self.assertEqual(before, REPO_MANIFEST.read_bytes(),
                         "Repo-Manifest wurde veraendert!")

    def test_source_checkout_manifest_base_is_pkg_root(self):
        os.environ.pop("TB_MANIFEST_DIR", None)
        from toolboxv2 import tb_root_dir
        from toolboxv2._state_dir import manifest_base_dir
        self.assertEqual(manifest_base_dir(), tb_root_dir)


if __name__ == "__main__":
    unittest.main()
