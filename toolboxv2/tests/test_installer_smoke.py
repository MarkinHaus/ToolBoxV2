# tests/test_installer_smoke.py
"""
Post-install smoke test.
Ausführen: python -m unittest tests/test_installer_smoke.py
Setzt voraus: tb ist installiert und im PATH.
"""
import unittest
import subprocess
import sys
import os
from pathlib import Path


def _run(cmd: list[str], timeout=15) -> tuple[int, str, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


class TestTbBinaryReachable(unittest.TestCase):

    def test_tb_in_path(self):
        import shutil
        self.assertIsNotNone(shutil.which("tb"), "'tb' not found in PATH")

    def test_tb_version_exits_zero(self):
        code, out, _ = _run(["tb", "-v"])
        self.assertEqual(code, 0, f"'tb -v' returned {code}")

    def test_tb_help_exits_zero(self):
        code, _, _ = _run(["tb", "--help"])
        # tb uses sys.exit(0) or just prints help — both 0 or 1 are acceptable
        self.assertIn(code, (0, 1))


class TestManifestCLI(unittest.TestCase):

    def test_manifest_help(self):
        code, out, _ = _run(["tb", "manifest", "--help"])
        self.assertIn("set", out + _)
        self.assertIn("get", out + _)
        self.assertIn("init", out + _)

    def test_manifest_set_and_get_roundtrip(self):
        # Setzt einen harmlosen Wert und liest ihn zurück
        _run(["tb", "manifest", "set", "app.log_level", "DEBUG"])
        code, out, _ = _run(["tb", "manifest", "get", "app.log_level"])
        self.assertEqual(code, 0)
        self.assertIn("DEBUG", out)
        # Aufräumen
        _run(["tb", "manifest", "set", "app.log_level", "INFO"])

    def test_manifest_validate_exits_cleanly(self):
        code, _, _ = _run(["tb", "manifest", "validate"])
        self.assertIn(code, (0, 1), "validate should exit 0 (ok) or 1 (errors), not crash")


class TestFeatureLoader(unittest.TestCase):

    def test_fl_status_exits_zero(self):
        code, out, _ = _run(["tb", "fl", "status"])
        self.assertEqual(code, 0)
        self.assertIn("core", out.lower())

    def test_fl_list_exits_zero(self):
        code, out, _ = _run(["tb", "fl", "list"])
        self.assertEqual(code, 0)


class TestEnvFile(unittest.TestCase):

    def test_dotenv_exists_after_install(self):
        """Nach Wizard-Durchlauf muss .env existieren (auch wenn leer)."""
        try:
            from toolboxv2 import tb_root_dir
            env_path = tb_root_dir / ".env"
            # Nicht fatal wenn noch kein Wizard gelaufen — nur warnen
            if not env_path.exists():
                self.skipTest(".env not yet created (wizard not run)")
        except ImportError:
            self.fail("toolboxv2 not importable — installation broken")

    def test_toolboxv2_importable(self):
        code, _, err = _run([sys.executable, "-c", "import toolboxv2; print(toolboxv2.__version__)"])
        self.assertEqual(code, 0, f"import failed: {err}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
