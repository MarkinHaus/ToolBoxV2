"""`tb --version` prints the version and exits cleanly.

Regression: main() prints the version and closes the app; main_helper then saw
a dead app with the default runner (local-cli) and raised
"FIX DAS SOFORT WENN RUNNER local-cli muss app.alive=False == TRUE sein".
"""
import os
import subprocess
import sys
import tempfile
import unittest


class CliVersionTest(unittest.TestCase):

    def _run(self, flag):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "TB_DATA_DIR": tmp, "HOME": tmp, "USERPROFILE": tmp}
            for k in ("TOOLBOXV2_REMOTE_BASE", "TB_INSTALL_DIR"):
                env.pop(k, None)
            return subprocess.run(
                [sys.executable, "-m", "toolboxv2", flag],
                env=env, capture_output=True, text=True, timeout=180,
                stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")

    def test_version_exits_cleanly(self):
        for flag in ("--version", "-v"):
            with self.subTest(flag=flag):
                proc = self._run(flag)
                out = proc.stdout + proc.stderr
                self.assertEqual(0, proc.returncode, out[-1500:])
                self.assertIn("Version", out)
                self.assertNotIn("FIX DAS SOFORT", out)
                self.assertNotIn("Traceback", out)


if __name__ == "__main__":
    unittest.main()
