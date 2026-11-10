"""installer.sh JSON-Lines protocol as consumed by Tauri (installer.rs + installer.html).

Dry run, exactly like Tauri calls it: `installer.sh --json-output --config <yaml>`.
A curl stub answers only the two connectivity probes; everything else is real.

Run: python -m unittest toolboxv2.tests.test_core.test_installer_protocol -v
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "installer.sh"
TAURI_COPY = ROOT / "toolboxv2" / "simple-core" / "src-tauri" / "installer.sh"


@unittest.skipUnless(shutil.which("bash") and shutil.which("curl"), "bash/curl required")
class TestInstallerProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tb_inst_"))
        (cls.tmp / "home").mkdir()
        stub = cls.tmp / "bin" / "curl"
        stub.parent.mkdir()
        real_curl = shutil.which("curl")
        stub.write_text("#!/usr/bin/env bash\n"
                        'for a in "$@"; do case "$a" in https://1.1.1.1|https://github.com) exit 0;; esac; done\n'
                        f'exec {real_curl} "$@"\n')
        stub.chmod(0o755)
        cfg = cls.tmp / "tb-install.yaml"
        cfg.write_text(f"install_mode: uv\ninstall_path: {cls.tmp / 'tbhome'}\nprofile: developer\n"
                       "environment: development\nassume: yes\naction: install\nui: none\n")
        env = {**os.environ, "HOME": str(cls.tmp / "home"),
               "PATH": f"{stub.parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        cls.proc = subprocess.run(
            ["bash", str(INSTALLER), "--json-output", "--config", str(cfg), "--dry-run"],
            cwd=cls.tmp, env=env, capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL)
        cls.events = [json.loads(l) for l in cls.proc.stdout.splitlines() if l.strip()]

    def test_exit_ok_and_config_path_parsed(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-2000:])
        self.assertNotIn("Config file not found", self.proc.stderr)

    def test_stdout_is_only_json_with_type(self):
        # json.loads in setUpClass already proved every line is JSON
        self.assertTrue(self.events)
        for ev in self.events:
            self.assertIn(ev["type"], {"progress", "error", "log", "done"}, ev)

    def test_progress_is_fraction_and_monotonic_enough(self):
        prog = [ev["progress"] for ev in self.events if ev["type"] == "progress"]
        self.assertTrue(all(0 <= p <= 1 for p in prog))
        self.assertGreaterEqual(prog[-1], 0.8)

    def test_ends_with_done(self):
        last = self.events[-1]
        self.assertEqual(last["type"], "done")
        self.assertEqual(last["profile"], "developer")
        self.assertEqual(last["progress"], 1)

    def test_fail_emits_error_event(self):
        proc = subprocess.run(["bash", str(INSTALLER), "--json-output", "--config", "/nope.yaml",
                               "--dry-run"], cwd=self.tmp, capture_output=True, text=True,
                              timeout=120, stdin=subprocess.DEVNULL, env={**os.environ, "HOME": str(self.tmp / "home"),
                                                "PATH": f"{self.tmp / 'bin'}{os.pathsep}{os.environ['PATH']}"})
        self.assertNotEqual(proc.returncode, 0)
        events = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("/nope.yaml", events[-1]["msg"])

    def test_bundled_uv_sidecar_is_used(self):
        """TB_UV_BIN (Tauri sidecar) wins over PATH and the astral.sh bootstrap."""
        fake = self.tmp / "sidecar" / "uv"
        fake.parent.mkdir(exist_ok=True)
        fake.write_text("#!/usr/bin/env bash\nexit 0\n")
        fake.chmod(0o755)
        cfg = self.tmp / "tb-install.yaml"
        proc = subprocess.run(
            ["bash", str(INSTALLER), "--json-output", "--config", str(cfg), "--dry-run"],
            cwd=self.tmp, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
            env={**os.environ, "HOME": str(self.tmp / "home"), "TB_UV_BIN": str(fake),
                 "PATH": f"{self.tmp / 'bin'}{os.pathsep}/usr/bin{os.pathsep}/bin"})
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        self.assertIn(f"Runtime: bundled uv at {fake}", proc.stderr)  # dry-run: no copy
        self.assertNotIn("bootstrapping uv", proc.stderr)

    def test_tauri_bundle_copy_in_sync(self):
        self.assertEqual(INSTALLER.read_bytes(), TAURI_COPY.read_bytes(),
                         "src-tauri/installer.sh drifted from installer.sh")


class TestOnboardingUiSingleSource(unittest.TestCase):
    def test_web_copy_matches_tauri_bundle(self):
        web = ROOT / "toolboxv2" / "web" / "installer" / "index.html"
        tauri = ROOT / "toolboxv2" / "simple-core" / "src-tauri" / "installer.html"
        self.assertEqual(web.read_bytes(), tauri.read_bytes(),
                         "web/installer/index.html drifted from src-tauri/installer.html")


if __name__ == "__main__":
    unittest.main()
