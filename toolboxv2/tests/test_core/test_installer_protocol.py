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


_WSL = False
try:
    _b = subprocess.run(["bash", "-c", "uname -r"], capture_output=True, text=True, timeout=15)
    _WSL = "microsoft" in _b.stdout.lower()
except Exception:
    pass


@unittest.skipUnless(shutil.which("bash") and shutil.which("curl") and not _WSL,
                     "bash/curl required (WSL bash cannot see Windows paths)")
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


@unittest.skipUnless(shutil.which("bash") and not _WSL, "needs bash with native paths")
class TestCoreUpdateInPlace(unittest.TestCase):
    """`--update --path` as the Tauri app calls it for a core update: reinstall
    the bundled wheel, record the build id, keep everything else."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tb_update_"))
        cls.install = cls.tmp / "my install"
        (cls.install / "bin").mkdir(parents=True)
        (cls.tmp / "home").mkdir()
        cls.uv_log = cls.tmp / "uv.log"
        uv = cls.install / "bin" / "uv"
        uv.write_text("#!/usr/bin/env bash\n"
                      f'echo "$@" >> "{cls.uv_log}"\n'
                      'if [ "$1 $2" = "tool list" ]; then echo "toolboxv2 v9.9.9"; fi\n'
                      'if [ "$1 $2 $3" = "tool dir --bin" ]; then echo "$UV_TOOL_BIN_DIR"; fi\n'
                      "exit 0\n")
        uv.chmod(0o755)
        (cls.install / "install.manifest").write_text(
            'installer_version: "1.0.0"\n'
            'install_mode: "uv"\n'
            'source_from: "git"\n'
            'source_branch: "master"\n'
            'tb_version: "0.0.1"\n'
            'core_build: "old-build"\n'
            f'toolbox_home: "{cls.install}"\n'
            'runtime: "uv"\n'
            'python_path: ""\n'
            f'uv_path: "{uv}"\n'
            'features: "mini core cli web isaa desktop"\n'
            'optional_installed: "minio"\n', encoding="utf-8")
        cls.env_before = 'TB_ENV="production"\nTB_JWT_SECRET=abc\n'
        (cls.install / ".env").write_text(cls.env_before, encoding="utf-8")
        (cls.install / "tb-manifest.yaml").write_text("app:\n  profile: developer\n", encoding="utf-8")
        wheel = cls.tmp / "toolboxv2-9.9.9-py3-none-any.whl"
        wheel.write_bytes(b"")
        cls.wheel = wheel
        env = {**os.environ, "HOME": str(cls.tmp / "home"), "TB_WHEEL": str(wheel),
               "TB_CORE_BUILD": "new-build"}
        for k in ("TB_INSTALL_DIR", "TOOLBOX_HOME"):
            env.pop(k, None)
        cls.proc = subprocess.run(
            ["bash", str(INSTALLER), "--json-output", "--update", "--yes", "--path", str(cls.install)],
            cwd=cls.tmp, env=env, capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL)
        cls.manifest = (cls.install / "install.manifest").read_text(encoding="utf-8")

    def test_exit_ok_and_done_event(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-2000:])
        events = [json.loads(l) for l in self.proc.stdout.splitlines() if l.strip()]
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["version"], "9.9.9")
        self.assertEqual(events[-1]["path"], str(self.install))
        self.assertEqual(events[-1]["profile"], "", "an update applies no profile")

    def test_wheel_reinstalled_with_the_installed_features(self):
        log = self.uv_log.read_text(encoding="utf-8")
        self.assertIn(f"tool install ToolBoxV2[cli,web,isaa,desktop] @ file://{self.wheel} --force", log)

    def test_manifest_records_new_build_and_keeps_the_rest(self):
        self.assertIn('core_build: "new-build"', self.manifest)
        self.assertIn('tb_version: "9.9.9"', self.manifest)
        self.assertIn('install_mode: "uv"', self.manifest)
        self.assertIn('features: "mini core cli web isaa desktop"', self.manifest,
                      "features must not grow on every update")
        self.assertIn('optional_installed: "minio"', self.manifest)

    def test_profile_and_env_untouched(self):
        self.assertEqual(self.env_before, (self.install / ".env").read_text(encoding="utf-8"))
        self.assertEqual("app:\n  profile: developer\n",
                         (self.install / "tb-manifest.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("-init", self.uv_log.read_text(encoding="utf-8"))
        self.assertNotIn("Applying profile", self.proc.stderr)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(not _WSL, "wheel dry-run probe uses Windows paths (WSL cannot see them)")
class TestWheelAndFeaturesDryRun(unittest.TestCase):
    """T5/F1-F3: tb_features aus Config + TB_WHEEL als Wheel-Quelle (dry run)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tb_wheel_"))
        (cls.tmp / "home").mkdir()
        stub = cls.tmp / "bin" / "curl"
        stub.parent.mkdir()
        real_curl = shutil.which("curl")
        stub.write_text("#!/usr/bin/env bash\n"
                        'for a in "$@"; do case "$a" in https://1.1.1.1|https://github.com) exit 0;; esac; done\n'
                        f'exec {real_curl} "$@"\n')
        stub.chmod(0o755)
        wheel = cls.tmp / "toolboxv2-9.9.9-py3-none-any.whl"
        wheel.write_bytes(b"")
        cfg = cls.tmp / "tb-install.yaml"
        cfg.write_text(f"install_mode: uv\ninstall_path: {cls.tmp / 'tbhome'}\nprofile: developer\n"
                       "environment: development\nassume: yes\naction: install\nui: none\n"
                       "tb_features: cli,web,isaa\n")
        env = {**os.environ, "HOME": str(cls.tmp / "home"), "TB_WHEEL": str(wheel),
               "PATH": f"{stub.parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        cls.proc = subprocess.run(
            ["bash", str(INSTALLER), "--json-output", "--config", str(cfg), "--dry-run"],
            cwd=cls.tmp, env=env, capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL)

    def test_exit_ok_and_json_only(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-2000:])
        events = [json.loads(l) for l in self.proc.stdout.splitlines() if l.strip()]
        self.assertTrue(events)
        self.assertEqual(events[-1]["type"], "done")

    def test_dry_run_names_pkg_with_extras_and_file_url(self):
        self.assertIn("ToolBoxV2[cli,web,isaa] @ file://", self.proc.stderr)
        self.assertIn("toolboxv2-9.9.9-py3-none-any.whl", self.proc.stderr)
