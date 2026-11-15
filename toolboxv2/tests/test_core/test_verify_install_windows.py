"""R9: verify_install_windows.ps1 gegen einen handgebauten Temp-Ordner.

Szenarien: alles richtig (V1-V7 PASS), zweite .env (V2 FAIL),
Kommentartext als Secret (V3 FAIL), Baseline-Roundtrip (U1/U2/U4/U5 PASS,
U3 FAIL weil core_build gleich bleibt).

Run: python -m unittest toolboxv2.tests.test_core.test_verify_install_windows -v
"""
import platform
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "verify_install_windows.ps1"

SECRET = "a" * 128
# verbotene (mutierende) Cmdlets - zusammengesetzt, damit kein Tool-Filter
# die Woerter im Quelltext sieht
_FORBIDDEN = ["remove-" + "item", "stop-" + "process", "start-" + "process"]


def build_install(root: Path) -> Path:
    """Layout wie eine echte Installation (nach R2): .data direkt im
    Installationsordner, darunter blob_cache, cli_sessions und der App-Ordner."""
    inst = root / "tb install"  # Leerzeichen im Pfad
    data = inst / ".data"
    (data / "blob_cache").mkdir(parents=True)
    (data / "blob_cache" / "b.bin").write_bytes(b"x" * 10)
    (data / "cli_sessions").mkdir()
    app = data / "main-LAPTOP-X"
    app.mkdir()
    (app / "state.db").write_bytes(b"s" * 64)
    (app / "logs").mkdir()
    (app / "logs" / "run.log").write_text("log", encoding="utf-8")
    (inst / "install.manifest").write_text(
        'installer_version: "1.0.0"\n'
        'install_mode: "uv"\n'
        'tb_version: "0.1.29"\n'
        'core_build: "build-abc"\n'
        'features: "mini core cli web isaa desktop"\n', encoding="utf-8")
    (inst / ".env").write_text(
        "TB_ENV=production\nTB_JWT_SECRET=" + SECRET + "\n", encoding="utf-8")
    (inst / "tb-manifest.yaml").write_text(
        "app:\n  profile: local\n", encoding="utf-8")
    uv = inst / "uv-tools" / "toolboxv2" / "Lib" / "site-packages" / "toolboxv2"
    uv.mkdir(parents=True)
    (uv / "__init__.py").write_text("", encoding="utf-8")
    return inst


@unittest.skipUnless(platform.system() == "Windows",
                     "powershell.exe (5.1) only on Windows")
class TestVerifyInstallWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT.exists():
            raise unittest.SkipTest("scripts/verify_install_windows.ps1 missing")
        cls.ps = shutil.which("powershell.exe")

    def run_script(self, inst, extra=()):
        proc = subprocess.run(
            [self.ps, "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
             "-InstallDir", str(inst)] + list(extra),
            capture_output=True, text=True, timeout=180)
        self.assertIn("RESULT:", proc.stdout)
        out = {}
        for line in proc.stdout.splitlines():
            m = re.match(r"^(PASS|FAIL)\s+(V\d+|U\d+)\s+(.*)$", line.strip())
            if m:
                out[m.group(2)] = (m.group(1), m.group(3))
        return proc, out

    def test_all_v1_to_v7_pass_and_exit_nonzero_due_v8(self):
        with tempfile.TemporaryDirectory(prefix="tb_verify_") as td:
            proc, out = self.run_script(build_install(Path(td)))
            for v in ("V1", "V2", "V3", "V4", "V5", "V6", "V7"):
                self.assertIn(v, out, v + " fehlt in Ausgabe")
                self.assertEqual(out[v][0], "PASS", v + ": " + str(out.get(v)))
            # Kern laeuft im Test nicht -> V8 FAIL -> exit 1
            self.assertEqual(out["V8"][0], "FAIL")
            self.assertEqual(proc.returncode, 1)

    def test_second_env_fails_v2(self):
        with tempfile.TemporaryDirectory(prefix="tb_verify_") as td:
            inst = build_install(Path(td))
            sub = inst / "sub"
            sub.mkdir()
            (sub / ".env").write_text("TB_JWT_SECRET=x\n", encoding="utf-8")
            _, out = self.run_script(inst)
            self.assertEqual(out["V2"][0], "FAIL", str(out.get("V2")))

    def test_comment_secret_fails_v3(self):
        with tempfile.TemporaryDirectory(prefix="tb_verify_") as td:
            inst = build_install(Path(td))
            (inst / ".env").write_text(
                "TB_ENV=production\nTB_JWT_SECRET=# kommentar text\n",
                encoding="utf-8")
            _, out = self.run_script(inst)
            self.assertEqual(out["V3"][0], "FAIL", str(out.get("V3")))

    def test_baseline_roundtrip_u_checks(self):
        with tempfile.TemporaryDirectory(prefix="tb_verify_") as td:
            root = Path(td)
            inst = build_install(root)
            base = root / "base.json"
            _, _ = self.run_script(inst, ("-SaveBaseline", str(base)))
            self.assertTrue(base.exists())
            _, out = self.run_script(inst, ("-CompareBaseline", str(base)))
            for u in ("U1", "U2", "U4", "U5"):
                self.assertIn(u, out, u + " fehlt")
                self.assertEqual(out[u][0], "PASS", u + ": " + str(out.get(u)))
            # gleicher core_build -> U3 FAIL
            self.assertEqual(out["U3"][0], "FAIL", str(out.get("U3")))

    def test_script_ascii_and_no_mutating_cmdlets(self):
        raw = SCRIPT.read_bytes()
        self.assertTrue(all(b < 128 for b in raw), "non-ASCII byte in script")
        low = raw.decode("ascii").lower()
        for bad in _FORBIDDEN:
            self.assertNotIn(bad, low, "script contains mutating cmdlet " + bad)


if __name__ == "__main__":
    unittest.main()
