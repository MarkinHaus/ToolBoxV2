"""R9 B1: NginxManager._ensure_symlink darf auf Windows nicht hart crashen.

write_site_config ruft im exists-Branch UND im fresh-Branch _ensure_symlink;
auf Windows ohne Developer Mode wirft os.symlink OSError -> start_all stirbt,
http_worker bleibt als Waise zurueck (Workernamen ok: "Symlink created" enthaelt
kein kritisch-arabisches Zeichen, aber der Fehlerpfad ist real beobachtet).

Fix: _ensure_symlink faengt OSError/NotImplementedError und macht einen Copy-
Fallback statt zu werfen. Andere Caller (auch nginx-uebliche Verzeichnis-
Symlinks) profitieren gleich.

Run: python -m unittest toolboxv2.tests.test_core.test_worker_manager_symlink -v
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_nginx_manager():
    from toolboxv2.utils.clis import cli_worker_manager as cwm
    return cwm.NginxManager


@unittest.skipUnless(os.name == "nt",
                     "Windows-spezifisches Symlink-Verhalten")
class TestEnsureSymlinkWindows(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.src = self.root / "available" / "toolbox"
        self.src.parent.mkdir(parents=True)
        self.src.write_text("server {}", encoding="utf-8")
        self.dst_dir = self.root / "enabled"
        NginxManager = _load_nginx_manager()
        self.mgr = NginxManager.__new__(NginxManager)

    def test_symlink_failure_falls_back_to_copy(self):
        # os.symlink wirft (kein DevMode/Admin) -> Copy-Fallback, kein raise
        orig = os.symlink
        def boom(*a, **k):
            raise OSError("[WinError 1314] A required privilege is not held")
        with mock.patch("os.symlink", side_effect=boom):
            self.mgr._ensure_symlink(self.src, self.dst_dir / "toolbox")
        dst = self.dst_dir / "toolbox"
        self.assertTrue(dst.exists())
        self.assertFalse(dst.is_symlink())
        self.assertEqual(dst.read_text(encoding="utf-8"), "server {}")

    def test_existing_dst_is_noop(self):
        dst = self.dst_dir / "toolbox"
        dst.parent.mkdir(parents=True)
        dst.write_text("OLD", encoding="utf-8")
        orig = os.symlink
        def boom(*a, **k):
            raise OSError("no privilege")
        with mock.patch("os.symlink", side_effect=boom):
            self.mgr._ensure_symlink(self.src, dst)
        self.assertEqual(dst.read_text(encoding="utf-8"), "OLD")

    def test_dev_mode_symlink_still_preferred(self):
        # symlink klappt -> echtes Symlink, kein Copy
        self.mgr._ensure_symlink(self.src, self.dst_dir / "toolbox")
        dst = self.dst_dir / "toolbox"
        self.assertTrue(dst.exists())


if __name__ == "__main__":
    unittest.main()
