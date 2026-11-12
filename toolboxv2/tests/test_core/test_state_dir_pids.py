"""R9 B2: PID-Dateien liegen im State-Root (.info/pids), nie im Paketordner.

setup_app/main (__main__.py) bauten info_folder aus dem Paket-abspath ->
bei Wheel-Installationen landete .info/pids in site-packages (V6-FAIL).

Fix: _state_dir.pids_dir() als eine Quelle; __main__ nutzt sie.

Run: python -m unittest toolboxv2.tests.test_core.test_state_dir_pids -v
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestPidsDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old = {k: os.environ.get(k)
                     for k in ("TB_INSTALL_DIR", "TB_DATA_DIR")}
        os.environ.pop("TB_INSTALL_DIR", None)
        os.environ.pop("TB_DATA_DIR", None)
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_pids_dir_under_tb_data_dir(self):
        from toolboxv2 import _state_dir
        os.environ["TB_DATA_DIR"] = self._tmp.name
        self.assertEqual(_state_dir.pids_dir(),
                         Path(self._tmp.name) / ".info" / "pids")

    def test_pids_dir_under_packaged_install_dir(self):
        from toolboxv2 import _state_dir
        os.environ["TB_INSTALL_DIR"] = self._tmp.name
        with mock.patch.object(_state_dir, "is_packaged_install",
                               return_value=True):
            self.assertEqual(_state_dir.pids_dir(),
                             Path(self._tmp.name) / ".info" / "pids")

    def test_main_uses_state_dir_helper(self):
        import toolboxv2
        src = (Path(toolboxv2.__file__).parent / "__main__.py").read_text(
            encoding="utf-8")
        self.assertNotIn('abspath + "\\\\.info', src,
                         "__main__ baut .info noch aus Paket-abspath")
        self.assertIn("pids_dir", src)
        compile(src, "__main__.py", "exec")
        gca = (Path(toolboxv2.__file__).parent / "utils" / "system" / "getting_and_closing_app.py").read_text(encoding="utf-8")
        self.assertNotIn("{app.start_dir}", gca.split("pid_file")[1][:100])


if __name__ == "__main__":
    unittest.main()


