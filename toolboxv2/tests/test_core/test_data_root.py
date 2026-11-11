import os, tempfile, unittest
from pathlib import Path
from unittest import mock


class DataRootTest(unittest.TestCase):
    KEYS = ("TB_DATA_DIR", "TB_INSTALL_DIR")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self.KEYS}
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        for k in self.KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def test_tb_data_dir_wins_verbatim(self):
        from toolboxv2._state_dir import data_root
        dd = self.tmp / "dockermount"
        os.environ["TB_DATA_DIR"] = str(dd)
        self.assertEqual(dd, Path(data_root()))
        self.assertFalse((Path(data_root())).name == ".data")

    def test_packaged_falls_back_to_user_state_dir_data(self):
        import toolboxv2._state_dir as sd
        install = self.tmp / "install"
        os.environ["TB_INSTALL_DIR"] = str(install)
        with mock.patch.object(sd, "is_packaged_install", lambda: True):
            self.assertEqual(install / ".data", Path(sd.data_root()))

    def test_source_falls_back_to_tb_root_data(self):
        import toolboxv2._state_dir as sd
        with mock.patch.object(sd, "is_packaged_install", lambda: False):
            self.assertEqual(Path(sd.__file__).resolve().parent.parent / ".data",
                             Path(sd.data_root()))

    def test_info_dir_rule(self):
        from toolboxv2._state_dir import info_dir
        os.environ["TB_DATA_DIR"] = str(self.tmp / "dd")
        self.assertEqual(self.tmp / "dd" / ".info", Path(info_dir()))


class SitePackagesCleanTest(unittest.TestCase):
    """R2: packaged code must not write .data/.config/.info into site-packages."""

    def _pkg(self):
        import toolboxv2._state_dir as sd
        install = Path(os.environ["TB_INSTALL_DIR"])
        return mock.patch.object(sd, "is_packaged_install", lambda: True)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        install = self._tmp.name
        os.environ["TB_INSTALL_DIR"] = install
        self.install = Path(install)
        for k in ("TB_DATA_DIR",):
            self._old = os.environ.pop(k, None)

    def tearDown(self):
        self._tmp.cleanup()
        if self._old is not None:
            os.environ["TB_DATA_DIR"] = self._old

    def test_blob_storage_default_goes_to_data_dir(self):
        import inspect
        import toolboxv2.utils.extras.blobs as bl
        src = inspect.getsource(bl)
        self.assertNotIn("tb_root_dir / \".data\"", src,
                         "R2: blobs must use data_root(), not tb_root_dir")

    def test_login_cli_sessions_not_tb_root(self):
        import inspect
        import toolboxv2.mods.CloudM.LogInSystem as li
        src = inspect.getsource(li)
        self.assertNotIn("tb_root_dir / \".data\"", src)

    def test_session_fallback_no_pkg_relative(self):
        import inspect
        import toolboxv2.utils.system.session as ss
        src = inspect.getsource(ss)
        self.assertNotIn("Path(__file__).parent.parent.parent / \".data\"", src)

    def test_info_family_uses_info_dir(self):
        import inspect
        import toolboxv2.utils.clis.service_manager as sm
        src = inspect.getsource(sm)
        self.assertNotIn("tb_root_dir / \".info\"", src)


class TbDataDirNotNestedTest(unittest.TestCase):
    """R2: App data_dir must be TB_DATA_DIR verbatim, not TB_DATA_DIR/.data."""

    def test_app_prefix_file_uses_data_dir(self):
        import inspect
        import toolboxv2.utils.toolbox as tb
        src = inspect.getsource(tb)
        self.assertIn("TB_DATA_DIR", src)
        # no second lapp writer that appends .data to TB_DATA_DIR
        self.assertNotIn('os.getenv("TB_DATA_DIR") or dir_name, ".data"', src)


if __name__ == "__main__":
    unittest.main()
