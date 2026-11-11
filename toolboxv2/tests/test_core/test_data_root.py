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

    def test_tb_data_dir_is_the_state_root(self):
        """Explicit TB_DATA_DIR (e.g. Docker volume): exactly ONE .data level below it."""
        import toolboxv2._state_dir as sd
        dd = self.tmp / "dockermount"
        os.environ["TB_DATA_DIR"] = str(dd)
        self.assertEqual(dd, sd.state_root())
        self.assertEqual(dd / ".data", sd.data_root())
        self.assertEqual(dd / ".info", sd.info_dir())

    def test_packaged_state_root_is_install_dir(self):
        import toolboxv2._state_dir as sd
        install = self.tmp / "install"
        os.environ["TB_INSTALL_DIR"] = str(install)
        with mock.patch.object(sd, "is_packaged_install", lambda: True):
            self.assertEqual(install, sd.state_root())
            self.assertEqual(install / ".data", sd.data_root())
            self.assertEqual(install / ".info", sd.info_dir())

    def test_source_checkout_layout_is_unchanged(self):
        """Source checkout keeps the historic layout: <package dir>/.data, /.info."""
        import toolboxv2
        import toolboxv2._state_dir as sd
        with mock.patch.object(sd, "is_packaged_install", lambda: False):
            self.assertEqual(Path(toolboxv2.tb_root_dir), sd.state_root())
            self.assertEqual(Path(toolboxv2.tb_root_dir) / ".data", sd.data_root())
            self.assertEqual(Path(toolboxv2.tb_root_dir) / ".info", sd.info_dir())

    def test_app_dirs_are_one_level_below_state_root(self):
        """The running App uses <state root>/.data/<id> -- never .data/.data."""
        import toolboxv2
        import toolboxv2._state_dir as sd
        app = toolboxv2.get_app("test_data_root")
        for d in (app.data_dir, app.config_dir, app.info_dir):
            parts = Path(d).parts
            self.assertFalse(
                any(parts[i] == ".data" and parts[i + 1] == ".data" for i in range(len(parts) - 1)),
                f"nested .data in {d}")
        self.assertEqual(".data", Path(app.data_dir).parent.name)
        self.assertEqual(".config", Path(app.config_dir).parent.name)
        self.assertEqual(".info", Path(app.info_dir).parent.name)


class InstalledRootTest(unittest.TestCase):
    """One rule for installer and core: the core finds the dir the installer used."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._saved = os.environ.pop("TB_INSTALL_DIR", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["TB_INSTALL_DIR"] = self._saved
        self._tmp.cleanup()

    def _fake_pkg(self, layout: str, manifest: bool) -> Path:
        install = self.tmp / "custom dir"
        pkg = install / layout / "toolboxv2"
        pkg.mkdir(parents=True)
        if manifest:
            (install / "install.manifest").write_text("install_mode: uv\n", encoding="utf-8")
        return pkg / "_state_dir.py"

    def _root_for(self, fake_file: Path):
        import toolboxv2._state_dir as sd
        with mock.patch.object(sd, "__file__", str(fake_file)):
            return sd._installed_root(), sd.user_state_dir()

    def test_uv_layout_unix(self):
        f = self._fake_pkg("uv-tools/toolboxv2/lib/python3.13/site-packages", True)
        root, usd = self._root_for(f)
        self.assertEqual((self.tmp / "custom dir").resolve(), root)
        self.assertEqual(root, usd)

    def test_uv_layout_windows(self):
        f = self._fake_pkg("uv-tools/toolboxv2/Lib/site-packages", True)
        self.assertEqual((self.tmp / "custom dir").resolve(), self._root_for(f)[0])

    def test_venv_layout(self):
        f = self._fake_pkg(".venv/lib/python3.13/site-packages", True)
        self.assertEqual((self.tmp / "custom dir").resolve(), self._root_for(f)[0])

    def test_plain_project_venv_is_not_an_installation(self):
        """No install.manifest -> not ours -> per-user default, never the project dir."""
        import toolboxv2._state_dir as sd
        f = self._fake_pkg(".venv/lib/python3.13/site-packages", False)
        root, usd = self._root_for(f)
        self.assertIsNone(root)
        self.assertEqual(sd._default_user_dir(), usd)

    def test_tb_install_dir_wins(self):
        import toolboxv2._state_dir as sd
        f = self._fake_pkg("uv-tools/toolboxv2/lib/python3.13/site-packages", True)
        os.environ["TB_INSTALL_DIR"] = str(self.tmp / "forced")
        with mock.patch.object(sd, "__file__", str(f)):
            self.assertEqual(self.tmp / "forced", sd.user_state_dir())


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


class NoStateInPackageDirTest(unittest.TestCase):
    """Writers of local state go through _state_dir, not through tb_root_dir."""

    def test_file_handler_default_base_is_state_root(self):
        import inspect
        import toolboxv2.utils.system.file_handler as fh
        src = inspect.getsource(fh.FileHandlerV2._get_default_base_path)
        self.assertIn("state_root", src)
        self.assertNotIn("tb_root_dir", src)


if __name__ == "__main__":
    unittest.main()
