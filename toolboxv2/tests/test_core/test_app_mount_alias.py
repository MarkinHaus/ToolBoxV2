"""R9 B4: Auto-Mount und start_app muessen dieselbe Source-Sprache sprechen.

isaa wird per Default-Flag schon beim Kern-Start in den Owner (8467)
geladen (_auto_mount_features). Bisher registrierte das Mount unter
"feature:isaa", waehrend start_app unter "app:isaa" sucht -> zweiter
Mount-Versuch lieferte Routen-Konflikte statt {"already": true}.

Fix: _auto_mount_features registriert unter "app:<feature-name>".

Run: python -m unittest toolboxv2.tests.test_core.test_app_mount_alias -v
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _make_feature_dir(root: Path, name: str = "isaa"):
    d = root / "features" / name
    d.mkdir(parents=True)
    (d / "feature.yaml").write_text(
        "name: " + name + "\n"
        "mount:\n"
        "  module: fake.pkg.appmod\n"
        "  prefix: " + name + "\n"
        "  app_var: app\n", encoding="utf-8")


class TestAutoMountSourceAlias(unittest.TestCase):
    def test_auto_mount_registers_app_source(self):
        import toolboxv2.utils.workers.fast.local_ui as lui

        class FakeTB:
            pass

        class FakeFastApp:
            def __init__(self):
                self.calls = []

            def mount_app(self, other, fallback_prefix, source):
                self.calls.append((other, fallback_prefix, source))
                return []

        fake_mod = types.ModuleType("fake.pkg.appmod")
        fake_mod.app = FakeTB()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_feature_dir(root)
            old = (lui.tb_root_dir, lui.importlib, lui.FastTB)
            lui.tb_root_dir = root
            lui.importlib = types.SimpleNamespace(
                import_module=lambda name: fake_mod)
            lui.FastTB = FakeTB
            try:
                app = FakeFastApp()
                lui._auto_mount_features(app)
            finally:
                (lui.tb_root_dir, lui.importlib, lui.FastTB) = old
        self.assertEqual(len(app.calls), 1, "feature not mounted")
        self.assertEqual(app.calls[0][2], "app:isaa",
                         "source must be app:<name>, got "
                         + repr(app.calls[0][2]))

    def test_start_app_already_true_when_auto_mounted(self):
        from toolboxv2.utils.workers.fast import tray_api
        from toolboxv2.utils.system import user_apps

        class Owner:
            _mounted_sources = {"app:isaa": {"prefix": "/isaa"}}

        entry = {"mount": {"module": "fake.pkg.appmod", "prefix": "isaa"}}
        orig_get = user_apps.get_user_app
        tray_api.set_owner_app(Owner())
        user_apps.get_user_app = lambda name: entry
        try:
            res = tray_api._cmd_start_app("isaa", _persist=False)
        finally:
            user_apps.get_user_app = orig_get
            tray_api.set_owner_app(None)
        self.assertEqual(res.get("merged"), True)
        self.assertEqual(res.get("already"), True, str(res))


if __name__ == "__main__":
    unittest.main()
