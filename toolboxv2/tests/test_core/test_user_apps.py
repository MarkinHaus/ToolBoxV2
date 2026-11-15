"""User-facing app registry + lazy start into the local owner.

Run: python -m unittest toolboxv2.tests.test_core.test_user_apps -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


def _feature(root: Path, name: str, user_facing, installed=True):
    d = root / name
    d.mkdir(parents=True)
    (d / "feature.yaml").write_text(yaml.safe_dump({"name": name, "user_facing": user_facing}),
                                    encoding="utf-8")
    if installed:
        (d / ".installed").touch()


ISAA = {"name": "isaa", "function": "ui", "label": "ISAA", "icon": "smart_toy",
        "mount": {"module": "toolboxv2.mods.isaa.ui.app", "attr": "app", "prefix": "isaa"}}


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="tb_apps_"))

    def test_only_installed_features(self):
        from toolboxv2.utils.system.user_apps import all_user_apps
        _feature(self.root, "isaa", [ISAA])
        _feature(self.root, "other", [{"name": "x"}], installed=False)
        self.assertEqual([a["name"] for a in all_user_apps(self.root)], ["isaa"])

    def test_mount_parsed_and_invalid_entries_skipped(self):
        from toolboxv2.utils.system.user_apps import all_user_apps
        _feature(self.root, "f", [ISAA, {"label": "no name"}, {"name": "cli_only", "mount": {}}])
        apps = {a["name"]: a for a in all_user_apps(self.root)}
        self.assertEqual(set(apps), {"isaa", "cli_only"})
        self.assertEqual(apps["isaa"]["mount"]["prefix"], "isaa")
        self.assertIsNone(apps["cli_only"]["mount"])  # CLI-only, not launchable on web

    def test_isaa_feature_declares_lazy_app(self):
        from toolboxv2 import tb_root_dir
        data = yaml.safe_load((tb_root_dir / "features/isaa/feature.yaml").read_text(encoding="utf-8"))
        entry = next(e for e in data["user_facing"] if e["name"] == "isaa")
        self.assertEqual(entry["mount"]["module"], "toolboxv2.mods.isaa.ui.app")

    def test_obs_feature_declares_lazy_app(self):
        from toolboxv2 import tb_root_dir
        data = yaml.safe_load((tb_root_dir / "features/isaa/feature.yaml").read_text(encoding="utf-8"))
        entry = next(e for e in data["user_facing"] if e["name"] == "obs")
        self.assertEqual(entry["mount"]["module"], "toolboxv2.mods.isaa.extras.live_obs_server")
        self.assertEqual(entry["mount"]["prefix"], "obs")


class _Owner:
    """Interface-level stand-in for a FastTB owner."""
    def __init__(self, conflicts=None):
        self._mounted_sources = {}
        self.conflicts = conflicts or []

    def mount_app(self, joiner, prefix, source):
        if not self.conflicts:
            self._mounted_sources[source] = {"prefix": prefix}
        return self.conflicts


class TestStartApp(unittest.TestCase):
    def setUp(self):
        from toolboxv2.utils.workers.fast import tray_api
        self.tray = tray_api
        root = Path(tempfile.mkdtemp(prefix="tb_apps_"))
        _feature(root, "isaa", [ISAA, {"name": "cli_only"}])
        from toolboxv2.utils.system import user_apps
        self.p_dir = mock.patch.object(user_apps, "_features_dir", return_value=root)
        self.p_dir.start()
        self.p_load = mock.patch.object(tray_api, "reload_from_src", return_value=object())
        self.loader = self.p_load.start()

    def tearDown(self):
        self.p_dir.stop()
        self.p_load.stop()
        self.tray.set_owner_app(None)

    def test_mounts_registered_app_once(self):
        owner = _Owner()
        self.tray.set_owner_app(owner)
        res = self.tray._cmd_start_app("isaa")
        self.assertEqual(res, {"merged": True, "prefix": "/isaa", "source": "app:isaa"})
        self.loader.assert_called_once_with("toolboxv2.mods.isaa.ui.app", "app", include=None, exclude=None)
        again = self.tray._cmd_start_app("isaa")
        self.assertTrue(again["already"])
        self.assertEqual(self.loader.call_count, 1)  # idempotent, no second import

    def test_unknown_or_cli_only_app_refused(self):
        self.tray.set_owner_app(_Owner())
        self.assertFalse(self.tray._cmd_start_app("nope")["merged"])
        self.assertFalse(self.tray._cmd_start_app("cli_only")["merged"])
        self.loader.assert_not_called()

    def test_conflict_is_reported(self):
        self.tray.set_owner_app(_Owner(conflicts=["GET /static"]))
        res = self.tray._cmd_start_app("isaa")
        self.assertFalse(res["merged"])
        self.assertEqual(res["conflicts"], ["GET /static"])

    def test_no_owner(self):
        self.assertFalse(self.tray._cmd_start_app("isaa")["merged"])

    def test_registered_as_command(self):
        self.tray.register_collective_commands(_Owner())
        self.assertIs(self.tray._command_handlers["start_app"], self.tray._cmd_start_app)


class TestIsaaLazyAttr(unittest.TestCase):
    def test_unknown_attr_raises(self):
        try:
            import toolboxv2.mods.isaa.ui.app as ui
        except ImportError as e:
            self.skipTest(f"isaa ui deps missing: {e}")
        with self.assertRaises(AttributeError):
            ui.nope  # noqa: B018


if __name__ == "__main__":
    unittest.main()
