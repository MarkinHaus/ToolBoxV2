"""Tests for zero-friction onboarding and local admin (unittest only)."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


class TestProfileType(unittest.TestCase):
    def test_local_profile_exists(self):
        from toolboxv2.utils.manifest.schema import ProfileType
        self.assertEqual(ProfileType.LOCAL.value, "local")


class TestManifestAutoInit(unittest.TestCase):
    def test_create_default_sets_local_profile(self):
        from toolboxv2.utils.manifest.loader import ManifestLoader
        with tempfile.TemporaryDirectory() as tmp:
            loader = ManifestLoader(tmp)
            self.assertFalse(loader.exists())
            manifest = loader.load_or_create_default()
            self.assertTrue(loader.exists())
            self.assertEqual(manifest.app.profile.value, "local")

    def test_existing_manifest_untouched(self):
        from toolboxv2.utils.manifest.loader import ManifestLoader
        with tempfile.TemporaryDirectory() as tmp:
            loader = ManifestLoader(tmp)
            loader.create_default(save=True)
            mtime = Path(loader.manifest_path).stat().st_mtime_ns
            loader2 = ManifestLoader(tmp)
            loader2.load_or_create_default()
            self.assertEqual(Path(loader2.manifest_path).stat().st_mtime_ns, mtime)


class TestEnsureLocalAdmin(unittest.TestCase):
    def test_creates_anonymous_root_once(self):
        from toolboxv2.mods.CloudM.auth import local_admin

        saved = {}

        async def fake_save(app, user):
            saved["user"] = user

        find_mock = AsyncMock(return_value=None)
        with patch.object(local_admin, "_find_user_by_email", find_mock), \
             patch.object(local_admin, "_save_user", side_effect=fake_save):
            user = asyncio.run(local_admin.ensure_local_admin(app=None))

        self.assertEqual(user.level, -1)
        self.assertTrue(user.settings["anonymous"])
        self.assertTrue(user.settings["local_admin"])
        self.assertEqual(user.email, local_admin.LOCAL_ADMIN_EMAIL)
        self.assertIs(saved["user"], user)

    def test_idempotent_returns_existing(self):
        from toolboxv2.mods.CloudM.auth import local_admin
        from toolboxv2.mods.CloudM.auth.models import UserData

        existing = UserData(user_id="usr_x", username="root",
                            email=local_admin.LOCAL_ADMIN_EMAIL, level=-1)
        save_mock = AsyncMock()
        with patch.object(local_admin, "_find_user_by_email",
                          AsyncMock(return_value=existing)), \
             patch.object(local_admin, "_save_user", save_mock):
            user = asyncio.run(local_admin.ensure_local_admin(app=None))
            self.assertIs(user, existing)
        save_mock.assert_not_awaited()


class TestUpdateUserDataRename(unittest.TestCase):
    def test_rename_flips_anonymous_flag(self):
        from toolboxv2.mods.CloudM.auth import api_session
        from toolboxv2.mods.CloudM.auth.models import UserData

        user = UserData(user_id="usr_x", username="root",
                        email="local-admin@toolbox.local", level=-1,
                        settings={"anonymous": True, "local_admin": True})
        save_mock = AsyncMock()
        with patch.object(api_session, "_load_user", AsyncMock(return_value=user)), \
             patch.object(api_session, "_save_user", save_mock):
            res = asyncio.run(api_session.update_user_data(
                app=None, user_id="usr_x", username="markin"))

        self.assertFalse(res.as_result().is_error()
                         if hasattr(res, "as_result") else res.is_error())
        self.assertEqual(user.username, "markin")
        self.assertFalse(user.settings["anonymous"])
        self.assertEqual(user.level, -1)  # level untouched by rename
        save_mock.assert_awaited_once()


class TestDashboardMagicLinkWiring(unittest.TestCase):
    def test_request_my_magic_link_passes_email_kwarg(self):
        import toolboxv2.mods.CloudM.UserDashboard as ud

        captured = {}

        class _FakeApi:
            def as_result(self):
                from toolboxv2 import Result
                return Result.ok({"message": "Magic link sent"})

        async def fake_request_magic_link(app, **kwargs):
            captured.update(kwargs)
            return _FakeApi()

        class _FakeUser:
            email = "a@b.de"
            username = "markin"

        async def fake_current_user(app, request):
            return _FakeUser()

        with patch.object(ud, "request_magic_link", fake_request_magic_link), \
             patch.object(ud, "get_current_user_from_request", fake_current_user):
            res = asyncio.run(ud.request_my_magic_link(app=None, request=object()))

        self.assertEqual(captured.get("email"), "a@b.de")
        self.assertNotIn("username", captured)
        self.assertFalse(res.is_error())

class TestCustomServiceCommandBuild(unittest.TestCase):
    def test_named_custom_service_builds_tb_args(self):
        import sys
        from toolboxv2.utils.clis import service_manager as sm

        captured = {}

        class FakeMgr(sm.ServiceManager):
            def __init__(self):
                pass  # skip filesystem setup
            def load_config(self):
                return {"services": {"myflow": {"custom": True, "args": ["run", "x"]}}}
            def is_service_running(self, name):
                return (False, None)
            def get_service_args(self, name):
                return ["run", "x"]
            def configure_service(self, *a, **k):
                pass

        mgr = FakeMgr()
        with patch.object(sm.subprocess, "Popen") as popen, \
             patch.object(sm.time, "sleep", lambda *_: None):
            popen.return_value.pid = 4242
            # second is_service_running (verify) — patch instance to report running
            calls = {"n": 0}
            def running(name):
                calls["n"] += 1
                return (calls["n"] > 1, 4242 if calls["n"] > 1 else None)
            mgr.is_service_running = running
            mgr.pids_dir = __import__("pathlib").Path(tempfile.mkdtemp())
            mgr.start_service("myflow", args=["run", "x"])

        cmd = popen.call_args[0][0]
        self.assertEqual(cmd[:3], [sys.executable, "-m", "toolboxv2"])
        self.assertNotIn("myflow", cmd)         # custom → no service name in cmd
        self.assertEqual(cmd[3:], ["run", "x"])  # user args appended

if __name__ == "__main__":
    unittest.main()
