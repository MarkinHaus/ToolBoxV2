"""WP1 K3: account-link flow - onboarding gates, wizard API endpoints,
account page, local UI routes. Everything unit-level: remote login and
device linking are mocked, no browser and no network involved.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HOSTS = ("toolboxv2._state_dir.user_state_dir",
         "toolboxv2.mods.CloudM.auth.device_account.device_owner_file")


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _FakeSession:
    def __init__(self):
        self.base = None
        self.username = None
        self.saved = None

    def _save_session_token(self, access, refresh, user_id):
        self.saved = (access, refresh, user_id)

    async def login(self, verbose=False):
        return True


class AccountFlowTest(unittest.TestCase):
    """onboarding_state + wizard_api endpoints on a real FS sandbox."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.app = SimpleNamespace(data_dir=str(self.tmp), session=_FakeSession())
        self.owner = {"user_id": "usr_acc1", "username": "markin"}

        import toolboxv2.utils.clis.first_run as fr
        import toolboxv2.mods.CloudM.auth.device_account as da
        self.fr = fr
        self.da = da
        self.db = {}

        async def fake_find_by_email(app, email):
            uid = self.db.get(f"AUTH_USER_EMAIL::{email}")
            if not uid:
                return None
            data = self.db.get(f"AUTH_USER::{uid}")
            return self.da.UserData.from_dict(data) if data else None

        async def fake_load(app, user_id):
            data = self.db.get(f"AUTH_USER::{user_id}")
            return self.da.UserData.from_dict(data) if data else None

        async def fake_save(app, user):
            self.db[f"AUTH_USER::{user.user_id}"] = user.to_dict()
            if user.email:
                self.db[f"AUTH_USER_EMAIL::{user.email}"] = user.user_id

        async def fake_delete(app, key):
            self.db.pop(key, None)

        self._db_patches = [
            mock.patch.object(da, "_find_user_by_email", fake_find_by_email),
            mock.patch.object(da, "_load_user", fake_load),
            mock.patch.object(da, "_save_user", fake_save),
            mock.patch.object(da, "_db_delete", fake_delete),
        ]
        for p in self._db_patches:
            p.start()
            self.addCleanup(p.stop)

        self._patches = [
            mock.patch.object(da, "device_owner_file", lambda a: self.tmp / "device_owner.json"),
            mock.patch.object(fr, "_read_device_owner",
                              lambda a=None: self.da.read_device_owner(self.app)),
            mock.patch.object(fr, "_start_account_link",
                              lambda a=None, method=None, timeout=180.0: self._link(method)),
        ]
        import toolboxv2._state_dir as sd
        self._patches.append(
            mock.patch.object(sd, "manifest_base_dir", lambda: self.tmp))
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _link(self, method):
        run(self.da.link_account(self.app, {
            "access_token": "at", "refresh_token": "rt",
            "user_id": self.owner["user_id"], "username": self.owner["username"],
        }))
        return self.da.read_device_owner(self.app)

    def test_needs_account_true_without_owner_local_profile(self):
        self.fr.apply_profile("local")
        state = self.fr.onboarding_state()
        self.assertTrue(state["needs_account"])

    def test_needs_account_false_for_server_profile(self):
        self.fr.apply_profile("server")
        self._link(None)
        state = self.fr.onboarding_state()
        self.assertFalse(state["needs_account"])

    def test_needs_account_false_without_profile(self):
        state = self.fr.onboarding_state()
        self.assertFalse(state["needs_account"])

    def test_needs_account_false_after_link(self):
        self._link(None)
        state = self.fr.onboarding_state()
        self.assertFalse(state["needs_account"])
        self.assertTrue(state["account"])

    def test_account_state_shape(self):
        self.assertFalse(self.fr.account_state()["linked"])
        self._link(None)
        st = self.fr.account_state()
        self.assertTrue(st["linked"])
        self.assertEqual(st["username"], "markin")
        self.assertNotIn("access_token", json.dumps(st))

    def test_first_run_and_needs_account_independent(self):
        self._link(None)
        state = self.fr.onboarding_state()
        self.assertFalse(state["needs_account"])
        self.assertTrue(state["first_run"])

    def test_wizard_state_endpoint_reports_account(self):
        from toolboxv2.utils.workers.fast import wizard_api
        with mock.patch("toolboxv2.utils.clis.first_run.onboarding_state", self.fr.onboarding_state):
            routes = {}
            class _App:
                def get(self, path, auth=False):
                    def deco(fn):
                        routes[path] = fn
                        return fn
                    return deco
                def post(self, path, auth=False):
                    def deco(fn):
                        routes[path] = fn
                        return fn
                    return deco
            wizard_api.register_wizard_routes(_App())
            self._link(None)
            res = routes["/api/wizard/state"](None)
        self.assertFalse(res["needs_account"])
        self.assertEqual(res["account"], "markin")

    def test_start_endpoint_rejects_unknown_method(self):
        from toolboxv2.utils.workers.fast import wizard_api
        with mock.patch("toolboxv2.utils.clis.first_run.onboarding_state", self.fr.onboarding_state):
            routes = {}
            class _App:
                def get(self, path, auth=False):
                    def deco(fn):
                        routes[path] = fn
                        return fn
                    return deco
                def post(self, path, auth=False):
                    def deco(fn):
                        routes[path] = fn
                        return fn
                    return deco
            wizard_api.register_wizard_routes(_App())
            res = routes["/api/account/start"](SimpleNamespace(json_data={"method": "nope"}))
        self.assertEqual(res[0], 400)


if __name__ == "__main__":
    unittest.main()
