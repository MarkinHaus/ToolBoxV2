"""WP1 K1: Device account - link, remember, migrate local data.

Spec: Arbeitspaket WP1 (2026-11-15), K1. device_owner.json holds exactly
user_id, username, base_url, linked_at, migrated_from - never a token.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from toolboxv2.mods.CloudM.auth.models import UserData


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _FakeSession:
    def __init__(self):
        self.base = None
        self.username = None
        self.saved = None
        self.login_result = True

    def _save_session_token(self, access, refresh, user_id):
        self.saved = (access, refresh, user_id)

    async def login(self, verbose=False):
        return self.login_result


class DeviceAccountTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.session = _FakeSession()
        self.app = SimpleNamespace(data_dir=str(self.tmp), session=self.session)
        self.db = {}  # key -> dict/str (AUTH_USER::*, AUTH_USER_EMAIL::*)

        import toolboxv2.mods.CloudM.auth.device_account as da

        async def fake_find_by_email(app, email):
            key = f"AUTH_USER_EMAIL::{email}"
            uid = self.db.get(key)
            if not uid:
                return None
            data = self.db.get(f"AUTH_USER::{uid.strip(chr(34))}")
            return UserData.from_dict(data) if data else None

        async def fake_load(app, user_id):
            data = self.db.get(f"AUTH_USER::{user_id}")
            return UserData.from_dict(data) if data else None

        async def fake_save(app, user):
            self.db[f"AUTH_USER::{user.user_id}"] = user.to_dict()
            if user.email:
                self.db[f"AUTH_USER_EMAIL::{user.email}"] = user.user_id

        async def fake_delete(app, key):
            self.db.pop(key, None)

        patches = [
            mock.patch.object(da, "_find_user_by_email", fake_find_by_email),
            mock.patch.object(da, "_load_user", fake_load),
            mock.patch.object(da, "_save_user", fake_save),
            mock.patch.object(da, "_db_delete", fake_delete),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _payload(self, user_id="usr_acc1", username="markin"):
        return {"access_token": "at", "refresh_token": "rt", "user_id": user_id,
                "username": username}

    def _admin_in_db(self):
        self.db["AUTH_USER_EMAIL::local-admin@toolbox.local"] = '"usr_admin1"'
        self.db["AUTH_USER::usr_admin1"] = UserData(
            user_id="usr_admin1", username="root",
            email="local-admin@toolbox.local", level=-1,
            settings={"anonymous": True, "local_admin": True, "theme": "dark"},
            mod_data={"m": {"a": 1}},
        ).to_dict()

    def test_link_without_admin_creates_file_and_user_level_minus1(self):
        from toolboxv2.mods.CloudM.auth.device_account import link_account, read_device_owner
        user = run(link_account(self.app, self._payload()))
        f = self.tmp / "device_owner.json"
        self.assertTrue(f.is_file())
        self.assertEqual(user.level, -1)
        owner = read_device_owner(self.app)
        self.assertEqual(owner["user_id"], "usr_acc1")
        self.assertEqual(owner["username"], "markin")

    def test_migrates_local_admin_settings_and_mod_data(self):
        from toolboxv2.mods.CloudM.auth.device_account import link_account, read_device_owner
        self._admin_in_db()
        user = run(link_account(self.app, self._payload()))
        self.assertEqual(user.settings.get("theme"), "dark")
        self.assertNotIn("anonymous", user.settings)
        self.assertNotIn("local_admin", user.settings)
        self.assertEqual(user.mod_data, {"m": {"a": 1}})
        self.assertNotIn("AUTH_USER::usr_admin1", self.db)
        self.assertNotIn("AUTH_USER_EMAIL::local-admin@toolbox.local", self.db)
        self.assertEqual(read_device_owner(self.app)["migrated_from"], "usr_admin1")

    def test_account_wins_on_conflict(self):
        from toolboxv2.mods.CloudM.auth.device_account import link_account
        self.db["AUTH_USER_EMAIL::local-admin@toolbox.local"] = '"usr_admin1"'
        self.db["AUTH_USER::usr_admin1"] = UserData(
            user_id="usr_admin1", username="root",
            email="local-admin@toolbox.local", level=-1,
            settings={"theme": "dark"}, mod_data={},
        ).to_dict()
        self.db["AUTH_USER::usr_acc1"] = UserData(
            user_id="usr_acc1", username="markin", email="", level=-1,
            settings={"theme": "light"}, mod_data={},
        ).to_dict()
        user = run(link_account(self.app, self._payload()))
        self.assertEqual(user.settings["theme"], "light")

    def test_link_same_account_twice_is_noop(self):
        from toolboxv2.mods.CloudM.auth.device_account import link_account
        run(link_account(self.app, self._payload()))
        f = self.tmp / "device_owner.json"
        before = f.read_text(encoding="utf-8")
        user = run(link_account(self.app, self._payload()))
        self.assertEqual(user.user_id, "usr_acc1")
        self.assertEqual(f.read_text(encoding="utf-8"), before)

    def test_link_other_account_raises(self):
        from toolboxv2.mods.CloudM.auth.device_account import link_account
        run(link_account(self.app, self._payload()))
        with self.assertRaises(RuntimeError):
            run(link_account(self.app, self._payload(user_id="usr_other")))

    def test_login_failure_raises_and_writes_nothing(self):
        from toolboxv2.mods.CloudM.auth.device_account import link_account
        self.session.login_result = False
        with self.assertRaises(RuntimeError):
            run(link_account(self.app, self._payload()))
        self.assertFalse((self.tmp / "device_owner.json").exists())
        self.assertEqual(self.db, {})

    def test_owner_file_never_contains_token(self):
        from toolboxv2.mods.CloudM.auth.device_account import link_account
        run(link_account(self.app, self._payload()))
        raw = (self.tmp / "device_owner.json").read_text(encoding="utf-8").lower()
        self.assertNotIn("token", raw)
        data = json.loads(raw)
        self.assertEqual(
            sorted(data), ["base_url", "linked_at", "migrated_from", "user_id", "username"])

    def test_unlink_removes_file(self):
        from toolboxv2.mods.CloudM.auth.device_account import link_account, unlink_account
        run(link_account(self.app, self._payload()))
        run(unlink_account(self.app))
        self.assertFalse((self.tmp / "device_owner.json").exists())


if __name__ == "__main__":
    unittest.main()

