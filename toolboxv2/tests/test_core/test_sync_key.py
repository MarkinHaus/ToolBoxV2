"""WP1 Auftrag 08 Teil 8: get_sync_key + secret-free user responses.

Spec: Auftrag 08 (2026-11-16). user_pass_sync leaves the server only via
CloudM.Auth.get_sync_key for the authenticated owner - never via
get_user_data, never for a user_id taken from parameters.
"""
import asyncio
import base64
import unittest
from types import SimpleNamespace
from unittest import mock

from toolboxv2.mods.CloudM.auth.models import UserData


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _user(uid="usr_a"):
    return UserData.from_dict({
        "user_id": uid, "username": "markin", "email": "m@example.com",
        "level": 0, "user_pass_sync": "",
    })


class SyncKeyTest(unittest.TestCase):
    def setUp(self):
        import toolboxv2.mods.CloudM.auth.api_session as api

        self.api = api
        self.app = SimpleNamespace()
        self.db = {}
        self.saved = 0

        async def fake_load(app, user_id):
            data = self.db.get(user_id)
            return UserData.from_dict(data) if data else None

        async def fake_save(app, user):
            self.saved += 1
            self.db[user.user_id] = user.to_dict()

        self.patchers = [
            mock.patch.object(api, "_load_user", fake_load),
            mock.patch.object(api, "_save_user", fake_save),
        ]
        for p in self.patchers:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patchers])

    def _auth_ok(self, uid="usr_a"):
        from toolboxv2 import Result
        return Result.ok({"authenticated": True, "user_id": uid})

    def test_unauthenticated_returns_401_and_sets_nothing(self):
        from toolboxv2 import Result
        with mock.patch.object(self.api, "validate_session",
                               async_lambda := (lambda app=None, request=None,
                                       token=None, session_token=None, data=None:
                                       _async_err())) as _:
            pass
        # simpler: patch with async func directly
        async def err(app=None, request=None, token=None, session_token=None, data=None):
            return Result.default_user_error("No token provided")
        with mock.patch.object(self.api, "validate_session", err):
            res = run(self.api.get_sync_key(self.app, request=None, data={}))
        self.assertTrue(res.is_error())
        self.assertEqual(self.saved, 0)
        self.assertEqual(self.db, {})

    def test_generates_32_byte_key_and_saves(self):
        self.db["usr_a"] = _user().to_dict()
        with mock.patch.object(self.api, "validate_session",
                               self._mk_auth("usr_a")):
            res = run(self.api.get_sync_key(self.app, request=None, data={}))
        key = res.get()["key"]
        self.assertEqual(len(base64.urlsafe_b64decode(key)), 32)
        self.assertEqual(self.saved, 1)
        self.assertEqual(self.db["usr_a"]["user_pass_sync"], key)

    def test_second_call_returns_same_key_without_saving(self):
        self.db["usr_a"] = _user().to_dict()
        with mock.patch.object(self.api, "validate_session",
                               self._mk_auth("usr_a")):
            k1 = run(self.api.get_sync_key(self.app, request=None, data={})).get()["key"]
            k2 = run(self.api.get_sync_key(self.app, request=None, data={})).get()["key"]
        self.assertEqual(k1, k2)
        self.assertEqual(self.saved, 1)

    def test_get_user_data_has_no_secrets(self):
        self.db["usr_a"] = _user().to_dict()
        self.db["usr_a"]["user_pass_sync"] = "k" * 43
        res = run(self.api.get_user_data(self.app, data={"user_id": "usr_a"}))
        d = res.get()
        self.assertNotIn("user_pass_sync", d)
        self.assertNotIn("minio_credentials", d)

    def test_foreign_user_id_in_data_is_ignored(self):
        self.db["usr_a"] = _user("usr_a").to_dict()
        self.db["fremd"] = _user("fremd").to_dict()
        self.db["fremd"]["user_pass_sync"] = "F" * 43
        with mock.patch.object(self.api, "validate_session",
                               self._mk_auth("usr_a")):
            key = run(self.api.get_sync_key(self.app, request=None,
                                            data={"user_id": "fremd"})).get()["key"]
        self.assertNotEqual(key, "F" * 43)
        self.assertEqual(self.db["usr_a"]["user_pass_sync"], key)

    def test_from_dict_old_record_without_field(self):
        d = _user().to_dict()
        d.pop("user_pass_sync")
        u = UserData.from_dict(d)
        self.assertEqual(u.user_pass_sync, "")

    def _mk_auth(self, uid):
        from toolboxv2 import Result
        ok = Result.ok({"authenticated": True, "user_id": uid})

        async def auth(app=None, request=None, token=None,
                       session_token=None, data=None):
            return ok
        return auth


if __name__ == "__main__":
    unittest.main()
