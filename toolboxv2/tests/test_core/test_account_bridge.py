"""Tests for account_bridge + sync_all (WP1 Auftrag 08, Teil 10)."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from toolboxv2.mods.CloudM.LiveSync import data_api_sync as das
from toolboxv2.mods.CloudM.LiveSync.account_bridge import (
    export_account, import_account)
from toolboxv2.mods.CloudM.auth.models import UserData


from toolboxv2.utils.system.types import ToolBoxError

_ERR_NONE = ToolBoxError.none


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _user(uid="usr_a"):
    return UserData.from_dict({
        "user_id": uid, "username": "markin", "email": "m@example.com",
        "level": 0,
        "settings": {"theme": "light", "anonymous": "x", "local_admin": "y"},
        "mod_data": {"m": {"a": "va", "b": "vb"}, "plain": "scalar"},
    })


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mirror = Path(self.tmp.name) / "mirror"

    def test_roundtrip_without_admin_keys(self):
        u = _user()
        export_account(u, self.mirror)
        u2 = UserData.from_dict({
            "user_id": "usr_b", "username": "x", "email": "x@x", "level": 0})
        self.assertTrue(import_account(u2, self.mirror))
        self.assertEqual(u2.settings["theme"], "light")
        self.assertEqual(u2.mod_data["m"]["a"], "va")
        self.assertNotIn("anonymous", u2.settings)
        self.assertNotIn("local_admin", u2.settings)

    def test_changed_value_writes_only_that_file(self):
        u = _user()
        export_account(u, self.mirror)
        before = {}
        for f in (self.mirror / "mod_data" / "m").iterdir():
            before[f.name] = f.read_bytes()
        u.mod_data["m"]["a"] = "changed"
        n = export_account(u, self.mirror)
        self.assertEqual(n, 1)
        f = self.mirror / "mod_data" / "m" / "a.json"
        self.assertNotEqual(f.read_bytes(), before["a.json"])
        self.assertEqual(
            (self.mirror / "mod_data" / "m" / "b.json").read_bytes(),
            before["b.json"])

    def test_removed_key_deletes_file(self):
        u = _user()
        export_account(u, self.mirror)
        del u.settings["theme"]
        export_account(u, self.mirror)
        self.assertFalse((self.mirror / "settings" / "theme.json").exists())

    def test_slash_umlaut_roundtrip(self):
        u = _user()
        u.mod_data["m"] = {"a/b": "umlaut-äöü"}
        export_account(u, self.mirror)
        u2 = UserData.from_dict({
            "user_id": "usr_b", "username": "x", "email": "x@x", "level": 0})
        import_account(u2, self.mirror)
        self.assertEqual(u2.mod_data["m"]["a/b"], "umlaut-äöü")

    def test_non_json_value_skipped_keyname_logged(self):
        u = _user()
        u.settings["bad"] = object()
        with self.assertLogs("toolboxv2.mods.CloudM.LiveSync.account_bridge",
                             level="WARNING") as cm:
            export_account(u, self.mirror)
        self.assertFalse((self.mirror / "settings" / "bad.json").exists())
        joined = "|".join(cm.output)
        self.assertIn("bad", joined)


class _FakeTransport:
    def __init__(self, files=None):
        self.files = files if files is not None else {}
        self._clock = 1000.0
        self.puts = 0

    async def list(self, prefix):
        out = {}
        for path, meta in self.files.items():
            if path.startswith(prefix):
                out[path] = (len(meta["enc"]), meta["updated_at"])
        return out

    async def put(self, path, ciphertext):
        self._clock += 1.0
        self.puts += 1
        self.files[path] = {"enc": ciphertext, "updated_at": self._clock}

    async def get(self, path):
        return self.files[path]["enc"]


def _enc_of(payload):
    import base64
    if isinstance(payload, dict) and "enc" in payload:
        return base64.b64decode(payload["enc"])
    raise AssertionError(payload)


class _Env:
    """Two devices (A/B) sharing one FakeTransport + one server-side store."""

    def __init__(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import generate_encryption_key
        self.key = generate_encryption_key()
        self.transport = _FakeTransport()
        import tempfile
        root = Path(tempfile.mkdtemp())
        self.apps = {}
        self.store = {}
        for name in ("A", "B"):
            uid = "usr_" + name
            app = SimpleNamespace()
            app.session = SimpleNamespace(base="http://127.0.0.1:8467")
            app.data_dir = str(root / name)
            self.apps[name] = app
            self.store[uid] = _user(uid).to_dict()

    def user(self, name):
        return UserData.from_dict(self.store["usr_" + name])

    def patch(self, name):
        uid = "usr_" + name
        owner = {"user_id": uid,
                 "username": self.store[uid].get("username", "")}

        async def fake_load(app, user_id):
            d = self.store.get(user_id)
            return UserData.from_dict(d) if d else None

        async def fake_save(app, user):
            self.store[user.user_id] = user.to_dict()

        def fake_owner(a):
            return owner

        async def fake_run_http(mod, fn, method="POST", **kw):
            if fn == "get_sync_key":
                return {"error": _ERR_NONE,
                        "result": {"data": {"key": self.key}, "data_info": "200",
                                   "data_type": "json", "data_to": "remote"},
                        "info": {"exec_code": 200}, "origin": []}
            if fn == "list_data":
                prefix = kw["prefix"]
                return {"error": _ERR_NONE, "result": {"data": [
                    {"path": p, "size": len(m["enc"]),
                     "updated_at": m["updated_at"]}
                    for p, m in self.transport.files.items()
                    if p.startswith(prefix)], "data_info": "200",
                    "data_type": "json", "data_to": "remote"},
                    "info": {"exec_code": 200}, "origin": []}
            if fn == "set_data":
                self.transport.put(kw["path"], _enc_of(kw["data"]))
                return {"error": _ERR_NONE,
                        "result": {"data": True, "data_info": "200",
                                   "data_type": "json", "data_to": "remote"},
                        "info": {"exec_code": 200}, "origin": []}
            if fn == "get_data":
                return {"error": _ERR_NONE, "result": {"data": self.transport.files[kw["path"]]["enc"],
                        "data_info": "200", "data_type": "json", "data_to": "remote"},
                        "info": {"exec_code": 200}, "origin": []}
            raise AssertionError(fn)

        for a in self.apps.values():
            a.run_http = fake_run_http
        return mock.patch.multiple(
            "toolboxv2.mods.CloudM.LiveSync.data_api_sync",
            read_device_owner=fake_owner,
            DataApiTransport=lambda a: self.transport,
            _load_user=fake_load,
            _save_user=fake_save,
        )


class SyncAllTest(unittest.TestCase):
    def setUp(self):
        das._LAST.clear()
        self.env = _Env()

    def tearDown(self):
        das._LAST.clear()

    def _sync_all(self, name):
        with self.env.patch(name):
            r = run(das.sync_all(self.env.apps[name]))
        self.assertNotIn("error", r, msg=str(r))
        return r

    # 1: no account -> skipped
    def test_no_account_skipped(self):
        with mock.patch(
                "toolboxv2.mods.CloudM.LiveSync.data_api_sync.read_device_owner",
                return_value=None):
            r = run(das.sync_all(self.env.apps["A"]))
        self.assertEqual(r, {"skipped": True, "reason": "no-account"})

    # 2: A sets theme, A sync, B sync -> B has theme
    def test_setting_travels_a_to_b(self):
        self.env.store["usr_A"]["settings"]["theme"] = "dark"
        self._sync_all("A")
        self._sync_all("B")
        self.assertEqual(self.env.user("B").settings["theme"], "dark")

    # 3: B changes only mod_data m.a -> A gets it, exactly 1 put at B
    def test_mod_data_partial(self):
        self._sync_all("A")  # baseline both sides
        self._sync_all("B")
        t = self.env.transport
        puts_before = t.puts
        self.env.store["usr_B"]["mod_data"]["m"]["a"] = "new-a"
        self._sync_all("B")
        self.assertEqual(t.puts - puts_before, 1)
        self._sync_all("A")
        u_a = self.env.user("A")
        self.assertEqual(u_a.mod_data["m"]["a"], "new-a")
        self.assertEqual(u_a.mod_data["m"]["b"], "vb")

    # 4: both change m.a differently -> B mirror keeps conflict copy of B
    def test_conflict_copy_keeps_local(self):
        self._sync_all("A")
        self._sync_all("B")
        self.env.store["usr_A"]["mod_data"]["m"]["a"] = "from-A"
        self.env.store["usr_B"]["mod_data"]["m"]["a"] = "from-B"
        self._sync_all("A")
        self._sync_all("B")
        self.assertEqual(self.env.user("B").mod_data["m"]["a"], "from-A")
        from toolboxv2.mods.CloudM.LiveSync.conflict import is_conflict_copy
        b_mirror = Path(self.env.apps["B"].data_dir) / "account_sync" / "mirror"
        conflicts = [p for p in b_mirror.rglob("*.json") if is_conflict_copy(p.name)]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(json.loads(conflicts[0].read_bytes()), "from-B")

    # 5: running lock -> skipped running
    def test_running_lock(self):
        with das._run_lock():
            r = run(das.sync_all(self.env.apps["A"]))
        self.assertEqual(r, {"skipped": True, "reason": "running"})

    # 6: get_sync_key only on the very first run (cached afterwards)
    def test_sync_key_cached(self):
        calls = {"n": 0}

        async def fake_load(app, user_id):
            return self.env.user("A")

        saved = {"n": 0}

        async def fake_save(app, user):
            saved["n"] += 1
            self.env.store[user.user_id] = user.to_dict()

        def fake_owner(a):
            return {"user_id": "usr_A", "username": "markin"}

        async def fake_http(mod, fn, method="POST", **kw):
            if fn == "get_sync_key":
                calls["n"] += 1
                return {"error": _ERR_NONE,
                        "result": {"data": {"key": self.env.key}, "data_info": "200",
                                   "data_type": "json", "data_to": "remote"},
                        "info": {"exec_code": 200}, "origin": []}
            raise AssertionError(fn)

        self.env.apps["A"].run_http = fake_http
        with mock.patch.multiple(
                "toolboxv2.mods.CloudM.LiveSync.data_api_sync",
                read_device_owner=fake_owner,
                _load_user=fake_load, _save_user=fake_save):
            k1 = run(das.ensure_sync_key(self.env.apps["A"], "usr_A"))
            k2 = run(das.ensure_sync_key(self.env.apps["A"], "usr_A"))
        self.assertEqual(calls["n"], 1)
        self.assertEqual(saved["n"], 1)
        self.assertEqual(k1, k2)
        self.assertTrue(k1)

    # 7: get_sync_key 401 -> reauth_required, nothing written
    def test_reauth_required(self):
        def fake_owner(a):
            return {"user_id": "usr_A", "username": "markin"}

        async def fake_load(app, user_id):
            return self.env.user("A")

        saved = {"n": 0}

        async def fake_save(app, user):
            saved["n"] += 1

        async def fake_http(mod, fn, method="POST", **kw):
            if fn == "get_sync_key":
                return {"error": ToolBoxError.internal_error,
                        "result": {"data": "404", "data_info": "401",
                                   "data_type": "json", "data_to": "remote"},
                        "info": {"exec_code": 401}, "origin": []}
            raise AssertionError(fn)

        self.env.apps["A"].run_http = fake_http
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import TransferError
        with mock.patch.multiple(
                "toolboxv2.mods.CloudM.LiveSync.data_api_sync",
                read_device_owner=fake_owner,
                _load_user=fake_load, _save_user=fake_save):
            with self.assertRaises(TransferError) as cm:
                run(das.ensure_sync_key(self.env.apps["A"], "usr_A"))
        self.assertIn("reauth", str(cm.exception))
        self.assertEqual(saved["n"], 0)
        self.assertEqual(self.env.store["usr_A"].get("user_pass_sync", ""), "")

    # 8: no key/value leaks into logs or result
    def test_logs_clean(self):
        self.env.store["usr_A"]["settings"]["secret-theme"] = "topsecretvalue"
        with self.assertNoLogs("toolboxv2.mods.CloudM.LiveSync.data_api_sync",
                               level="DEBUG"):
            r = self._sync_all("A")
        blob = json.dumps(r, default=str)
        self.assertNotIn("topsecretvalue", blob)
        self.assertNotIn("k" * 43, blob)


if __name__ == "__main__":
    unittest.main()
