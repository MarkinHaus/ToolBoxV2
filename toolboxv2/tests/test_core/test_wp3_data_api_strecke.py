"""WP3 Auftrag 7b - Betriebstest der Data-API-Strecke (nichts dazwischen ersetzt).

Strecke: Geraet (echte App, eigener Datenordner) -> DataApiTransport ->
App.run_http -> Session.fetch -> HTTP -> echter server_worker (Unterprozess,
eigener Datenordner) -> CloudM.UserDataAPI -> Speicher.

Es gibt KEIN Double. Einzige Fremdkoerper sind die Prozessgrenzen selbst: der
Worker laeuft als eigener Prozess auf 127.0.0.1.

Nutzer und Anmeldung (ohne Eingriff in den Server): Ein Hilfsskript startet die
echte App auf dem Datenordner des Servers und legt die Nutzer ueber die
vorhandene `_save_user` an; die Tokens kommen aus der vorhandenen
`_generate_tokens`, signiert mit dem Geheimnis, das auch der Worker per
TB_JWT_SECRET bekommt (zufaellig je Lauf, nur im Speicher). Das Geraet setzt das
Token wie `link_account`: `session._save_session_token`. (`session.login()` wird
nicht aufgerufen: es liest `result.authenticated`, der Server liefert
`result.data.authenticated`, siehe Bericht.)
"""
import asyncio
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
CASE_LIMIT = 60       # Regel 12: Sekunden je Fall
SERVER_START_LIMIT = 90

PREFIX = "livesync/global/"

_SEED = r"""
import asyncio, json, sys, time, uuid
import jwt as pyjwt
from toolboxv2 import get_app
from toolboxv2.mods.CloudM.auth.config import JWT_ALGORITHM, get_jwt_secret
from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_tokens
from toolboxv2.mods.CloudM.auth.models import UserData
from toolboxv2.mods.CloudM.auth.user_store import _save_user

app = get_app("wp3-7b-seed")
out = {}


async def go():
    for name in ("alice", "bob"):
        u = UserData(user_id="usr_" + uuid.uuid4().hex[:12], username=name,
                     email=name + "@example.test")
        await _save_user(app, u)
        out[name] = {"user_id": u.user_id,
                     "token": _generate_tokens(u)["access_token"]}
    # gueltig signiert, aber abgelaufen
    payload = pyjwt.decode(out["alice"]["token"], get_jwt_secret(),
                           algorithms=[JWT_ALGORITHM])
    payload["exp"] = time.time() - 3600
    out["alice_expired"] = {"user_id": out["alice"]["user_id"],
                            "token": pyjwt.encode(payload, get_jwt_secret(),
                                                  algorithm=JWT_ALGORITHM)}


asyncio.run(go())
json.dump(out, open(sys.argv[1], "w"))
app.exit()
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """Echter HTTP-Worker im Unterprozess, eigener Datenordner."""

    def __init__(self, extra_env=None):
        self.root = Path(tempfile.mkdtemp(prefix="tb_7b_srv_"))
        self.data = self.root / "data"
        self.home = self.root / "home"
        self.data.mkdir()
        self.home.mkdir()
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc = None
        self.users = {}
        self.env = dict(
            os.environ,
            MOCK_EXTERNAL_SERVICES="true", IS_OFFLINE_DB="true",
            HOME=str(self.home), USERPROFILE=str(self.home),
            TB_DATA_DIR=str(self.data),
            TB_JWT_SECRET=secrets.token_hex(24),   # zufaellig, nur dieser Lauf
            PYTHONPATH=str(REPO),
        )
        self.env.pop("TB_TEST_DATA_LOCK", None)
        for k in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"):
            self.env.pop(k, None)   # Standard: ohne MinIO
        self.env.update(extra_env or {})

    def start(self):
        seed = self.root / "seed.py"
        seed.write_text(_SEED, encoding="utf-8")
        users_file = self.root / "users.json"
        r = subprocess.run([sys.executable, str(seed), str(users_file)],
                           env=self.env, cwd=str(self.root), capture_output=True,
                           text=True, timeout=SERVER_START_LIMIT,
                           encoding="utf-8", errors="replace")
        if not users_file.is_file():
            raise AssertionError(f"Nutzer anlegen fehlgeschlagen:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        self.users = json.loads(users_file.read_text())
        self.log = open(self.root / "server.log", "wb")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "toolboxv2.utils.workers.server_worker",
             "-H", "127.0.0.1", "-p", str(self.port), "-w", "t7b"],
            env=self.env, cwd=str(self.root), stdout=self.log,
            stderr=subprocess.STDOUT)
        deadline = time.time() + SERVER_START_LIMIT
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError("Worker beendet: " + self.tail())
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                time.sleep(0.5)
        raise AssertionError("Worker meldet nicht /health: " + self.tail())

    def tail(self) -> str:
        self.log.flush()
        return (self.root / "server.log").read_text(errors="replace")[-1500:]

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=20)
        self.log.close()

    def cleanup(self):
        self.stop()
        shutil.rmtree(self.root, ignore_errors=True)


class _DeviceCase(unittest.IsolatedAsyncioTestCase):
    """Basis: eine App je Fall als 'Geraet', eigener Datenordner."""

    def _new_device(self, server: _Server, who: str):
        """Echte App auf eigenem Datenordner; Token wie link_account."""
        app = self._blank_device(server)
        u = server.users[who]
        s = app.session
        s.base = server.base
        s._save_session_token(u["token"], "", u["user_id"])
        return app

    def _blank_device(self, server: _Server):
        """Echte App auf eigenem Datenordner, ohne Konto und ohne Token."""
        from toolboxv2 import get_app
        from toolboxv2.tests.a_util import reset_app_singleton
        root = Path(tempfile.mkdtemp(prefix="tb_7b_dev_"))
        self.addCleanup(shutil.rmtree, root, True)
        (root / "data").mkdir()
        (root / "home").mkdir()
        env = patch.dict(os.environ, {
            "TB_DATA_DIR": str(root / "data"), "HOME": str(root / "home"),
            "USERPROFILE": str(root / "home"),
            "MOCK_EXTERNAL_SERVICES": "true", "IS_OFFLINE_DB": "true",
            "TOOLBOXV2_REMOTE_BASE": server.base})   # Ziel von link_account
        env.start()
        self.addCleanup(env.stop)
        reset_app_singleton(force_exit=True)
        self.addCleanup(reset_app_singleton, True)
        return get_app("wp3-7b-device")

    async def _limit(self, coro):
        return await asyncio.wait_for(coro, CASE_LIMIT)


class TestDataApiStrecke(_DeviceCase):
    """Fall 1-4: Zugriffe ueber den ECHTEN DataApiTransport."""

    @classmethod
    def setUpClass(cls):
        cls.server = _Server()
        cls.addClassCleanup(cls.server.cleanup)
        cls.server.start()

    def _transport(self, who="alice"):
        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import DataApiTransport
        return DataApiTransport(self._new_device(self.server, who))

    async def test_01_put_list_get_delete(self):
        t = self._transport("alice")
        path = PREFIX + "a/b.bin"
        blob = bytes(range(256)) + b"cipher"
        await self._limit(t.put(path, blob))
        listing = await self._limit(t.list(PREFIX))
        self.assertIn(path, listing,
                      f"list(prefix={PREFIX!r}) liefert Pfade in anderer Form: {sorted(listing)}")
        size, updated_at = listing[path]
        self.assertGreater(size, 0)
        self.assertGreater(updated_at, 0)
        self.assertEqual(await self._limit(t.get(path)), blob)
        await self._limit(t.delete(path))
        self.assertNotIn(path, await self._limit(t.list(PREFIX)))

    async def test_02_zweiter_nutzer_sieht_nichts(self):
        a, b = self._transport("alice"), self._transport("bob")
        path = PREFIX + "nur-alice.bin"
        await self._limit(a.put(path, b"geheim"))
        self.assertIn(path, await self._limit(a.list(PREFIX)))
        self.assertNotIn(path, await self._limit(b.list(PREFIX)))
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import TransferError
        with self.assertRaises(TransferError):
            await self._limit(b.get(path))
        await self._limit(a.delete(path))

    async def test_03_abgelaufenes_token_ist_reauth_required(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import TransferError
        t = self._transport("alice_expired")
        with self.assertRaises(TransferError) as cm:
            await self._limit(t.list(PREFIX))
        self.assertEqual(str(cm.exception), "reauth_required")

    async def test_04_ungueltiges_token_ist_reauth_required(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import TransferError
        t = self._transport("alice")
        t.app.session._save_session_token("kein.gueltiges.token", "", "x")
        with self.assertRaises(TransferError) as cm:
            await self._limit(t.put(PREFIX + "x.bin", b"x"))
        self.assertEqual(str(cm.exception), "reauth_required")


class TestLinkAccountStrecke(_DeviceCase):
    """Echtes link_account (device_account.py) gegen den echten Worker."""

    @classmethod
    def setUpClass(cls):
        cls.server = _Server()
        cls.addClassCleanup(cls.server.cleanup)
        cls.server.start()

    async def test_gueltiges_token_verknuepft_das_geraet(self):
        from toolboxv2.mods.CloudM.auth.device_account import (
            device_owner_file, link_account, read_device_owner)
        app = self._blank_device(self.server)
        u = self.server.users["alice"]
        await self._limit(link_account(app, {
            "access_token": u["token"], "refresh_token": "",
            "user_id": u["user_id"], "username": "alice"}))
        self.assertTrue(device_owner_file(app).is_file())
        self.assertEqual(read_device_owner(app)["user_id"], u["user_id"])
        self.assertTrue(app.session.valid)

    async def test_ungueltiges_token_schreibt_nichts(self):
        from toolboxv2.mods.CloudM.auth.device_account import (
            device_owner_file, link_account)
        app = self._blank_device(self.server)
        old_base = app.session.base
        u = self.server.users["alice"]
        with self.assertRaises(RuntimeError):
            await self._limit(link_account(app, {
                "access_token": "kein.gueltiges.token", "refresh_token": "",
                "user_id": u["user_id"], "username": "alice"}))
        self.assertFalse(device_owner_file(app).exists())
        self.assertFalse(app.session._load_session_token())
        self.assertEqual(app.session.base, old_base)
        self.assertFalse(app.session.valid)


class TestServerGestoppt(_DeviceCase):
    """Fall 5: Worker laeuft, antwortet, wird beendet -> 'offline'."""

    async def test_gestoppter_server_ist_offline(self):
        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import DataApiTransport
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import TransferError
        server = _Server()
        self.addCleanup(server.cleanup)
        server.start()
        t = DataApiTransport(self._new_device(server, "alice"))
        await self._limit(t.list(PREFIX))          # lief vorher
        server.stop()
        with self.assertRaises(TransferError) as cm:
            await self._limit(t.list(PREFIX))
        self.assertEqual(str(cm.exception), "offline")


class TestMinioNichtErreichbar(_DeviceCase):
    """MinIO ist konfiguriert, lauscht aber nicht: Fehler an den Aufrufer, kein
    stiller Rueckfall auf lokalen Speicher, kein 'ok' ohne Speicherung."""

    async def test_put_und_list_scheitern_laut(self):
        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import DataApiTransport
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import TransferError
        server = _Server({"MINIO_ENDPOINT": f"127.0.0.1:{_free_port()}",
                          "MINIO_ACCESS_KEY": "testkey", "MINIO_SECRET_KEY": "testsecret"})
        self.addCleanup(server.cleanup)
        server.start()
        t = DataApiTransport(self._new_device(server, "alice"))
        with self.assertRaises(TransferError):
            await self._limit(t.put(PREFIX + "x.bin", b"x"))
        with self.assertRaises(TransferError):
            await self._limit(t.list(PREFIX))


class TestSyncFolderStrecke(_DeviceCase):
    """Fall 6: echter sync_folder, zwei Geraete, gleicher Nutzer."""

    @classmethod
    def setUpClass(cls):
        cls.server = _Server()
        cls.addClassCleanup(cls.server.cleanup)
        cls.server.start()

    async def _sync(self, app, folder: Path, index_db: Path, key: str, device: str):
        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import (
            DataApiTransport, LocalIndex, sync_folder)
        idx = LocalIndex(str(index_db))
        await idx.init()
        try:
            return await self._limit(sync_folder(
                DataApiTransport(app), folder, PREFIX, idx, key, device))
        finally:
            await idx.close()

    async def test_datei_von_geraet_a_kommt_auf_geraet_b(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import generate_encryption_key
        key = generate_encryption_key()     # fester Testschluessel fuer beide
        a_app = self._new_device(self.server, "alice")
        a = Path(tempfile.mkdtemp(prefix="tb_7b_a_"))
        self.addCleanup(shutil.rmtree, a, True)
        (a / "sub").mkdir()
        (a / "sub" / "notiz.txt").write_bytes("Inhalt äöü\n".encode() * 50)
        out_a = await self._sync(a_app, a, a.with_name(a.name + "-index.db"), key, "geraet-a")
        self.assertEqual(out_a["pushed"], ["sub/notiz.txt"], out_a)

        b_app = self._new_device(self.server, "alice")
        b = Path(tempfile.mkdtemp(prefix="tb_7b_b_"))
        self.addCleanup(shutil.rmtree, b, True)
        out_b = await self._sync(b_app, b, b.with_name(b.name + "-index.db"), key, "geraet-b")
        self.assertEqual(out_b["pulled"], ["sub/notiz.txt"], out_b)
        self.assertEqual((b / "sub" / "notiz.txt").read_bytes(),
                         (a / "sub" / "notiz.txt").read_bytes())


if __name__ == "__main__":
    unittest.main()
