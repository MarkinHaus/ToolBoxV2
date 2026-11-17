"""Tests for LiveSync data_api_sync (WP1 Auftrag 08, Teil 9)."""
import asyncio
import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from toolboxv2.mods.CloudM.LiveSync import data_api_sync as das
from toolboxv2.mods.CloudM.LiveSync.crypto import (
    decrypt_bytes,
    encrypt_bytes,
    generate_encryption_key,
)
from toolboxv2.mods.CloudM.LiveSync.index import LocalIndex
from toolboxv2.mods.CloudM.LiveSync.minio_helper import TransferError
from toolboxv2.mods.CloudM.LiveSync.conflict import is_conflict_copy


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class FakeTransport:
    """In-memory stand-in with the exact transport interface."""

    def __init__(self):
        self.files = {}     # path -> {"enc": bytes, "updated_at": float}
        self._clock = 1000.0

    async def list(self, prefix):
        out = {}
        for path, meta in self.files.items():
            if path.startswith(prefix):
                out[path] = (len(meta["enc"]), meta["updated_at"])
        return out

    async def put(self, path, ciphertext):
        self._clock += 1.0
        self.files[path] = {"enc": ciphertext, "updated_at": self._clock}

    async def get(self, path):
        meta = self.files.get(path)
        if meta is None:
            raise TransferError("missing")
        return meta["enc"]


class OfflineTransport(FakeTransport):
    async def list(self, prefix):
        raise TransferError("offline")


class DataApiSyncTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.local = self.root / "A"
        self.local.mkdir()
        self.key = generate_encryption_key()
        self.transport = FakeTransport()
        self.index = LocalIndex(str(self.root / "index.db"))
        run(self.index.init())
        self.addCleanup(lambda: run(self.index.close()))

    def _sync(self, transport=None):
        return run(das.sync_folder(
            transport or self.transport, self.local, das.PREFIX_GLOBAL,
            self.index, self.key, "dev-A"))

    def _write(self, rel, data):
        p = self.local / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    # --- 1: roundtrip push + pull on "device B" ---
    def test_sync_roundtrip_push_pull(self):
        self._write("a.txt", b"hello")
        r1 = self._sync()
        self.assertEqual(r1["pushed"], ["a.txt"])
        # device B: empty dir, own index
        b_dir = self.root / "B"
        b_dir.mkdir()
        b_index = LocalIndex(str(self.root / "b.db"))
        run(b_index.init())
        self.addCleanup(lambda: run(b_index.close()))
        r2 = run(das.sync_folder(self.transport, b_dir, das.PREFIX_GLOBAL,
                                 b_index, self.key, "dev-B"))
        self.assertEqual(r2["pulled"], ["a.txt"])
        self.assertEqual((b_dir / "a.txt").read_bytes(), b"hello")

    # --- 2: second run transfers nothing ---
    def test_put_0_get_0_when_nothing_changed(self):
        self._write("a.txt", b"hello")
        self._sync()
        r2 = self._sync()
        self.assertEqual(r2["pushed"], [])
        self.assertEqual(r2["pulled"], [])
        self.assertEqual(r2["conflicts"], [])
        self.assertEqual(r2["unchanged"], 1)

    # --- 3: remote new file is pulled ---
    def test_remote_new_file_pulled(self):
        self.transport.files[das.PREFIX_GLOBAL + "neu.txt"] = {
            "enc": encrypt_bytes(b"remote", self.key), "updated_at": 5.0}
        r = self._sync()
        self.assertEqual(r["pulled"], ["neu.txt"])
        self.assertEqual((self.local / "neu.txt").read_bytes(), b"remote")

    # --- 4: conflict -> local backed up, remote wins ---
    def test_conflict_local_backed_up(self):
        self._write("a.txt", b"v1")
        self._sync()
        self._write("a.txt", b"local edit")
        self.transport.files[das.PREFIX_GLOBAL + "a.txt"] = {
            "enc": encrypt_bytes(b"remote edit", self.key), "updated_at": 9000.0}
        r = self._sync()
        self.assertEqual(r["conflicts"], ["a.txt"])
        self.assertEqual((self.local / "a.txt").read_bytes(), b"remote edit")
        backups = [p for p in self.local.rglob("*") if is_conflict_copy(p.name)]
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"local edit")

    # --- 5: same plaintext both sides -> no conflict ---
    def test_same_plaintext_no_conflict(self):
        self._write("a.txt", b"v1")
        self._sync()
        self.transport.files[das.PREFIX_GLOBAL + "a.txt"] = {
            "enc": encrypt_bytes(b"v1", self.key), "updated_at": 9100.0}
        r = self._sync()
        self.assertEqual(r["conflicts"], [])
        self.assertEqual(r["unchanged"], 1)
        self.assertEqual(r["pushed"], [])
        self.assertEqual(r["pulled"], [])

    # --- 6: oversize skipped ---
    def test_oversize_skipped(self):
        p = self._write("big.bin", b"")
        with open(p, "rb+") as f:
            f.truncate(das.MAX_BYTES + 1)
        r = self._sync()
        self.assertEqual(r["skipped"], ["big.bin"])
        self.assertEqual(r["pushed"], [])

    # --- 7: unsafe remote path never written ---
    def test_unsafe_remote_path_never_written(self):
        bad = das.PREFIX_GLOBAL + "../escape.txt"
        self.transport.files[bad] = {
            "enc": encrypt_bytes(b"x", self.key), "updated_at": 1.0}
        r = self._sync()
        self.assertIn(bad, r["skipped"])
        self.assertFalse((self.local / "escape.txt").exists())
        self.assertTrue((self.local.parent / "escape.txt").exists() is False)

    # --- 8: offline -> TransferError propagates ---
    def test_offline_raises_transfer_error(self):
        with self.assertRaises(TransferError):
            self._sync(transport=OfflineTransport())

    # --- 9: transport rejects insecure base ---
    def test_transport_rejects_insecure_base(self):
        app = SimpleNamespace(session=SimpleNamespace(base="http://example.com"))
        with self.assertRaises(TransferError):
            das.DataApiTransport(app)
        ok = SimpleNamespace(session=SimpleNamespace(base="http://127.0.0.1:8467"))
        self.assertIsInstance(das.DataApiTransport(ok), das.DataApiTransport)

    # --- 10: server only sees {"v":1,"enc":...} ---
    def test_payload_form(self):
        self._write("a.txt", b"hello")
        self._sync()
        path = das.PREFIX_GLOBAL + "a.txt"
        self.assertIn(path, self.transport.files)
        raw = self.transport.files[path]["enc"]
        self.assertNotIn(b"hello", raw)  # ciphertext, not plaintext

    # --- 11: index holds remote meta + plaintext checksum ---
    def test_index_updated_with_remote_meta(self):
        self._write("a.txt", b"hello")
        self._sync()
        row = run(self.index.get_file("a.txt"))
        self.assertIsNotNone(row)
        meta = self.transport.files[das.PREFIX_GLOBAL + "a.txt"]
        self.assertEqual(row["mtime"], meta["updated_at"])
        self.assertEqual(row["size"], len(meta["enc"]))
        self.assertEqual(row["sync_state"], "synced")
        self.assertEqual(row["remote_key"], das.PREFIX_GLOBAL + "a.txt")
        from toolboxv2.mods.CloudM.LiveSync.crypto import compute_checksum
        self.assertEqual(row["checksum"], compute_checksum(b"hello"))

    # --- 12: local deletion is not propagated ---
    def test_local_delete_not_propagated(self):
        self._write("a.txt", b"hello")
        self._sync()
        (self.local / "a.txt").unlink()
        r = self._sync()
        self.assertEqual(r["pushed"], [])
        self.assertEqual(r["pulled"], [])
        self.assertEqual(r["conflicts"], [])
        self.assertIn(das.PREFIX_GLOBAL + "a.txt", self.transport.files)


if __name__ == "__main__":
    unittest.main()
