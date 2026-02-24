"""Tests for toolboxv2.utils.extras.blobs — Unit + E2E (MinIO-gated)."""

import unittest
import os
import pickle
import json
import time
import socket
import tempfile
import shutil
from unittest.mock import MagicMock, patch, PropertyMock

from toolboxv2 import Code
from toolboxv2.utils.extras.blobs import (
    BlobStorage,
    BlobFile,
    CryptoLayer,
    WatchCallback,
    WatchManager,
    StorageMode,
    ConnectionState,
    ServerStatus,
    create_offline_storage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minio_reachable(host="127.0.0.1", port=9000, timeout=1) -> bool:
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


MINIO_AVAILABLE = _minio_reachable()
SKIP_E2E = not MINIO_AVAILABLE


def _make_offline_storage(tmp: str) -> BlobStorage:
    """Deterministic offline storage for unit tests — no MinIO needed."""
    return create_offline_storage(storage_directory=tmp)


# ===========================================================================
# 1) Pure dataclass / enum tests
# ===========================================================================

class TestServerStatus(unittest.TestCase):
    def test_mark_healthy(self):
        s = ServerStatus(endpoint="x")
        s.mark_healthy()
        self.assertTrue(s.is_healthy())
        self.assertEqual(s.error_count, 0)

    def test_mark_error_increments(self):
        s = ServerStatus(endpoint="x")
        s.mark_error("oops")
        s.mark_error("oops2")
        self.assertEqual(s.error_count, 2)
        self.assertEqual(s.state, ConnectionState.ERROR)

    def test_mark_degraded(self):
        s = ServerStatus(endpoint="x")
        s.mark_degraded()
        self.assertEqual(s.state, ConnectionState.DEGRADED)
        self.assertFalse(s.is_healthy())


# ===========================================================================
# 2) CryptoLayer (unit, no external deps)
# ===========================================================================

class TestCryptoLayer(unittest.TestCase):
    def test_sign_and_verify(self):
        c = CryptoLayer()
        data = b"hello world"
        sig = c.sign(data)
        self.assertTrue(c.verify(data, sig))
        self.assertFalse(c.verify(b"tampered", sig))

    def test_encrypt_decrypt_roundtrip(self):
        c = CryptoLayer()
        data = b"secret payload 1234"
        enc = c.encrypt(data)
        dec = c.decrypt(enc)
        self.assertEqual(dec, data)

    def test_encrypt_with_custom_key(self):
        c = CryptoLayer()
        key = Code.generate_symmetric_key()
        data = b"keyed data"
        enc = c.encrypt(data, key=key)
        dec = c.decrypt(enc, key=key)
        self.assertEqual(dec, data)


# ===========================================================================
# 3) WatchCallback
# ===========================================================================

class TestWatchCallback(unittest.TestCase):
    def test_not_expired_initially(self):
        wc = WatchCallback(callback=lambda x: None, blob_id="b", max_idle_timeout=10)
        self.assertFalse(wc.is_expired())

    def test_expired_after_timeout(self):
        wc = WatchCallback(callback=lambda x: None, blob_id="b", max_idle_timeout=0)
        time.sleep(0.05)
        self.assertTrue(wc.is_expired())

    def test_update_timestamp_resets(self):
        wc = WatchCallback(callback=lambda x: None, blob_id="b", max_idle_timeout=10)
        old = wc.last_update
        time.sleep(0.05)
        wc.update_timestamp()
        self.assertGreater(wc.last_update, old)


# ===========================================================================
# 4) WatchManager (mocked storage)
# ===========================================================================

class TestWatchManager(unittest.TestCase):
    def setUp(self):
        self.mock_storage = MagicMock()
        self.mgr = WatchManager(self.mock_storage)

    def tearDown(self):
        self.mgr.remove_all_watches()

    def test_add_starts_thread(self):
        self.mgr.add_watch("b1", MagicMock())
        self.assertIn("b1", self.mgr._watches)
        self.assertTrue(self.mgr._running)

    def test_remove_specific_callback(self):
        cb1, cb2 = MagicMock(), MagicMock()
        self.mgr.add_watch("b1", cb1)
        self.mgr.add_watch("b1", cb2)
        self.assertEqual(len(self.mgr._watches["b1"]), 2)
        self.mgr.remove_watch("b1", cb1)
        self.assertEqual(len(self.mgr._watches["b1"]), 1)

    def test_remove_all(self):
        self.mgr.add_watch("b1", MagicMock())
        self.mgr.add_watch("b2", MagicMock())
        self.mgr.remove_all_watches()
        self.assertEqual(len(self.mgr._watches), 0)

    def test_dispatch_callbacks(self):
        cb = MagicMock()
        self.mgr.add_watch("b1", cb)
        with patch("toolboxv2.utils.extras.blobs.BlobFile") as MockBF:
            self.mgr._dispatch_callbacks("b1")
            cb.assert_called_once()

    def test_cleanup_expired(self):
        cb = MagicMock()
        self.mgr.add_watch("b1", cb, max_idle_timeout=0)
        time.sleep(0.05)
        self.mgr._cleanup_expired_callbacks()
        self.assertNotIn("b1", self.mgr._watches)


# ===========================================================================
# 5) BlobFile._path_splitter
# ===========================================================================

class TestPathSplitter(unittest.TestCase):
    def test_simple(self):
        bid, folder, datei = BlobFile._path_splitter("abc123/file.txt")
        self.assertEqual(bid, "abc123")
        self.assertEqual(folder, "")
        self.assertEqual(datei, "file.txt")

    def test_with_folder(self):
        bid, folder, datei = BlobFile._path_splitter("abc123/sub/file.txt")
        self.assertEqual(bid, "abc123")
        self.assertEqual(folder, "sub")
        self.assertEqual(datei, "file.txt")

    def test_nested_folders(self):
        bid, folder, datei = BlobFile._path_splitter("abc123/a/b/c/file.txt")
        self.assertEqual(bid, "abc123")
        self.assertEqual(folder, "a|b|c")
        self.assertEqual(datei, "file.txt")

    def test_no_file_raises(self):
        with self.assertRaises(ValueError):
            BlobFile._path_splitter("abc123")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            BlobFile._path_splitter("")


# ===========================================================================
# 6) BlobStorage OFFLINE mode — real SQLite, no MinIO
# ===========================================================================

class TestBlobStorageOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="blob_test_")
        self.storage = _make_offline_storage(self.tmp)

    def tearDown(self):
        self.storage.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_mode_is_offline(self):
        self.assertEqual(self.storage.mode, StorageMode.OFFLINE)

    def test_create_and_read(self):
        blob_id = self.storage.create_blob(b"hello", blob_id="t1", encrypt=False)
        self.assertEqual(blob_id, "t1")
        data = self.storage.read_blob("t1", decrypt=False)
        self.assertEqual(data, b"hello")

    def test_create_and_read_encrypted(self):
        blob_id = self.storage.create_blob(b"secret", blob_id="t2", encrypt=True)
        raw = self.storage.read_blob("t2", decrypt=False)
        self.assertNotEqual(raw, b"secret")  # should be encrypted
        plain = self.storage.read_blob("t2", decrypt=True)
        self.assertEqual(plain, b"secret")

    def test_update_blob(self):
        self.storage.create_blob(b"v1", blob_id="t3", encrypt=False)
        result = self.storage.update_blob("t3", b"v2", encrypt=False)
        self.assertIn("version", result)
        data = self.storage.read_blob("t3", decrypt=False)
        self.assertEqual(data, b"v2")

    def test_delete_blob(self):
        self.storage.create_blob(b"x", blob_id="t4", encrypt=False)
        self.assertTrue(self.storage.delete_blob("t4"))
        self.assertIsNone(self.storage.read_blob("t4"))

    def test_get_blob_meta(self):
        self.storage.create_blob(b"meta_test", blob_id="t5", encrypt=False)
        meta = self.storage.get_blob_meta("t5")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["blob_id"], "t5")

    def test_get_blob_meta_nonexistent(self):
        self.assertIsNone(self.storage.get_blob_meta("nope"))

    def test_list_blobs(self):
        self.storage.create_blob(b"a", blob_id="list_a", encrypt=False)
        self.storage.create_blob(b"b", blob_id="list_b", encrypt=False)
        blobs = self.storage.list_blobs()
        ids = [b["blob_id"] for b in blobs]
        self.assertIn("list_a", ids)
        self.assertIn("list_b", ids)

    def test_read_nonexistent_returns_none(self):
        self.assertIsNone(self.storage.read_blob("doesnt_exist"))

    def test_status(self):
        status = self.storage.get_server_status()
        self.assertEqual(status["mode"], "offline")
        self.assertIn("state", status)


# ===========================================================================
# 7) BlobFile read/write integration (offline)
# ===========================================================================

class TestBlobFileOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="blobfile_test_")
        self.storage = _make_offline_storage(self.tmp)
        # Pre-create a blob container
        self.storage.create_blob(pickle.dumps({}), blob_id="container", encrypt=True)

    def tearDown(self):
        self.storage.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_and_read_text(self):
        with BlobFile("container/file.txt", "w", storage=self.storage) as f:
            f.write("hello world")

        with BlobFile("container/file.txt", "r", storage=self.storage) as f:
            data = f.read()
        self.assertEqual(data, b"hello world")

    def test_write_and_read_json(self):
        payload = {"key": "value", "num": 42}
        with BlobFile("container/data.json", "w", storage=self.storage) as f:
            f.write_json(payload)

        with BlobFile("container/data.json", "r", storage=self.storage) as f:
            result = f.read_json()
        self.assertEqual(result, payload)

    def test_write_and_read_pickle(self):
        payload = {"set": {1, 2, 3}, "tuple": (4, 5)}
        with BlobFile("container/data.pkl", "w", storage=self.storage) as f:
            f.write_pickle(payload)

        with BlobFile("container/data.pkl", "r", storage=self.storage) as f:
            result = f.read_pickle()
        self.assertEqual(result, payload)

    def test_subfolder(self):
        with BlobFile("container/sub/deep.txt", "w", storage=self.storage) as f:
            f.write("nested")

        with BlobFile("container/sub/deep.txt", "r", storage=self.storage) as f:
            self.assertEqual(f.read(), b"nested")

    def test_exists(self):
        bf = BlobFile("container/check.txt", "r", storage=self.storage)
        self.assertFalse(bf.exists())

        with BlobFile("container/check.txt", "w", storage=self.storage) as f:
            f.write("exists now")

        self.assertTrue(BlobFile("container/check.txt", "r", storage=self.storage).exists())

    def test_write_mode_required(self):
        with BlobFile("container/ro.txt", "r", storage=self.storage) as f:
            with self.assertRaises(OSError):
                f.write("nope")

    def test_read_mode_required(self):
        with BlobFile("container/wo.txt", "w", storage=self.storage) as f:
            with self.assertRaises(OSError):
                f.read()

    def test_read_empty_json_returns_dict(self):
        with BlobFile("container/empty.json", "r", storage=self.storage) as f:
            self.assertEqual(f.read_json(), {})

    def test_read_empty_pickle_returns_dict(self):
        with BlobFile("container/empty.pkl", "r", storage=self.storage) as f:
            self.assertEqual(f.read_pickle(), {})


# ===========================================================================
# 8) E2E Tests — nur wenn MinIO erreichbar
# ===========================================================================

@unittest.skipIf(SKIP_E2E, "MinIO not reachable on 127.0.0.1:9000")
class TestBlobStorageE2EMinIO(unittest.TestCase):
    """Full roundtrip against a live MinIO instance."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="blob_e2e_")
        cls.storage = BlobStorage(
            mode=StorageMode.SERVER,
            minio_endpoint=os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"),
            minio_access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            minio_secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            minio_secure=False,
            storage_directory=cls.tmp,
            bucket="test-blob-e2e",
            user_id="e2e_test_user",
        )

    @classmethod
    def tearDownClass(cls):
        # Cleanup all test blobs
        for b in cls.storage.list_blobs():
            try:
                cls.storage.delete_blob(b["blob_id"])
            except Exception:
                pass
        cls.storage.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_e2e_create_read_delete(self):
        bid = self.storage.create_blob(b"e2e_data", blob_id="e2e_1", encrypt=False)
        data = self.storage.read_blob(bid, decrypt=False)
        self.assertEqual(data, b"e2e_data")

        meta = self.storage.get_blob_meta(bid)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["blob_id"], "e2e_1")

        self.assertTrue(self.storage.delete_blob(bid))
        # After delete, local cache also gone
        self.assertIsNone(self.storage.read_blob(bid, use_cache=False))

    def test_e2e_encrypted_roundtrip(self):
        bid = self.storage.create_blob(b"secret_e2e", blob_id="e2e_enc", encrypt=True)
        raw = self.storage.read_blob(bid, decrypt=False)
        self.assertNotEqual(raw, b"secret_e2e")
        plain = self.storage.read_blob(bid, decrypt=True)
        self.assertEqual(plain, b"secret_e2e")
        self.storage.delete_blob(bid)

    def test_e2e_update(self):
        self.storage.create_blob(b"v1", blob_id="e2e_upd", encrypt=False)
        res = self.storage.update_blob("e2e_upd", b"v2", encrypt=False)
        self.assertGreaterEqual(res["version"], 1)
        self.assertEqual(self.storage.read_blob("e2e_upd", decrypt=False), b"v2")
        self.storage.delete_blob("e2e_upd")

    def test_e2e_list(self):
        self.storage.create_blob(b"a", blob_id="e2e_list_a", encrypt=False)
        self.storage.create_blob(b"b", blob_id="e2e_list_b", encrypt=False)
        ids = [b["blob_id"] for b in self.storage.list_blobs()]
        self.assertIn("e2e_list_a", ids)
        self.assertIn("e2e_list_b", ids)
        self.storage.delete_blob("e2e_list_a")
        self.storage.delete_blob("e2e_list_b")

    def test_e2e_blobfile_roundtrip(self):
        self.storage.create_blob(pickle.dumps({}), blob_id="e2e_bf", encrypt=True)

        with BlobFile("e2e_bf/test/doc.json", "w", storage=self.storage) as f:
            f.write_json({"e2e": True})

        with BlobFile("e2e_bf/test/doc.json", "r", storage=self.storage) as f:
            self.assertEqual(f.read_json(), {"e2e": True})

        self.storage.delete_blob("e2e_bf")

    def test_e2e_server_status_healthy(self):
        status = self.storage.get_server_status()
        self.assertEqual(status["mode"], "server")
        self.assertTrue(status["is_healthy"])

    def test_e2e_read_from_minio_bypassing_cache(self):
        """Verify data is actually in MinIO, not just SQLite."""
        self.storage.create_blob(b"minio_check", blob_id="e2e_cache", encrypt=False)
        # Clear local cache by deleting from local_db
        self.storage.local_db.delete("e2e_cache", hard_delete=True)
        # Should still read from MinIO
        data = self.storage.read_blob("e2e_cache", use_cache=False, decrypt=False)
        self.assertEqual(data, b"minio_check")
        self.storage.delete_blob("e2e_cache")


# ===========================================================================
# 9) E2E BlobFile with custom key — MinIO-gated
# ===========================================================================

@unittest.skipIf(SKIP_E2E, "MinIO not reachable on 127.0.0.1:9000")
class TestBlobFileE2ECustomKey(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="blob_e2e_key_")
        cls.storage = BlobStorage(
            mode=StorageMode.SERVER,
            minio_endpoint=os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"),
            minio_access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            minio_secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            minio_secure=False,
            storage_directory=cls.tmp,
            bucket="test-blob-e2e",
            user_id="e2e_key_user",
        )
        cls.storage.create_blob(pickle.dumps({}), blob_id="keyed", encrypt=True)

    @classmethod
    def tearDownClass(cls):
        cls.storage.delete_blob("keyed")
        cls.storage.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _get_valid_key(self):
        """Get a key that passes the BlobFile validation."""
        return Code.generate_symmetric_key()

    def test_custom_key_write_read(self):
        key = self._get_valid_key()
        with BlobFile("keyed/secret.txt", "w", storage=self.storage, key=key) as f:
            f.write("keyed content")
        with BlobFile("keyed/secret.txt", "r", storage=self.storage, key=key) as f:
            self.assertEqual(f.read(), b"keyed content")


if __name__ == "__main__":
    unittest.main(verbosity=2)
