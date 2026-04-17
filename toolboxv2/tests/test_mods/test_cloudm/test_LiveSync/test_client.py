"""Tests for client.py — SyncClient logic (unit tests, mocked WS + MinIO)."""
import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDebouncer(unittest.TestCase):
    """Test the debounce batch collector."""

    def test_deduplicates_same_path(self):
        from toolboxv2.mods.CloudM.LiveSync.client import DebounceBatch
        db = DebounceBatch(delay=0.1)
        db.add("notes.md", "modified")
        db.add("notes.md", "modified")
        db.add("notes.md", "modified")
        self.assertEqual(len(db.pending), 1)

    def test_keeps_different_paths(self):
        from toolboxv2.mods.CloudM.LiveSync.client import DebounceBatch
        db = DebounceBatch(delay=0.1)
        db.add("a.md", "modified")
        db.add("b.md", "created")
        db.add("c.md", "modified")
        self.assertEqual(len(db.pending), 3)

    def test_flush_returns_and_clears(self):
        from toolboxv2.mods.CloudM.LiveSync.client import DebounceBatch
        db = DebounceBatch(delay=0.0)
        db.add("a.md", "modified")
        db.add("b.md", "created")
        items = db.flush()
        self.assertEqual(len(items), 2)
        self.assertEqual(len(db.pending), 0)

    def test_not_ready_before_delay(self):
        from toolboxv2.mods.CloudM.LiveSync.client import DebounceBatch
        db = DebounceBatch(delay=10.0)
        db.add("a.md", "modified")
        self.assertFalse(db.is_ready())

    def test_ready_after_delay(self):
        from toolboxv2.mods.CloudM.LiveSync.client import DebounceBatch
        db = DebounceBatch(delay=0.0)
        db.add("a.md", "modified")
        self.assertTrue(db.is_ready())

    def test_deleted_overrides_modified(self):
        from toolboxv2.mods.CloudM.LiveSync.client import DebounceBatch
        db = DebounceBatch(delay=0.0)
        db.add("a.md", "modified")
        db.add("a.md", "deleted")
        items = db.flush()
        self.assertEqual(items["a.md"], "deleted")


class TestSyncClient(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.vault = os.path.join(self.tmpdir, "vault")
        os.makedirs(self.vault)

    def _make_client(self):
        from toolboxv2.mods.CloudM.LiveSync.client import SyncClient
        from toolboxv2.mods.CloudM.LiveSync.config import SyncConfig
        cfg = SyncConfig(
            share_id="test-share",
            vault_path=self.vault,
            minio_endpoint="localhost:9000",
            ws_endpoint="ws://localhost:8765",
            encryption_key="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        )
        return SyncClient(cfg)

    def test_init(self):
        client = self._make_client()
        self.assertEqual(client.config.share_id, "test-share")
        self.assertFalse(client._running)

    def test_compute_local_diff_new_files(self):
        # Create local files
        with open(os.path.join(self.vault, "a.md"), "w") as f:
            f.write("aaa")
        with open(os.path.join(self.vault, "b.md"), "w") as f:
            f.write("bbb")

        client = self._make_client()
        run(client.index.init())

        # Server has no files
        server_checksums = {}
        to_download, to_upload = run(client._compute_diff(server_checksums))
        self.assertEqual(len(to_upload), 2)
        self.assertEqual(len(to_download), 0)
        run(client.index.close())

    def test_compute_local_diff_missing_remote(self):
        client = self._make_client()
        run(client.index.init())

        # Server has files we don't
        server_checksums = {"notes.md": "aabb", "todo.md": "ccdd"}
        to_download, to_upload = run(client._compute_diff(server_checksums))
        self.assertEqual(len(to_download), 2)
        self.assertEqual(len(to_upload), 0)
        run(client.index.close())

    def test_compute_diff_with_conflicts(self):
        with open(os.path.join(self.vault, "notes.md"), "w") as f:
            f.write("local version")

        client = self._make_client()
        run(client.index.init())

        from toolboxv2.mods.CloudM.LiveSync.crypto import compute_checksum
        local_cs = compute_checksum(b"local version")

        # Server has different checksum for same file
        server_checksums = {"notes.md": "different_checksum"}
        to_download, to_upload = run(client._compute_diff(server_checksums))

        # Should appear in download list (server wins on reconnect)
        self.assertIn("notes.md", [d[0] for d in to_download])
        run(client.index.close())

    def test_upload_pipeline(self):
        """Test the upload path: read → checksum → encrypt → upload → notify."""
        with open(os.path.join(self.vault, "test.md"), "w") as f:
            f.write("hello world")

        client = self._make_client()
        run(client.index.init())
        client._minio = MagicMock()
        client._ws = AsyncMock()

        run(client._upload_file("test.md"))

        # MinIO put_object called twice (file + metadata)
        self.assertEqual(client._minio.put_object.call_count, 2)
        # WS notification sent
        client._ws.send.assert_called_once()

        # Index updated
        row = run(client.index.get_file("test.md"))
        self.assertIsNotNone(row)
        self.assertEqual(row["sync_state"], "synced")
        run(client.index.close())

    def test_download_pipeline(self):
        """Test download path: fetch → decrypt → atomic write → index update."""
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_bytes

        client = self._make_client()
        run(client.index.init())

        # Mock MinIO download
        original = b"# Remote content\n"
        encrypted = encrypt_bytes(original, client.config.encryption_key)

        mock_resp = MagicMock()
        mock_resp.read.return_value = encrypted
        client._minio = MagicMock()
        client._minio.get_object.return_value = mock_resp

        run(client._download_file("remote.md", "test-share/remote.md.enc"))

        # File written
        local = os.path.join(self.vault, "remote.md")
        self.assertTrue(os.path.exists(local))
        with open(local, "rb") as f:
            self.assertEqual(f.read(), original)

        # Index updated
        row = run(client.index.get_file("remote.md"))
        self.assertIsNotNone(row)
        self.assertEqual(row["sync_state"], "synced")

        # No .sync-tmp left
        self.assertFalse(os.path.exists(local + ".sync-tmp"))
        run(client.index.close())

    def test_download_checksum_verification(self):
        """Downloaded file must match expected checksum."""
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_bytes, compute_checksum

        client = self._make_client()
        run(client.index.init())

        original = b"content"
        encrypted = encrypt_bytes(original, client.config.encryption_key)
        expected_cs = compute_checksum(original)

        mock_resp = MagicMock()
        mock_resp.read.return_value = encrypted
        client._minio = MagicMock()
        client._minio.get_object.return_value = mock_resp

        run(client._download_file("file.md", "test-share/file.md.enc", expected_checksum=expected_cs))

        self.assertTrue(os.path.exists(os.path.join(self.vault, "file.md")))
        run(client.index.close())

    def test_handle_file_deleted_moves_to_trash(self):
        # Create a local file
        fpath = os.path.join(self.vault, "delete_me.md")
        with open(fpath, "w") as f:
            f.write("doomed")

        client = self._make_client()
        run(client.index.init())
        run(client.index.upsert_file("delete_me.md", 1.0, 6, "aabb", "synced"))

        run(client._handle_remote_delete("delete_me.md"))

        self.assertFalse(os.path.exists(fpath))
        # Should be in .sync-trash
        trash = os.path.join(self.vault, ".sync-trash")
        self.assertTrue(os.path.exists(trash))
        self.assertTrue(len(os.listdir(trash)) > 0)

        # Index cleared
        self.assertIsNone(run(client.index.get_file("delete_me.md")))
        run(client.index.close())


class TestReconnectBackoff(unittest.TestCase):
    def test_exponential_backoff(self):
        from toolboxv2.mods.CloudM.LiveSync.client import _backoff_delay
        d1 = _backoff_delay(0, base=1.0, maximum=60.0)
        d2 = _backoff_delay(1, base=1.0, maximum=60.0)
        d3 = _backoff_delay(5, base=1.0, maximum=60.0)
        self.assertAlmostEqual(d1, 1.0, places=0)
        self.assertGreater(d2, d1)
        self.assertLessEqual(d3, 60.0)

    def test_backoff_caps_at_max(self):
        from toolboxv2.mods.CloudM.LiveSync.client import _backoff_delay
        d = _backoff_delay(100, base=1.0, maximum=60.0)
        self.assertLessEqual(d, 60.0)


if __name__ == "__main__":
    unittest.main()
