"""Tests for minio_helper.py — MinIO operations (unit tests with mocks)."""
import io
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


class TestMinIOClientFactory(unittest.TestCase):
    def test_create_client(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import create_minio_client
        creds = {
            "endpoint": "localhost:9000",
            "access_key": "testkey",
            "secret_key": "testsecret",
            "secure": False,
        }
        with patch("toolboxv2.mods.CloudM.LiveSync.minio_helper.Minio") as MockMinio:
            client = create_minio_client(creds)
            MockMinio.assert_called_once_with(
                "localhost:9000",
                access_key="testkey",
                secret_key="testsecret",
                secure=False,
            )


class TestUploadFile(unittest.TestCase):
    def test_upload_bytes(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import upload_bytes
        mock_client = MagicMock()
        upload_bytes(
            mock_client,
            bucket="livesync",
            key="share1/notes.md.enc",
            data=b"encrypted_data",
            metadata={"x-amz-meta-original-hash": "aabb"},
        )
        mock_client.put_object.assert_called_once()
        call_args = mock_client.put_object.call_args
        self.assertEqual(call_args[0][0], "livesync")
        self.assertEqual(call_args[0][1], "share1/notes.md.enc")

    def test_upload_metadata_stored(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import upload_metadata
        mock_client = MagicMock()
        meta = {"checksum": "aabb", "mtime": 1234.0, "source_client": "c1"}
        upload_metadata(
            mock_client,
            bucket="livesync",
            share_prefix="share1",
            rel_path="notes.md",
            metadata=meta,
        )
        mock_client.put_object.assert_called_once()
        call_args = mock_client.put_object.call_args
        key = call_args[0][1]
        self.assertIn(".meta/", key)
        self.assertTrue(key.endswith(".json"))


class TestDownloadFile(unittest.TestCase):
    def test_download_bytes(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import download_bytes
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"encrypted_stuff"
        mock_client.get_object.return_value = mock_resp

        data = download_bytes(mock_client, "livesync", "share1/notes.md.enc")
        self.assertEqual(data, b"encrypted_stuff")
        mock_resp.close.assert_called_once()
        mock_resp.release_conn.assert_called_once()


class TestDeleteFile(unittest.TestCase):
    def test_delete_object(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import delete_object
        mock_client = MagicMock()
        delete_object(mock_client, "livesync", "share1/notes.md.enc")
        mock_client.remove_object.assert_called_once_with("livesync", "share1/notes.md.enc")


class TestEnsureBucket(unittest.TestCase):
    def test_creates_if_not_exists(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import ensure_bucket
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = False
        ensure_bucket(mock_client, "livesync")
        mock_client.make_bucket.assert_called_once_with("livesync")

    def test_skips_if_exists(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import ensure_bucket
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        ensure_bucket(mock_client, "livesync")
        mock_client.make_bucket.assert_not_called()


class TestMinIOKeyHelpers(unittest.TestCase):
    def test_make_object_key(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import make_object_key
        key = make_object_key("share123", "sub/notes.md")
        self.assertEqual(key, "share123/sub/notes.md.enc")

    def test_make_meta_key(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import make_meta_key
        key = make_meta_key("share123", "sub/notes.md")
        self.assertEqual(key, "share123/.meta/sub/notes.md.json")

    def test_rel_path_from_object_key(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import rel_path_from_object_key
        rp = rel_path_from_object_key("share123", "share123/sub/notes.md.enc")
        self.assertEqual(rp, "sub/notes.md")

    def test_rel_path_returns_none_for_meta(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import rel_path_from_object_key
        rp = rel_path_from_object_key("share123", "share123/.meta/notes.md.json")
        self.assertIsNone(rp)


class TestListRemoteFiles(unittest.TestCase):
    def test_list_encrypted_objects(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import list_remote_files
        mock_client = MagicMock()

        obj1 = MagicMock()
        obj1.object_name = "share1/notes.md.enc"
        obj1.last_modified = MagicMock()
        obj1.last_modified.timestamp.return_value = 1000.0
        obj1.size = 512

        obj2 = MagicMock()
        obj2.object_name = "share1/.meta/notes.md.json"  # should be skipped
        obj2.last_modified = MagicMock()

        obj3 = MagicMock()
        obj3.object_name = "share1/img.png.enc"
        obj3.last_modified = MagicMock()
        obj3.last_modified.timestamp.return_value = 2000.0
        obj3.size = 1024

        mock_client.list_objects.return_value = [obj1, obj2, obj3]

        result = list_remote_files(mock_client, "livesync", "share1")
        self.assertEqual(len(result), 2)
        self.assertEqual(result["notes.md"]["minio_key"], "share1/notes.md.enc")
        self.assertEqual(result["img.png"]["mtime"], 2000.0)


class TestHealthcheck(unittest.TestCase):
    def test_healthcheck_ok(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import healthcheck
        mock_client = MagicMock()
        mock_client.list_buckets.return_value = []
        ok, msg = healthcheck(mock_client)
        self.assertTrue(ok)

    def test_healthcheck_fail(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import healthcheck
        mock_client = MagicMock()
        mock_client.list_buckets.side_effect = Exception("connection refused")
        ok, msg = healthcheck(mock_client)
        self.assertFalse(ok)
        self.assertIn("connection refused", msg)


if __name__ == "__main__":
    unittest.main()
