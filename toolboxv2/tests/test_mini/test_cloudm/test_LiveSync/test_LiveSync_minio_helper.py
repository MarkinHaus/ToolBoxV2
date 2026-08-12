"""Tests for minio_helper.py — MinIO operations (unit tests with mocks)."""
import io
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from toolboxv2.tests.a_util import IsolatedTestCase


class TestMinIOClientFactory(IsolatedTestCase):
    def test_create_client(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import create_minio_client
        creds = {
            "endpoint": "localhost:9000",
            "access_key": "testkey",
            "secret_key": "testsecret",
            "secure": False,
        }
        from toolboxv2.mods.CloudM.LiveSync import minio_helper
        minio_helper.MINIO_AVAILABLE = True
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


class TestCredentialVending(unittest.TestCase):
    """Tests for vend_user_credentials_for_user and vend_credentials_for_share."""

    def test_vend_user_credentials_for_user(self):
        """vend_user_credentials_for_user delegates to CredentialBroker.vend_user_credentials."""
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import vend_user_credentials_for_user

        env = {
            "endpoint": "localhost:9000",
            "access_key": "admin",
            "secret_key": "secret",
            "secure": False,
        }
        creds = {
            "endpoint": "localhost:9000",
            "access_key": "sa-user",
            "secret_key": "secret",
            "secure": False,
            "buckets": {"private": "tb-users-private", "public": "tb-users-public", "shared": "tb-shared"},
            "user_prefix": "markin",
            "policy_applied": True,
            "expires_in": 86400,
        }

        # CredentialBroker is imported inside the function, so we patch
        # at the source module path.
        with patch("toolboxv2.mods.CloudM.auth.minio_policy.CredentialBroker") as MockBroker:
            broker_instance = MockBroker.return_value
            broker_instance.vend_user_credentials.return_value = creds

            result = vend_user_credentials_for_user("markin", env)

            self.assertEqual(result, creds)
            MockBroker.assert_called_once()
            broker_instance.vend_user_credentials.assert_called_once_with("markin")

    def test_vend_user_credentials_for_user_requires_user_id(self):
        """Empty user_id raises ValueError."""
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import vend_user_credentials_for_user

        env = {"endpoint": "x", "access_key": "x", "secret_key": "x", "secure": False}
        with self.assertRaises(ValueError):
            vend_user_credentials_for_user("", env)

    def test_vend_user_credentials_for_user_requires_env_fields(self):
        """Missing env fields raise ValueError."""
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import vend_user_credentials_for_user

        with self.assertRaises(ValueError):
            vend_user_credentials_for_user("markin", {})

    def test_no_share_credential_vending_exists(self):
        """
        Share members must never be handed S3 credentials.

        The old vend_credentials_for_share() either returned keys that were
        never registered with MinIO, or fell back to the root credentials.
        Both are gone; presigned URLs replaced it.
        """
        from toolboxv2.mods.CloudM.LiveSync import minio_helper

        self.assertFalse(hasattr(minio_helper, "vend_credentials_for_share"))

    def test_presign_helpers_sign_one_object(self):
        """presign_get/put delegate to MinIO for exactly the given key."""
        from datetime import timedelta
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import presign_get, presign_put

        client = MagicMock()
        client.presigned_get_object.return_value = "https://s3/get"
        client.presigned_put_object.return_value = "https://s3/put"

        self.assertEqual(presign_get(client, "b", "share/a.md.enc", 60), "https://s3/get")
        client.presigned_get_object.assert_called_once_with(
            "b", "share/a.md.enc", expires=timedelta(seconds=60))

        self.assertEqual(presign_put(client, "b", "share/a.md.enc", 60), "https://s3/put")
        client.presigned_put_object.assert_called_once_with(
            "b", "share/a.md.enc", expires=timedelta(seconds=60))

    def test_http_upload_rejects_error_status(self):
        """A failed presigned PUT must raise, not pass silently."""
        import urllib.error
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import http_upload, TransferError

        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
                "https://s3/put", 403, "Forbidden", {}, None)):
            with self.assertRaises(TransferError) as ctx:
                http_upload("https://s3/put", b"data")
        self.assertEqual(ctx.exception.status, 403)
