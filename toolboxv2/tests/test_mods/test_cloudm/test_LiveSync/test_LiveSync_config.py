"""Tests for config.py"""
import base64
import json
import os
import unittest


class TestSyncConfig(unittest.TestCase):
    def test_create_config(self):
        from toolboxv2.mods.CloudM.LiveSync.config import SyncConfig
        cfg = SyncConfig(
            share_id="abc123",
            vault_path="/tmp/vault",
            minio_endpoint="localhost:9000",
            ws_endpoint="ws://localhost:8765",
            encryption_key="dGVzdGtleQ==",
        )
        self.assertEqual(cfg.share_id, "abc123")
        self.assertEqual(cfg.vault_path, "/tmp/vault")
        self.assertEqual(cfg.bucket, "livesync")

    def test_config_defaults(self):
        from toolboxv2.mods.CloudM.LiveSync.config import SyncConfig
        cfg = SyncConfig(
            share_id="x", vault_path="/tmp",
            minio_endpoint="h:9000", ws_endpoint="ws://h:8765",
            encryption_key="k",
        )
        self.assertEqual(cfg.bucket, "livesync")
        self.assertEqual(cfg.max_file_size, 50 * 1024 * 1024)
        self.assertEqual(cfg.debounce_seconds, 2.0)
        self.assertEqual(cfg.max_concurrent_transfers, 5)


class TestShareToken(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken
        tok = ShareToken(
            share_id="abc123",
            minio_endpoint="server.example:9000",
            bucket="livesync",
            prefix="abc123",
            encryption_key="c29tZWtleQ==",
            ws_endpoint="ws://server.example:8765",
        )
        encoded = tok.encode()
        self.assertIsInstance(encoded, str)

        restored = ShareToken.decode(encoded)
        self.assertEqual(restored.share_id, "abc123")
        self.assertEqual(restored.minio_endpoint, "server.example:9000")
        self.assertEqual(restored.encryption_key, "c29tZWtleQ==")
        self.assertEqual(restored.ws_endpoint, "ws://server.example:8765")
        self.assertEqual(restored.version, 1)

    def test_decode_invalid_token(self):
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken
        with self.assertRaises(ValueError):
            ShareToken.decode("not-valid-base64-json!!!")

    def test_token_contains_no_minio_credentials(self):
        """Token must NEVER contain MinIO access/secret keys."""
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken
        tok = ShareToken(
            share_id="x", minio_endpoint="h:9000", bucket="livesync",
            prefix="x", encryption_key="k", ws_endpoint="ws://h:8765",
        )
        encoded = tok.encode()
        raw = base64.urlsafe_b64decode(encoded).decode()
        data = json.loads(raw)
        self.assertNotIn("access_key", data)
        self.assertNotIn("secret_key", data)


class TestEnvConfig(unittest.TestCase):
    def test_load_from_env(self):
        from toolboxv2.mods.CloudM.LiveSync.config import load_env_config
        os.environ["MINIO_ENDPOINT"] = "test-host:9000"
        os.environ["MINIO_ROOT_USER"] = "testadmin"
        os.environ["MINIO_ROOT_PASSWORD"] = "testsecret"
        os.environ["LIVESYNC_WS_PORT"] = "9999"

        cfg = load_env_config()
        self.assertEqual(cfg["endpoint"], "test-host:9000")
        self.assertEqual(cfg["access_key"], "testadmin")
        self.assertEqual(cfg["secret_key"], "testsecret")
        self.assertEqual(cfg["ws_port"], 9999)

    def test_defaults_when_env_missing(self):
        from toolboxv2.mods.CloudM.LiveSync.config import load_env_config
        for k in ["MINIO_ENDPOINT", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD", "LIVESYNC_WS_PORT"]:
            os.environ.pop(k, None)
        cfg = load_env_config()
        self.assertEqual(cfg["ws_port"], 8765)


if __name__ == "__main__":
    unittest.main()
