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
        self.assertEqual(cfg.bucket, "tb-shared")

    def test_config_defaults(self):
        from toolboxv2.mods.CloudM.LiveSync.config import SyncConfig
        cfg = SyncConfig(
            share_id="x", vault_path="/tmp",
            minio_endpoint="h:9000", ws_endpoint="ws://h:8765",
            encryption_key="k",
        )
        self.assertEqual(cfg.bucket, "tb-shared")
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

        self.assertTrue(encoded.startswith("v4:"))

        restored = ShareToken.decode(encoded)
        self.assertEqual(restored.share_id, "abc123")
        self.assertEqual(restored.minio_endpoint, "server.example:9000")
        self.assertEqual(restored.encryption_key, "c29tZWtleQ==")
        self.assertEqual(restored.ws_endpoint, "ws://server.example:8765")
        self.assertEqual(restored.version, 4)

        verified = ShareToken.verify(encoded)
        self.assertEqual(verified.share_id, "abc123")

    def test_decode_invalid_token(self):
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken
        with self.assertRaises(ValueError):
            ShareToken.decode("v4:invalid!!!.garbage")

    def test_legacy_token_rejected(self):
        """A v3 token has no signature and must not be accepted at all."""
        import base64, json
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken
        legacy = "v3:" + base64.urlsafe_b64encode(
            json.dumps({"share_id": "x"}).encode()).decode()
        with self.assertRaises(ValueError):
            ShareToken.decode(legacy)
        with self.assertRaises(ValueError):
            ShareToken.verify(legacy)

    def test_forged_token_rejected(self):
        """A self-made payload without a valid HMAC must fail verification."""
        import base64, json, time
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken
        payload = base64.urlsafe_b64encode(json.dumps({
            "v": 4, "sid": "victim", "bkt": "tb-shared", "pfx": "victim",
            "key": "k", "ws": "ws://h:1", "s3": "h:9000",
            "exp": time.time() + 999,
        }).encode()).decode().rstrip("=")
        forged = f"v4:{payload}.{'A' * 43}"
        with self.assertRaises(ValueError):
            ShareToken.verify(forged)

    def test_tampered_payload_rejected(self):
        """Editing the payload of a genuine token invalidates the signature."""
        import base64, json
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken
        tok = ShareToken(
            share_id="orig", minio_endpoint="h:9000", bucket="tb-shared",
            prefix="orig", encryption_key="k", ws_endpoint="ws://h:8765",
        )
        payload_b64, signature = tok.encode()[3:].split(".", 1)
        data = json.loads(base64.urlsafe_b64decode(
            payload_b64 + "=" * (-len(payload_b64) % 4)))
        data["sid"] = "other"
        swapped = base64.urlsafe_b64encode(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).decode().rstrip("=")
        with self.assertRaises(ValueError):
            ShareToken.verify(f"v4:{swapped}.{signature}")

    def test_expired_token_rejected(self):
        """verify() enforces the expiry; decode() does not need to."""
        import time
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken
        tok = ShareToken(
            share_id="x", minio_endpoint="h:9000", bucket="tb-shared",
            prefix="x", encryption_key="k", ws_endpoint="ws://h:8765",
            expires_at=time.time() - 10,
        )
        with self.assertRaises(ValueError):
            ShareToken.verify(tok.encode())

    def test_raw_token_reaches_sync_config(self):
        """to_sync_config must carry the raw token or AUTH fails."""
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken
        tok = ShareToken(
            share_id="x", minio_endpoint="h:9000", bucket="tb-shared",
            prefix="x", encryption_key="k", ws_endpoint="ws://h:8765",
        )
        encoded = tok.encode()
        cfg = ShareToken.decode(encoded).to_sync_config("/tmp/v", raw_token=encoded)
        self.assertEqual(cfg.share_token, encoded)

    def test_token_contains_no_minio_credentials(self):
        """Token must NEVER contain MinIO access/secret keys."""
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken
        tok = ShareToken(
            share_id="x", minio_endpoint="h:9000", bucket="tb-shared",
            prefix="x", encryption_key="k", ws_endpoint="ws://h:8765",
        )
        encoded = tok.encode()
        # The token carries the AES key but never S3 credentials.
        restored = ShareToken.decode(encoded)
        self.assertNotIn("access_key", [a for a in dir(restored) if not a.startswith('_')])
        self.assertNotIn("secret_key", [a for a in dir(restored) if not a.startswith('_')])


class TestEnvConfig(unittest.TestCase):
    def test_load_from_env(self):
        from toolboxv2.mods.CloudM.LiveSync.config import load_env_config

        saved = {k: os.environ.get(k) for k in [
            "MINIO_ENDPOINT", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD", "LIVESYNC_WS_PORT"
        ]}

        os.environ["MINIO_ENDPOINT"] = "test-host:9000"
        os.environ["MINIO_ROOT_USER"] = "testadmin"
        os.environ["MINIO_ROOT_PASSWORD"] = "testsecret"
        os.environ["LIVESYNC_WS_PORT"] = "9999"

        cfg = load_env_config()
        self.assertEqual(cfg["endpoint"], "test-host:9000")
        self.assertEqual(cfg["access_key"], "testadmin")
        self.assertEqual(cfg["secret_key"], "testsecret")
        self.assertEqual(cfg["ws_port"], 9999)

        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_defaults_when_env_missing(self):
        from toolboxv2.mods.CloudM.LiveSync.config import load_env_config
        for k in ["MINIO_ENDPOINT", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD", "LIVESYNC_WS_PORT"]:
            os.environ.pop(k, None)
        cfg = load_env_config()
        self.assertEqual(cfg["ws_port"], 8765)


if __name__ == "__main__":
    unittest.main()
