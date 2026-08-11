"""Tests for __init__.py — Supervisor interface (subprocess, share mgmt)."""
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock


class TestShareTokenCreation(unittest.TestCase):
    def test_create_share_produces_token(self):
        from toolboxv2.mods.CloudM.LiveSync import create_share_token
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken

        token_str = create_share_token(
            share_id="abc123",
            encryption_key="dGVzdGtleQ==",
            minio_endpoint="server:9000",
            ws_endpoint="ws://server:8765",
        )
        self.assertIsInstance(token_str, str)

        # Token decodes correctly
        tok = ShareToken.decode(token_str)
        self.assertEqual(tok.share_id, "abc123")
        self.assertEqual(tok.ws_endpoint, "ws://server:8765")


class TestSupervisorStatus(unittest.TestCase):
    def test_status_when_not_running(self):
        from toolboxv2.mods.CloudM.LiveSync import  get_sync_status
        status = get_sync_status()
        self.assertEqual(status["running"], False)
        self.assertIsNone(status["pid"])

    def test_status_has_required_fields(self):
        from toolboxv2.mods.CloudM.LiveSync import  get_sync_status
        status = get_sync_status()
        self.assertIn("running", status)
        self.assertIn("pid", status)
        self.assertIn("shares", status)


class TestShareRegistry(unittest.TestCase):
    """
    The registry is an encrypted file now, not a process dict: a restart used
    to drop every share, and replica mode needs the token back after a crash.
    """

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._saved = os.environ.get("DEVICE_KEY_DIR")
        os.environ["DEVICE_KEY_DIR"] = self._dir

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("DEVICE_KEY_DIR", None)
        else:
            os.environ["DEVICE_KEY_DIR"] = self._saved
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_register_and_list_shares(self):
        from toolboxv2.mods.CloudM.LiveSync import register_share, list_shares

        register_share("s1", "/tmp/vault1", "token1")
        register_share("s2", "/tmp/vault2", "token2", mode="replica")

        shares = list_shares()
        self.assertEqual(len(shares), 2)
        ids = {s["share_id"] for s in shares}
        self.assertEqual(ids, {"s1", "s2"})
        self.assertEqual({s["share_id"]: s["mode"] for s in shares}["s2"], "replica")

    def test_stop_share_removes(self):
        from toolboxv2.mods.CloudM.LiveSync import (
            register_share, stop_share, list_shares)

        register_share("s1", "/tmp/vault1", "token1")
        result = stop_share("s1")
        self.assertTrue(result["ok"])
        self.assertEqual(len(list_shares()), 0)

    def test_stop_nonexistent_share(self):
        from toolboxv2.mods.CloudM.LiveSync import stop_share
        result = stop_share("nonexistent")
        self.assertFalse(result["ok"])

    def test_survives_process_restart(self):
        """A fresh read of the store must still see the share."""
        from toolboxv2.mods.CloudM.LiveSync import register_share
        from toolboxv2.mods.CloudM.LiveSync import share_store

        register_share("s1", "/tmp/vault1", "token1", mode="replica", ws_port=9001)
        record = share_store.load_shares()["s1"]
        self.assertEqual(record["token"], "token1")
        self.assertEqual(record["ws_port"], 9001)

    def test_store_file_is_encrypted_and_private(self):
        """The token must not be readable in the file, and not world-readable."""
        from toolboxv2.mods.CloudM.LiveSync import register_share
        from toolboxv2.mods.CloudM.LiveSync import share_store

        register_share("s1", "/tmp/vault1", "supersecret-token-value")
        path = share_store.store_path()
        raw = path.read_bytes()
        self.assertNotIn(b"supersecret-token-value", raw)
        self.assertNotIn(b"/tmp/vault1", raw)
        self.assertEqual(oct(path.stat().st_mode)[-3:], "600")

    def test_unknown_mode_rejected(self):
        from toolboxv2.mods.CloudM.LiveSync import share_store
        with self.assertRaises(ValueError):
            share_store.save_share("s1", "/tmp/v", "t", mode="whatever")

    def test_replica_start_without_token_refuses(self):
        from toolboxv2.mods.CloudM.LiveSync import start_sync
        res = start_sync("/tmp/vault1", "unknown-share", 8765, mode="replica")
        self.assertFalse(res["ok"])
        self.assertIn("replica mode needs a stored token", res["error"])


class TestHealthcheck(unittest.TestCase):
    def test_selftest_checks_all_deps(self):
        from toolboxv2.mods.CloudM.LiveSync import  run_selftest
        report = run_selftest()
        self.assertIn("websockets", report)
        self.assertIn("watchdog", report)
        self.assertIn("minio", report)
        self.assertIn("cryptography", report)
        self.assertIn("aiosqlite", report)

    def test_selftest_all_installed(self):
        from toolboxv2.mods.CloudM.LiveSync import run_selftest
        report = run_selftest()
        missing = [d for d, ok in report.items() if not ok]
        if missing:
            self.skipTest(f"Optional deps not installed: {', '.join(missing)}")

if __name__ == "__main__":
    unittest.main()
