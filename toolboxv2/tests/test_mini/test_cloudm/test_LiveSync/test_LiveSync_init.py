"""Tests for __init__.py — Supervisor interface (subprocess, share mgmt)."""
import os
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
    def test_register_and_list_shares(self):
        from toolboxv2.mods.CloudM.LiveSync import  _share_registry, register_share, list_shares
        _share_registry.clear()

        register_share("s1", "/tmp/vault1", "token1")
        register_share("s2", "/tmp/vault2", "token2")

        shares = list_shares()
        self.assertEqual(len(shares), 2)
        ids = {s["share_id"] for s in shares}
        self.assertIn("s1", ids)
        self.assertIn("s2", ids)

        _share_registry.clear()

    def test_stop_share_removes(self):
        from toolboxv2.mods.CloudM.LiveSync import  _share_registry, register_share, stop_share, list_shares
        _share_registry.clear()

        register_share("s1", "/tmp/vault1", "token1")
        result = stop_share("s1")
        self.assertTrue(result["ok"])
        self.assertEqual(len(list_shares()), 0)

        _share_registry.clear()

    def test_stop_nonexistent_share(self):
        from toolboxv2.mods.CloudM.LiveSync import  _share_registry, stop_share
        _share_registry.clear()
        result = stop_share("nonexistent")
        self.assertFalse(result["ok"])


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
        from toolboxv2.mods.CloudM.LiveSync import  run_selftest
        report = run_selftest()
        # All deps should be installed in test env
        for dep, ok in report.items():
            self.assertTrue(ok, f"{dep} not available")


if __name__ == "__main__":
    unittest.main()
