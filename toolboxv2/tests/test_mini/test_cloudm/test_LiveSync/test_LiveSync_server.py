"""Tests for server.py — SyncService server logic (unit tests, no real WS)."""
import asyncio
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSyncServer(unittest.TestCase):
    """Test SyncServer core logic without starting real WS."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.vault = os.path.join(self.tmpdir, "vault")
        os.makedirs(self.vault)
        # Create some test files
        with open(os.path.join(self.vault, "notes.md"), "w") as f:
            f.write("# Hello\n")
        os.makedirs(os.path.join(self.vault, "sub"))
        with open(os.path.join(self.vault, "sub", "deep.md"), "w") as f:
            f.write("deep content")

    def _make_server(self):
        from toolboxv2.mods.CloudM.LiveSync.server import SyncServer
        return SyncServer(
            vault_path=self.vault,
            share_id="test-share",
            env_config={
                "endpoint": "localhost:9000",
                "access_key": "admin",
                "secret_key": "secret",
                "secure": False,
                "bucket": "livesync",
                "ws_host": "127.0.0.1",
                "ws_port": 0,  # don't bind
            },
        )

    def test_init_opens_index_without_scanning(self):
        """The index starts empty; clients fill it, not a disk walk."""
        server = self._make_server()
        run(server._init_index())
        checksums = run(server.index.get_all_checksums())
        self.assertEqual(checksums, {})
        run(server.index.close())

    def test_handle_file_changed_updates_index(self):
        server = self._make_server()
        run(server._init_index())

        # Simulate a client reporting a file change
        run(server._process_file_changed(
            client_id="c1",
            path="new_file.md",
            checksum="deadbeef",
            minio_key="test-share/new_file.md.enc",
            file_type="text",
        ))

        row = run(server.index.get_file("new_file.md"))
        self.assertIsNotNone(row)
        self.assertEqual(row["checksum"], "deadbeef")
        run(server.index.close())

    def test_conflict_detection(self):
        server = self._make_server()
        run(server._init_index())

        # Set initial state
        run(server.index.upsert_file("notes.md", 1.0, 10, "aabb", "synced"))

        # Client sends change with different base checksum
        has_conflict = run(server._check_conflict("notes.md", "ccdd"))
        self.assertTrue(has_conflict)

        # Same checksum = no conflict
        has_conflict = run(server._check_conflict("notes.md", "aabb"))
        self.assertFalse(has_conflict)

        # New file = no conflict
        has_conflict = run(server._check_conflict("brand_new.md", "xxxx"))
        self.assertFalse(has_conflict)

        run(server.index.close())

    def test_broadcast_skips_originator(self):
        server = self._make_server()
        run(server._init_index())

        # Add mock clients
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        server.clients["c1"] = {"ws": ws1, "client_id": "c1", "device_type": "desktop"}
        server.clients["c2"] = {"ws": ws2, "client_id": "c2", "device_type": "mobile"}

        # Broadcast from c1
        from toolboxv2.mods.CloudM.LiveSync.protocol import SyncMessage
        msg = SyncMessage.file_changed("notes.md", "aabb", "test/notes.md.enc", source_client="c1")
        run(server._broadcast(msg, skip_client="c1"))

        ws1.send.assert_not_called()
        ws2.send.assert_called_once()
        run(server.index.close())

    def test_broadcast_handles_dead_connection(self):
        server = self._make_server()
        run(server._init_index())

        ws_dead = AsyncMock()
        ws_dead.send.side_effect = Exception("connection closed")
        ws_ok = AsyncMock()

        server.clients["dead"] = {"ws": ws_dead, "client_id": "dead", "device_type": "desktop"}
        server.clients["ok"] = {"ws": ws_ok, "client_id": "ok", "device_type": "desktop"}

        from toolboxv2.mods.CloudM.LiveSync.protocol import SyncMessage
        msg = SyncMessage.ping()
        # Should not raise
        run(server._broadcast(msg))
        ws_ok.send.assert_called_once()
        run(server.index.close())


class TestConflictDetection(unittest.TestCase):
    """
    A conflict is two clients editing from the same base — not simply a new
    revision. The old check compared the INCOMING checksum against the stored
    one, which differ on every ordinary update, so every update was reported
    as a conflict.
    """

    def setUp(self):
        self.vault = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.vault, ignore_errors=True)

    def _server(self):
        from toolboxv2.mods.CloudM.LiveSync.server import SyncServer
        srv = SyncServer(self.vault, "s1", {
            "endpoint": "127.0.0.1:9000", "access_key": "x",
            "secret_key": "y", "secure": False, "bucket": "tb-shared"})
        run(srv.index.init())
        return srv

    def test_sequential_update_is_not_a_conflict(self):
        srv = self._server()
        run(srv.index.upsert_file("a.md", 1.0, 3, "v1", "synced", "k"))
        # client had v1 and now sends v2 — the normal case
        self.assertFalse(run(srv._check_conflict("a.md", "v1")))
        run(srv.index.close())

    def test_divergent_base_is_a_conflict(self):
        srv = self._server()
        run(srv.index.upsert_file("a.md", 1.0, 3, "v2_from_b", "synced", "k"))
        # client edited from v1 while B already pushed v2
        self.assertTrue(run(srv._check_conflict("a.md", "v1")))
        run(srv.index.close())

    def test_new_file_is_not_a_conflict(self):
        srv = self._server()
        self.assertFalse(run(srv._check_conflict("new.md", "")))
        run(srv.index.close())

    def test_claiming_new_when_we_hold_a_version_is_a_conflict(self):
        srv = self._server()
        run(srv.index.upsert_file("a.md", 1.0, 3, "v1", "synced", "k"))
        self.assertTrue(run(srv._check_conflict("a.md", "")))
        run(srv.index.close())

    def test_legacy_client_without_base_is_not_flagged(self):
        """A client that sends no base at all cannot be judged, not always."""
        srv = self._server()
        run(srv.index.upsert_file("a.md", 1.0, 3, "v1", "synced", "k"))
        self.assertFalse(run(srv._check_conflict("a.md", None)))
        run(srv.index.close())


class TestServerModes(unittest.TestCase):
    """
    The server never touches share content itself.

    relay: broker only. replica: broker plus an ordinary SyncClient on the
    same folder. Watching its own vault and broadcasting changes it could not
    upload is gone — that announced files no client could ever fetch.
    """

    def setUp(self):
        self.vault = tempfile.mkdtemp()
        with open(os.path.join(self.vault, "preexisting.md"), "w") as f:
            f.write("was here before the server")

    def tearDown(self):
        shutil.rmtree(self.vault, ignore_errors=True)

    def _env(self):
        return {"endpoint": "127.0.0.1:9000", "access_key": "x",
                "secret_key": "y", "secure": False, "bucket": "tb-shared"}

    def test_default_mode_is_relay(self):
        from toolboxv2.mods.CloudM.LiveSync.server import SyncServer
        server = SyncServer(self.vault, "s1", self._env())
        self.assertEqual(server.mode, "relay")

    def test_unknown_mode_rejected(self):
        from toolboxv2.mods.CloudM.LiveSync.server import SyncServer
        with self.assertRaises(ValueError):
            SyncServer(self.vault, "s1", self._env(), mode="whatever")

    def test_index_does_not_scan_the_vault(self):
        """
        Files sitting in the server folder must not enter the index: there is
        no object for them in storage, so the entry would advertise a download
        that can never succeed.
        """
        from toolboxv2.mods.CloudM.LiveSync.server import SyncServer
        server = SyncServer(self.vault, "s1", self._env())
        run(server._init_index())
        checksums = run(server.index.get_all_checksums())
        self.assertEqual(checksums, {})
        run(server.index.close())

    def test_server_has_no_vault_watchdog(self):
        from toolboxv2.mods.CloudM.LiveSync import server as server_mod
        self.assertFalse(hasattr(server_mod, "AsyncWatchdogHandler"))
        self.assertFalse(hasattr(server_mod.SyncServer, "_start_watchdog"))
        self.assertFalse(hasattr(server_mod.SyncServer, "_on_server_file_changed"))

    def test_replica_without_stored_token_falls_back_to_relay(self):
        """No token in the store → no silent half-broken replica."""
        from toolboxv2.mods.CloudM.LiveSync.server import SyncServer
        server = SyncServer(self.vault, "no-such-share", self._env(), mode="replica")
        with patch("toolboxv2.mods.CloudM.LiveSync.server.get_share",
                   return_value=None):
            run(server._start_replica_client(8765))
        self.assertEqual(server.mode, "relay")
        self.assertIsNone(server._replica_client)

    def test_replica_client_connects_to_loopback(self):
        """
        The token carries the LAN endpoint peers use; the embedded client is
        on this machine and must not take the detour.
        """
        from toolboxv2.mods.CloudM.LiveSync.server import SyncServer
        from toolboxv2.mods.CloudM.LiveSync import create_share_token

        token = create_share_token(
            share_id="s1", encryption_key="c29tZWtleQ==",
            minio_endpoint="10.0.0.5:9000",
            ws_endpoint="ws://10.0.0.5:8765", bucket="tb-shared",
        )
        server = SyncServer(self.vault, "s1", self._env(), mode="replica")
        created = {}

        class _FakeClient:
            def __init__(self, config):
                created["config"] = config

            async def run(self):
                await asyncio.sleep(0)

        with patch("toolboxv2.mods.CloudM.LiveSync.server.get_share",
                   return_value={"token": token, "mode": "replica"}), \
                patch("toolboxv2.mods.CloudM.LiveSync.client.SyncClient", _FakeClient):
            run(server._start_replica_client(9123))
            run(server._stop_replica_client())

        self.assertEqual(created["config"].ws_endpoint, "ws://127.0.0.1:9123")
        self.assertEqual(created["config"].share_token, token)
        self.assertEqual(created["config"].vault_path, self.vault)


if __name__ == "__main__":
    unittest.main()
