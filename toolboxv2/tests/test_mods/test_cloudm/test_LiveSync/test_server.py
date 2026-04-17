"""Tests for server.py — SyncService server logic (unit tests, no real WS)."""
import asyncio
import json
import os
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

    def test_init_builds_checksum_index(self):
        server = self._make_server()
        run(server._init_index())
        checksums = run(server.index.get_all_checksums())
        self.assertIn("notes.md", checksums)
        self.assertIn("sub/deep.md", checksums)
        run(server.index.close())

    def test_ignores_system_dirs(self):
        # Create .obsidian dir
        obs = os.path.join(self.vault, ".obsidian")
        os.makedirs(obs)
        with open(os.path.join(obs, "config.json"), "w") as f:
            f.write("{}")

        server = self._make_server()
        run(server._init_index())
        checksums = run(server.index.get_all_checksums())
        for path in checksums:
            self.assertNotIn(".obsidian", path)
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


class TestWatchdogQueue(unittest.TestCase):
    """Test the thread-safe watchdog → asyncio queue bridge."""

    def test_queue_receives_events(self):
        loop = asyncio.new_event_loop()
        q = asyncio.Queue()

        from toolboxv2.mods.CloudM.LiveSync.server import AsyncWatchdogHandler
        handler = AsyncWatchdogHandler(loop, q, "/tmp/vault")

        # Simulate a watchdog event
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/vault/notes.md"

        handler.on_modified(event)

        async def check():
            item = await asyncio.wait_for(q.get(), timeout=1.0)
            return item

        result = loop.run_until_complete(check())
        self.assertEqual(result[0], "modified")
        self.assertEqual(result[1], "notes.md")
        loop.close()

    def test_queue_ignores_system_files(self):
        loop = asyncio.new_event_loop()
        q = asyncio.Queue()

        from toolboxv2.mods.CloudM.LiveSync.server import AsyncWatchdogHandler
        handler = AsyncWatchdogHandler(loop, q, "/tmp/vault")

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/vault/.obsidian/config.json"
        handler.on_modified(event)

        event2 = MagicMock()
        event2.is_directory = False
        event2.src_path = "/tmp/vault/file.tmp"
        handler.on_modified(event2)

        self.assertTrue(q.empty())
        loop.close()

    def test_queue_ignores_directories(self):
        loop = asyncio.new_event_loop()
        q = asyncio.Queue()

        from toolboxv2.mods.CloudM.LiveSync.server import AsyncWatchdogHandler
        handler = AsyncWatchdogHandler(loop, q, "/tmp/vault")

        event = MagicMock()
        event.is_directory = True
        event.src_path = "/tmp/vault/subdir"
        handler.on_modified(event)

        self.assertTrue(q.empty())
        loop.close()


if __name__ == "__main__":
    unittest.main()
