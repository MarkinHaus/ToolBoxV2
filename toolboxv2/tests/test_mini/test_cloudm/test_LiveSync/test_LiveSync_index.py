"""Tests for index.py — async SQLite local index."""
import asyncio
import os
import tempfile
import time
import unittest


def run(coro):
    """Helper to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestLocalIndex(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_index.db")

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def _make_index(self):
        from toolboxv2.mods.CloudM.LiveSync.index import LocalIndex
        idx = LocalIndex(self.db_path)
        run(idx.init())
        return idx

    def test_init_creates_tables(self):
        idx = self._make_index()
        # Should not raise
        result = run(idx.get_file("nonexistent"))
        self.assertIsNone(result)
        run(idx.close())

    def test_upsert_and_get(self):
        idx = self._make_index()
        run(idx.upsert_file(
            rel_path="notes.md",
            mtime=1000.0,
            size=512,
            checksum="aabb1122",
            sync_state="synced",
            remote_key="share/notes.md.enc",
        ))
        row = run(idx.get_file("notes.md"))
        self.assertIsNotNone(row)
        self.assertEqual(row["rel_path"], "notes.md")
        self.assertEqual(row["checksum"], "aabb1122")
        self.assertEqual(row["sync_state"], "synced")
        self.assertEqual(row["remote_key"], "share/notes.md.enc")
        run(idx.close())

    def test_upsert_overwrites(self):
        idx = self._make_index()
        run(idx.upsert_file("a.md", 1.0, 10, "old", "synced"))
        run(idx.upsert_file("a.md", 2.0, 20, "new", "synced"))
        row = run(idx.get_file("a.md"))
        self.assertEqual(row["checksum"], "new")
        self.assertEqual(row["size"], 20)
        run(idx.close())

    def test_delete_file(self):
        idx = self._make_index()
        run(idx.upsert_file("a.md", 1.0, 10, "cc", "synced"))
        run(idx.delete_file("a.md"))
        self.assertIsNone(run(idx.get_file("a.md")))
        run(idx.close())

    def test_get_all_checksums(self):
        idx = self._make_index()
        run(idx.upsert_file("a.md", 1.0, 10, "aa", "synced"))
        run(idx.upsert_file("b.md", 2.0, 20, "bb", "synced"))
        run(idx.upsert_file("c.md", 3.0, 30, "cc", "synced"))
        cs = run(idx.get_all_checksums())
        self.assertEqual(cs, {"a.md": "aa", "b.md": "bb", "c.md": "cc"})
        run(idx.close())

    def test_get_pending(self):
        idx = self._make_index()
        run(idx.upsert_file("a.md", 1.0, 10, "aa", "synced"))
        run(idx.upsert_file("b.md", 2.0, 20, "bb", "pending_upload"))
        run(idx.upsert_file("c.md", 3.0, 30, "cc", "pending_download"))
        pending = run(idx.get_pending())
        paths = {r["rel_path"] for r in pending}
        self.assertIn("b.md", paths)
        self.assertIn("c.md", paths)
        self.assertNotIn("a.md", paths)
        run(idx.close())

    def test_set_sync_state(self):
        idx = self._make_index()
        run(idx.upsert_file("a.md", 1.0, 10, "aa", "synced"))
        run(idx.set_sync_state("a.md", "pending_upload"))
        row = run(idx.get_file("a.md"))
        self.assertEqual(row["sync_state"], "pending_upload")
        run(idx.close())


class TestSyncLog(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_index.db")

    def _make_index(self):
        from toolboxv2.mods.CloudM.LiveSync.index import LocalIndex
        idx = LocalIndex(self.db_path)
        run(idx.init())
        return idx

    def test_log_sync_event(self):
        idx = self._make_index()
        run(idx.log_sync_event("notes.md", "upload", "aabb", "client-1"))
        run(idx.log_sync_event("notes.md", "download", "ccdd", "client-2"))
        logs = run(idx.get_sync_log(limit=10))
        self.assertEqual(len(logs), 2)
        # Most recent first
        self.assertEqual(logs[0]["action"], "download")
        self.assertEqual(logs[1]["action"], "upload")
        run(idx.close())

    def test_log_limit(self):
        idx = self._make_index()
        for i in range(20):
            run(idx.log_sync_event(f"f{i}.md", "upload", f"cs{i}", "c1"))
        logs = run(idx.get_sync_log(limit=5))
        self.assertEqual(len(logs), 5)
        run(idx.close())

    def test_log_for_share(self):
        idx = self._make_index()
        run(idx.log_sync_event("a.md", "upload", "aa", "c1"))
        run(idx.log_sync_event("b.md", "conflict", "bb", "c2"))
        logs = run(idx.get_sync_log(limit=50))
        actions = [l["action"] for l in logs]
        self.assertIn("conflict", actions)
        run(idx.close())


class TestExportIndex(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_index.db")

    def _make_index(self):
        from toolboxv2.mods.CloudM.LiveSync.index import LocalIndex
        idx = LocalIndex(self.db_path)
        run(idx.init())
        return idx

    def test_export_gzipped(self):
        import gzip
        idx = self._make_index()
        for i in range(100):
            run(idx.upsert_file(f"file{i}.md", float(i), i * 10, f"cs{i:04d}", "synced"))

        data = run(idx.export_gzipped())
        self.assertIsInstance(data, bytes)
        # Should be valid gzip
        decompressed = gzip.decompress(data)
        self.assertGreater(len(decompressed), 0)
        run(idx.close())

    def test_import_gzipped_roundtrip(self):
        from toolboxv2.mods.CloudM.LiveSync.index import LocalIndex
        idx = self._make_index()
        for i in range(50):
            run(idx.upsert_file(f"f{i}.md", float(i), i, f"c{i:04d}", "synced"))
        exported = run(idx.export_gzipped())
        run(idx.close())

        # Import into fresh index
        db2_path = os.path.join(self.tmpdir, "imported.db")
        idx2 = LocalIndex(db2_path)
        run(idx2.init())
        run(idx2.import_gzipped(exported))
        cs = run(idx2.get_all_checksums())
        self.assertEqual(len(cs), 50)
        self.assertEqual(cs["f0.md"], "c0000")
        run(idx2.close())


if __name__ == "__main__":
    unittest.main()
