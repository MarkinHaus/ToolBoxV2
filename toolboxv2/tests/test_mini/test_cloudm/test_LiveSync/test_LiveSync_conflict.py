"""Tests for conflict.py — conflict detection + resolution."""
import os
import tempfile
import unittest


class TestDetectConflict(unittest.TestCase):
    def test_same_checksum_no_conflict(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import detect_conflict
        self.assertFalse(detect_conflict("aabb", "aabb"))

    def test_different_checksum_conflict(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import detect_conflict
        self.assertTrue(detect_conflict("aabb", "ccdd"))

    def test_empty_checksum_no_conflict(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import detect_conflict
        # Missing checksum = new file, not a conflict
        self.assertFalse(detect_conflict("", "aabb"))
        self.assertFalse(detect_conflict("aabb", ""))


class TestResolveMdConflict(unittest.TestCase):
    def test_merge_markers_present(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import resolve_md_conflict
        local = "# Title\nHallo Welt"
        remote = "# Title\nHallo!"
        merged = resolve_md_conflict(
            local, remote, "desktop-markin", "handy", 1713379200.0, 1713379201.5
        )
        self.assertIn("<<<<<<< LOCAL", merged)
        self.assertIn("=======", merged)
        self.assertIn(">>>>>>> REMOTE", merged)
        self.assertIn("Hallo Welt", merged)
        self.assertIn("Hallo!", merged)
        self.assertIn("desktop-markin", merged)
        self.assertIn("handy", merged)

    def test_merge_returns_string(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import resolve_md_conflict
        result = resolve_md_conflict("a", "b", "c1", "c2", 0.0, 0.0)
        self.assertIsInstance(result, str)


class TestResolveBinaryConflict(unittest.TestCase):
    def test_latest_wins(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import resolve_binary_conflict
        local_meta = {"checksum": "aa", "mtime": 1000.0, "client_id": "c1"}
        remote_meta = {"checksum": "bb", "mtime": 2000.0, "client_id": "c2"}
        winner, loser = resolve_binary_conflict(local_meta, remote_meta)
        self.assertEqual(winner["client_id"], "c2")
        self.assertEqual(loser["client_id"], "c1")

    def test_local_wins_if_newer(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import resolve_binary_conflict
        local_meta = {"checksum": "aa", "mtime": 3000.0, "client_id": "c1"}
        remote_meta = {"checksum": "bb", "mtime": 2000.0, "client_id": "c2"}
        winner, loser = resolve_binary_conflict(local_meta, remote_meta)
        self.assertEqual(winner["client_id"], "c1")

    def test_same_mtime_deterministic(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import resolve_binary_conflict
        local_meta = {"checksum": "aa", "mtime": 1000.0, "client_id": "c1"}
        remote_meta = {"checksum": "bb", "mtime": 1000.0, "client_id": "c2"}
        w1, _ = resolve_binary_conflict(local_meta, remote_meta)
        w2, _ = resolve_binary_conflict(local_meta, remote_meta)
        self.assertEqual(w1["client_id"], w2["client_id"])


class TestBackupFile(unittest.TestCase):
    def test_creates_backup(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import create_backup
        tmpdir = tempfile.mkdtemp()
        src = os.path.join(tmpdir, "notes.md")
        with open(src, "w") as f:
            f.write("original content")

        backup_path = create_backup(src)
        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(backup_path.endswith(".backup"))
        with open(backup_path) as f:
            self.assertEqual(f.read(), "original content")
        # Original still exists
        self.assertTrue(os.path.exists(src))

    def test_backup_nonexistent_returns_none(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import create_backup
        result = create_backup("/nonexistent/path/file.md")
        self.assertIsNone(result)

    def test_conflict_backup_path(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import make_conflict_backup_name
        name = make_conflict_backup_name(os.path.join("sub","notes.md"), "aabb1122")
        self.assertEqual(name, os.path.join("sub","notes.conflict.aabb1122.md"))

    def test_conflict_backup_path_no_ext(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import make_conflict_backup_name
        name = make_conflict_backup_name("README", "ccdd")
        self.assertEqual(name, "README.conflict.ccdd")


class TestMoveToSyncTrash(unittest.TestCase):
    def test_move_to_trash(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import move_to_sync_trash
        tmpdir = tempfile.mkdtemp()
        vault = os.path.join(tmpdir, "vault")
        os.makedirs(vault)
        src = os.path.join(vault, "old.md")
        with open(src, "w") as f:
            f.write("delete me")

        trash_path = move_to_sync_trash(vault, "old.md")
        self.assertFalse(os.path.exists(src))
        self.assertTrue(os.path.exists(trash_path))
        self.assertIn(".sync-trash", trash_path)

    def test_trash_preserves_content(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import move_to_sync_trash
        tmpdir = tempfile.mkdtemp()
        vault = os.path.join(tmpdir, "vault")
        os.makedirs(vault)
        src = os.path.join(vault, "file.md")
        with open(src, "w") as f:
            f.write("important")

        trash_path = move_to_sync_trash(vault, "file.md")
        with open(trash_path) as f:
            self.assertEqual(f.read(), "important")


if __name__ == "__main__":
    unittest.main()
