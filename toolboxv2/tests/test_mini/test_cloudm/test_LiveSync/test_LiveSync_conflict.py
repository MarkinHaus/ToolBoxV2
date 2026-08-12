"""Tests for conflict.py — conflict detection + resolution."""
import os
import shutil
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


class TestConflictCopies(unittest.TestCase):
    """
    One rule for every file type: the loser is kept beside the winner as
    <name>.conflict.<device>.<timestamp>.<ext>. No merge markers — they turn a
    conflicted .md into something no editor renders and have no counterpart
    for a .docx.
    """

    def test_name_keeps_extension(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import make_conflict_name
        for rel, ext in (
            ("notes.md", ".md"), ("script.py", ".py"), ("report.pdf", ".pdf"),
            ("photo.png", ".png"), ("data.json", ".json"),
            ("sheet.xlsx", ".xlsx"), ("clip.mp4", ".mp4"),
        ):
            name = make_conflict_name(rel, "laptop", 1713379200.0)
            self.assertTrue(name.endswith(ext), f"{rel} -> {name}")
            self.assertIn(".conflict.laptop.", name)

    def test_name_keeps_compound_extension(self):
        """Path.suffix alone would produce backup.tar.conflict.<...>.gz."""
        from toolboxv2.mods.CloudM.LiveSync.conflict import make_conflict_name
        name = make_conflict_name("backup.tar.gz", "laptop", 1713379200.0)
        self.assertTrue(name.endswith(".tar.gz"), name)

    def test_name_without_extension(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import make_conflict_name
        name = make_conflict_name("Makefile", "laptop", 1713379200.0)
        self.assertTrue(name.startswith("Makefile.conflict.laptop."), name)

    def test_name_stays_in_subdirectory(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import make_conflict_name
        name = make_conflict_name("sub/deep/notes.md", "laptop", 1713379200.0)
        self.assertTrue(name.startswith("sub/deep/"), name)

    def test_device_id_sanitised(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import make_conflict_name
        name = make_conflict_name("a.md", "my.laptop/../etc", 1713379200.0)
        self.assertNotIn("/", name.split(".conflict.")[1])
        self.assertNotIn("..", name)

    def test_two_conflicts_do_not_collide(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import make_conflict_name
        a = make_conflict_name("a.md", "laptop", 1713379200.0)
        b = make_conflict_name("a.md", "laptop", 1713379260.0)
        self.assertNotEqual(a, b)

    def test_conflict_copies_never_sync(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import make_conflict_name
        from toolboxv2.mods.CloudM.LiveSync.protocol import should_ignore
        for rel in ("notes.md", "sub/report.pdf", "Makefile"):
            self.assertTrue(should_ignore(make_conflict_name(rel, "laptop")))

    def test_save_conflict_copy_moves_the_file(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import save_conflict_copy
        vault = tempfile.mkdtemp()
        src = os.path.join(vault, "notes.md")
        with open(src, "w") as f:
            f.write("local version")

        saved = save_conflict_copy(vault, "notes.md", "laptop")
        self.assertIsNotNone(saved)
        self.assertFalse(os.path.exists(src))
        with open(saved) as f:
            self.assertEqual(f.read(), "local version")
        shutil.rmtree(vault, ignore_errors=True)

    def test_save_conflict_copy_binary_roundtrip(self):
        """Binary content must survive byte for byte."""
        from toolboxv2.mods.CloudM.LiveSync.conflict import save_conflict_copy
        vault = tempfile.mkdtemp()
        payload = bytes(range(256)) * 8
        src = os.path.join(vault, "image.png")
        with open(src, "wb") as f:
            f.write(payload)

        saved = save_conflict_copy(vault, "image.png", "laptop")
        with open(saved, "rb") as f:
            self.assertEqual(f.read(), payload)
        shutil.rmtree(vault, ignore_errors=True)

    def test_save_conflict_copy_missing_file(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import save_conflict_copy
        vault = tempfile.mkdtemp()
        self.assertIsNone(save_conflict_copy(vault, "nope.md", "laptop"))
        shutil.rmtree(vault, ignore_errors=True)


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

    def test_backup_preserves_binary_content(self):
        from toolboxv2.mods.CloudM.LiveSync.conflict import create_backup
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "doc.pdf")
        payload = b"%PDF-1.7\n" + bytes(range(256))
        with open(path, "wb") as f:
            f.write(payload)

        backup = create_backup(path)
        with open(backup, "rb") as f:
            self.assertEqual(f.read(), payload)
        shutil.rmtree(tmpdir, ignore_errors=True)


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
