"""
VFS V2 – Mount-specific tests.

Covers:
  - Basic mount/unmount lifecycle
  - Shadow-file scanning (metadata-only)
  - Lazy content loading
  - Write to a mounted file → disk updated
  - Create file inside a mount → local file created, shadow_index populated   [BUG #5]
  - Readonly mount rejects writes
  - Delete from mount removes the local file
  - Unmount with save_changes persists dirty files
  - refresh_mount picks up externally added files
  - mkdir under a mount creates local directory

Run with:
    python -m pytest test_vfs_mount.py -v
    # or
    python -m unittest test_vfs_mount -v
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest

from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
    FileBackingType,
    VirtualFileSystemV2,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_vfs() -> VirtualFileSystemV2:
    return VirtualFileSystemV2(session_id="test-mount", agent_name="mount-agent")


def _populate_dir(base: str, files: dict[str, str]) -> None:
    """Write a dict of {rel_path: content} into *base*."""
    for rel, content in files.items():
        full = os.path.join(base, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)


# ===========================================================================
# 1. Basic mount / unmount
# ===========================================================================

class TestMountLifecycle(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vfs = _make_vfs()

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_mount_success(self):
        result = self.vfs.mount(self.tmp.name, vfs_path="/proj")
        self.assertTrue(result["success"], result)
        self.assertIn("/proj", self.vfs.mounts)

    def test_mount_nonexistent_dir_fails(self):
        result = self.vfs.mount("/nonexistent_xyz_abc", vfs_path="/proj")
        self.assertFalse(result["success"])

    def test_mount_duplicate_fails(self):
        self.vfs.mount(self.tmp.name, vfs_path="/proj")
        result = self.vfs.mount(self.tmp.name, vfs_path="/proj")
        self.assertFalse(result["success"])
        self.assertIn("Already mounted", result["error"])

    def test_unmount_removes_entries(self):
        _populate_dir(self.tmp.name, {"a.txt": "hello"})
        self.vfs.mount(self.tmp.name, vfs_path="/proj")

        result = self.vfs.unmount("/proj", save_changes=False)
        self.assertTrue(result["success"], result)
        self.assertNotIn("/proj", self.vfs.mounts)
        self.assertFalse(self.vfs._is_file("/proj/a.txt"))

    def test_unmount_nonexistent_fails(self):
        result = self.vfs.unmount("/not_a_mount")
        self.assertFalse(result["success"])

    def test_mount_creates_vfs_directory(self):
        _populate_dir(self.tmp.name, {"sub/file.py": "pass"})
        self.vfs.mount(self.tmp.name, vfs_path="/proj")
        self.assertTrue(self.vfs._is_directory("/proj"))

    def test_mount_returns_file_count(self):
        _populate_dir(self.tmp.name, {
            "a.py": "x=1",
            "b.py": "y=2",
            "c.txt": "z",
        })
        result = self.vfs.mount(self.tmp.name, vfs_path="/proj")
        self.assertEqual(result["files_indexed"], 3)


# ===========================================================================
# 2. Shadow file scanning
# ===========================================================================

class TestShadowScan(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vfs = _make_vfs()
        _populate_dir(self.tmp.name, {
            "main.py": "print('hi')",
            "utils/helper.py": "def f(): pass",
            "data/config.json": '{"k": 1}',
        })
        self.vfs.mount(self.tmp.name, vfs_path="/proj")

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_files_indexed_as_shadow(self):
        f = self.vfs.files.get("/proj/main.py")
        self.assertIsNotNone(f, "main.py must be indexed after mount")
        self.assertEqual(f.backing_type, FileBackingType.SHADOW)

    def test_shadow_content_not_loaded_initially(self):
        f = self.vfs.files["/proj/main.py"]
        self.assertFalse(f.is_loaded, "Shadow file content must not be pre-loaded")

    def test_subdirectory_files_indexed(self):
        self.assertIn("/proj/utils/helper.py", self.vfs.files)

    def test_shadow_index_populated(self):
        self.assertIn("/proj/main.py", self.vfs._shadow_index)
        self.assertIn("/proj/utils/helper.py", self.vfs._shadow_index)

    def test_local_path_stored(self):
        f = self.vfs.files["/proj/main.py"]
        self.assertIsNotNone(f.local_path)
        self.assertTrue(os.path.exists(f.local_path))

    def test_lazy_load_on_read(self):
        result = self.vfs.read("/proj/main.py")
        self.assertTrue(result["success"], result)
        self.assertIn("print", result["content"])

    def test_lazy_load_on_open(self):
        result = self.vfs.open("/proj/main.py")
        self.assertTrue(result["success"], result)
        f = self.vfs.files["/proj/main.py"]
        self.assertTrue(f.is_loaded)

    def test_excluded_pycache_not_indexed(self):
        pycache = os.path.join(self.tmp.name, "__pycache__")
        os.makedirs(pycache, exist_ok=True)
        _populate_dir(self.tmp.name, {"__pycache__/main.cpython-311.pyc": "\x00\x01"})
        # Re-scan via refresh
        self.vfs.refresh_mount("/proj")
        for path in self.vfs.files:
            self.assertNotIn("__pycache__", path, f"Excluded path found: {path}")


# ===========================================================================
# 3. Writing to a mounted file
# ===========================================================================

class TestMountWrite(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vfs = _make_vfs()
        _populate_dir(self.tmp.name, {"script.py": "x = 1\n"})
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_write_updates_disk(self):
        """After write with auto_sync, the local file on disk must be updated."""
        self.vfs.write("/proj/script.py", "x = 42\n")
        local_path = self.vfs.files["/proj/script.py"].local_path
        with open(local_path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "x = 42\n")

    def test_write_clears_is_dirty_after_sync(self):
        """
        After a successful auto_sync, is_dirty must be False.

        BUG #3 – write() response falsely returns `"is_dirty": True` even
        though _sync_to_local already cleared the flag.
        """
        self.vfs.write("/proj/script.py", "x = 99\n")
        f = self.vfs.files["/proj/script.py"]
        self.assertFalse(
            f.is_dirty,
            "is_dirty must be False after a successful auto_sync (BUG #3)"
        )

    def test_write_updates_updated_at(self):
        """
        write() with auto_sync must update f.updated_at.

        BUG #2 – the early-return path in write() skips `f.updated_at = …`
        """
        f = self.vfs.files["/proj/script.py"]
        before = f.updated_at
        time.sleep(0.01)            # ensure timestamp differs
        self.vfs.write("/proj/script.py", "x = 55\n")
        self.assertNotEqual(
            f.updated_at, before,
            "f.updated_at must be refreshed after write (BUG #2)"
        )

    def test_write_sets_vfs_dirty(self):
        """
        write() with auto_sync must keep vfs._dirty == True.

        BUG #2 – early return skips `self._dirty = True`.
        """
        self.vfs._dirty = False          # reset manually
        self.vfs.write("/proj/script.py", "new content\n")
        self.assertTrue(
            self.vfs._dirty,
            "VFS-level _dirty must be True after write (BUG #2)"
        )

    def test_write_empty_string_succeeds(self):
        """
        Writing an empty string to a mounted file must not fail.

        BUG #1 – _sync_to_local uses `if not f._content` which treats ''
        as falsy, causing sync to return an error for empty files.
        """
        result = self.vfs.write("/proj/script.py", "")
        self.assertTrue(result["success"], f"Write empty string failed: {result}")
        local_path = self.vfs.files["/proj/script.py"].local_path
        with open(local_path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "")


# ===========================================================================
# 4. Creating a file inside a mount
# ===========================================================================

class TestMountCreate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_create_makes_local_file(self):
        """create() under a mount must physically create the file on disk."""
        self.vfs.create("/proj/new.py", "print('new')\n")
        expected = os.path.join(self.tmp.name, "new.py")
        self.assertTrue(os.path.exists(expected), "Local file not created")
        with open(expected, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "print('new')\n")

    def test_create_adds_to_shadow_index(self):
        """
        create() under a mount must add the new entry to _shadow_index.

        BUG #5 – create() writes to disk but never calls
        `self._shadow_index[vfs_file_path] = local_file`.
        """
        self.vfs.create("/proj/new.py", "# hello\n")
        self.assertIn(
            "/proj/new.py",
            self.vfs._shadow_index,
            "New mount file must appear in _shadow_index (BUG #5)"
        )

    def test_create_registers_local_path(self):
        """The VFSFile created in a mount must have its local_path set."""
        self.vfs.create("/proj/thing.txt", "content")
        f = self.vfs.files["/proj/thing.txt"]
        self.assertIsNotNone(f.local_path)
        self.assertTrue(os.path.exists(f.local_path))

    def test_create_in_subdir_creates_local_subdir(self):
        """create() with a sub-path must create the local subdirectory."""
        self.vfs.create("/proj/sub/deep.py", "pass\n")
        expected = os.path.join(self.tmp.name, "sub", "deep.py")
        self.assertTrue(os.path.exists(expected))

    def test_create_empty_file_in_mount(self):
        """Creating an empty file in a mount must succeed without errors."""
        result = self.vfs.create("/proj/empty.py", "")
        self.assertTrue(result["success"], result)
        local = os.path.join(self.tmp.name, "empty.py")
        self.assertTrue(os.path.exists(local))

    def test_create_then_write_syncs(self):
        """After create() in a mount, write() must still auto-sync correctly."""
        self.vfs.create("/proj/mod.py", "v = 1\n")
        self.vfs.write("/proj/mod.py", "v = 999\n")
        local_path = self.vfs.files["/proj/mod.py"].local_path
        with open(local_path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "v = 999\n")


# ===========================================================================
# 5. Readonly mount
# ===========================================================================

class TestReadonlyMount(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate_dir(self.tmp.name, {"readme.md": "# README"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/ro", readonly=True)

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_write_to_readonly_mount_fails(self):
        result = self.vfs.write("/ro/readme.md", "overwritten")
        self.assertFalse(result["success"])

    def test_create_in_readonly_mount_fails(self):
        result = self.vfs.create("/ro/newfile.txt", "content")
        self.assertFalse(result["success"])

    def test_delete_from_readonly_mount_fails(self):
        result = self.vfs.delete("/ro/readme.md")
        self.assertFalse(result["success"])


# ===========================================================================
# 6. Delete from mount
# ===========================================================================

class TestMountDelete(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate_dir(self.tmp.name, {"to_delete.py": "# bye"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_delete_removes_vfs_entry(self):
        self.vfs.delete("/proj/to_delete.py")
        self.assertFalse(self.vfs._is_file("/proj/to_delete.py"))

    def test_delete_removes_local_file(self):
        local_path = self.vfs.files["/proj/to_delete.py"].local_path
        self.vfs.delete("/proj/to_delete.py")
        self.assertFalse(os.path.exists(local_path), "Local file must be deleted")

    def test_delete_removes_from_shadow_index(self):
        self.vfs.delete("/proj/to_delete.py")
        self.assertNotIn("/proj/to_delete.py", self.vfs._shadow_index)


# ===========================================================================
# 7. Unmount with save_changes
# ===========================================================================

class TestUnmountSaveChanges(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate_dir(self.tmp.name, {"work.py": "v = 1"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=False)

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def _write_and_mark_dirty(self):
        f = self.vfs.files["/proj/work.py"]
        # Bypass auto_sync by writing directly to the VFSFile
        f._content = "v = 999"
        f.is_dirty = True
        f.backing_type = FileBackingType.MODIFIED

    def test_unmount_save_changes_true_persists(self):
        self._write_and_mark_dirty()
        self.vfs.unmount("/proj", save_changes=True)

        local = os.path.join(self.tmp.name, "work.py")
        with open(local, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "v = 999")

    def test_unmount_save_changes_false_does_not_persist(self):
        self._write_and_mark_dirty()
        self.vfs.unmount("/proj", save_changes=False)

        local = os.path.join(self.tmp.name, "work.py")
        with open(local, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "v = 1")   # original unchanged


# ===========================================================================
# 8. refresh_mount
# ===========================================================================

class TestRefreshMount(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate_dir(self.tmp.name, {"existing.py": "x = 1"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj")

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_refresh_finds_new_file(self):
        # Add a file externally (after mount)
        _populate_dir(self.tmp.name, {"added_later.py": "y = 2"})
        self.vfs.refresh_mount("/proj")
        self.assertIn("/proj/added_later.py", self.vfs.files)

    def test_refresh_updates_changed_content(self):
        # Externally modify a file
        local = os.path.join(self.tmp.name, "existing.py")
        # Ensure mtime difference
        time.sleep(0.05)
        with open(local, "w", encoding="utf-8") as fh:
            fh.write("x = 999")

        # Load file first so it has a known mtime
        self.vfs.open("/proj/existing.py")
        # Now refresh
        self.vfs.refresh_mount("/proj")
        result = self.vfs.read("/proj/existing.py")
        self.assertIn("999", result["content"])

    def test_refresh_skips_dirty_files(self):
        """Dirty VFS files must not be overwritten on refresh."""
        f = self.vfs.files["/proj/existing.py"]
        f._content = "x = DIRTY"
        f.is_dirty = True
        f.backing_type = FileBackingType.MODIFIED

        # External change
        local = os.path.join(self.tmp.name, "existing.py")
        with open(local, "w", encoding="utf-8") as fh:
            fh.write("x = EXTERNAL")

        self.vfs.refresh_mount("/proj")
        self.assertEqual(f._content, "x = DIRTY",
                         "Dirty VFS content must not be overwritten by refresh")

    def test_refresh_nonexistent_mount_fails(self):
        result = self.vfs.refresh_mount("/no_such_mount")
        self.assertFalse(result["success"])


# ===========================================================================
# 9. mkdir under a mount
# ===========================================================================

class TestMountMkdir(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj")

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_mkdir_creates_local_directory(self):
        self.vfs.mkdir("/proj/newdir")
        expected = os.path.join(self.tmp.name, "newdir")
        self.assertTrue(os.path.isdir(expected),
                        "mkdir under mount must create local directory")

    def test_mkdir_parents_creates_nested_local_dirs(self):
        self.vfs.mkdir("/proj/a/b/c", parents=True)
        expected = os.path.join(self.tmp.name, "a", "b", "c")
        self.assertTrue(os.path.isdir(expected))


if __name__ == "__main__":
    unittest.main()
