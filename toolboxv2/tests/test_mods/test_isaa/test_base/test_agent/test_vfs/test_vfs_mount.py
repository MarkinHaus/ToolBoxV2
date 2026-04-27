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

from toolboxv2.tests.test_mods.test_isaa.test_base.test_agent.test_vfs.test_vfs_pooling import _populate, _write_disk
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

# ===========================================================================
# EBENE 3b — Generic Shared-Mount (register_shared_mount / shared_write etc.)
# ===========================================================================

import shutil
import tempfile
import threading as _threading

from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs, GLOBAL_VFS_PATH



def _make_vfs_ext(session_id: str = "test", agent: str = "agent") -> VirtualFileSystemV2:
    """Helper mit optionalen Parametern — überschreibt den file-level _make_vfs nicht."""
    return VirtualFileSystemV2(session_id=session_id, agent_name=agent)

def _reset_gvfs_extra():
    """Reset extra-mount state ohne den /global/-Store zu berühren."""
    gvfs = get_global_vfs()
    with gvfs._store_lock:
        # Nur Extra-Mounts + deren Store-Einträge löschen
        for abs_path, info in list(gvfs._extra_mounts.items()):
            prefix = f"{info['mount_key']}::"
            for k in [k for k in gvfs._shared_store if k.startswith(prefix)]:
                del gvfs._shared_store[k]
        gvfs._extra_mounts.clear()
    return gvfs


class TestGenericSharedMountRegister(unittest.TestCase):
    """register_shared_mount / unregister_shared_mount Lifecycle."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vfs_gmt_")
        self.gvfs = _reset_gvfs_extra()

    def tearDown(self):
        _reset_gvfs_extra()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_register_returns_mount_key(self):
        key = self.gvfs.register_shared_mount(self.tmp)
        self.assertIsNotNone(key)
        self.assertTrue(len(key) > 0)

    def test_register_custom_key(self):
        key = self.gvfs.register_shared_mount(self.tmp, mount_key="worktree-abc")
        self.assertEqual(key, "worktree-abc")

    def test_register_same_path_twice_returns_same_key(self):
        k1 = self.gvfs.register_shared_mount(self.tmp, mount_key="wt-1")
        k2 = self.gvfs.register_shared_mount(self.tmp, mount_key="wt-other")
        # Zweite Registrierung gibt bestehenden Key zurück, überschreibt ihn nicht
        self.assertEqual(k1, k2)

    def test_get_mount_key_for_known_path(self):
        key = self.gvfs.register_shared_mount(self.tmp, mount_key="wt-findme")
        found = self.gvfs.get_mount_key_for(self.tmp)
        self.assertEqual(found, "wt-findme")

    def test_get_mount_key_for_unknown_path(self):
        self.assertIsNone(self.gvfs.get_mount_key_for("/nonexistent/xyz"))

    def test_unregister_removes_mount(self):
        self.gvfs.register_shared_mount(self.tmp, mount_key="wt-rm")
        self.gvfs.unregister_shared_mount(self.tmp)
        self.assertIsNone(self.gvfs.get_mount_key_for(self.tmp))

    def test_unregister_cleans_store_entries(self):
        key = self.gvfs.register_shared_mount(self.tmp, mount_key="wt-clean")
        self.gvfs.shared_write(key, "a.py", "x=1", local_base=self.tmp)
        self.gvfs.shared_write(key, "b.py", "y=2", local_base=self.tmp)
        self.gvfs.unregister_shared_mount(self.tmp)
        with self.gvfs._store_lock:
            leftover = [k for k in self.gvfs._shared_store if k.startswith("wt-clean::")]
        self.assertEqual(leftover, [])

    def test_unregister_nonexistent_does_not_raise(self):
        try:
            self.gvfs.unregister_shared_mount("/never/registered")
        except Exception as e:
            self.fail(f"unregister must be idempotent: {e}")


class TestGenericSharedMountHydrate(unittest.TestCase):
    """_hydrate_extra_mount lädt vorhandene Disk-Dateien in den Store."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vfs_hydrate_")
        _populate(self.tmp, {
            "src/main.py": "x = 1",
            "src/utils.py": "def f(): pass",
            "README.md": "# Title",
        })
        self.gvfs = _reset_gvfs_extra()

    def tearDown(self):
        _reset_gvfs_extra()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_hydrate_true_loads_disk_files(self):
        key = self.gvfs.register_shared_mount(
            self.tmp, mount_key="wt-hydrate", hydrate=True
        )
        entry = self.gvfs.shared_read(key, "src/main.py")
        self.assertIsNotNone(entry, "Hydrated file must be in store")
        self.assertEqual(entry["content"], "x = 1")

    def test_hydrate_loads_nested(self):
        key = self.gvfs.register_shared_mount(
            self.tmp, mount_key="wt-nested", hydrate=True
        )
        self.assertIsNotNone(self.gvfs.shared_read(key, "src/utils.py"))
        self.assertIsNotNone(self.gvfs.shared_read(key, "README.md"))

    def test_hydrate_false_skips_loading(self):
        key = self.gvfs.register_shared_mount(
            self.tmp, mount_key="wt-nohyd", hydrate=False
        )
        entry = self.gvfs.shared_read(key, "src/main.py")
        self.assertIsNone(entry, "hydrate=False must leave store empty")

    def test_hydrate_excludes_git_dir(self):
        git_dir = os.path.join(self.tmp, ".git")
        os.makedirs(git_dir, exist_ok=True)
        _write_disk(os.path.join(git_dir, "config"), "[core]")
        key = self.gvfs.register_shared_mount(
            self.tmp, mount_key="wt-git", hydrate=True
        )
        with self.gvfs._store_lock:
            git_entries = [k for k in self.gvfs._shared_store
                           if ".git" in k and k.startswith("wt-git::")]
        self.assertEqual(git_entries, [], ".git must be excluded from hydration")

    def test_hydrate_skips_large_files(self):
        large = os.path.join(self.tmp, "big.bin")
        # 6MB > 5MB threshold
        with open(large, "wb") as f:
            f.write(b"x" * 6 * 1024 * 1024)
        key = self.gvfs.register_shared_mount(
            self.tmp, mount_key="wt-large", hydrate=True
        )
        entry = self.gvfs.shared_read(key, "big.bin")
        self.assertIsNone(entry, "Files >5MB must be skipped in hydration")


class TestGenericSharedWrite(unittest.TestCase):
    """shared_write: Store + Disk + Version."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vfs_sw_")
        self.gvfs = _reset_gvfs_extra()
        self.key = self.gvfs.register_shared_mount(
            self.tmp, mount_key="wt-sw", hydrate=False
        )

    def tearDown(self):
        _reset_gvfs_extra()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_populates_store(self):
        self.gvfs.shared_write(self.key, "app.py", "v=1", local_base=self.tmp)
        entry = self.gvfs.shared_read(self.key, "app.py")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["content"], "v=1")

    def test_write_persists_to_disk(self):
        self.gvfs.shared_write(self.key, "app.py", "v=1", local_base=self.tmp)
        disk_path = os.path.join(self.tmp, "app.py")
        self.assertTrue(os.path.exists(disk_path))
        with open(disk_path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "v=1")

    def test_write_creates_subdirectory(self):
        self.gvfs.shared_write(self.key, "src/utils.py", "pass", local_base=self.tmp)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "src", "utils.py")))

    def test_write_records_author(self):
        self.gvfs.shared_write(
            self.key, "authored.py", "x=1", local_base=self.tmp, author="agent-x"
        )
        entry = self.gvfs.shared_read(self.key, "authored.py")
        self.assertEqual(entry["author"], "agent-x")

    def test_write_increments_version(self):
        r1 = self.gvfs.shared_write(self.key, "a.py", "1", local_base=self.tmp)
        r2 = self.gvfs.shared_write(self.key, "b.py", "2", local_base=self.tmp)
        self.assertLess(r1["version"], r2["version"])

    def test_write_traversal_rejected(self):
        result = self.gvfs.shared_write(self.key, "../escape.py", "bad", local_base=self.tmp)
        self.assertFalse(result["success"])

    def test_overwrite_updates_store(self):
        self.gvfs.shared_write(self.key, "x.py", "v=1", local_base=self.tmp)
        self.gvfs.shared_write(self.key, "x.py", "v=99", local_base=self.tmp)
        entry = self.gvfs.shared_read(self.key, "x.py")
        self.assertEqual(entry["content"], "v=99")


class TestGenericSharedRead(unittest.TestCase):
    """shared_read: Store-Hit, Miss, after-write."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vfs_sr_")
        self.gvfs = _reset_gvfs_extra()
        self.key = self.gvfs.register_shared_mount(
            self.tmp, mount_key="wt-sr", hydrate=False
        )

    def tearDown(self):
        _reset_gvfs_extra()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_read_miss_returns_none(self):
        self.assertIsNone(self.gvfs.shared_read(self.key, "ghost.py"))

    def test_read_after_write_returns_content(self):
        self.gvfs.shared_write(self.key, "x.py", "hello", local_base=self.tmp)
        entry = self.gvfs.shared_read(self.key, "x.py")
        self.assertEqual(entry["content"], "hello")

    def test_read_returns_copy_not_reference(self):
        self.gvfs.shared_write(self.key, "x.py", "original", local_base=self.tmp)
        e1 = self.gvfs.shared_read(self.key, "x.py")
        e1["content"] = "mutated"
        e2 = self.gvfs.shared_read(self.key, "x.py")
        self.assertEqual(e2["content"], "original",
                         "shared_read must return a copy, not the live dict")

    def test_read_after_hydration(self):
        """
        Hydration läuft beim ersten register_shared_mount mit hydrate=True.
        Wir nutzen dafür ein separates tempdir, damit kein bereits-registrierter
        Mount aus setUp im Weg ist.
        """
        tmp2 = tempfile.mkdtemp(prefix="vfs_sr_hyd_")
        try:
            _write_disk(os.path.join(tmp2, "hydrated.py"), "from_disk")
            key2 = self.gvfs.register_shared_mount(
                tmp2, mount_key="wt-sr-hyd", hydrate=True
            )
            entry = self.gvfs.shared_read(key2, "hydrated.py")
            self.assertIsNotNone(entry)
            self.assertEqual(entry["content"], "from_disk")
        finally:
            self.gvfs.unregister_shared_mount(tmp2)
            shutil.rmtree(tmp2, ignore_errors=True)

class TestGenericSharedDelete(unittest.TestCase):
    """shared_delete: Store + Disk cleanup."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vfs_sd_")
        self.gvfs = _reset_gvfs_extra()
        self.key = self.gvfs.register_shared_mount(
            self.tmp, mount_key="wt-sd", hydrate=False
        )
        self.gvfs.shared_write(self.key, "doomed.py", "bye", local_base=self.tmp)

    def tearDown(self):
        _reset_gvfs_extra()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_delete_removes_from_store(self):
        self.gvfs.shared_delete(self.key, "doomed.py", local_base=self.tmp)
        self.assertIsNone(self.gvfs.shared_read(self.key, "doomed.py"))

    def test_delete_removes_from_disk(self):
        disk = os.path.join(self.tmp, "doomed.py")
        self.assertTrue(os.path.exists(disk))
        self.gvfs.shared_delete(self.key, "doomed.py", local_base=self.tmp)
        self.assertFalse(os.path.exists(disk))

    def test_delete_traversal_rejected(self):
        result = self.gvfs.shared_delete(self.key, "../bad.py", local_base=self.tmp)
        self.assertFalse(result["success"])

    def test_delete_nonexistent_still_succeeds(self):
        # Store leer, Disk leer — trotzdem kein Crash
        result = self.gvfs.shared_delete(self.key, "never_existed.py", local_base=self.tmp)
        self.assertTrue(result["success"])


class TestGenericSharedMountVFSIntegration(unittest.TestCase):
    """
    vfs.read() / vfs.write() / vfs.delete() routen über den Shared-Store
    wenn das gemountete local_path via register_shared_mount bekannt ist.
    Zwei VFS auf gleichem Pfad — A schreibt → B sieht instant.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vfs_int_")
        _populate(self.tmp, {"shared.py": "v = 0"})
        self.gvfs = _reset_gvfs_extra()
        self.key = self.gvfs.register_shared_mount(
            self.tmp, mount_key=f"wt-int", hydrate=True
        )
        self.vfs_a = _make_vfs_ext("int-a", "agent-a")
        self.vfs_b = _make_vfs_ext("int-b", "agent-b")
        self.vfs_a.mount(self.tmp, vfs_path="/project", auto_sync=True)
        self.vfs_b.mount(self.tmp, vfs_path="/project", auto_sync=True)
        self.gvfs.register_vfs(self.vfs_a)
        self.gvfs.register_vfs(self.vfs_b)

    def tearDown(self):
        for vfs in [self.vfs_a, self.vfs_b]:
            try: vfs.unmount("/project", save_changes=False)
            except Exception: pass
        _reset_gvfs_extra()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_vfs_write_populates_shared_store(self):
        self.vfs_a.write("/project/shared.py", "v = A")
        entry = self.gvfs.shared_read(self.key, "shared.py")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["content"], "v = A")

    def test_vfs_write_records_agent_as_author(self):
        self.vfs_a.write("/project/shared.py", "v = A")
        entry = self.gvfs.shared_read(self.key, "shared.py")
        self.assertEqual(entry["author"], "agent-a")

    def test_vfs_read_hits_shared_store(self):
        # Direkt in Store schreiben (bypass VFS)
        self.gvfs.shared_write(self.key, "shared.py", "v = STORE",
                                local_base=self.tmp, author="external")
        # A's VFS-Cache muss invalidiert worden sein
        f = self.vfs_a.files.get("/project/shared.py")
        if f is not None:
            f.is_dirty = False  # sicherstellen dass Store-Pfad greift
        result = self.vfs_a.read("/project/shared.py")
        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "v = STORE",
                         "vfs.read muss aus Shared-Store lesen")

    def test_a_writes_b_sees_instantly(self):
        """Kein time.sleep, kein Poll. Store ist synchron."""
        self.vfs_a.write("/project/shared.py", "v = FROM_A")
        # B's Cache invalidiert → nächstes read aus Store
        f_b = self.vfs_b.files.get("/project/shared.py")
        if f_b is not None:
            # Cache muss None sein (invalidiert)
            self.assertIsNone(f_b._content,
                              "B's Cache muss nach A's write invalidiert sein")
        # Read liefert neuen Content
        result = self.vfs_b.read("/project/shared.py")
        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "v = FROM_A")

    def test_dirty_vfs_file_not_invalidated(self):
        """Wenn B gerade editiert (dirty), darf A's write B's Cache nicht löschen."""
        f_b = self.vfs_b.files.get("/project/shared.py")
        if f_b is None:
            self.skipTest("File nicht im Mount — scan nicht durchgelaufen")
        f_b._content = "B unsaved work"
        f_b.is_dirty = True
        f_b.backing_type = FileBackingType.MODIFIED

        self.vfs_a.write("/project/shared.py", "v = FROM_A")

        self.assertEqual(f_b._content, "B unsaved work",
                         "Dirty-Content darf durch Shared-Write nicht überschrieben werden")
        self.assertTrue(f_b.is_dirty)

    def test_vfs_delete_removes_from_store(self):
        self.vfs_a.write("/project/shared.py", "v = 1")
        self.assertTrue(self.gvfs.shared_read(self.key, "shared.py") is not None)

        self.vfs_a.delete("/project/shared.py")
        self.assertIsNone(self.gvfs.shared_read(self.key, "shared.py"))

    def test_new_file_via_vfs_create_enters_store(self):
        self.vfs_a.create("/project/new.py", "brand new")
        # Nach create+write muss Shared-Store den Content haben
        self.vfs_a.write("/project/new.py", "brand new")
        entry = self.gvfs.shared_read(self.key, "new.py")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["content"], "brand new")


class TestGenericSharedMountMultiWorktree(unittest.TestCase):
    """
    Zwei unabhängige Worktrees (verschiedene mount_keys) teilen keinen Store.
    Innerhalb eines Worktrees: alle Agents instant-synced.
    """

    def setUp(self):
        self.tmp_a = tempfile.mkdtemp(prefix="vfs_wt_a_")
        self.tmp_b = tempfile.mkdtemp(prefix="vfs_wt_b_")
        _populate(self.tmp_a, {"main.py": "a=1"})
        _populate(self.tmp_b, {"main.py": "b=1"})
        self.gvfs = _reset_gvfs_extra()
        self.key_a = self.gvfs.register_shared_mount(
            self.tmp_a, mount_key="wt-A", hydrate=True
        )
        self.key_b = self.gvfs.register_shared_mount(
            self.tmp_b, mount_key="wt-B", hydrate=True
        )

    def tearDown(self):
        _reset_gvfs_extra()
        shutil.rmtree(self.tmp_a, ignore_errors=True)
        shutil.rmtree(self.tmp_b, ignore_errors=True)

    def test_different_worktrees_isolated_stores(self):
        """Schreiben in Worktree A darf Worktree B nicht beeinflussen."""
        self.gvfs.shared_write(self.key_a, "main.py", "a=CHANGED", local_base=self.tmp_a)
        entry_b = self.gvfs.shared_read(self.key_b, "main.py")
        self.assertEqual(entry_b["content"], "b=1",
                         "Worktree B muss isoliert von Worktree A bleiben")

    def test_unregister_a_does_not_affect_b(self):
        self.gvfs.unregister_shared_mount(self.tmp_a)
        entry_b = self.gvfs.shared_read(self.key_b, "main.py")
        self.assertIsNotNone(entry_b, "Worktree B muss nach unregister(A) noch zugänglich sein")

    def test_two_vfs_on_same_worktree_share_store(self):
        vfs1 = _make_vfs_ext("s1", "coder-1")
        vfs2 = _make_vfs_ext("s2", "coder-2")
        try:
            vfs1.mount(self.tmp_a, vfs_path="/project", auto_sync=True)
            vfs2.mount(self.tmp_a, vfs_path="/project", auto_sync=True)
            self.gvfs.register_vfs(vfs1)
            self.gvfs.register_vfs(vfs2)

            vfs1.write("/project/main.py", "a=SHARED")
            entry = self.gvfs.shared_read(self.key_a, "main.py")
            self.assertIsNotNone(entry)
            self.assertEqual(entry["content"], "a=SHARED")

            # vfs2's Cache invalidiert → read gibt neuen Content
            result = vfs2.read("/project/main.py")
            self.assertTrue(result["success"])
            self.assertEqual(result["content"], "a=SHARED")
        finally:
            for vfs in [vfs1, vfs2]:
                try: vfs.unmount("/project", save_changes=False)
                except Exception: pass


class TestGenericSharedMountThreadSafety(unittest.TestCase):
    """Concurrent reads/writes auf Generic Shared-Mount crashen nicht."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vfs_ts_")
        self.gvfs = _reset_gvfs_extra()
        self.key = self.gvfs.register_shared_mount(
            self.tmp, mount_key="wt-ts", hydrate=False
        )

    def tearDown(self):
        _reset_gvfs_extra()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_concurrent_writes_different_files(self):
        errors = []

        def worker(wid):
            try:
                for i in range(10):
                    r = self.gvfs.shared_write(
                        self.key, f"t{wid}_{i}.py", f"v={wid}_{i}",
                        local_base=self.tmp, author=f"agent-{wid}"
                    )
                    if not r.get("success"):
                        errors.append(f"t{wid}_{i}: {r.get('error')}")
            except Exception as e:
                errors.append(str(e))

        threads = [_threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(errors, [])
        with self.gvfs._store_lock:
            entries = [k for k in self.gvfs._shared_store if k.startswith("wt-ts::")]
        self.assertEqual(len(entries), 200)

    def test_concurrent_writes_same_file_last_wins(self):
        written = set()
        errors = []

        def worker(wid):
            try:
                val = f"val_{wid}"
                written.add(val)
                self.gvfs.shared_write(self.key, "contested.py", val, local_base=self.tmp)
            except Exception as e:
                errors.append(str(e))

        threads = [_threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(errors, [])
        entry = self.gvfs.shared_read(self.key, "contested.py")
        self.assertIsNotNone(entry)
        self.assertIn(entry["content"], written)

if __name__ == "__main__":
    unittest.main()
