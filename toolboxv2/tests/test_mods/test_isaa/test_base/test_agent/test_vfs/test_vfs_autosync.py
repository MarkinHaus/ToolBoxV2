"""
VFS V2 – Auto-Sync tests.

Covers every path where content changes should propagate from VFS to disk
(or be intentionally skipped when auto_sync=False).

Bugs explicitly targeted:

  BUG #1  _sync_to_local uses `if not f._content` → empty string ("") is
          treated as missing content, blocking sync of empty files.
          Fix: change to `if f._content is None`.

  BUG #2  write() with a successful auto_sync returns early, skipping both
          `f.updated_at = datetime.now().isoformat()` and `self._dirty = True`.

  BUG #3  write() response dict contains `"is_dirty": True` even though
          _sync_to_local() already cleared the flag (f.is_dirty = False).

  BUG #4  append() and edit() call _sync_to_local() but throw away the
          result — sync errors are never surfaced to the caller.

Run with:
    python -m pytest test_vfs_autosync.py -v
    # or
    python -m unittest test_vfs_autosync -v
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
    FileBackingType,
    VirtualFileSystemV2,
    VFSFile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vfs() -> VirtualFileSystemV2:
    return VirtualFileSystemV2(session_id="test-sync", agent_name="sync-agent")


def _populate(base: str, files: dict[str, str]) -> None:
    for rel, content in files.items():
        full = os.path.join(base, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)


def _read_local(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ===========================================================================
# 1. write() – auto_sync enabled
# ===========================================================================

class TestWriteAutoSync(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate(self.tmp.name, {"app.py": "x = 1\n"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)
        self.local_path = os.path.join(self.tmp.name, "app.py")

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_write_updates_disk(self):
        """Primary contract: write() must persist content to disk."""
        self.vfs.write("/proj/app.py", "x = 42\n")
        self.assertEqual(_read_local(self.local_path), "x = 42\n")

    def test_write_returns_success(self):
        result = self.vfs.write("/proj/app.py", "x = 7\n")
        self.assertTrue(result["success"], result)

    def test_write_clears_is_dirty(self):
        """
        After a successful auto_sync, f.is_dirty must be False.

        BUG #3 – the response dict says `"is_dirty": True`, but _sync_to_local
        already cleared the flag. The *actual* field state should be False.
        """
        self.vfs.write("/proj/app.py", "x = 0\n")
        f = self.vfs.files["/proj/app.py"]
        self.assertFalse(
            f.is_dirty,
            "f.is_dirty must be False after a successful auto_sync (BUG #3)"
        )

    def test_write_refreshes_updated_at(self):
        """
        write() must update f.updated_at.

        BUG #2 – the early-return path in write() exits before reaching
        `f.updated_at = datetime.now().isoformat()`.
        """
        f = self.vfs.files["/proj/app.py"]
        original_ts = f.updated_at
        time.sleep(0.02)

        self.vfs.write("/proj/app.py", "x = 5\n")

        self.assertNotEqual(
            f.updated_at, original_ts,
            "f.updated_at must be updated after write() (BUG #2)"
        )

    def test_write_sets_vfs_dirty_flag(self):
        """
        After write(), vfs._dirty must be True.

        BUG #2 – the early-return path skips `self._dirty = True`.
        """
        self.vfs._dirty = False
        self.vfs.write("/proj/app.py", "x = 3\n")
        self.assertTrue(
            self.vfs._dirty,
            "VFS-level _dirty must be True after write() (BUG #2)"
        )

    def test_write_empty_string_syncs_without_error(self):
        """
        Writing '' (empty string) must not fail.

        BUG #1 – `if not f._content` treats '' as missing content and returns
        {"success": False, "error": "No content to sync"}.
        Fix: `if f._content is None`.
        """
        result = self.vfs.write("/proj/app.py", "")
        self.assertTrue(
            result["success"],
            f"write('') must succeed (BUG #1): {result}"
        )
        self.assertEqual(_read_local(self.local_path), "")

    def test_write_overwrites_previous_content_on_disk(self):
        self.vfs.write("/proj/app.py", "first write\n")
        self.vfs.write("/proj/app.py", "second write\n")
        self.assertEqual(_read_local(self.local_path), "second write\n")

    def test_write_response_is_dirty_field_is_false_after_sync(self):
        """
        The response metadata `is_dirty` must reflect the *actual* post-sync
        state (False), not a stale pre-sync value.

        BUG #3 – currently the response hardcodes `"is_dirty": True`.
        """
        result = self.vfs.write("/proj/app.py", "clean write\n")
        # If the key exists, it should be False (synced); True indicates the bug.
        if "is_dirty" in result:
            self.assertFalse(
                result["is_dirty"],
                "Response 'is_dirty' must be False after successful sync (BUG #3)"
            )


# ===========================================================================
# 2. write() – auto_sync disabled
# ===========================================================================

class TestWriteAutoSyncDisabled(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate(self.tmp.name, {"app.py": "x = 1\n"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=False)
        self.local_path = os.path.join(self.tmp.name, "app.py")

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_write_does_not_update_disk(self):
        """With auto_sync=False, the local file must stay unchanged."""
        self.vfs.write("/proj/app.py", "x = 99\n")
        self.assertEqual(_read_local(self.local_path), "x = 1\n")

    def test_write_marks_file_dirty(self):
        """With auto_sync=False, the file must be marked dirty."""
        self.vfs.write("/proj/app.py", "x = 77\n")
        f = self.vfs.files["/proj/app.py"]
        self.assertTrue(f.is_dirty)

    def test_sync_all_flushes_dirty_files(self):
        """sync_all() must persist all dirty files to disk."""
        self.vfs.write("/proj/app.py", "synced content\n")
        result = self.vfs.sync_all()
        self.assertTrue(result["success"], result)
        self.assertIn("/proj/app.py", result["synced"])
        self.assertEqual(_read_local(self.local_path), "synced content\n")

    def test_sync_all_clears_dirty_flag(self):
        self.vfs.write("/proj/app.py", "synced\n")
        self.vfs.sync_all()
        f = self.vfs.files["/proj/app.py"]
        self.assertFalse(f.is_dirty)


# ===========================================================================
# 3. append() – auto_sync
# ===========================================================================

class TestAppendAutoSync(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate(self.tmp.name, {"log.txt": "line1\n"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)
        self.local_path = os.path.join(self.tmp.name, "log.txt")
        # Pre-load so the file is not in SHADOW state before append
        self.vfs.open("/proj/log.txt")

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_append_updates_disk(self):
        """append() with auto_sync must persist the new content."""
        self.vfs.append("/proj/log.txt", "line2\n")
        content = _read_local(self.local_path)
        self.assertIn("line1", content)
        self.assertIn("line2", content)

    def test_append_clears_is_dirty(self):
        """After append + auto_sync, f.is_dirty must be False."""
        self.vfs.append("/proj/log.txt", "appended\n")
        f = self.vfs.files["/proj/log.txt"]
        self.assertFalse(f.is_dirty,
                         "is_dirty must be cleared after append auto_sync")

    def test_append_sync_error_is_reported(self):
        """
        If _sync_to_local fails, append() must communicate the failure.

        BUG #4 – append() calls `self._sync_to_local(path)` but ignores the
        return value entirely, so callers can never detect a sync error.
        """
        with patch.object(
            self.vfs, "_sync_to_local",
            return_value={"success": False, "error": "disk full"}
        ):
            result = self.vfs.append("/proj/log.txt", "boom\n")
            # Either the overall result must indicate failure …
            # … or a sync_error key must be present.
            has_failure_signal = (
                not result.get("success", True)
                or "sync_error" in result
                or "error" in result
            )
            self.assertTrue(
                has_failure_signal,
                "append() must surface sync errors to caller (BUG #4). "
                f"Got: {result}"
            )

    def test_append_empty_string(self):
        """Appending '' must succeed."""
        result = self.vfs.append("/proj/log.txt", "")
        self.assertTrue(result["success"], result)


# ===========================================================================
# 4. edit() – auto_sync
# ===========================================================================

class TestEditAutoSync(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        content = "\n".join(f"line{i}" for i in range(1, 11)) + "\n"
        _populate(self.tmp.name, {"source.py": content})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)
        self.local_path = os.path.join(self.tmp.name, "source.py")
        self.vfs.open("/proj/source.py")

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_edit_updates_disk(self):
        """edit() with auto_sync must persist the change."""
        self.vfs.edit("/proj/source.py", 3, 3, "REPLACED_LINE")
        disk = _read_local(self.local_path)
        self.assertIn("REPLACED_LINE", disk)

    def test_edit_clears_is_dirty(self):
        """After edit + auto_sync, f.is_dirty must be False."""
        self.vfs.edit("/proj/source.py", 1, 1, "FIRST_REPLACED")
        f = self.vfs.files["/proj/source.py"]
        self.assertFalse(f.is_dirty)

    def test_edit_sync_error_is_reported(self):
        """
        If _sync_to_local fails, edit() must communicate the failure.

        BUG #4 – same as append: return value of _sync_to_local is discarded.
        """
        with patch.object(
            self.vfs, "_sync_to_local",
            return_value={"success": False, "error": "permission denied"}
        ):
            result = self.vfs.edit("/proj/source.py", 2, 2, "oops")
            has_failure_signal = (
                not result.get("success", True)
                or "sync_error" in result
                or "error" in result
            )
            self.assertTrue(
                has_failure_signal,
                "edit() must surface sync errors to caller (BUG #4). "
                f"Got: {result}"
            )


# ===========================================================================
# 5. sync_all()
# ===========================================================================

class TestSyncAll(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate(self.tmp.name, {
            "a.py": "a = 1",
            "b.py": "b = 2",
            "c.py": "c = 3",
        })
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=False)

    def tearDown(self):
        self.tmp.cleanup()

    def _dirty_write(self, vfs_path: str, content: str):
        """Directly mark a file dirty without going through write()."""
        f = self.vfs.files[vfs_path]
        f._content = content
        f.is_dirty = True
        f.backing_type = FileBackingType.MODIFIED

    # -----------------------------------------------------------------------
    def test_sync_all_persists_all_dirty(self):
        self._dirty_write("/proj/a.py", "a = 100")
        self._dirty_write("/proj/b.py", "b = 200")
        result = self.vfs.sync_all()
        self.assertTrue(result["success"], result)
        self.assertIn("/proj/a.py", result["synced"])
        self.assertIn("/proj/b.py", result["synced"])
        self.assertEqual(_read_local(os.path.join(self.tmp.name, "a.py")), "a = 100")
        self.assertEqual(_read_local(os.path.join(self.tmp.name, "b.py")), "b = 200")

    def test_sync_all_skips_clean_files(self):
        self._dirty_write("/proj/a.py", "a = 10")
        result = self.vfs.sync_all()
        self.assertNotIn("/proj/b.py", result["synced"])
        self.assertNotIn("/proj/c.py", result["synced"])

    def test_sync_all_clears_dirty_flags(self):
        self._dirty_write("/proj/a.py", "a = 77")
        self.vfs.sync_all()
        self.assertFalse(self.vfs.files["/proj/a.py"].is_dirty)

    def test_sync_all_reports_errors(self):
        self._dirty_write("/proj/a.py", "x")
        with patch.object(
            self.vfs, "_sync_to_local",
            return_value={"success": False, "error": "no space"}
        ):
            result = self.vfs.sync_all()
        self.assertFalse(result["success"])
        self.assertTrue(len(result["errors"]) > 0)

    def test_sync_all_empty_string_content(self):
        """
        sync_all() must handle files with empty-string content.

        BUG #1 – _sync_to_local: `if not f._content` blocks empty-string sync.
        """
        self._dirty_write("/proj/c.py", "")   # empty content
        result = self.vfs.sync_all()
        self.assertIn(
            "/proj/c.py", result["synced"],
            "sync_all must sync files with empty-string content (BUG #1)"
        )
        self.assertEqual(
            _read_local(os.path.join(self.tmp.name, "c.py")), ""
        )


# ===========================================================================
# 6. _sync_to_local() directly
# ===========================================================================

class TestSyncToLocal(unittest.TestCase):
    """Unit-level tests for the internal _sync_to_local method."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate(self.tmp.name, {"file.txt": "original"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=False)
        self.local = os.path.join(self.tmp.name, "file.txt")
        self.vfs_path = "/proj/file.txt"

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_sync_to_local_basic(self):
        f = self.vfs.files[self.vfs_path]
        f._content = "updated"
        result = self.vfs._sync_to_local(self.vfs_path)
        self.assertTrue(result["success"], result)
        self.assertEqual(_read_local(self.local), "updated")

    def test_sync_to_local_clears_dirty(self):
        f = self.vfs.files[self.vfs_path]
        f._content = "data"
        f.is_dirty = True
        self.vfs._sync_to_local(self.vfs_path)
        self.assertFalse(f.is_dirty)

    def test_sync_to_local_empty_string(self):
        """
        _sync_to_local must not treat '' as missing content.

        BUG #1 – `if not f._content` is True for '', causing a false error.
        """
        f = self.vfs.files[self.vfs_path]
        f._content = ""      # empty, but valid
        result = self.vfs._sync_to_local(self.vfs_path)
        self.assertTrue(
            result["success"],
            f"_sync_to_local must accept empty-string content (BUG #1): {result}"
        )
        self.assertEqual(_read_local(self.local), "")

    def test_sync_to_local_none_content_fails(self):
        """_sync_to_local must fail gracefully if content is genuinely None."""
        f = self.vfs.files[self.vfs_path]
        f._content = None   # truly missing
        result = self.vfs._sync_to_local(self.vfs_path)
        self.assertFalse(result["success"])

    def test_sync_to_local_updates_mtime(self):
        f = self.vfs.files[self.vfs_path]
        f._content = "new"
        self.vfs._sync_to_local(self.vfs_path)
        # mtime must equal what's on disk
        disk_mtime = os.path.getmtime(self.local)
        self.assertAlmostEqual(f.local_mtime, disk_mtime, places=3)

    def test_sync_to_local_no_backing_fails(self):
        """Files without local_path are not syncable."""
        self.vfs.create("/plain.txt", "pure memory")
        result = self.vfs._sync_to_local("/plain.txt")
        self.assertFalse(result["success"])


# ===========================================================================
# 7. Mixed: write to newly created mount file
# ===========================================================================

class TestNewMountFileSync(unittest.TestCase):
    """
    Agent workflow: mount a directory, create a new file, write to it,
    then verify the content ends up on disk correctly.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)

    def tearDown(self):
        self.tmp.cleanup()

    # -----------------------------------------------------------------------
    def test_create_then_write_is_stable(self):
        self.vfs.create("/proj/new.py", "v = 1\n")
        self.vfs.write("/proj/new.py", "v = 2\n")

        local = os.path.join(self.tmp.name, "new.py")
        self.assertEqual(_read_local(local), "v = 2\n")

    def test_multiple_writes_remain_consistent(self):
        self.vfs.create("/proj/counter.py", "n = 0\n")
        for i in range(1, 6):
            self.vfs.write("/proj/counter.py", f"n = {i}\n")

        local = os.path.join(self.tmp.name, "counter.py")
        self.assertEqual(_read_local(local), "n = 5\n")
        self.assertFalse(self.vfs.files["/proj/counter.py"].is_dirty)

    def test_create_append_sync(self):
        self.vfs.create("/proj/log.txt", "")
        self.vfs.open("/proj/log.txt")   # load content
        self.vfs.append("/proj/log.txt", "entry1\n")
        self.vfs.append("/proj/log.txt", "entry2\n")

        local = os.path.join(self.tmp.name, "log.txt")
        content = _read_local(local)
        self.assertIn("entry1", content)
        self.assertIn("entry2", content)

    def test_is_dirty_false_after_all_writes_synced(self):
        self.vfs.create("/proj/final.py", "")
        self.vfs.write("/proj/final.py", "done\n")
        f = self.vfs.files["/proj/final.py"]
        self.assertFalse(f.is_dirty,
                         "File must not be dirty after auto_sync write")


if __name__ == "__main__":
    unittest.main()
