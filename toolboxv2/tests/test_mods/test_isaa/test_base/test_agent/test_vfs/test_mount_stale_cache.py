"""
test_mount_stale_cache.py
=========================
Validates that VFS mount operations return consistent, fresh content
when the underlying disk files change externally.

Bug context:
  - grep_vfs reads directly from disk (bypassing vfs.read())
  - vfs.read() uses cached _content if mtime unchanged
  - _sync_from_local skips files with matching mtime even if _content=None
  → grep and cat/wc see different versions of the same file

These tests use real tempdir mounts to reproduce the exact failure mode.

Run with:
    python -m unittest test_mount_stale_cache -v
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
import textwrap
import unittest
from unittest.mock import MagicMock

from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
    VirtualFileSystemV2,
    VFSFile,
    FileBackingType,
)
from toolboxv2.mods.isaa.base.patch.power_vfs import grep_vfs, find_files
from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import make_vfs_shell


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vfs() -> VirtualFileSystemV2:
    return VirtualFileSystemV2(session_id="test-stale", agent_name="test-stale")


def _write_disk(path: str, content: str, bump_mtime: bool = True):
    """Write content to disk file, optionally bump mtime to ensure detection."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    if bump_mtime:
        # Ensure mtime is at least 1s newer (filesystem granularity)
        stat = os.stat(path)
        os.utime(path, (stat.st_atime + 2, stat.st_mtime + 2))


def _read_disk(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ===========================================================================
# A. grep vs cat/wc consistency after external disk change
# ===========================================================================

class TestGrepCatConsistencyAfterDiskChange(unittest.TestCase):
    """H1: After an external disk change, grep and cat/wc must return
    content from the same version of the file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="vfs_stale_")
        self.vfs = _make_vfs()

        # Initial file on disk — version 1
        self.disk_file = os.path.join(self.tmpdir, "module.py")
        self.v1_content = "\n".join([f"# line {i}" for i in range(1, 51)])
        _write_disk(self.disk_file, self.v1_content, bump_mtime=False)

        # Mount
        result = self.vfs.mount(
            local_path=self.tmpdir,
            vfs_path="/proj",
            auto_sync=True,
        )
        self.assertTrue(result["success"], result)

        # Session for shell commands
        self.session = MagicMock()
        self.session.vfs = self.vfs
        self.sh = make_vfs_shell(self.session)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_h1_grep_and_cat_see_same_version_after_disk_change(self):
        """After disk file grows, both grep and cat must see the new content."""
        # Version 2: add a unique marker at line 80
        v2_lines = [f"# line {i}" for i in range(1, 51)]
        v2_lines.extend([f"def func_{i}(): pass" for i in range(51, 81)])
        v2_lines.append("class UNIQUE_MARKER_V2:")
        v2_lines.extend([f"# tail {i}" for i in range(82, 101)])
        v2_content = "\n".join(v2_lines)
        _write_disk(self.disk_file, v2_content)

        # Refresh mount to pick up changes
        self.vfs.refresh_mount("/proj")

        # grep must find the marker
        grep_result = self.sh("", "grep -n UNIQUE_MARKER_V2 /proj/module.py")
        self.assertTrue(grep_result["success"],
                        f"grep must find UNIQUE_MARKER_V2: {grep_result}")
        self.assertIn("UNIQUE_MARKER_V2", grep_result["stdout"])

        # cat/wc must also see the new version
        wc_result = self.sh("", "wc -l /proj/module.py")
        self.assertTrue(wc_result["success"])
        line_count = int(wc_result["stdout"].strip().split()[0])
        self.assertEqual(line_count, len(v2_lines),
                         f"wc must report {len(v2_lines)} lines, got {line_count}")

        # cat must contain the marker too
        cat_result = self.sh("", "cat /proj/module.py")
        self.assertIn("UNIQUE_MARKER_V2", cat_result["stdout"],
                       "cat must see UNIQUE_MARKER_V2")

    def test_h2_grep_and_wc_agree_on_line_count(self):
        """grep -c and wc -l must return the same number."""
        v2_lines = [f"def func_{i}(): pass" for i in range(200)]
        _write_disk(self.disk_file, "\n".join(v2_lines))
        self.vfs.refresh_mount("/proj")

        # wc -l
        wc_result = self.sh("", "wc -l /proj/module.py")
        wc_count = int(wc_result["stdout"].strip().split()[0])

        # grep -c (count matches for 'def')
        grep_result = self.sh("", "grep -n def /proj/module.py")
        grep_lines = [l for l in grep_result["stdout"].splitlines() if l.strip()]

        # wc must see 200 lines
        self.assertEqual(wc_count, 200, f"wc says {wc_count}, expected 200")
        # grep must find 200 matches
        self.assertEqual(len(grep_lines), 200,
                         f"grep found {len(grep_lines)} matches, expected 200")


# ===========================================================================
# B. _sync_from_local skips content-never-loaded files
# ===========================================================================

class TestSyncFromLocalNeverLoaded(unittest.TestCase):
    """H3: _sync_from_local must not skip files where _content is None,
    even if mtime is unchanged."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="vfs_sync_")
        self.vfs = _make_vfs()

        self.disk_file = os.path.join(self.tmpdir, "data.py")
        _write_disk(self.disk_file, "original = True\n", bump_mtime=False)

        self.vfs.mount(local_path=self.tmpdir, vfs_path="/proj")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_h3_sync_loads_content_when_none(self):
        """After mount, shadow files have _content=None. sync_from_local
        must load them even if mtime matches."""
        f = self.vfs.files.get("/proj/data.py")
        self.assertIsNotNone(f, "File must exist in VFS after mount")
        self.assertIsNone(f._content, "Shadow file must not be loaded yet")

        # sync_from_local — mtime matches but content is None
        result = self.vfs._sync_from_local("/proj/data.py")
        self.assertTrue(result["success"])

        # Content must now be loaded (not skipped)
        self.assertNotIn("skipped", result,
                         "Must not skip when _content is None")
        self.assertIsNotNone(f._content, "_content must be loaded")
        self.assertIn("original = True", f._content)

    def test_h4_refresh_mount_loads_never_read_files(self):
        """refresh_mount must populate _content for files never read via cat."""
        f = self.vfs.files.get("/proj/data.py")
        self.assertIsNone(f._content)

        self.vfs.refresh_mount("/proj")

        # After refresh, if we read, we must get disk content
        read_result = self.vfs.read("/proj/data.py")
        self.assertTrue(read_result["success"])
        self.assertIn("original = True", read_result["content"])


# ===========================================================================
# C. read() detects disk changes via mtime for mounted files
# ===========================================================================

class TestReadDetectsDiskChanges(unittest.TestCase):
    """H5: vfs.read() must detect when disk file has newer mtime and reload."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="vfs_read_")
        self.vfs = _make_vfs()

        self.disk_file = os.path.join(self.tmpdir, "config.py")
        _write_disk(self.disk_file, "VERSION = 1\n", bump_mtime=False)

        self.vfs.mount(local_path=self.tmpdir, vfs_path="/proj")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_h5_read_reloads_after_disk_change(self):
        """First read loads v1, disk changes to v2, second read must see v2."""
        # First read — loads content
        r1 = self.vfs.read("/proj/config.py")
        self.assertTrue(r1["success"])
        self.assertIn("VERSION = 1", r1["content"])

        # External change with bumped mtime
        _write_disk(self.disk_file, "VERSION = 2\nNEW_SETTING = True\n")

        # Second read — must detect mtime change and reload
        r2 = self.vfs.read("/proj/config.py")
        self.assertTrue(r2["success"])
        self.assertIn("VERSION = 2", r2["content"],
                       "read() must detect disk change and reload")
        self.assertIn("NEW_SETTING", r2["content"])

    def test_h6_read_does_not_reload_dirty_files(self):
        """If agent has UNsynced edits (auto_sync=False), read() must NOT
        overwrite with disk content."""
        # Re-mount without auto_sync
        self.vfs.unmount("/proj", save_changes=False)
        self.vfs.mount(local_path=self.tmpdir, vfs_path="/proj", auto_sync=False)

        # Load content
        self.vfs.read("/proj/config.py")

        # Agent edits in VFS — NOT synced to disk (auto_sync=False)
        self.vfs.edit("/proj/config.py", 1, 1, "VERSION = 99  # agent edit")

        f = self.vfs.files["/proj/config.py"]
        self.assertTrue(f.is_dirty, "Edit must leave file dirty when auto_sync=False")

        # External change on disk
        _write_disk(self.disk_file, "VERSION = 2\n")

        # Read must return agent's unsaved version, not disk
        r = self.vfs.read("/proj/config.py")
        self.assertTrue(r["success"])
        self.assertIn("VERSION = 99", r["content"],
                      "read() must preserve dirty (unsynced) agent edits")


# ===========================================================================
# D. Full shell roundtrip: write on disk → refresh → grep/cat/wc consistent
# ===========================================================================

class TestShellConsistencyAfterRefresh(unittest.TestCase):
    """H7: After refresh_mount, all shell commands must see the same
    file version — no divergence between grep and cat/wc/sed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="vfs_shell_")
        self.vfs = _make_vfs()

        # Create a realistic project structure
        os.makedirs(os.path.join(self.tmpdir, "src"), exist_ok=True)
        _write_disk(
            os.path.join(self.tmpdir, "src", "main.py"),
            textwrap.dedent("""\
                import os

                def main():
                    print("hello")

                if __name__ == "__main__":
                    main()
            """),
            bump_mtime=False,
        )
        _write_disk(
            os.path.join(self.tmpdir, "src", "utils.py"),
            "def helper(): pass\n",
            bump_mtime=False,
        )

        self.vfs.mount(local_path=self.tmpdir, vfs_path="/proj")

        self.session = MagicMock()
        self.session.vfs = self.vfs
        self.sh = make_vfs_shell(self.session)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_h7_all_commands_consistent_after_external_rewrite(self):
        """Externally rewrite main.py from 8 lines to 50 lines,
        refresh, then verify grep/cat/wc/sed all agree."""
        new_lines = ["import os", "import sys", ""]
        for i in range(3, 48):
            new_lines.append(f"def func_{i}(): return {i}")
        new_lines.append("")
        new_lines.append("class FinalClass:")
        new_lines.append("    pass")
        new_content = "\n".join(new_lines)
        expected_line_count = len(new_lines)

        _write_disk(os.path.join(self.tmpdir, "src", "main.py"), new_content)
        self.vfs.refresh_mount("/proj")

        # wc -l
        wc_r = self.sh("", "wc -l /proj/src/main.py")
        wc_count = int(wc_r["stdout"].strip().split()[0])
        self.assertEqual(wc_count, expected_line_count,
                         f"wc: {wc_count} != {expected_line_count}")

        # grep must find FinalClass
        grep_r = self.sh("", "grep -n FinalClass /proj/src/main.py")
        self.assertTrue(grep_r["success"], f"grep must find FinalClass: {grep_r}")
        self.assertIn("FinalClass", grep_r["stdout"])

        # cat must contain FinalClass
        cat_r = self.sh("", "cat /proj/src/main.py")
        self.assertIn("FinalClass", cat_r["stdout"],
                       "cat must see FinalClass")

        # sed must return correct line range
        sed_r = self.sh("", f"sed -n '{expected_line_count - 1},{expected_line_count}p' /proj/src/main.py")
        self.assertTrue(sed_r["success"])
        self.assertIn("FinalClass", sed_r["stdout"],
                       "sed on last lines must see FinalClass")

    def test_h8_new_file_on_disk_visible_after_refresh(self):
        """A file created externally must appear in grep/cat after refresh."""
        _write_disk(
            os.path.join(self.tmpdir, "src", "new_module.py"),
            "class BrandNewClass:\n    secret = 42\n",
        )

        self.vfs.refresh_mount("/proj")

        # grep must find it
        grep_r = self.sh("", "grep -rn BrandNewClass /proj/src")
        self.assertTrue(grep_r["success"],
                        f"grep must find BrandNewClass: {grep_r}")

        # cat must work
        cat_r = self.sh("", "cat /proj/src/new_module.py")
        self.assertTrue(cat_r["success"])
        self.assertIn("secret = 42", cat_r["stdout"])

    def test_h9_deleted_file_on_disk_gone_after_refresh(self):
        """A file deleted externally must disappear after refresh."""
        # Ensure file exists in VFS
        cat_r = self.sh("", "cat /proj/src/utils.py")
        self.assertTrue(cat_r["success"])

        # Delete on disk
        os.remove(os.path.join(self.tmpdir, "src", "utils.py"))
        self.vfs.refresh_mount("/proj")

        # cat must fail
        cat_r = self.sh("", "cat /proj/src/utils.py")
        self.assertFalse(cat_r["success"],
                         "cat on deleted file must fail after refresh")

        # grep must not find it
        grep_r = self.sh("", "grep -rn helper /proj/src")
        if grep_r["success"]:
            self.assertNotIn("utils.py", grep_r["stdout"],
                             "grep must not find content from deleted file")


# ===========================================================================
# E. grep_vfs direct — layer isolation test
# ===========================================================================

class TestGrepVfsLayerIsolation(unittest.TestCase):
    """H10: grep_vfs must return content consistent with vfs.read(),
    not its own disk-read bypass."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="vfs_grep_")
        self.vfs = _make_vfs()

        self.disk_file = os.path.join(self.tmpdir, "target.py")
        _write_disk(self.disk_file, "MARKER_V1 = True\n", bump_mtime=False)

        self.vfs.mount(local_path=self.tmpdir, vfs_path="/proj")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_h10_grep_vfs_matches_vfs_read(self):
        """After disk change + refresh, grep_vfs and vfs.read() must agree."""
        _write_disk(self.disk_file, "MARKER_V2 = True\nMARKER_V1 = False\n")
        self.vfs.refresh_mount("/proj")

        # vfs.read content
        read_r = self.vfs.read("/proj/target.py")
        read_content = read_r["content"]

        # grep_vfs content
        grep_results = grep_vfs(
            vfs=self.vfs,
            pattern="MARKER",
            file_pattern="target.py",
            path="/proj",
        )

        grep_found_v2 = any("MARKER_V2" in m["match"] for m in grep_results)
        read_has_v2 = "MARKER_V2" in read_content

        self.assertEqual(grep_found_v2, read_has_v2,
                         f"grep sees V2={grep_found_v2}, read sees V2={read_has_v2} — divergence!")

    def test_h11_grep_does_not_see_old_content_after_shrink(self):
        """File shrinks from 100 lines to 10 — grep must not find old content."""
        # Write 100-line version
        v1 = "\n".join([f"OLD_LINE_{i} = {i}" for i in range(100)])
        _write_disk(self.disk_file, v1, bump_mtime=False)
        self.vfs.refresh_mount("/proj")

        # Force a read to populate cache
        self.vfs.read("/proj/target.py")

        # Shrink to 10 lines — no OLD_LINE_50
        v2 = "\n".join([f"NEW_LINE_{i} = {i}" for i in range(10)])
        _write_disk(self.disk_file, v2)
        self.vfs.refresh_mount("/proj")

        # grep must NOT find OLD_LINE_50
        grep_results = grep_vfs(
            vfs=self.vfs,
            pattern="OLD_LINE_50",
            file_pattern="target.py",
            path="/proj",
        )
        self.assertEqual(len(grep_results), 0,
                         "grep must not find OLD_LINE_50 after file shrunk")

        # read must also not have it
        read_r = self.vfs.read("/proj/target.py")
        self.assertNotIn("OLD_LINE_50", read_r["content"],
                         "read must not have OLD_LINE_50 after shrink")

        # wc must report 10
        self.assertEqual(len(read_r["content"].splitlines()), 10)


# ===========================================================================
# F. Same-mtime edge case (the sneakiest bug)
# ===========================================================================

class TestSameMtimeContentChange(unittest.TestCase):
    """H12: If disk content changes but mtime stays the same (sub-second
    writes, git operations), the VFS must still eventually serve fresh content."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="vfs_mtime_")
        self.vfs = _make_vfs()

        self.disk_file = os.path.join(self.tmpdir, "tricky.py")
        _write_disk(self.disk_file, "STATE = 'alpha'\n", bump_mtime=False)

        self.vfs.mount(local_path=self.tmpdir, vfs_path="/proj")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_h12_same_mtime_different_content_detected_on_refresh(self):
        """Write different content with same mtime — refresh_mount must
        still pick it up (this tests the _content=None fix)."""
        # Read to populate cache
        r1 = self.vfs.read("/proj/tricky.py")
        self.assertIn("alpha", r1["content"])
        f = self.vfs.files["/proj/tricky.py"]
        saved_mtime = f.local_mtime

        # Write different content, preserve exact mtime
        with open(self.disk_file, "w") as fh:
            fh.write("STATE = 'beta'\n")
        os.utime(self.disk_file, (saved_mtime, saved_mtime))

        # Verify mtime is truly identical
        self.assertEqual(os.path.getmtime(self.disk_file), saved_mtime)

        # Invalidate _content to simulate the scenario where VFS
        # knows the file exists but hasn't loaded it yet
        f._content = None
        f.backing_type = FileBackingType.SHADOW

        # refresh_mount — _sync_from_local must NOT skip this
        self.vfs.refresh_mount("/proj")

        # read must return 'beta'
        r2 = self.vfs.read("/proj/tricky.py")
        self.assertTrue(r2["success"])
        self.assertIn("beta", r2["content"],
                       "Must load fresh content even when mtime unchanged "
                       "if _content was None")


if __name__ == "__main__":
    unittest.main()
