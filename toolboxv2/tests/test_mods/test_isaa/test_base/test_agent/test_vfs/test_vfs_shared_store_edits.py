"""
test_vfs_shared_store_edits.py
==============================
Tests for the interaction between edit()/append()/read() and the
GlobalVFSManager Shared-Store (Ebene 3).

Covers:
  - /global/ paths (always shared)
  - Registered shared mounts via register_shared_mount() (Ebene 3b)
  - Multi-agent visibility after edit/append
  - The core bug: edit() bypasses shared_write → stale read()

Run with:
    python -m unittest test_vfs_shared_store_edits -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
    FileBackingType,
    VirtualFileSystemV2,
)
from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs
from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import make_vfs_shell, make_vfs_view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vfs(name="agent1", session="sess1"):
    vfs = VirtualFileSystemV2(session_id=session, agent_name=name)
    return vfs


def _make_shell(vfs):
    session = MagicMock()
    session.vfs = vfs
    return make_vfs_shell(session)


def _make_view(vfs):
    session = MagicMock()
    session.vfs = vfs
    return make_vfs_view(session)


def _safe_read(tc, vfs, path):
    """Read + assert success, returns content string.
    Converts KeyError into clear assertion failure."""
    r = vfs.read(path)
    tc.assertTrue(r.get("success"),
                  f"read({path}) failed: {r.get('error', r)}")
    return r["content"]


# ===========================================================================
# A. edit() on /global/ — the primary bug
# ===========================================================================

class TestEditGlobalSharedStore(unittest.TestCase):
    """edit() on /global/ files must update the Shared-Store so that
    subsequent read() returns the edited content, not stale store data."""

    def setUp(self):
        self.gvfs = get_global_vfs()
        self.vfs1 = _make_vfs("agent1", "s1")
        self.vfs2 = _make_vfs("agent2", "s2")

        # Mount /global/ on both VFS instances
        self.gvfs.register_vfs(self.vfs1)
        self.gvfs.register_vfs(self.vfs2)

        self.vfs1.mount(str(self.gvfs.data_dir), vfs_path="/global", auto_sync=True)
        self.sh1 = _make_shell(self.vfs1)

        # Create file via vfs.write (goes through shared_write correctly)
        self.vfs1.write("/global/test_edit.txt", "line1\nline2\nline3\nline4\nline5")
        self.vfs1.refresh_mount("/global")

        self.vfs2.mount(str(self.gvfs.data_dir), vfs_path="/global", auto_sync=True)
        self.vfs2.refresh_mount("/global")

    def tearDown(self):
        self.gvfs.unregister_vfs(self.vfs1)
        self.gvfs.unregister_vfs(self.vfs2)
        # Clean up the test file
        test_file = os.path.join(str(self.gvfs.data_dir), "test_edit.txt")
        if os.path.exists(test_file):
            os.remove(test_file)

    def test_edit_visible_to_same_agent_via_read(self):
        """After edit(), the same agent's read() must return edited content."""
        r = self.vfs1.edit("/global/test_edit.txt", 2, 2, "EDITED_LINE2")
        self.assertTrue(r["success"], r)

        content = _safe_read(self, self.vfs1, "/global/test_edit.txt")
        self.assertIn("EDITED_LINE2", content,
                       "BUG: edit() updated _content but read() returned stale Shared-Store data")
        self.assertNotIn("line2", content.split("\n")[1],
                         "line2 must be replaced by EDITED_LINE2")

    def test_edit_visible_to_other_agent(self):
        """After agent1 edits, agent2 must see the edited content."""
        self.vfs1.edit("/global/test_edit.txt", 3, 3, "AGENT1_EDIT")

        content2 = _safe_read(self, self.vfs2, "/global/test_edit.txt")
        self.assertIn("AGENT1_EDIT", content2,
                       "BUG: edit() did not update Shared-Store → agent2 sees stale content")

    def test_edit_preserves_surrounding_lines(self):
        """Edit line 3 must not affect lines 1,2,4,5."""
        self.vfs1.edit("/global/test_edit.txt", 3, 3, "REPLACED")
        content = _safe_read(self, self.vfs1, "/global/test_edit.txt")
        lines = content.splitlines()
        self.assertEqual(lines[0], "line1")
        self.assertEqual(lines[1], "line2")
        self.assertEqual(lines[2], "REPLACED")
        self.assertEqual(lines[3], "line4")

    def test_sequential_edits_both_apply(self):
        """Two consecutive edits on different lines must both persist."""
        self.vfs1.edit("/global/test_edit.txt", 1, 1, "FIRST_EDIT")
        self.vfs1.edit("/global/test_edit.txt", 4, 4, "SECOND_EDIT")

        content = _safe_read(self, self.vfs1, "/global/test_edit.txt")
        self.assertIn("FIRST_EDIT", content)
        self.assertIn("SECOND_EDIT", content)
        # Verify via second agent too
        content2 = _safe_read(self, self.vfs2, "/global/test_edit.txt")
        self.assertIn("FIRST_EDIT", content2)
        self.assertIn("SECOND_EDIT", content2)

    def test_edit_multiline_replacement(self):
        """Replace lines 2-3 with a multi-line block."""
        self.vfs1.edit("/global/test_edit.txt", 2, 3, "new_a\nnew_b\nnew_c")
        content = _safe_read(self, self.vfs1, "/global/test_edit.txt")
        lines = content.splitlines()
        self.assertEqual(lines[0], "line1")
        self.assertEqual(lines[1], "new_a")
        self.assertEqual(lines[2], "new_b")
        self.assertEqual(lines[3], "new_c")
        self.assertEqual(lines[4], "line4")

    def test_edit_then_cat_via_shell(self):
        """Full agent workflow: edit via shell, then cat to verify."""
        sh = self.sh1
        r = sh("", 'edit /global/test_edit.txt 2 2 "SHELL_EDIT"')
        self.assertTrue(r["success"], r)

        r = sh("", "cat /global/test_edit.txt")
        self.assertTrue(r["success"])
        self.assertIn("SHELL_EDIT", r["stdout"],
                       "BUG: cat after edit shows stale content from Shared-Store")

    def test_edit_then_grep_finds_new_content(self):
        """grep must find content that was just edited in."""
        self.sh1("", 'edit /global/test_edit.txt 2 2 "UNIQUE_MARKER_XYZ"')
        r = self.sh1("", "grep -n UNIQUE_MARKER_XYZ /global/test_edit.txt")
        self.assertTrue(r["success"],
                        "BUG: grep cannot find edited content — Shared-Store is stale")
        self.assertIn("UNIQUE_MARKER_XYZ", r["stdout"])

    def test_edit_then_wc_reflects_line_change(self):
        """After replacing 1 line with 3 lines, wc -l must increase by 2."""
        before = _safe_read(self, self.vfs1, "/global/test_edit.txt")
        before_lines = len(before.splitlines())

        self.vfs1.edit("/global/test_edit.txt", 2, 2, "a\nb\nc")
        content = _safe_read(self, self.vfs1, "/global/test_edit.txt")
        after_lines = len(content.splitlines())
        self.assertEqual(after_lines, before_lines + 2,
                         "Edit expanding 1→3 lines must increase line count by 2")


# ===========================================================================
# B. edit() on registered shared mount (Ebene 3b)
# ===========================================================================

class TestEditRegisteredSharedMount(unittest.TestCase):
    """Same bugs apply to mounts registered via register_shared_mount()."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gvfs = get_global_vfs()
        self.mount_key = self.gvfs.register_shared_mount(self.tmp, hydrate=False)

        # Agent 1
        self.vfs1 = _make_vfs("agent1", "s1")
        self.vfs1.mount(self.tmp, vfs_path="/project", auto_sync=True)
        self.gvfs.register_vfs(self.vfs1)
        self.sh1 = _make_shell(self.vfs1)

        # Write initial file via vfs.write (ensures shared-store + disk + files dict)
        self.vfs1.write("/project/app.py", "import os\nimport sys\n\ndef main():\n    pass")
        self.vfs1.refresh_mount("/project")

        # Agent 2
        self.vfs2 = _make_vfs("agent2", "s2")
        self.vfs2.mount(self.tmp, vfs_path="/project", auto_sync=True)
        self.gvfs.register_vfs(self.vfs2)
        self.vfs2.refresh_mount("/project")

    def tearDown(self):
        self.gvfs.unregister_vfs(self.vfs1)
        self.gvfs.unregister_vfs(self.vfs2)
        self.gvfs.unregister_shared_mount(self.tmp)
        shutil.rmtree(self.tmp)

    def test_edit_visible_to_same_agent(self):
        """edit() on shared mount → same agent's read() sees it."""
        self.vfs1.edit("/project/app.py", 4, 5, "def main():\n    print('hello')")
        content = _safe_read(self, self.vfs1, "/project/app.py")
        self.assertIn("print('hello')", content,
                       "BUG: edit on shared mount not visible to same agent")

    def test_edit_visible_to_other_agent(self):
        """edit() on shared mount → other agent's read() sees it."""
        self.vfs1.edit("/project/app.py", 1, 1, "import os  # edited")
        content2 = _safe_read(self, self.vfs2, "/project/app.py")
        self.assertIn("# edited", content2,
                       "BUG: edit on shared mount not visible to agent2 via Shared-Store")

    def test_edit_then_append_preserves_edit(self):
        """edit() then append() — both changes must be visible.
        This is the cascade bug: if edit doesn't update store,
        append reads stale store content and overwrites the edit."""
        self.vfs1.edit("/project/app.py", 1, 1, "import os  # EDITED")
        self.vfs1.append("/project/app.py", "\n# appended line")

        content = _safe_read(self, self.vfs1, "/project/app.py")
        self.assertIn("# EDITED", content,
                       "BUG: append after edit lost the edit (stale store read)")
        self.assertIn("# appended line", content)

    def test_sequential_edits_across_agents(self):
        """Agent1 edits line 1, Agent2 edits line 2 — both must persist."""
        self.vfs1.edit("/project/app.py", 1, 1, "# agent1 was here")

        # Agent2 must see agent1's edit before making its own
        content2 = _safe_read(self, self.vfs2, "/project/app.py")
        self.assertIn("# agent1 was here", content2,
                       "Prerequisite: agent2 must see agent1's edit")

        self.vfs2.edit("/project/app.py", 2, 2, "# agent2 was here")

        # Both edits must be visible
        final1 = _safe_read(self, self.vfs1, "/project/app.py")
        final2 = _safe_read(self, self.vfs2, "/project/app.py")
        for content in (final1, final2):
            self.assertIn("# agent1 was here", content)
            self.assertIn("# agent2 was here", content)

    def test_edit_disk_file_matches_vfs(self):
        """After edit on shared mount, disk file must match VFS content."""
        self.vfs1.edit("/project/app.py", 1, 1, "# disk must match")
        vfs_content = _safe_read(self, self.vfs1, "/project/app.py")
        with open(os.path.join(self.tmp, "app.py"), "r") as f:
            disk_content = f.read()
        self.assertEqual(vfs_content, disk_content,
                         "BUG: VFS content and disk content diverged after edit")


# ===========================================================================
# C. append() after edit() — the cascade bug
# ===========================================================================

class TestAppendAfterEditCascade(unittest.TestCase):
    """When edit() doesn't update the store, append() reads stale content
    from the store and concatenates to it — losing the edit entirely."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gvfs = get_global_vfs()
        self.mount_key = self.gvfs.register_shared_mount(self.tmp, hydrate=False)

        self.vfs = _make_vfs("agent1", "s1")
        self.vfs.mount(self.tmp, vfs_path="/proj", auto_sync=True)
        self.gvfs.register_vfs(self.vfs)
        self.sh = _make_shell(self.vfs)

        self.vfs.write("/proj/log.txt", "line1\nline2\nline3")
        self.vfs.refresh_mount("/proj")

    def tearDown(self):
        self.gvfs.unregister_vfs(self.vfs)
        self.gvfs.unregister_shared_mount(self.tmp)
        shutil.rmtree(self.tmp)

    def test_append_after_edit_keeps_both(self):
        """edit line 2, then append — edit must survive."""
        self.vfs.edit("/proj/log.txt", 2, 2, "EDITED")
        self.vfs.append("/proj/log.txt", "\nappended")
        content = _safe_read(self, self.vfs, "/proj/log.txt")
        self.assertIn("EDITED", content,
                       "BUG: append after edit overwrote the edit with stale store content")
        self.assertIn("appended", content)

    def test_multiple_edits_then_append(self):
        """3 edits then 1 append — all must persist."""
        self.vfs.edit("/proj/log.txt", 1, 1, "E1")
        self.vfs.edit("/proj/log.txt", 2, 2, "E2")
        self.vfs.edit("/proj/log.txt", 3, 3, "E3")
        self.vfs.append("/proj/log.txt", "\nFINAL")
        content = _safe_read(self, self.vfs, "/proj/log.txt")
        for marker in ("E1", "E2", "E3", "FINAL"):
            self.assertIn(marker, content,
                          f"BUG: {marker} lost after edit+append cascade")

    def test_echo_append_after_edit(self):
        """Shell echo >> after edit — edit must survive."""
        self.sh("", 'edit /proj/log.txt 2 2 "SHELL_EDIT"')
        self.sh("", 'echo "\nshell_append" >> /proj/log.txt')
        r = self.sh("", "cat /proj/log.txt")
        self.assertIn("SHELL_EDIT", r["stdout"],
                       "BUG: echo >> after edit lost the edit")
        self.assertIn("shell_append", r["stdout"])


# ===========================================================================
# D. Agent roundtrip: grep → edit → verify on shared mount
# ===========================================================================

class TestAgentRoundtripSharedMount(unittest.TestCase):
    """The real-world agent workflow that currently fails:
    find a line via grep, edit it, verify via cat/grep."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gvfs = get_global_vfs()
        self.mount_key = self.gvfs.register_shared_mount(self.tmp, hydrate=False)

        self.vfs = _make_vfs("coder", "s1")
        self.vfs.mount(self.tmp, vfs_path="/work", auto_sync=True)
        self.gvfs.register_vfs(self.vfs)
        self.sh = _make_shell(self.vfs)
        self.view = _make_view(self.vfs)

        # Create a realistic file via vfs.write (avoids shlex quote issues)
        html = "\n".join([
            "<!DOCTYPE html>",
            "<html>",
            "<head><title>Test</title></head>",
            "<body>",
            "<script>",
            "let score = 0;",
            "let speed = 1.0;",
            "const DEBUG = false;",
            "",
            "function resetGame() {",
            "    score = 0;",
            "    speed = 1.0;",
            "    lastPlatX = 200;",
            "    nextGap = 80;",
            "}",
            "",
            "function spawnPlatform() {",
            "    const py = H - 60 - Math.random() * 200;",
            "    const pw = 60 + Math.random() * 80;",
            "    lastPlatX = px + pw;",
            "}",
            "",
            "function update() {",
            "    score += 1;",
            "    if (score > 100) speed = 2.0;",
            "}",
            "</script>",
            "</body>",
            "</html>",
        ])
        self.vfs.write("/work/game.html", html)
        self.vfs.refresh_mount("/work")

    def tearDown(self):
        self.gvfs.unregister_vfs(self.vfs)
        self.gvfs.unregister_shared_mount(self.tmp)
        shutil.rmtree(self.tmp)

    def test_grep_edit_cat_roundtrip(self):
        """grep to find line, edit it, cat to verify."""
        # Step 1: Find
        r = self.sh("", "grep -n 'const DEBUG' /work/game.html")
        self.assertTrue(r["success"])
        import re
        m = re.search(r"(\d+):.*const DEBUG", r["stdout"])
        self.assertIsNotNone(m)
        line_no = int(m.group(1))

        # Step 2: Edit
        r = self.sh("", f'edit /work/game.html {line_no} {line_no} "const DEBUG = true;"')
        self.assertTrue(r["success"])

        # Step 3: Verify via cat
        r = self.sh("", "cat /work/game.html")
        self.assertIn("const DEBUG = true;", r["stdout"],
                       "BUG: cat after edit returns stale content")
        self.assertNotIn("const DEBUG = false;", r["stdout"])

    def test_grep_edit_grep_verify(self):
        """grep → edit → grep again — must find new content."""
        r = self.sh("", "grep -n 'speed = 1.0' /work/game.html")
        self.assertTrue(r["success"])
        import re
        m = re.search(r"(\d+):.*speed = 1.0", r["stdout"])
        line_no = int(m.group(1))

        self.sh("", f'edit /work/game.html {line_no} {line_no} "let speed = 2.5;"')

        r = self.sh("", "grep -n 'speed = 2.5' /work/game.html")
        self.assertTrue(r["success"],
                        "BUG: grep cannot find edited content — stale Shared-Store")

    def test_multiple_edits_file_integrity(self):
        """Multiple edits on different lines — file line count must be stable."""
        before_r = self.sh("", "wc -l /work/game.html")
        before_count = int(before_r["stdout"].strip().split()[0])

        # Edit 3 different single lines (1-for-1 replacement)
        self.sh("", 'edit /work/game.html 6 6 "let score = 100;"')
        self.sh("", 'edit /work/game.html 7 7 "let speed = 5.0;"')
        self.sh("", 'edit /work/game.html 8 8 "const DEBUG = true;"')

        after_r = self.sh("", "wc -l /work/game.html")
        after_count = int(after_r["stdout"].strip().split()[0])

        self.assertEqual(before_count, after_count,
                         f"BUG: line count changed from {before_count} to {after_count} "
                         f"after 1-for-1 edits — file was truncated/corrupted")

        content = _safe_read(self, self.vfs, "/work/game.html")
        self.assertIn("let score = 100;", content)
        self.assertIn("let speed = 5.0;", content)
        self.assertIn("const DEBUG = true;", content)

    def test_edit_expanding_lines_preserves_rest(self):
        """Replace 1 line with 3 — rest of file must be intact."""
        self.sh("", 'edit /work/game.html 13 13 "    lastPlatX = 200;\n    lastPlatY = 100;\n    lastPlatW = 60;"')

        content = _safe_read(self, self.vfs, "/work/game.html")
        self.assertIn("lastPlatY = 100;", content)
        # Lines after the edit must still be there
        self.assertIn("nextGap = 80;", content)
        self.assertIn("function spawnPlatform()", content)
        self.assertIn("</html>", content)

    def test_view_after_edit_shows_new_content(self):
        """vfs_view scroll_to after edit must show edited content."""
        self.sh("", 'edit /work/game.html 8 8 "const DEBUG = true; // CHANGED"')

        r = self.view("/work/game.html", scroll_to="DEBUG", context_lines=6)
        self.assertTrue(r["success"])
        self.assertIn("// CHANGED", r["content"],
                       "BUG: vfs_view after edit shows stale content")


# ===========================================================================
# E. Disk consistency — edit must match on disk
# ===========================================================================

class TestEditDiskConsistency(unittest.TestCase):
    """After every edit, the on-disk file must exactly match VFS content."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gvfs = get_global_vfs()
        self.mount_key = self.gvfs.register_shared_mount(self.tmp, hydrate=False)

        self.vfs = _make_vfs("agent", "s1")
        self.vfs.mount(self.tmp, vfs_path="/p", auto_sync=True)
        self.gvfs.register_vfs(self.vfs)

        self.vfs.write("/p/data.py", "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n")
        self.vfs.refresh_mount("/p")

    def tearDown(self):
        self.gvfs.unregister_vfs(self.vfs)
        self.gvfs.unregister_shared_mount(self.tmp)
        shutil.rmtree(self.tmp)

    def _disk(self):
        with open(os.path.join(self.tmp, "data.py"), "r") as f:
            return f.read()

    def _vfs(self):
        r = self.vfs.read("/p/data.py")
        self.assertTrue(r.get("success"), f"VFS read failed: {r}")
        return r["content"]

    def _store(self):
        store_key = f"{self.mount_key}::data.py"
        entry = self.gvfs._shared_store.get(store_key)
        return entry["content"] if entry else None

    def test_all_three_layers_match_after_edit(self):
        """VFS _content, Shared-Store, and disk must all agree after edit."""
        self.vfs.edit("/p/data.py", 2, 2, "b = 999")

        vfs_content = self._vfs()
        disk_content = self._disk()
        store_content = self._store()

        self.assertIn("b = 999", vfs_content)
        self.assertEqual(vfs_content, disk_content,
                         "BUG: VFS and disk diverged after edit")
        if store_content is not None:
            self.assertEqual(vfs_content, store_content,
                             "BUG: VFS and Shared-Store diverged after edit")

    def test_three_layers_match_after_sequential_edits(self):
        """Multiple edits — all three layers must stay in sync."""
        self.vfs.edit("/p/data.py", 1, 1, "a = 10")
        self.vfs.edit("/p/data.py", 3, 3, "c = 30")
        self.vfs.edit("/p/data.py", 5, 5, "e = 50")

        vfs_content = self._vfs()
        disk_content = self._disk()
        store_content = self._store()

        for layer_name, layer_content in [("VFS", vfs_content), ("Disk", disk_content)]:
            self.assertIn("a = 10", layer_content, f"{layer_name} missing edit 1")
            self.assertIn("c = 30", layer_content, f"{layer_name} missing edit 2")
            self.assertIn("e = 50", layer_content, f"{layer_name} missing edit 3")

        self.assertEqual(vfs_content, disk_content, "VFS ≠ Disk after 3 edits")
        if store_content is not None:
            self.assertEqual(vfs_content, store_content, "VFS ≠ Store after 3 edits")


# ===========================================================================
# F. write_chunk with shared mount
# ===========================================================================

class TestWriteChunkSharedMount(unittest.TestCase):
    """write_chunk on shared mounts must update the Shared-Store
    at each chunk so other agents see partial progress."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gvfs = get_global_vfs()
        self.mount_key = self.gvfs.register_shared_mount(self.tmp, hydrate=False)

        self.vfs1 = _make_vfs("writer", "s1")
        self.vfs1.mount(self.tmp, vfs_path="/proj", auto_sync=True)
        self.gvfs.register_vfs(self.vfs1)
        self.sh1 = _make_shell(self.vfs1)

        self.vfs2 = _make_vfs("reader", "s2")
        self.vfs2.mount(self.tmp, vfs_path="/proj", auto_sync=True)
        self.gvfs.register_vfs(self.vfs2)

    def tearDown(self):
        self.gvfs.unregister_vfs(self.vfs1)
        self.gvfs.unregister_vfs(self.vfs2)
        self.gvfs.unregister_shared_mount(self.tmp)
        shutil.rmtree(self.tmp)

    def test_chunks_visible_to_other_agent_after_completion(self):
        """After all chunks are written, agent2 must see the full content."""
        self.sh1("", 'write_chunk /proj/big.py 0 3 "# Part 1\nimport os\n"')
        self.sh1("", 'write_chunk /proj/big.py 1 3 "# Part 2\ndef main():\n    pass\n"')
        self.sh1("", "write_chunk /proj/big.py 2 3 \"# Part 3\nif __name__ == '__main__':\n    main()\"")

        self.vfs1.refresh_mount("/proj")
        self.vfs2.refresh_mount("/proj")
        content2 = _safe_read(self, self.vfs2, "/proj/big.py")
        self.assertIn("import os", content2)
        self.assertIn("def main():", content2)
        self.assertIn("__main__", content2)

    def test_edit_after_chunk_write_persists(self):
        """Write file via chunks, then edit a line — edit must stick."""
        self.sh1("", 'write_chunk /proj/cfg.py 0 2 "DEBUG = False\nPORT = 8080\n"')
        self.sh1("", 'write_chunk /proj/cfg.py 1 2 "HOST = localhost\nLOG = INFO"')
        self.vfs1.refresh_mount("/proj")

        # Now edit
        self.sh1("", 'edit /proj/cfg.py 1 1 "DEBUG = True"')

        content = _safe_read(self, self.vfs1, "/proj/cfg.py")
        self.assertIn("DEBUG = True", content,
                       "BUG: edit after chunk-write lost due to stale store")
        self.assertNotIn("DEBUG = False", content)


if __name__ == "__main__":
    unittest.main()
