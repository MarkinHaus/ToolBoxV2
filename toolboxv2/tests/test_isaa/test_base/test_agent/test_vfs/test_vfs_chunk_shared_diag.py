"""
test_vfs_chunk_shared_diag.py
=============================
Diagnostic tests to isolate WHY write_chunk chunks 1+ are not visible
to agent2 via Shared-Store. Each test targets ONE hypothesis.

Run with:
    python -m unittest test_vfs_chunk_shared_diag -v
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2
from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs
from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import make_vfs_shell


def _make_shell(vfs):
    s = MagicMock()
    s.vfs = vfs
    return make_vfs_shell(s)


class TestChunkSharedStoreDiag(unittest.TestCase):
    """Isolate the root cause of chunk desync in shared mounts."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gvfs = get_global_vfs()
        self.mount_key = self.gvfs.register_shared_mount(self.tmp, hydrate=False)

        self.vfs = VirtualFileSystemV2(session_id="s1", agent_name="a1")
        self.vfs.mount(self.tmp, vfs_path="/proj", auto_sync=True)
        self.gvfs.register_vfs(self.vfs)
        self.sh = _make_shell(self.vfs)

    def tearDown(self):
        self.gvfs.unregister_vfs(self.vfs)
        self.gvfs.unregister_shared_mount(self.tmp)
        shutil.rmtree(self.tmp)

    def _store_content(self, relative):
        """Read directly from the Shared-Store RAM dict."""
        store_key = f"{self.mount_key}::{relative}"
        entry = self.gvfs._shared_store.get(store_key)
        return entry["content"] if entry else None

    def _disk_content(self, relative):
        path = os.path.join(self.tmp, relative)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return f.read()

    # =================================================================
    # H1: Does write (chunk 0) update the Shared-Store?
    # =================================================================

    def test_h1_chunk0_updates_store(self):
        """Chunk 0 uses vfs.write → must land in Shared-Store."""
        self.sh("", 'write_chunk /proj/f.txt 0 2 "chunk0_content\n"')
        store = self._store_content("f.txt")
        self.assertIsNotNone(store, "H1 FAIL: chunk 0 did not reach Shared-Store")
        self.assertIn("chunk0_content", store)

    # =================================================================
    # H2: Does append (chunk 1) update the Shared-Store?
    # =================================================================

    def test_h2_chunk1_updates_store(self):
        """Chunk 1 uses vfs.append → must update Shared-Store with concatenated content."""
        self.sh("", 'write_chunk /proj/f.txt 0 2 "chunk0\n"')
        store_after_0 = self._store_content("f.txt")
        self.assertIsNotNone(store_after_0, "Prerequisite: chunk 0 must be in store")

        self.sh("", 'write_chunk /proj/f.txt 1 2 "chunk1"')
        store_after_1 = self._store_content("f.txt")
        self.assertIsNotNone(store_after_1,
                             "H2 FAIL: Shared-Store has no entry after chunk 1")
        self.assertIn("chunk0", store_after_1,
                      "H2 FAIL: chunk 0 content lost from store after chunk 1")
        self.assertIn("chunk1", store_after_1,
                      "H2 FAIL: chunk 1 content not in store — append did not update store")

    # =================================================================
    # H3: Does append go through the Shared-Store path at all?
    # =================================================================

    def test_h3_append_shared_path_reached(self):
        """Verify _get_shared_store_info returns non-None for /proj/ paths."""
        self.vfs.write("/proj/probe.txt", "initial")
        self.vfs.refresh_mount("/proj")

        info = self.vfs._get_shared_store_info("/proj/probe.txt")
        self.assertIsNotNone(info,
                             "H3 FAIL: _get_shared_store_info returns None — "
                             "append will take local-only path, never updating store")
        mount_key, relative, local_base = info
        self.assertEqual(mount_key, self.mount_key,
                         "H3 FAIL: mount_key mismatch — store writes go to wrong key")
        self.assertEqual(relative, "probe.txt")

    # =================================================================
    # H4: Does the sidecar file (.__chunks__) interfere?
    # =================================================================

    def test_h4_sidecar_does_not_corrupt_store(self):
        """The .__chunks__ sidecar must not pollute or corrupt the store."""
        self.sh("", 'write_chunk /proj/f.txt 0 3 "a\n"')

        # Sidecar should exist in VFS
        self.assertTrue(self.vfs._is_file("/proj/f.txt.__chunks__"),
                        "Prerequisite: sidecar must exist during chunking")

        # Sidecar should NOT be in the Shared-Store (it's internal bookkeeping)
        sidecar_in_store = self._store_content("f.txt.__chunks__")
        # Even if it IS in the store, it must not break f.txt reads
        # But ideally it shouldn't be there

        # Main file must still be readable
        store = self._store_content("f.txt")
        self.assertIsNotNone(store, "H4 FAIL: main file disappeared from store")
        self.assertIn("a", store)

    # =================================================================
    # H5: Does refresh_mount wipe the _content after chunks?
    # =================================================================

    def test_h5_refresh_does_not_lose_content(self):
        """refresh_mount must not revert file content to disk-only state
        if Shared-Store has newer content."""
        self.sh("", 'write_chunk /proj/f.txt 0 2 "part1\n"')
        self.sh("", 'write_chunk /proj/f.txt 1 2 "part2"')

        # Before refresh: check VFS _content
        f = self.vfs.files.get("/proj/f.txt")
        self.assertIsNotNone(f, "Prerequisite: file must exist in VFS")
        pre_refresh_content = f._content

        # Refresh
        self.vfs.refresh_mount("/proj")

        # After refresh: read must still return full content
        r = self.vfs.read("/proj/f.txt")
        self.assertTrue(r.get("success"), f"read after refresh failed: {r}")
        self.assertIn("part1", r["content"])
        self.assertIn("part2", r["content"],
                      "H5 FAIL: refresh_mount lost chunk 1 content — "
                      f"pre-refresh _content was: {repr(pre_refresh_content)}")

    # =================================================================
    # H6: Does vfs.read on vfs2 actually hit the store?
    # =================================================================

    def test_h6_vfs2_reads_from_store(self):
        """A second VFS instance reading a shared-mount file must
        get content from the Shared-Store, not stale disk."""
        self.vfs.write("/proj/shared.txt", "version_1")

        vfs2 = VirtualFileSystemV2(session_id="s2", agent_name="a2")
        vfs2.mount(self.tmp, vfs_path="/proj", auto_sync=True)
        self.gvfs.register_vfs(vfs2)
        vfs2.refresh_mount("/proj")

        try:
            r = vfs2.read("/proj/shared.txt")
            self.assertTrue(r.get("success"), f"vfs2 read failed: {r}")
            self.assertEqual(r["content"], "version_1",
                             "H6 FAIL: vfs2 did not read from Shared-Store")

            # Now update via vfs1
            self.vfs.write("/proj/shared.txt", "version_2")

            # vfs2 should see the update via store (no refresh needed)
            r2 = vfs2.read("/proj/shared.txt")
            self.assertTrue(r2.get("success"))
            self.assertEqual(r2["content"], "version_2",
                             "H6 FAIL: vfs2 sees stale content after vfs1 write — "
                             "Shared-Store not being used for cross-agent reads")
        finally:
            self.gvfs.unregister_vfs(vfs2)

    # =================================================================
    # H7: append on shared mount — is the concat correct?
    # =================================================================

    def test_h7_append_concat_correct(self):
        """Direct append (not chunk) on shared mount must concat correctly."""
        self.vfs.write("/proj/acc.txt", "line1\n")
        self.vfs.refresh_mount("/proj")

        self.vfs.append("/proj/acc.txt", "line2\n")

        store = self._store_content("acc.txt")
        self.assertIsNotNone(store, "H7 FAIL: no store entry after append")
        self.assertIn("line1", store)
        self.assertIn("line2", store,
                      "H7 FAIL: append did not concatenate to store content")

        disk = self._disk_content("acc.txt")
        self.assertEqual(store, disk,
                         "H7 FAIL: store and disk diverged after append")

    # =================================================================
    # H8: Three appends accumulate correctly in store?
    # =================================================================

    def test_h8_three_appends_accumulate(self):
        """Three sequential appends must all be in the store."""
        self.vfs.write("/proj/acc.txt", "base\n")
        self.vfs.refresh_mount("/proj")

        self.vfs.append("/proj/acc.txt", "a1\n")
        self.vfs.append("/proj/acc.txt", "a2\n")
        self.vfs.append("/proj/acc.txt", "a3\n")

        store = self._store_content("acc.txt")
        for marker in ("base", "a1", "a2", "a3"):
            self.assertIn(marker, store,
                          f"H8 FAIL: '{marker}' missing from store after 3 appends")

    # =================================================================
    # H9: mount_key consistency between register and VFS lookup
    # =================================================================

    def test_h9_mount_key_consistent(self):
        """The mount_key from register_shared_mount must match what
        _get_shared_store_info returns."""
        self.vfs.write("/proj/check.txt", "x")
        self.vfs.refresh_mount("/proj")

        info = self.vfs._get_shared_store_info("/proj/check.txt")
        self.assertIsNotNone(info)
        resolved_key = info[0]

        self.assertEqual(resolved_key, self.mount_key,
                         f"H9 FAIL: mount_key mismatch — "
                         f"register returned '{self.mount_key}', "
                         f"_get_shared_store_info returned '{resolved_key}'. "
                         f"Likely Path.resolve() vs os.path.abspath difference on Windows")


if __name__ == "__main__":
    unittest.main()
