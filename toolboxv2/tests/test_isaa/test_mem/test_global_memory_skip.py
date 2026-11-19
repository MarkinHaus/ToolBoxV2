"""Regression tests: the VFS auto-indexer must NOT index /global/.memory/*.

Bug: every tool-output archived by _retroactive_offload() into
/global/.memory/archive/ was chunked + embedded into VFSIndex/global,
poisoning recall with shell dumps (93 MB DB).

H1: live path (_eligible via on_change hook + backfill) is open.
H2: bulk path (index_global_memory rglob scan) is open.
Fix contract: _global_path_eligible() gates BOTH paths; normal /global
content stays eligible.
"""
import unittest
from types import SimpleNamespace


class FakeVFS:
    def __init__(self):
        self.files = {}
        self._shadow_index = {}

    def _get_mount_for_path(self, path):
        return None

    def _get_shared_store_info(self, path):
        return None


def _mk_file(vfs, path, name, content="x"):
    vfs.files[path] = SimpleNamespace(
        filename=name, content=content, size_bytes=len(content), readonly=False
    )


class TestGlobalMemorySkip(unittest.TestCase):
    def _indexer(self):
        from toolboxv2.mods.isaa.base.Agent.vfs_memory_bridge import (
            VFSMemoryIndexer,
        )

        vfs = FakeVFS()
        session = SimpleNamespace(
            vfs=vfs, _memory=None, agent_name="t", session_id="s1"
        )
        idx = VFSMemoryIndexer(session)
        return idx, vfs

    def test_h1_memory_archive_not_eligible(self):
        """H1: live on_change path must skip /global/.memory/*."""
        idx, vfs = self._indexer()
        p = "/global/.memory/archive/123_call_shell.txt"
        _mk_file(vfs, p, "123_call_shell.txt")
        self.assertFalse(
            idx._eligible(p),
            "H1 FAIL: .memory archive file is live-indexed",
        )

    def test_h1_memory_logs_not_eligible_any_depth(self):
        """Any segment position: /global/x/.memory/y also skipped."""
        idx, vfs = self._indexer()
        p = "/global/runs/.memory/logs/x.md"
        _mk_file(vfs, p, "x.md")
        self.assertFalse(idx._eligible(p))

    def test_normal_global_content_stays_eligible(self):
        """Regression: real knowledge files must remain indexable."""
        idx, vfs = self._indexer()
        cases = [
            ("/global/notes/plan.md", "plan.md"),
            ("/global/90_Agents/AGENTS.md", "AGENTS.md"),
            ("/global/sub/vfs_embed_analysis/findings.md", "findings.md"),
        ]
        for p, name in cases:
            _mk_file(vfs, p, name)
            self.assertTrue(idx._eligible(p), f"over-blocked: {p}")

    def test_h2_gate_function_covers_archive(self):
        """H2: bulk scanner gate rejects archive rel-paths."""
        from toolboxv2.mods.isaa.base.Agent.vfs_memory_bridge import (
            _global_path_eligible,
        )

        self.assertFalse(_global_path_eligible(".memory/archive/x.txt"))
        self.assertFalse(_global_path_eligible("sub/dir/.memory/y.txt"))
        self.assertTrue(_global_path_eligible("notes/plan.md"))
        self.assertTrue(_global_path_eligible("vault_analysis_map.md"))


if __name__ == "__main__":
    unittest.main()
