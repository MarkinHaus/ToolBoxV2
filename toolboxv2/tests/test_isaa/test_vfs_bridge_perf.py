"""P1: ``VFSMemoryIndexer.backfill`` runs on the asyncio event loop and must NOT
block it while waiting for an active agent run to yield (``AGENT_YIELD``).

The old code called the blocking ``AGENT_YIELD.wait(timeout=5.0)`` inside the
coroutine, freezing the whole loop (Discord heartbeat blocked >1700s). The wait
must be cooperative (await).

Scope (per project rule): we test ONLY the new non-blocking behaviour. The VFS /
memory / store machinery is existing, working tb code — it is stubbed to no-ops,
not rebuilt. We drive backfill with a bare indexer whose only real logic is the
yield loop.
"""
from __future__ import annotations

import asyncio
import importlib
import time
import unittest

bridge = importlib.import_module("toolboxv2.mods.isaa.base.Agent.vfs_memory_bridge")
VFSMemoryIndexer = bridge.VFSMemoryIndexer
AGENT_YIELD = bridge.AGENT_YIELD


def _bare_indexer(paths):
    """An indexer with only what backfill() touches — no Session/Memory/VFS."""
    idx = VFSMemoryIndexer.__new__(VFSMemoryIndexer)
    idx.agent_name = "tagent"
    idx.agent_space = "VFSIndex/tagent"

    class _V:  # minimal object exposing .files keys
        files = {p: object() for p in paths}
    idx.vfs = _V()
    idx.memory = None
    # stub the parts that are not under test (assumed working)
    idx._eligible = lambda p: True
    idx._load_indexed_hashes = lambda space=None: {}
    idx._hashes = {}
    idx.save_index = lambda: None

    async def _noop_index(path, session_scoped=False):
        return None
    idx._index_path = _noop_index

    # neutralise store creation used before the loop
    idx.memory = type("M", (), {
        "_sanitize_name": staticmethod(lambda n: n),
        "_get_or_create_store": staticmethod(lambda n: None),
    })()
    bridge._AGENT_BACKFILL_DONE.discard("tagent")
    return idx


class _MetaFile:
    """Minimal VFSFile-like node: only the metadata _meta_hash may use.
    ``_content`` raises if read, proving the hash does NOT touch content."""
    def __init__(self, size=10, mtime=1.0, updated="2026-01-01"):
        self.size_bytes = size
        self.local_mtime = mtime
        self.updated_at = updated

    @property
    def _content(self):
        raise AssertionError("_meta_hash must not read file content")


def _meta_indexer(files):
    idx = VFSMemoryIndexer.__new__(VFSMemoryIndexer)
    idx.agent_name = "tagent"

    class _V:
        pass
    v = _V()
    v.files = files
    idx.vfs = v
    return idx


class MetaHash(unittest.TestCase):
    def test_meta_hash_is_stable_for_same_metadata(self):
        idx = _meta_indexer({"/a.txt": _MetaFile(size=10, mtime=1.0)})
        self.assertEqual(idx._meta_hash("/a.txt"), idx._meta_hash("/a.txt"))
        self.assertTrue(idx._meta_hash("/a.txt"))

    def test_meta_hash_changes_when_size_changes(self):
        a = _meta_indexer({"/a.txt": _MetaFile(size=10, mtime=1.0)})
        b = _meta_indexer({"/a.txt": _MetaFile(size=11, mtime=1.0)})
        self.assertNotEqual(a._meta_hash("/a.txt"), b._meta_hash("/a.txt"))

    def test_meta_hash_changes_when_mtime_changes(self):
        a = _meta_indexer({"/a.txt": _MetaFile(size=10, mtime=1.0)})
        b = _meta_indexer({"/a.txt": _MetaFile(size=10, mtime=2.0)})
        self.assertNotEqual(a._meta_hash("/a.txt"), b._meta_hash("/a.txt"))

    def test_meta_hash_does_not_read_content(self):
        idx = _meta_indexer({"/a.txt": _MetaFile()})
        idx._meta_hash("/a.txt")  # must not raise (would if content was read)

    def test_meta_hash_falls_back_to_updated_at_without_mtime(self):
        a = _meta_indexer({"/a.txt": _MetaFile(size=10, mtime=None, updated="t1")})
        b = _meta_indexer({"/a.txt": _MetaFile(size=10, mtime=None, updated="t2")})
        self.assertNotEqual(a._meta_hash("/a.txt"), b._meta_hash("/a.txt"))

    def test_meta_hash_missing_file_is_empty(self):
        self.assertEqual(_meta_indexer({})._meta_hash("/gone.txt"), "")


class BackfillNonBlocking(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        AGENT_YIELD.set()

    async def asyncTearDown(self):
        AGENT_YIELD.set()

    async def test_wait_is_cooperative_not_blocking(self):
        idx = _bare_indexer([f"/f{i}.txt" for i in range(3)])
        AGENT_YIELD.clear()  # simulate an active agent run

        ticks = {"n": 0}

        async def ticker():
            for _ in range(200):
                ticks["n"] += 1
                await asyncio.sleep(0.01)

        async def release():
            await asyncio.sleep(0.3)
            AGENT_YIELD.set()

        t = asyncio.create_task(ticker())
        r = asyncio.create_task(release())
        t0 = time.perf_counter()
        await asyncio.wait_for(idx.backfill(show_progress=False), timeout=3.0)
        elapsed = time.perf_counter() - t0
        await r
        t.cancel()
        # If the wait blocked the loop, the ticker couldn't advance and wall time
        # would explode (3 files x 5s). Cooperative awaiting keeps both small.
        self.assertGreaterEqual(ticks["n"], 10,
                                f"loop blocked during backfill (ticks={ticks['n']})")
        self.assertLess(elapsed, 2.5, f"backfill blocked the loop ({elapsed:.1f}s)")


# =============================================================================
# Clean transition: old content-sha1 -> metadata hash WITHOUT re-embedding
# =============================================================================

import hashlib as _hashlib


class _ContentFile:
    def __init__(self, content, size=None, mtime=1.0):
        self._content = content
        self.content = content
        self.size_bytes = size if size is not None else len(content)
        self.local_mtime = mtime
        self.updated_at = "2026-01-01"
        self.filename = "f.txt"
        self.readonly = False
        self._content_hash = None


class _FakeStore:
    """Minimal stand-in for the hybrid_memory store: records the migration
    UPDATE and whether a re-embed (invalidate/add) happened."""
    def __init__(self):
        self.updated = []       # (source, new_hash)
        self.invalidated = []

    def invalidate_by_source(self, source):
        self.invalidated.append(source)

    # the migration uses _exec to rewrite meta_custom.content_hash
    def _exec(self, sql, params=()):
        up = sql.strip().upper()
        if up.startswith("UPDATE"):
            self.updated.append(params)

            class _C:
                def fetchall(self_inner): return []
                def fetchone(self_inner): return None
            return _C()
        # SELECT id, meta_custom ... -> one existing entry for the source
        row = {"id": 1, "meta_custom": "{}"}

        class _C:
            def fetchall(self_inner): return [row]
            def fetchone(self_inner): return row
        return _C()
    space = "VFSIndex/tagent"


def _migration_indexer(path, content, stored_hash):
    idx = VFSMemoryIndexer.__new__(VFSMemoryIndexer)
    idx.agent_name = "tagent"
    idx.agent_space = "VFSIndex/tagent"
    idx.space = "VFSIndex/tagent.s1"

    class _V:
        pass
    v = _V()
    v.files = {path: _ContentFile(content)}
    idx.vfs = v

    store = _FakeStore()
    idx._store = lambda space=None: store
    idx._eligible = lambda p: True
    idx._get_content = lambda p: content
    idx._space_for = lambda p, session_scoped=False: idx.agent_space
    # pre-load the OLD stored hash (content-sha1 style) for this path
    idx._hashes = {path: stored_hash}

    added = {"n": 0}

    async def _add(space, chunks, metadata=None):
        added["n"] += 1
        return True
    idx.memory = type("M", (), {"add_data": staticmethod(_add)})()
    return idx, store, added


def _sha1(s: str) -> str:
    return _hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()


class HashTransition(unittest.IsolatedAsyncioTestCase):
    async def test_same_content_old_hash_migrates_without_reembed(self):
        content = "stable content"
        old = _sha1(content)                      # legacy content-sha1
        idx, store, added = _migration_indexer("/a.txt", content, old)
        meta = idx._meta_hash("/a.txt")
        self.assertNotEqual(old, meta)            # hashes differ (would re-embed naively)
        await idx._index_path("/a.txt")
        # NO re-embed, hash migrated in-memory + persisted via UPDATE
        self.assertEqual(added["n"], 0, "must not re-embed unchanged content")
        self.assertEqual(idx._hashes["/a.txt"], meta)
        self.assertTrue(store.updated, "migration must persist the new hash")

    async def test_changed_content_reembeds(self):
        old = _sha1("the ORIGINAL content")       # stored hash of old content
        idx, store, added = _migration_indexer("/a.txt", "totally new content", old)
        await idx._index_path("/a.txt")
        # content sha1 != stored -> genuine change -> re-embed
        self.assertEqual(added["n"], 1, "changed content must re-embed")

    async def test_post_migration_meta_hash_skips_cheaply(self):
        # once _hashes holds the meta hash, an unchanged file skips with no read
        content = "x"
        idx, store, added = _migration_indexer("/a.txt", content, _sha1(content))
        meta = idx._meta_hash("/a.txt")
        idx._hashes["/a.txt"] = meta              # already migrated
        read = {"n": 0}
        def _no_read(p):
            read["n"] += 1
            return content
        idx._get_content = _no_read
        await idx._index_path("/a.txt")
        self.assertEqual(added["n"], 0)
        self.assertEqual(read["n"], 0, "steady state must not read content")


if __name__ == "__main__":
    unittest.main()
