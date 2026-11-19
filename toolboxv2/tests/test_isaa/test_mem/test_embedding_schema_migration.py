"""Option-C completion: embedding BLOB in SQLite (schema migration + guards).

H1 (legacy DB): a pre-Option-C `entries` table without `embedding` column
    must be migrated on store open (ALTER TABLE) — `store.add()` must work.
H2 (NULL-embedding guard): `_load_indexed_hashes` must NOT preload hashes
    for entries with `embedding IS NULL` (otherwise backfill skips them and
    vector search silently dies for those files). Entries WITH embedding
    must preload even when the FAISS RAM index is empty (BLOB rebuild
    makes FAISS state irrelevant).
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from toolboxv2.mods.isaa.base.hybrid_memory import (
    HybridMemoryStore,
    SCHEMA_SQL,
)

LEGACY_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content_type TEXT DEFAULT 'text',
    version     INTEGER DEFAULT 1,
    supersedes  TEXT,
    is_active   INTEGER DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    ttl         INTEGER,
    access_count INTEGER DEFAULT 0,
    space       TEXT NOT NULL DEFAULT 'default',
    meta_role   TEXT,
    meta_source TEXT,
    meta_language TEXT,
    meta_category TEXT,
    meta_importance REAL DEFAULT 0.5,
    meta_custom TEXT
);
"""


class FakeVFS:
    def __init__(self):
        self.files = {}
        self._shadow_index = {}

    def _get_mount_for_path(self, path):
        return None

    def _get_shared_store_info(self, path):
        return None


class FakeMemory:
    def __init__(self, store):
        self._store = store

    def get(self, space=None):
        return [self._store]


def _mk_indexer(store):
    from toolboxv2.mods.isaa.base.Agent.vfs_memory_bridge import (
        VFSMemoryIndexer,
    )

    vfs = FakeVFS()
    session = SimpleNamespace(
        vfs=vfs, _memory=FakeMemory(store), agent_name="t", session_id="s1"
    )
    return VFSMemoryIndexer(session)


class TestEmbeddingSchemaMigration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        # fresh singleton namespace per test
        HybridMemoryStore._instances = {}

    def tearDown(self):
        # close SQLite handles BEFORE tmpdir cleanup (Windows file locks)
        for inst in list(HybridMemoryStore._instances.values()):
            try:
                inst.close()
            except Exception:
                pass
        HybridMemoryStore._instances = {}
        self._tmp.cleanup()

    def _vec(self, seed=1.0):
        return np.full((1, 768), seed, dtype=np.float32)

    def test_h1_legacy_db_gets_embedding_column(self):
        """H1: legacy schema (no embedding) is migrated on open; add() works."""
        db = self.tmp / "VFSIndex-legacy"
        db.mkdir()
        con = sqlite3.connect(db / "entries.db")
        con.executescript(LEGACY_SCHEMA)
        con.commit()
        con.close()

        store = HybridMemoryStore(str(db), embedding_dim=768, space="default")
        cols = [r[1] for r in store._exec("PRAGMA table_info(entries)").fetchall()]
        self.assertIn(
            "embedding", cols, "H1 FAIL: migration did not add embedding column"
        )
        eid = store.add("hello world", self._vec())
        self.assertTrue(eid)

    def test_h1_fresh_db_add_then_reload(self):
        """Sanity: new-schema add persists BLOB; reopen sees it (BLOB source)."""
        db = self.tmp / "VFSIndex-fresh"
        store = HybridMemoryStore(str(db), embedding_dim=768, space="default")
        store.add("persistent text", self._vec())
        row = store._exec(
            "SELECT embedding FROM entries WHERE content='persistent text'"
        ).fetchone()
        self.assertIsNotNone(row[0], "embedding BLOB must be persisted")

    def test_h2_null_embedding_not_preloaded(self):
        """H2: hash preload skips entries with NULL embedding (re-indexable)."""
        db = self.tmp / "VFSIndex-guard"
        store = HybridMemoryStore(str(db), embedding_dim=768, space="default")
        store.add(
            "some file content",
            self._vec(),
            meta={"source": "vfs:/global/x.md",
                  "vfs_path": "/global/x.md",
                  "category": "vfs",
                  "content_hash": "abc123"},
        )
        idx = _mk_indexer(store)
        hashes = idx._load_indexed_hashes()
        self.assertEqual(hashes.get("/global/x.md"), "abc123")

        # simulate legacy/migrated row: vector missing
        store._exec("UPDATE entries SET embedding = NULL")
        hashes = idx._load_indexed_hashes()
        self.assertNotIn(
            "/global/x.md", hashes,
            "H2 FAIL: NULL-embedding entry preloaded -> backfill would skip it",
        )

        # restore vector -> preload works again, even with FAISS wiped
        store._exec(
            "UPDATE entries SET embedding = ? WHERE meta_source = ?",
            (self._vec().tobytes(), "vfs:/global/x.md"),
        )
        store._faiss.index.reset()
        hashes = idx._load_indexed_hashes()
        self.assertEqual(
            hashes.get("/global/x.md"), "abc123",
            "H2 FAIL: BLOB source of truth ignored when FAISS is empty",
        )


    def test_h3_hash_current_respects_null_embedding(self):
        """H3: _global_hash_current False for NULL-embedding rows (healable)."""
        from toolboxv2.mods.isaa.base.Agent.vfs_memory_bridge import (
            _global_hash_current,
        )

        db = self.tmp / "VFSIndex-hashcur"
        store = HybridMemoryStore(str(db), embedding_dim=768, space="default")
        src = "vfs:/global/y.md"
        store.add(
            "y content", self._vec(),
            meta={"source": src, "vfs_path": "/global/y.md",
                  "category": "vfs", "content_hash": "hashY"},
        )
        self.assertTrue(_global_hash_current(store, src, "hashY"))

        store._exec("UPDATE entries SET embedding = NULL")
        self.assertFalse(
            _global_hash_current(store, src, "hashY"),
            "H3 FAIL: NULL-embedding row treated as indexed -> never healed",
        )


if __name__ == "__main__":
    unittest.main()
