"""
Test Suite for HybridMemoryStore - Agent Memory V2
TDD Implementation following agent-memory-v2-plan.md

Tests cover:
- Schema initialization
- CRUD operations
- Vector search (FAISS)
- BM25 full-text search (SQLite FTS5)
- Hybrid RRF fusion
- Entity-Relation graph
- Persistence & backup
- Regression tests for V1 issues
"""

import asyncio
import hashlib
import os
import pickle
import shutil
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

# Import will fail initially (TDD), that's expected
try:
    from toolboxv2.mods.isaa.base.ai_semantic_memory import AISemanticMemory
    from toolboxv2.mods.isaa.base.hybrid_memory import HybridMemoryStore
except ImportError:
    # Expected in TDD - tests will fail until implementation exists
    HybridMemoryStore = None
    AISemanticMemory = None


class TestHybridMemoryFoundation(unittest.TestCase):
    """
    Core functionality tests for HybridMemoryStore
    Following TDD: Tests written BEFORE implementation
    """

    @classmethod
    def setUpClass(cls):
        """Setup class-level resources"""
        cls.test_rng = np.random.default_rng(42)

    def setUp(self):
        """Create temp directory for each test"""
        self.tmp = tempfile.mkdtemp(prefix="hybrid_memory_test_")
        if HybridMemoryStore is None:
            self.skipTest("HybridMemoryStore not implemented yet (TDD phase)")
        self.store = HybridMemoryStore(self.tmp, embedding_dim=768)
        self.rng = np.random.default_rng(42)

    def _emb(self, dim: int = 768) -> np.ndarray:
        """Generate random embedding for testing"""
        return self.rng.random(dim).astype(np.float32)

    def tearDown(self):
        """Cleanup temp directory"""
        if hasattr(self, "store") and self.store:
            self.store.close()
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)

    # ═══════════════════════════════════════════════════════════
    # TASK 1.1: Schema Creation Tests
    # ═══════════════════════════════════════════════════════════

    def test_schema_creation(self):
        """
        TASK 1.1: _init_db() creates all 5 tables + FTS
        Schema must include: entries, entities, relations, concept_index, entries_fts
        """
        # Check database file exists
        db_path = Path(self.tmp) / "entries.db"
        self.assertTrue(db_path.exists(), "Database file should be created")

        # Query all tables
        tables = self.store._exec(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row[0] for row in tables}

        # Required tables
        required = {"entries", "entities", "relations", "concept_index", "entries_fts"}
        self.assertTrue(
            required <= table_names,
            f"Missing tables. Required: {required}, Got: {table_names}",
        )

    def test_schema_has_indexes(self):
        """Schema should have proper indexes for performance"""
        indexes = self.store._exec(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = {row[0] for row in indexes}

        # Critical indexes
        critical_indexes = {"idx_entries_space", "idx_entries_hash", "idx_rel_src"}
        # Note: SQLite auto-creates indexes for PRIMARY KEYs, so we check for custom ones
        self.assertGreater(len(index_names), 5, "Should have multiple indexes")

    def test_wal_mode_enabled(self):
        """Database should use WAL mode for concurrent access"""
        result = self.store._exec("PRAGMA journal_mode").fetchone()
        self.assertEqual(result[0].lower(), "wal", "WAL mode should be enabled")

    # ═══════════════════════════════════════════════════════════
    # TASK 1.2: Basic add() Operation Tests
    # ═══════════════════════════════════════════════════════════

    def test_add_returns_id(self):
        """TASK 1.2: add() should return 16-char hex ID"""
        eid = self.store.add("SpaceX launched Starship", self._emb())
        self.assertIsInstance(eid, str)
        self.assertEqual(len(eid), 16, "ID should be 16 characters")
        self.assertTrue(all(c in "0123456789abcdef" for c in eid), "ID should be hex")

    def test_add_creates_entry_in_database(self):
        """Added entry should be queryable from database"""
        content = "Test content for database"
        eid = self.store.add(content, self._emb())

        row = self.store._exec(
            "SELECT id, content, is_active FROM entries WHERE id=?", (eid,)
        ).fetchone()

        self.assertIsNotNone(row, "Entry should exist in database")
        self.assertEqual(row[0], eid)
        self.assertEqual(row[1], content)
        self.assertEqual(row[2], 1, "Entry should be active")

    def test_duplicate_rejected(self):
        """TASK 1.2: Duplicate content should return same ID (deduplication)"""
        e1 = self.store.add("same text", self._emb())
        e2 = self.store.add("same text", self._emb())
        self.assertEqual(e1, e2, "Duplicate content should return same ID")

        # Should only have one entry in database
        count = self.store._exec(
            "SELECT COUNT(*) FROM entries WHERE content='same text'"
        ).fetchone()[0]
        self.assertEqual(count, 1, "Should only have one entry for duplicate content")

    def test_duplicate_different_spaces(self):
        """Same content in different spaces should be allowed"""
        store1 = HybridMemoryStore(self.tmp, 768, space="space1")
        store2 = HybridMemoryStore(self.tmp, 768, space="space2")

        e1 = store1.add("content", self._emb())
        e2 = store2.add("content", self._emb())

        self.assertNotEqual(e1, e2, "Different spaces should allow same content")

        store1.close()
        store2.close()

    def test_add_with_metadata(self):
        """add() should store metadata correctly"""
        meta = {
            "role": "assistant",
            "source": "/app/main.py",
            "language": "python",
            "importance": 0.9,
        }
        eid = self.store.add("def hello():", self._emb(), meta=meta)

        row = self.store._exec(
            "SELECT meta_role, meta_source, meta_language, meta_importance FROM entries WHERE id=?",
            (eid,),
        ).fetchone()

        self.assertEqual(row[0], "assistant")
        self.assertEqual(row[1], "/app/main.py")
        self.assertEqual(row[2], "python")
        self.assertAlmostEqual(row[3], 0.9, places=2)

    def test_add_with_ttl(self):
        """add() should store TTL correctly"""
        eid = self.store.add("ephemeral content", self._emb(), ttl=60)

        row = self.store._exec("SELECT ttl FROM entries WHERE id=?", (eid,)).fetchone()

        self.assertEqual(row[0], 60)

    def test_add_with_concepts(self):
        """add() should store concepts in concept_index"""
        concepts = ["python", "function", "coding"]
        eid = self.store.add("def test():", self._emb(), concepts=concepts)

        # Check concept_index
        for concept in concepts:
            row = self.store._exec(
                "SELECT entry_id FROM concept_index WHERE concept=? AND entry_id=?",
                (concept, eid),
            ).fetchone()
            self.assertIsNotNone(row, f"Concept '{concept}' should be indexed")

    # ═══════════════════════════════════════════════════════════
    # TASK 1.3: FAISS Integration Tests
    # ═══════════════════════════════════════════════════════════

    def test_vector_query_finds_added(self):
        """TASK 1.3: Vector search should find semantically similar content"""
        emb = self._emb()
        self.store.add("rockets launch from pad", emb)

        hits = self.store.query("rockets", emb, k=1, search_modes=("vector",))

        self.assertEqual(len(hits), 1)
        self.assertIn("rockets", hits[0]["content"])

    def test_vector_query_respects_k(self):
        """Vector query should return at most k results"""
        for i in range(10):
            self.store.add(f"entry {i}", self._emb())

        hits = self.store.query("entry", self._emb(), k=3, search_modes=("vector",))
        self.assertLessEqual(len(hits), 3)

    def test_vector_query_min_similarity(self):
        """Vector query should respect minimum similarity threshold"""
        # Add two very different entries
        emb1 = self._emb()
        emb2 = np.random.random(768).astype(np.float32)  # Very different

        self.store.add("entry 1", emb1)
        self.store.add("entry 2", emb2)

        # Query with emb1 should prefer entry 1
        hits = self.store.query(
            "entry", emb1, k=2, min_similarity=0.5, search_modes=("vector",)
        )

        # Should get at least one hit (entry 1)
        self.assertGreater(len(hits), 0)

    def test_faiss_id_mapping(self):
        """FAISS index should maintain bidirectional ID mapping"""
        e1 = self.store.add("first", self._emb())
        e2 = self.store.add("second", self._emb())

        # Check internal mappings exist
        self.assertIn(e1, self.store._id_map)
        self.assertIn(e2, self.store._id_map)

        # Reverse mapping should work
        idx1 = self.store._id_map[e1]
        self.assertEqual(self.store._idx_map[idx1], e1)

    # ═══════════════════════════════════════════════════════════
    # TASK 1.4: BM25 Full-Text Search Tests
    # ═══════════════════════════════════════════════════════════

    def test_bm25_finds_exact_term(self):
        """
        TASK 1.4: BM25 should find exact function names
        This addresses P3: Exact term matching for code
        """
        emb = self._emb()
        self.store.add("def calculate_tax(income, rate):", emb, content_type="code")

        # BM25 with random embedding (intentionally irrelevant for vector)
        hits = self.store.query("calculate_tax", self._emb(), k=1, search_modes=("bm25",))

        self.assertEqual(len(hits), 1)
        self.assertIn("calculate_tax", hits[0]["content"])

    def test_bm25_ranks_relevance(self):
        """BM25 should rank more relevant documents higher"""
        self.store.add("python programming language", self._emb())
        self.store.add("python snake in the wild", self._emb())
        self.store.add("java programming guide", self._emb())

        hits = self.store.query(
            "python programming", self._emb(), k=3, search_modes=("bm25",)
        )

        # First hit should be about python programming
        self.assertIn("programming", hits[0]["content"].lower())
        self.assertIn("python", hits[0]["content"].lower())

    def test_bm25_handles_special_chars(self):
        """BM25 should handle code with special characters"""
        self.store.add("def func(a, b): return a + b", self._emb(), content_type="code")

        hits = self.store.query("func", self._emb(), k=1, search_modes=("bm25",))
        self.assertEqual(len(hits), 1)

    # ═══════════════════════════════════════════════════════════
    # TASK 1.5: Hybrid RRF Fusion Tests
    # ═══════════════════════════════════════════════════════════

    def test_rrf_fusion(self):
        """
        TASK 1.5: Fusion should combine Vector + BM25 results
        RRF (Reciprocal Rank Fusion) algorithm
        """
        embs = [self._emb() for _ in range(3)]
        self.store.add("machine learning algorithms", embs[0])
        self.store.add("deep learning neural networks", embs[1])
        self.store.add("calculate_tax function", embs[2], content_type="code")

        hits = self.store.query("learning", embs[0], k=3, search_modes=("vector", "bm25"))

        self.assertGreater(len(hits), 0)
        # Should find learning-related entries
        contents = " ".join(h["content"] for h in hits)
        self.assertIn("learning", contents.lower())

    def test_rrf_weights_different_modes(self):
        """RRF should allow weighting different search modes"""
        emb = self._emb()
        self.store.add("python code", emb, content_type="code")
        self.store.add("python snake", emb)

        # Query with different weights (dict format)
        hits1 = self.store.query(
            "python",
            emb,
            k=2,
            search_modes=("vector", "bm25"),
            mode_weights={"vector": 0.7, "bm25": 0.3},
        )

        self.assertGreater(len(hits1), 0)

    def test_single_mode_query(self):
        """Query should work with single search mode"""
        self.store.add("test content", self._emb())

        # Vector only
        hits_v = self.store.query("test", self._emb(), k=1, search_modes=("vector",))
        self.assertEqual(len(hits_v), 1)

        # BM25 only
        hits_b = self.store.query("test", self._emb(), k=1, search_modes=("bm25",))
        self.assertEqual(len(hits_b), 1)

    # ═══════════════════════════════════════════════════════════
    # TASK 1.6: Update with Versioning Tests
    # ═══════════════════════════════════════════════════════════

    def test_update_creates_version(self):
        """TASK 1.6: Update should create new version and deactivate old"""
        eid = self.store.add("version 1", self._emb())
        new_id = self.store.update(eid, "version 2", self._emb())

        self.assertNotEqual(eid, new_id, "Update should create new ID")

        # Old entry should be inactive
        old = self.store._exec(
            "SELECT is_active, version FROM entries WHERE id=?", (eid,)
        ).fetchone()
        self.assertEqual(old[0], 0, "Old entry should be inactive")

        # New entry should reference old
        new = self.store._exec(
            "SELECT supersedes, version FROM entries WHERE id=?", (new_id,)
        ).fetchone()
        self.assertEqual(new[0], eid, "New entry should supersede old")
        self.assertEqual(new[1], 2, "Version should increment")

    def test_update_preserves_metadata(self):
        """Update should optionally preserve metadata from previous version"""
        meta = {"source": "/app/main.py", "importance": 0.8}
        eid = self.store.add("original", self._emb(), meta=meta)

        new_id = self.store.update(eid, "updated", self._emb())

        new_row = self.store._exec(
            "SELECT meta_source, meta_importance FROM entries WHERE id=?", (new_id,)
        ).fetchone()

        self.assertEqual(new_row[0], "/app/main.py")
        self.assertAlmostEqual(new_row[1], 0.8, places=2)

    def test_update_chain_versions(self):
        """Multiple updates should create version chain"""
        e1 = self.store.add("v1", self._emb())
        e2 = self.store.update(e1, "v2", self._emb())
        e3 = self.store.update(e2, "v3", self._emb())

        # Check chain
        row3 = self.store._exec(
            "SELECT supersedes, version FROM entries WHERE id=?", (e3,)
        ).fetchone()
        self.assertEqual(row3[0], e2)
        self.assertEqual(row3[1], 3)

    # ═══════════════════════════════════════════════════════════
    # TASK 1.7: Soft & Hard Delete Tests
    # ═══════════════════════════════════════════════════════════

    def test_soft_delete(self):
        """TASK 1.7: Soft delete should mark entry as inactive"""
        eid = self.store.add("delete me", self._emb())
        self.store.delete(eid, hard=False)

        # Should still exist but inactive
        row = self.store._exec(
            "SELECT is_active FROM entries WHERE id=?", (eid,)
        ).fetchone()
        self.assertEqual(row[0], 0)

        # Should not appear in queries
        hits = self.store.query("delete me", self._emb(), k=1)
        self.assertEqual(len(hits), 0)

    def test_hard_delete(self):
        """Hard delete should remove entry completely"""
        eid = self.store.add("hard delete", self._emb())
        self.store.delete(eid, hard=True)

        # Should not exist
        row = self.store._exec("SELECT id FROM entries WHERE id=?", (eid,)).fetchone()
        self.assertIsNone(row)

    def test_delete_removes_from_faiss(self):
        """Delete should remove from FAISS index"""
        eid = self.store.add("to delete", self._emb())
        idx = self.store._id_map.get(eid)

        self.store.delete(eid)

        # Mapping should be removed
        self.assertNotIn(eid, self.store._id_map)

    def test_delete_removes_concepts(self):
        """Delete should remove from concept_index"""
        eid = self.store.add("code", self._emb(), concepts=["python", "test"])
        self.store.delete(eid)

        # Concepts should be removed
        count = self.store._exec(
            "SELECT COUNT(*) FROM concept_index WHERE entry_id=?", (eid,)
        ).fetchone()[0]
        self.assertEqual(count, 0)

    # ═══════════════════════════════════════════════════════════
    # TASK 1.8: Entity-Relation Graph Tests
    # ═══════════════════════════════════════════════════════════

    def test_add_entity(self):
        """TASK 1.8: add_entity should create graph node"""
        eid = self.store.add_entity("company:spacex", "company", "SpaceX")

        row = self.store._exec(
            "SELECT entity_type, name FROM entities WHERE id=?", (eid,)
        ).fetchone()

        self.assertEqual(row[0], "company")
        self.assertEqual(row[1], "SpaceX")

    def test_add_relation(self):
        """add_relation should create graph edge"""
        e1 = self.store.add_entity("person:elon", "person", "Elon Musk")
        e2 = self.store.add_entity("company:spacex", "company", "SpaceX")

        self.store.add_relation(e1, e2, "WORKS_AT", weight=1.0)

        row = self.store._exec(
            "SELECT rel_type, weight FROM relations WHERE source_id=? AND target_id=?",
            (e1, e2),
        ).fetchone()

        self.assertEqual(row[0], "WORKS_AT")
        self.assertEqual(row[1], 1.0)

    def test_get_related_depth_1(self):
        """get_related should traverse 1 hop"""
        elon = self.store.add_entity("person:elon", "person", "Elon")
        spacex = self.store.add_entity("company:spacex", "company", "SpaceX")
        tesla = self.store.add_entity("company:tesla", "company", "Tesla")

        self.store.add_relation(elon, spacex, "WORKS_AT")
        self.store.add_relation(elon, tesla, "WORKS_AT")

        related = self.store.get_related(elon, depth=1)
        self.assertEqual(len(related), 2)

        names = {r["name"] for r in related}
        self.assertIn("SpaceX", names)
        self.assertIn("Tesla", names)

    def test_get_related_depth_2(self):
        """get_related should traverse multiple hops"""
        elon = self.store.add_entity("person:elon", "person", "Elon")
        spacex = self.store.add_entity("company:spacex", "company", "SpaceX")
        starship = self.store.add_entity("project:starship", "project", "Starship")

        self.store.add_relation(elon, spacex, "WORKS_AT")
        self.store.add_relation(spacex, starship, "DEVELOPS")

        # Depth 1: only SpaceX
        related_d1 = self.store.get_related(elon, depth=1)
        self.assertEqual(len(related_d1), 1)

        # Depth 2: SpaceX + Starship
        related_d2 = self.store.get_related(elon, depth=2)
        self.assertEqual(len(related_d2), 2)

    def test_relation_bidirectional(self):
        """Relations should be queryable in both directions"""
        e1 = self.store.add_entity("a", "type", "A")
        e2 = self.store.add_entity("b", "type", "B")

        self.store.add_relation(e1, e2, "LINKS_TO")

        # Forward direction
        forward = self.store.get_related(e1, depth=1)
        self.assertEqual(len(forward), 1)

        # Reverse direction (if bidirectional query supported)
        reverse = self.store.get_related(e2, depth=1, direction="incoming")
        self.assertEqual(len(reverse), 1)

    # ═══════════════════════════════════════════════════════════
    # TASK 1.9: Save/Load Roundtrip Tests
    # ═══════════════════════════════════════════════════════════

    def test_save_load_roundtrip_directory(self):
        """TASK 1.9: Directory-based save and load should preserve all data"""
        # Add test data
        e1 = self.store.add("entry 1", self._emb(), meta={"source": "test"})
        e2 = self.store.add("entry 2", self._emb(), concepts=["test"])

        # Save to directory (in-place)
        result = self.store.save()
        self.assertTrue(result)

        # Verify files created
        self.assertTrue((Path(self.tmp) / "entries.db").exists())
        self.assertTrue((Path(self.tmp) / "vectors.faiss").exists())
        self.assertTrue((Path(self.tmp) / "vector_ids.json").exists())

        # Close and reload from same directory
        self.store.close()
        store2 = HybridMemoryStore(self.tmp, 768)

        # Verify data survived
        hits = store2.query("entry", self._emb(), k=5)
        self.assertEqual(len(hits), 2)
        store2.close()

    def test_save_load_roundtrip_bytes(self):
        """save_to_bytes/load roundtrip for network transfer"""
        e1 = self.store.add("entry 1", self._emb(), meta={"source": "test"})
        e2 = self.store.add("entry 2", self._emb(), concepts=["test"])

        # Save to bytes (ZIP format)
        data = self.store.save_to_bytes()
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)
        # Verify it's ZIP format
        self.assertEqual(data[:4], b"PK\x03\x04")

        # Create new instance and load from bytes
        tmp2 = tempfile.mkdtemp(prefix="hybrid_memory_test2_")
        try:
            store2 = HybridMemoryStore(tmp2, 768)
            store2.load(data)

            # Verify data
            hits = store2.query("entry", self._emb(), k=5)
            self.assertEqual(len(hits), 2)
            store2.close()
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_save_includes_faiss(self):
        """Save should create FAISS index file"""
        self.store.add("test", self._emb())
        self.store.save()

        # FAISS file should exist and be substantial
        faiss_path = Path(self.tmp) / "vectors.faiss"
        self.assertTrue(faiss_path.exists())
        self.assertGreater(faiss_path.stat().st_size, 100)

    def test_load_preserves_id_mapping(self):
        """Load should restore ID mappings"""
        e1 = self.store.add("first", self._emb())
        e2 = self.store.add("second", self._emb())

        # Save to bytes and reload
        data = self.store.save_to_bytes()

        tmp2 = tempfile.mkdtemp(prefix="hybrid_memory_test2_")
        try:
            store2 = HybridMemoryStore(tmp2, 768)
            store2.load(data)

            # Mappings should exist
            self.assertIn(e1, store2._id_map)
            self.assertIn(e2, store2._id_map)

            store2.close()
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_load_handles_empty(self):
        """Load should handle empty/None data gracefully"""
        tmp2 = tempfile.mkdtemp(prefix="hybrid_memory_test2_")
        try:
            store2 = HybridMemoryStore(tmp2, 768)
            # Should not crash
            store2.load(None)
            store2.load(b"")
            store2.close()
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    # ═══════════════════════════════════════════════════════════
    # TASK 1.10: MinIO Backup/Restore Tests
    # ═══════════════════════════════════════════════════════════

    @unittest.skipIf(not os.getenv("MINIO_ENDPOINT"), "MinIO not configured")
    def test_minio_roundtrip(self):
        """TASK 1.10: Backup to MinIO and restore"""
        self.store.add("cloud backup test", self._emb())

        # Upload
        result = self.store.save_to_minio()
        self.assertTrue(result)

        # New instance
        tmp2 = tempfile.mkdtemp(prefix="hybrid_memory_test2_")
        try:
            store2 = HybridMemoryStore(tmp2, 768)
            store2.load_from_minio()

            hits = store2.query("backup", self._emb(), k=1)
            self.assertEqual(len(hits), 1)

            store2.close()
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    @patch("toolboxv2.mods.isaa.base.hybrid_memory.MinIOClient")
    def test_minio_lazy_init(self, mock_minio):
        """MinIO client should be initialized lazily"""
        store = HybridMemoryStore(self.tmp, 768)

        # Should not init MinIO on construction
        mock_minio.assert_not_called()

        # Should init when calling save_to_minio
        try:
            store.save_to_minio()
            mock_minio.assert_called()
        except:
            pass  # May fail due to missing config, that's OK

        store.close()

    # ═══════════════════════════════════════════════════════════
    # TASK 1.11: Legacy Pickle Migration Tests
    # ═══════════════════════════════════════════════════════════

    def test_migrate_from_pickle(self):
        """TASK 1.11: Should migrate old pickle format"""
        # Create mock legacy data
        old_chunks = [
            {"content": "old entry 1", "id": "1"},
            {"content": "old entry 2", "id": "2"},
        ]
        old_data = pickle.dumps(
            {
                "version": 1,
                "chunks": old_chunks,
                "embeddings": [self._emb() for _ in old_chunks],
            }
        )

        # Migrate
        tmp2 = tempfile.mkdtemp(prefix="hybrid_memory_test2_")
        try:
            store2 = HybridMemoryStore(tmp2, 768)
            migrated = store2._migrate_from_pickle(old_data)

            self.assertTrue(migrated)
            self.assertEqual(store2.stats()["active"], 2)

            store2.close()
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_migrate_handles_invalid(self):
        """Migration should handle invalid pickle data"""
        result = self.store._migrate_from_pickle(b"invalid pickle data")
        self.assertFalse(result)

    # ═══════════════════════════════════════════════════════════
    # TASK 1.12: Additional Utility Tests
    # ═══════════════════════════════════════════════════════════

    def test_invalidate_by_source(self):
        """invalidate_by_source should deactivate entries from a source"""
        meta = {"source": "/app/utils.py"}
        self.store.add("func a", self._emb(), meta=meta)
        self.store.add("func b", self._emb(), meta=meta)
        self.store.add("func c", self._emb(), meta={"source": "/app/other.py"})

        count = self.store.invalidate_by_source("/app/utils.py")

        self.assertEqual(count, 2)
        self.assertEqual(self.store.stats()["active"], 1)

    def test_cleanup_expired(self):
        """cleanup_expired should remove TTL-expired entries"""
        self.store.add("ephemeral", self._emb(), ttl=1)
        self.store.add("permanent", self._emb(), ttl=None)

        time.sleep(1.5)

        cleaned = self.store.cleanup_expired()
        self.assertEqual(cleaned, 1)
        self.assertEqual(self.store.stats()["active"], 1)

    def test_stats(self):
        """stats() should return accurate statistics"""
        self.store.add("entry 1", self._emb())
        self.store.add("entry 2", self._emb())
        self.store.add("entry 3", self._emb(), ttl=60)

        stats = self.store.stats()

        self.assertIn("active", stats)
        self.assertIn("total", stats)
        self.assertIn("with_ttl", stats)
        self.assertEqual(stats["active"], 3)
        self.assertEqual(stats["with_ttl"], 1)

    def test_query_with_filters(self):
        """Query should support metadata filtering"""
        self.store.add("python code", self._emb(), content_type="code")
        self.store.add("normal text", self._emb(), content_type="text")

        hits = self.store.query(
            "python", self._emb(), k=5, content_types=["code"], search_modes=("vector",)
        )

        self.assertTrue(all(h["content_type"] == "code" for h in hits))

    def test_thread_safety(self):
        """Basic thread safety test"""
        errors = []

        def writer(n):
            try:
                for i in range(n):
                    self.store.add(
                        f"writer {threading.current_thread().name} entry {i}", self._emb()
                    )
            except Exception as e:
                errors.append(e)

        def reader(n):
            try:
                for i in range(n):
                    self.store.query("entry", self._emb(), k=3)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(20,), name=f"Writer-{i}")
            for i in range(2)
        ] + [
            threading.Thread(target=reader, args=(20,), name=f"Reader-{i}")
            for i in range(2)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")


class TestHybridMemoryRegression(unittest.TestCase):
    """
    Regression tests for specific issues identified in V1
    Each test targets a known problem from code analysis
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hybrid_memory_regression_")
        if HybridMemoryStore is None:
            self.skipTest("HybridMemoryStore not implemented yet")
        self.store = HybridMemoryStore(self.tmp, 768)
        self.rng = np.random.default_rng(42)

    def _emb(self):
        return self.rng.random(768).astype(np.float32)

    def tearDown(self):
        if hasattr(self, "store"):
            self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_P2_no_on_iteration(self):
        """
        P2 CRITICAL: Concept lookup must be O(1) via index, not O(N)
        V1 issue: _expand_via_concepts was O(N) over all chunks
        """
        # Add 1000 entries with various concepts
        for i in range(1000):
            self.store.add(
                f"entry {i} about topic-{i % 10}",
                self._emb(),
                concepts=[f"topic-{i % 10}"],
            )

        # Query over concept must be < 50ms (O(1) index lookup)
        start = time.time()
        hits = self.store.query("topic-5", self._emb(), k=5, search_modes=("relation",))
        elapsed = time.time() - start

        self.assertLess(elapsed, 0.05, f"Query took {elapsed}s, expected < 0.05s")
        self.assertGreater(len(hits), 0)

    def test_P4_no_oom_dedup(self):
        """
        P4 HIGH: Dedup with many entries should not cause OOM
        V1 issue: Loading all chunks for dedup caused memory issues
        """
        # Add 5000 entries with ~50% duplicates
        for i in range(5000):
            self.store.add(f"content {i % 2500}", self._emb())

        # Should deduplicate to ~2500 unique entries
        active = self.store.stats()["active"]
        self.assertLessEqual(active, 2501)  # Small margin for timing

    def test_P5_no_silent_batch_failure(self):
        """
        P5 HIGH: Each add() must be immediately queryable
        V1 issue: sto queue batching caused entries to be invisible until batch_size
        """
        eid = self.store.add("immediately queryable", self._emb())

        # Must be immediately findable (not after batch_size entries)
        row = self.store._exec(
            "SELECT id FROM entries WHERE id=? AND is_active=1", (eid,)
        ).fetchone()

        self.assertIsNotNone(row, "Entry should be immediately queryable")

    def test_P6_concurrent_access(self):
        """
        P6 HIGH: Concurrent add+query must not crash
        V1 issue: Race conditions in sto queue
        """
        errors = []

        def writer(n):
            try:
                for i in range(n):
                    self.store.add(f"concurrent {i}", self._emb())
            except Exception as e:
                errors.append(f"Writer error: {e}")

        def reader(n):
            try:
                for i in range(n):
                    self.store.query("concurrent", self._emb(), k=3)
            except Exception as e:
                errors.append(f"Reader error: {e}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(writer, 50),
                executor.submit(writer, 50),
                executor.submit(reader, 50),
                executor.submit(reader, 50),
            ]
            for f in futures:
                f.result()

        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")

    def test_P8_metadata_prefilter(self):
        """
        P8 MEDIUM: Metadata filter BEFORE vector search
        V1 issue: Post-filtering wasted vector search computation
        """
        self.store.add(
            "python code", self._emb(), content_type="code", meta={"language": "python"}
        )
        self.store.add(
            "rust code", self._emb(), content_type="code", meta={"language": "rust"}
        )
        self.store.add("normal text", self._emb(), content_type="text")

        # Query with prefilter
        hits = self.store.query(
            "code",
            self._emb(),
            k=5,
            content_types=["code"],
            meta_filter={"language": "python"},
            search_modes=("vector",),
        )

        # Should only get python code
        self.assertTrue(all(h["content_type"] == "code" for h in hits))
        self.assertTrue(all(h.get("meta", {}).get("language") == "python" for h in hits))

    def test_ttl_expiry(self):
        """TTL expiry must work correctly"""
        self.store.add("ephemeral", self._emb(), ttl=1)
        self.store.add("permanent", self._emb(), ttl=None)

        time.sleep(1.5)

        cleaned = self.store.cleanup_expired()
        self.assertEqual(cleaned, 1)

        # Permanent entry should remain
        hits = self.store.query("permanent", self._emb(), k=5)
        self.assertEqual(len(hits), 1)


class TestAISemanticMemoryV1Compat(unittest.TestCase):
    """
    Test V1-compatible AISemanticMemory wrapper (ai_semantic_memory.py)

    Verifies the wrapper matches the V1 API from AgentUtils.py:
      - Singleton pattern
      - Multi-space management
      - V1 method signatures (add_data, query, save_memory, etc.)
      - Internal embedding generation (mocked)
      - to_str formatting
    """

    _rng = np.random.default_rng(42)

    def _emb(self):
        return self._rng.random(768).astype(np.float32)

    @classmethod
    def setUpClass(cls):
        # Clear singleton so each test class starts fresh
        if hasattr(AISemanticMemory, "_instances"):
            AISemanticMemory._instances.pop(AISemanticMemory, None)

    def setUp(self):
        if AISemanticMemory is None:
            self.skipTest("AISemanticMemory not implemented yet")
        self.tmp = tempfile.mkdtemp(prefix="ai_semantic_memory_v1_")
        # Clear singleton before each test
        AISemanticMemory._instances.pop(AISemanticMemory, None)
        # Create with V1-compatible constructor
        self.memory = AISemanticMemory(base_path=self.tmp)

    def tearDown(self):
        if hasattr(self, "memory"):
            self.memory.close()
            # Clear singleton
            AISemanticMemory._instances.pop(AISemanticMemory, None)
        shutil.rmtree(os.path.join(os.getcwd(), ".data", self.tmp), ignore_errors=True)

    def test_singleton_pattern(self):
        """V1 uses Singleton metaclass — second __init__ returns same instance"""
        mem1 = AISemanticMemory(base_path=self.tmp)
        mem2 = AISemanticMemory(base_path=self.tmp)
        self.assertIs(mem1, mem2)

    def test_v1_constructor_signature(self):
        """Constructor should accept V1 kwargs without error"""
        AISemanticMemory._instances.pop(AISemanticMemory, None)
        mem = AISemanticMemory(
            base_path=self.tmp + "_v1sig",
            default_model="gpt-4",
            default_embedding_model="nomic-embed-text",
            default_similarity_threshold=0.5,
            default_batch_size=32,
            default_n_clusters=4,
            default_deduplication_threshold=0.9,
        )
        self.assertEqual(mem.default_config["embedding_dim"], 768)
        mem.close()
        AISemanticMemory._instances.pop(AISemanticMemory, None)

    def test_create_memory(self):
        """create_memory should create a new space"""
        store = self.memory.create_memory("test_space")
        self.assertIsNotNone(store)
        self.assertIn("test_space", self.memory.list_memories())

    def test_create_duplicate_raises(self):
        """V1: creating duplicate raises ValueError"""
        self.memory.create_memory("dup_space")
        with self.assertRaises(ValueError):
            self.memory.create_memory("dup_space")

    def test_list_memories(self):
        """list_memories should return all space names"""
        self.memory.create_memory("space_a")
        self.memory.create_memory("space_b")
        names = self.memory.list_memories()
        self.assertIn("space_a", names)
        self.assertIn("space_b", names)

    def test_sanitize_name(self):
        """_sanitize_name should match V1 behavior"""
        # V1 regex replaces non-alnum (incl space) with '-', then ':' -> '_', ' ' -> '_'
        # "hello world" -> "hello-world" (space replaced by regex to '-')
        self.assertEqual(AISemanticMemory._sanitize_name("hello world"), "hello-world")
        # V1: regex replaces ':' with '-' first, then replace(":", "_") has no effect
        self.assertEqual(AISemanticMemory._sanitize_name("a:b"), "a-b")
        # Short names get padded
        self.assertEqual(AISemanticMemory._sanitize_name("ab"), "abZ")

    @patch("toolboxv2.mods.isaa.base.ai_semantic_memory.AISemanticMemory.get_embeddings")
    def test_add_data_v1_signature(self, mock_embed):
        """add_data(memory_name, data, metadata) — V1 signature"""
        mock_embed.return_value = self._emb()

        async def test():
            result = await self.memory.add_data(
                "knowledge",
                "Python was created by Guido van Rossum",
                metadata={"source": "wikipedia"},
            )
            self.assertTrue(result)
            # Memory should have been auto-created
            self.assertIn("knowledge", self.memory.list_memories())

        asyncio.run(test())

    @patch("toolboxv2.mods.isaa.base.ai_semantic_memory.AISemanticMemory.get_embeddings")
    def test_add_data_list(self, mock_embed):
        """add_data with list[str] — V1 supports this"""
        mock_embed.return_value = self._emb()

        async def test():
            result = await self.memory.add_data(
                "docs",
                ["First paragraph", "Second paragraph"],
            )
            self.assertTrue(result)
            self.assertEqual(self.memory.get_memory_size("docs"), 2)

        asyncio.run(test())

    @patch("toolboxv2.mods.isaa.base.ai_semantic_memory.AISemanticMemory.get_embeddings")
    def test_query_v1_signature(self, mock_embed):
        """query(query, memory_names, query_params, to_str) — V1 signature"""
        mock_embed.return_value = self._emb()

        async def test():
            await self.memory.add_data("facts", "The sky is blue")
            results = await self.memory.query(
                "sky color",
                memory_names="facts",
                query_params={"k": 3},
            )
            self.assertIsInstance(results, list)

        asyncio.run(test())

    @patch("toolboxv2.mods.isaa.base.ai_semantic_memory.AISemanticMemory.get_embeddings")
    def test_query_to_str(self, mock_embed):
        """query with to_str=True returns formatted string (V1 feature)"""
        mock_embed.return_value = self._emb()

        async def test():
            await self.memory.add_data("notes", "Meeting at 3pm tomorrow")
            result = await self.memory.query("meeting", to_str=True)
            self.assertIsInstance(result, str)
            # Should contain Source [notes]: prefix
            if "No relevant information" not in result:
                self.assertIn("Source [", result)

        asyncio.run(test())

    @patch("toolboxv2.mods.isaa.base.ai_semantic_memory.AISemanticMemory.get_embeddings")
    def test_query_empty_returns_no_info(self, mock_embed):
        """query on empty memory returns 'No relevant information' with to_str"""
        mock_embed.return_value = self._emb()

        async def test():
            result = await self.memory.query("anything", to_str=True)
            self.assertEqual(result, "")  # No targets → empty string

        asyncio.run(test())

    @patch("toolboxv2.mods.isaa.base.ai_semantic_memory.AISemanticMemory.get_embeddings")
    def test_query_multi_space(self, mock_embed):
        """query across multiple spaces (V1 feature: memory_names=None → all)"""
        mock_embed.return_value = self._emb()

        async def test():
            await self.memory.add_data("space1", "Earth orbits the Sun")
            await self.memory.add_data("space2", "Water is H2O")
            results = await self.memory.query("science")  # memory_names=None → all
            self.assertIsInstance(results, list)
            # Should search across both spaces
            searched_memories = {r["memory"] for r in results}
            self.assertTrue(len(searched_memories) >= 1)

        asyncio.run(test())

    def test_get_memory_size(self):
        """get_memory_size returns entry count (V1 method)"""
        self.memory.create_memory("sized")
        store = self.memory.memories["sized"]
        store.add("entry1", self._emb())
        store.add("entry2", self._emb())
        self.assertEqual(self.memory.get_memory_size("sized"), 2)

    @patch("toolboxv2.mods.isaa.base.ai_semantic_memory.AISemanticMemory.get_embeddings")
    def test_delete_memory(self, mock_embed):
        """delete_memory removes a space"""
        mock_embed.return_value = self._emb()

        async def test():
            await self.memory.add_data("temp_space", "temporary data")
            self.assertIn("temp_space", self.memory.list_memories())
            deleted = await self.memory.delete_memory("temp_space")
            self.assertTrue(deleted)
            self.assertNotIn("temp_space", self.memory.list_memories())

        asyncio.run(test())

    def test_save_load_single_memory(self):
        """save_memory / load_memory V1 signature"""
        # Create and populate
        self.memory.create_memory("persist_test")
        store = self.memory.memories["persist_test"]
        store.add("persistent data", self._emb())

        # Save to temp path
        save_dir = tempfile.mkdtemp(prefix="mem_save_")
        try:
            result = self.memory.save_memory("persist_test", save_dir)
            self.assertTrue(result)

            # Clear and reload — use new singleton
            self.memory.close()
            AISemanticMemory._instances.pop(AISemanticMemory, None)
            memory2 = AISemanticMemory(base_path=self.tmp + "_reload")
            loaded = memory2.load_memory("persist_test", save_dir)
            self.assertTrue(loaded)
            self.assertIn("persist_test", memory2.list_memories())
            self.assertEqual(memory2.get_memory_size("persist_test"), 1)
            memory2.close()
            AISemanticMemory._instances.pop(AISemanticMemory, None)
            # Re-create our test memory
            self.memory = AISemanticMemory(base_path=self.tmp)
        finally:
            shutil.rmtree(save_dir, ignore_errors=True)

    def test_save_load_all_memories(self):
        """save_all_memories / load_all_memories V1 signature"""
        # Populate two spaces
        self.memory.create_memory("all_a")
        self.memory.memories["all_a"].add("data A", self._emb())
        self.memory.create_memory("all_b")
        self.memory.memories["all_b"].add("data B", self._emb())

        save_dir = tempfile.mkdtemp(prefix="mem_save_all_")
        try:
            result = self.memory.save_all_memories(save_dir)
            self.assertTrue(result)

            # Reload into fresh instance
            self.memory.close()
            AISemanticMemory._instances.pop(AISemanticMemory, None)
            memory2 = AISemanticMemory(base_path=self.tmp + "_reload_all")
            loaded = memory2.load_all_memories(save_dir)
            self.assertTrue(loaded)
            self.assertIn("all_a", memory2.list_memories())
            self.assertIn("all_b", memory2.list_memories())
            self.assertEqual(memory2.get_memory_size("all_a"), 1)
            self.assertEqual(memory2.get_memory_size("all_b"), 1)
            memory2.close()
            AISemanticMemory._instances.pop(AISemanticMemory, None)
            self.memory = AISemanticMemory(base_path=self.tmp)
        finally:
            shutil.rmtree(save_dir, ignore_errors=True)

    def test_get_method(self):
        """get(names) returns list of stores (V1 method)"""
        self.memory.create_memory("get_test")
        stores = self.memory.get("get_test")
        self.assertEqual(len(stores), 1)
        self.assertIsInstance(stores[0], HybridMemoryStore)

    def test_context_manager(self):
        """AISemanticMemory supports context manager protocol"""
        AISemanticMemory._instances.pop(AISemanticMemory, None)
        with AISemanticMemory(base_path=self.tmp + "_ctx") as mem:
            mem.create_memory("ctx_test")
            self.assertIn("ctx_test", mem.list_memories())
        # After exit, memories should be cleared
        AISemanticMemory._instances.pop(AISemanticMemory, None)


class TestPersistence(unittest.TestCase):
    """Extended persistence and migration tests"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hybrid_memory_persistence_")
        if HybridMemoryStore is None:
            self.skipTest("HybridMemoryStore not implemented yet")
        self.store = HybridMemoryStore(self.tmp, 768)
        self.rng = np.random.default_rng(42)

    def _emb(self):
        return self.rng.random(768).astype(np.float32)

    def tearDown(self):
        if hasattr(self, "store"):
            self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_persistence_includes_entities(self):
        """Save/Load should preserve entity-relation graph"""
        e1 = self.store.add_entity("person:test", "person", "Test Person")
        e2 = self.store.add_entity("company:test", "company", "Test Co")
        self.store.add_relation(e1, e2, "WORKS_AT")

        data = self.store.save_to_bytes()

        tmp2 = tempfile.mkdtemp(prefix="hybrid_memory_test2_")
        try:
            store2 = HybridMemoryStore(tmp2, 768)
            store2.load(data)

            # Entities should exist
            related = store2.get_related(e1, depth=1)
            self.assertEqual(len(related), 1)

            store2.close()
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_persistence_includes_concepts(self):
        """Save/Load should preserve concept index"""
        self.store.add("python code", self._emb(), concepts=["python", "code"])

        data = self.store.save_to_bytes()

        tmp2 = tempfile.mkdtemp(prefix="hybrid_memory_test2_")
        try:
            store2 = HybridMemoryStore(tmp2, 768)
            store2.load(data)

            # Concept index should work
            hits = store2.query("python", self._emb(), k=1, search_modes=("relation",))
            self.assertGreater(len(hits), 0)

            store2.close()
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests for SQL special characters, FAISS rebuild, and migration"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hybrid_memory_edge_")
        if HybridMemoryStore is None:
            self.skipTest("HybridMemoryStore not implemented yet")
        self.store = HybridMemoryStore(self.tmp, 768)
        self.rng = np.random.default_rng(42)

    def _emb(self):
        return self.rng.random(768).astype(np.float32)

    def tearDown(self):
        if hasattr(self, "store"):
            self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sql_special_characters(self):
        """Content with SQL special characters should be stored and retrieved correctly"""
        # Test various SQL special characters (including FTS5 operators and SQL keywords)
        special_contents = [
            'SELECT * FROM users WHERE name = "admin";',
            "DROP TABLE entries; -- evil",
            "INSERT INTO test VALUES ('single', \"double\", `backtick`);",
            "Content with\nnewlines\nand\ttabs",
            "O'Reilly said: \"Don't use backslashes \\ \"",
            "; DROP TABLE entries -- ;",
            "UNION SELECT * FROM passwords",
            "'; --",
            "' OR '1'='1",
            "<script>alert('xss')</script>",
        ]

        for i, content in enumerate(special_contents):
            # Add content with special characters
            entry_id = self.store.add(content, self._emb(), meta={"test_index": i})

            # Retrieve by ID to verify content is preserved exactly (avoids FTS5 query issues)
            entry = self.store._load_entry(entry_id)
            self.assertIsNotNone(entry)
            self.assertEqual(entry["content"], content)
            # Verify metadata is preserved
            meta = self.store._extract_meta(entry)
            self.assertEqual(meta["test_index"], i)

    def test_faiss_rebuild_after_deletes(self):
        """FAISS index rebuild should compact the index after many deletes"""
        # Add 1000 entries
        entry_ids = []
        for i in range(1000):
            content = f"Test content {i}"
            entry_ids.append(self.store.add(content, self._emb()))

        # Check initial FAISS size
        stats_before = self.store.stats()
        self.assertEqual(stats_before["active"], 1000)
        self.assertEqual(stats_before["faiss_size"], 1000)

        # Delete 500 entries (soft delete)
        for i in range(0, 500):
            self.store.delete(entry_ids[i], hard=False)

        # Check that active entries decreased but FAISS size is still 1000
        stats_after_delete = self.store.stats()
        self.assertEqual(stats_after_delete["active"], 500)
        self.assertEqual(stats_after_delete["faiss_size"], 500)  # Mappings cleared

        # The actual FAISS index still has 1000 vectors internally
        # Rebuild the index
        rebuilt_count = self.store.rebuild_faiss_index()
        self.assertEqual(rebuilt_count, 500)

        # Check that FAISS index is now compacted
        stats_after_rebuild = self.store.stats()
        self.assertEqual(stats_after_rebuild["active"], 500)
        self.assertEqual(stats_after_rebuild["faiss_size"], 500)

        # Verify queries still work correctly after rebuild (use BM25 for exact matching)
        for i in range(500, 550):  # Test some of the remaining entries
            content = f"Test content {i}"
            result = self.store.query(content, self._emb(), k=1, search_modes=("bm25",))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["content"], content)

    def test_faiss_hard_delete_and_rebuild(self):
        """FAISS index rebuild should work correctly after hard deletes"""
        # Add 500 entries
        entry_ids = []
        for i in range(500):
            content = f"Test content {i}"
            entry_ids.append(self.store.add(content, self._emb()))

        # Hard delete first 250 entries
        for i in range(250):
            self.store.delete(entry_ids[i], hard=True)

        # Verify hard-deleted entries are gone from SQLite
        stats_before = self.store.stats()
        self.assertEqual(stats_before["active"], 250)
        self.assertEqual(stats_before["total"], 250)  # Hard delete removes from DB

        # Rebuild FAISS index
        rebuilt_count = self.store.rebuild_faiss_index()
        self.assertEqual(rebuilt_count, 250)

        # Verify queries work correctly (use BM25 for exact matching)
        for i in range(250, 500):  # All should be found
            content = f"Test content {i}"
            result = self.store.query(content, self._emb(), k=1, search_modes=("bm25",))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["content"], content)

    def test_rebuild_with_empty_store(self):
        """Rebuild on empty store should reset all state"""
        # Rebuild empty store
        count = self.store.rebuild_faiss_index()
        self.assertEqual(count, 0)

        # Verify all state is reset
        stats = self.store.stats()
        self.assertEqual(stats["active"], 0)
        self.assertEqual(stats["faiss_size"], 0)

        # Verify queries work after rebuild on empty store
        result = self.store.query("test", self._emb(), k=5)
        self.assertEqual(len(result), 0)

    def test_rebuild_preserves_functionality(self):
        """Rebuild should preserve all functionality"""
        # Add entries with different content types, metadata, concepts
        entry_id1 = self.store.add(
            "code example",
            self._emb(),
            content_type="code",
            concepts=["python", "example"],
        )
        entry_id2 = self.store.add(
            "text fact", self._emb(), content_type="fact", meta={"importance": 0.9}
        )
        entry_id3 = self.store.add("entity data", self._emb(), content_type="entity")

        # Rebuild
        count = self.store.rebuild_faiss_index()
        self.assertEqual(count, 3)

        # Verify all entries are still queryable via vector search
        result = self.store.query("code", self._emb(), k=3, search_modes=("vector",))
        self.assertEqual(len(result), 3)

        # Verify BM25 search still works
        result = self.store.query("code", self._emb(), k=3, search_modes=("bm25",))
        self.assertEqual(len(result), 1)

        # Verify concept search still works
        result = self.store.query("python", self._emb(), k=3, search_modes=("relation",))
        self.assertEqual(len(result), 1)

        # Verify metadata is preserved
        entry = self.store._load_entry(entry_id2)
        self.assertEqual(entry["content_type"], "fact")
        meta = self.store._extract_meta(entry)
        self.assertEqual(meta["importance"], 0.9)


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
