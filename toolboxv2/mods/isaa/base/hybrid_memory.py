"""
HybridMemoryStore - Agent Memory V2 Implementation

Combines SQLite (metadata + FTS5 + relations) + FAISS (vectors) + MinIO (backup)
Following TDD approach and DRY principles - reusing patterns from:
- mobile_db.py (thread-local connections, WAL mode)
- blob_instance.py (MinIO integration)
- FaissVectorStore (vector operations)

Optimized for:
- Raw, changing data (code)
- Concept relations
- Live cycle management
- Discrete relationship connections
"""

import hashlib
import json
import os
import pickle
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# MinIO client import (optional, for cloud backup)
try:
    from minio import Minio as MinIOClient
except ImportError:
    MinIOClient = None

try:
    from toolboxv2.mods.isaa.base.VectorStores.FaissVectorStore import FaissVectorStore
except ImportError:
    raise ImportError(
        "HybridMemoryStore requires faiss-cpu or faiss-gpu. "
        "Install with: pip install faiss-cpu"
    )
from toolboxv2.mods.isaa.base.VectorStores.types import Chunk

# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDED SCHEMA (memory_store.sql)
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- Agent Memory V2 Schema — SQLite mit FTS5

CREATE TABLE IF NOT EXISTS entries (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content_type TEXT DEFAULT 'text',  -- text|code|fact|entity
    version     INTEGER DEFAULT 1,
    supersedes  TEXT,                  -- ID der vorherigen Version
    is_active   INTEGER DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    ttl         INTEGER,              -- Sekunden bis Ablauf, NULL=kein Ablauf
    access_count INTEGER DEFAULT 0,
    space       TEXT NOT NULL DEFAULT 'default',
    -- Flattened StructuredMeta (statt free dict → indexierbare Spalten)
    meta_role   TEXT,                 -- user|assistant|system|tool
    meta_source TEXT,                 -- file path, url
    meta_language TEXT,               -- python|rust|de|en
    meta_category TEXT,
    meta_importance REAL DEFAULT 0.5,
    meta_custom TEXT                  -- JSON für Rest
);

CREATE INDEX IF NOT EXISTS idx_entries_space   ON entries(space, is_active);
CREATE INDEX IF NOT EXISTS idx_entries_hash    ON entries(content_hash);
CREATE INDEX IF NOT EXISTS idx_entries_source  ON entries(meta_source) WHERE meta_source IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entries_type    ON entries(content_type, is_active);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,     -- "company:spacex", "person:elon"
    entity_type TEXT NOT NULL,        -- person|company|project|location|module
    name        TEXT NOT NULL,
    meta        TEXT,                 -- JSON
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS relations (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    rel_type    TEXT NOT NULL,        -- WORKS_AT|DEPENDS_ON|LOCATED_IN|PART_OF
    weight      REAL DEFAULT 1.0,
    meta        TEXT,
    created_at  REAL NOT NULL,
    PRIMARY KEY (source_id, target_id, rel_type)
);
CREATE INDEX IF NOT EXISTS idx_rel_src ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_tgt ON relations(target_id);

-- Inverted Index: Concept → Entry (ersetzt O(N) _expand_via_concepts)
CREATE TABLE IF NOT EXISTS concept_index (
    concept     TEXT NOT NULL,
    entry_id    TEXT NOT NULL REFERENCES entries(id),
    PRIMARY KEY (concept, entry_id)
);
CREATE INDEX IF NOT EXISTS idx_concept_entry ON concept_index(entry_id);

-- FTS5 für BM25 Full-Text Search
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    content,
    entry_id UNINDEXED,
    space UNINDEXED,
    tokenize='porter unicode61'
);
"""


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID MEMORY STORE
# ═══════════════════════════════════════════════════════════════════════════════


class HybridMemoryStore:
    """
    Hybrid Index: SQLite(Meta+FTS5+Relations) + FAISS(Vectors) + MinIO(Backup)

    Optimized for:
    - Raw, changing data (code)
    - Concept relations
    - Live cycle management
    - Discrete relationship connections

    Thread-safe via thread-local connections (pattern from mobile_db.py)
    """

    def __init__(self, db_dir: str, embedding_dim: int = 768, space: str = "default"):
        """
        Initialize HybridMemoryStore

        Args:
            db_dir: Directory for SQLite database
            embedding_dim: Dimension of embeddings (default 768)
            space: Namespace for entries (default 'default')
        """
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "entries.db"
        self.space = space
        self.dim = embedding_dim

        # Thread-Safety — Pattern from mobile_db.py:161
        self._local = threading.local()
        self._lock = threading.RLock()

        # FAISS — Reuse existing FaissVectorStore (no wrapper!)
        self._faiss = FaissVectorStore(embedding_dim)
        self._id_map = {}  # entry_id → faiss_idx
        self._idx_map = {}  # faiss_idx → entry_id
        self._next_idx = 0

        # MinIO — lazy init
        self._minio = None
        self._minio_bucket = "tb-agent-memory"
        self._minio_config = None

        # Initialize database
        self._init_db()

    # ════════════════════ SQLite Layer ════════════════════
    # Pattern from mobile_db.py:169-193

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection with WAL mode"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    @contextmanager
    def _tx(self):
        """Transaction context manager with auto-commit/rollback"""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _exec(self, sql: str, params: tuple = ()):
        """Execute SQL and return cursor"""
        return self._get_conn().execute(sql, params)

    def _init_db(self):
        """Initialize database schema from embedded SQL"""
        with self._tx() as conn:
            conn.executescript(SCHEMA_SQL)

        # Load FAISS + ID-Maps if they exist (v3 native or legacy pickle)
        self._load_faiss_from_dir()

    # ════════════════════ CRUD Operations ════════════════════

    def add(
        self,
        content: str,
        embedding: np.ndarray,
        content_type: str = "text",
        meta: Optional[Dict[str, Any]] = None,
        concepts: Optional[List[str]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """
        Atomic Add: Hash→Dedup→SQLite→FTS5→FAISS→ConceptIndex

        Args:
            content: Text content
            embedding: Vector embedding
            content_type: text|code|fact|entity
            meta: Metadata dict
            concepts: List of concept strings
            ttl: Time-to-live in seconds

        Returns:
            entry_id: 16-character hex ID

        Optimized for raw, changing data - immediately queryable
        """
        if meta is not None and ttl is None:
            ttl = meta.get("ttl")
        # Compute hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check for duplicates (P4: No OOM on dedup)
        existing = self._exec(
            "SELECT id FROM entries WHERE content_hash=? AND is_active=1 AND space=?",
            (content_hash, self.space),
        ).fetchone()

        if existing:
            return existing[0]  # Return existing ID

        # Generate new ID
        entry_id = uuid.uuid4().hex[:16]
        now = time.time()
        meta = meta or {}

        # Extract standard metadata fields
        meta_role = meta.get("role")
        meta_source = meta.get("source")
        meta_language = meta.get("language")
        meta_category = meta.get("category")
        meta_importance = meta.get("importance", 0.5)

        # Store ALL non-standard fields in meta_custom (Fix #1)
        standard_fields = {
            "role",
            "source",
            "language",
            "category",
            "importance",
            "custom",
        }
        custom_fields = {k: v for k, v in meta.items() if k not in standard_fields}
        meta_custom = json.dumps(custom_fields) if custom_fields else None

        with self._lock:
            with self._tx() as conn:
                # 1. Insert into SQLite entries table
                conn.execute(
                    """
                    INSERT INTO entries (
                        id, content, content_hash, content_type,
                        created_at, updated_at, ttl, space,
                        meta_role, meta_source, meta_language, meta_category,
                        meta_importance, meta_custom
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry_id,
                        content,
                        content_hash,
                        content_type,
                        now,
                        now,
                        ttl,
                        self.space,
                        meta_role,
                        meta_source,
                        meta_language,
                        meta_category,
                        meta_importance,
                        meta_custom,
                    ),
                )

                # 2. Insert into FTS5 for BM25 search
                conn.execute(
                    "INSERT INTO entries_fts(content, entry_id, space) VALUES (?, ?, ?)",
                    (content, entry_id, self.space),
                )

                # 3. Add concepts to concept_index (P2: O(1) lookup)
                if concepts:
                    for concept in concepts:
                        conn.execute(
                            "INSERT OR IGNORE INTO concept_index(concept, entry_id) VALUES (?, ?)",
                            (concept, entry_id),
                        )

                # 4. Add to FAISS with proper Chunk object
                faiss_idx = self._next_idx
                chunk = Chunk(
                    text=content,
                    embedding=embedding,
                    metadata={"id": entry_id},
                    content_hash=content_hash,
                )
                self._faiss.add_embeddings(embedding.reshape(1, -1), [chunk])
                self._id_map[entry_id] = faiss_idx
                self._idx_map[faiss_idx] = entry_id
                self._next_idx += 1

        return entry_id

    def query(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int = 5,
        search_modes: Tuple[str, ...] = ("vector", "bm25"),
        mode_weights: Optional[Dict[str, float]] = None,
        content_types: Optional[List[str]] = None,
        meta_filter: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0,
        space: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid query with proper RRF (Reciprocal Rank Fusion)

        Args:
            query_text: Query text for BM25 and concept extraction
            query_embedding: Query vector for FAISS
            k: Number of results
            search_modes: Tuple of "vector", "bm25", "relation"
            mode_weights: Dict of mode -> weight for RRF fusion (e.g. {"vector": 0.4, "bm25": 0.35, "relation": 0.25})
            content_types: Filter by content types (P8: Prefilter)
            meta_filter: Filter by metadata key-value pairs (P8: Prefilter)
            min_similarity: Minimum similarity threshold
            space: Override space (default: use instance space)

        Returns:
            List of result dicts with content, score, metadata
        """
        space = space or self.space
        entry_cache = {}  # entry_id -> loaded entry dict (avoid duplicate loads)
        ranked_lists = []  # list of (mode_name, [entry_id, ...]) for RRF
        query_text = query_text.replace("\\", "/").replace('"', '').replace("'", "")
        # Helper: check if entry passes filters
        def _passes_filters(entry: Dict) -> bool:
            if not entry or not entry["is_active"] or entry["space"] != space:
                return False
            if content_types and entry["content_type"] not in content_types:
                return False
            if meta_filter:
                meta = self._extract_meta(entry)
                if not all(meta.get(fkey) == fval for fkey, fval in meta_filter.items()):
                    return False
            return True

        def _ensure_cached(entry_id: str) -> Optional[Dict]:
            if entry_id not in entry_cache:
                entry_cache[entry_id] = self._load_entry(entry_id)
            return entry_cache.get(entry_id)

        # ── 1. Vector Search ──
        if "vector" in search_modes:
            vector_results = self._faiss.search(
                query_embedding,
                k=min(k * 3, 100),
                min_similarity=min_similarity,
            )
            vector_ranked = []
            for chunk in vector_results:
                entry_id = chunk.metadata.get("id")
                if entry_id:
                    entry = _ensure_cached(entry_id)
                    if entry and _passes_filters(entry):
                        vector_ranked.append(entry_id)
            ranked_lists.append(("vector", vector_ranked))

        # ── 2. BM25 via FTS5 ──
        if "bm25" in search_modes:
            safe_query = self._fts_escape(re.sub(r'[\\/"\'(){}\[\]^~*:!]', ' ', query_text).strip())
            _fts5_unsafe = re.compile(r'[\\/.:"\'(){}\[\]^~*!@#$&|<>=,;]')
            safe_query_text = _fts5_unsafe.sub(' ', query_text).strip()  # oder wie die Variable heißt
            safe_query_text = ' '.join(safe_query_text.split())  # doppelte Spaces entfernen
            if not safe_query_text:
                bm25_results = []
                print("No query text")
                from toolboxv2 import get_logger
                get_logger().error(f"No query text bm25_results len og query {len(query_text)} len save query {len(safe_query_text)}")
            else:
                bm25_results = self._exec(
                    """
                    SELECT entry_id, rank
                    FROM entries_fts
                    WHERE entries_fts MATCH ? AND space = ?
                    ORDER BY rank
                    LIMIT ?
                """,
                    (safe_query, space, k * 3),
                ).fetchall()

                bm25_ranked = []
                for row in bm25_results:
                    entry_id = row["entry_id"]
                    entry = _ensure_cached(entry_id)
                    if entry and _passes_filters(entry):
                        bm25_ranked.append(entry_id)
                ranked_lists.append(("bm25", bm25_ranked))

        # ── 3. Relation Search (query-specific concept matching) ──
        if "relation" in search_modes:
            # Extract query concepts: words > 3 chars, lowercased
            query_concepts = [w.lower() for w in query_text.split() if len(w) > 3]
            if query_concepts:
                placeholders = ",".join("?" * len(query_concepts))
                # Build WHERE clause for content_type prefilter
                type_clause = ""
                type_params = []
                if content_types:
                    type_placeholders = ",".join("?" * len(content_types))
                    type_clause = f" AND e.content_type IN ({type_placeholders})"
                    type_params = list(content_types)

                rel_rows = self._exec(
                    f"""
                    SELECT DISTINCT ci.entry_id
                    FROM concept_index ci
                    JOIN entries e ON e.id = ci.entry_id
                    WHERE ci.concept IN ({placeholders})
                      AND e.space = ? AND e.is_active = 1
                      {type_clause}
                    LIMIT ?
                """,
                    (*query_concepts, space, *type_params, k * 3),
                ).fetchall()

                rel_ranked = []
                for row in rel_rows:
                    entry_id = row["entry_id"]
                    entry = _ensure_cached(entry_id)
                    if entry and _passes_filters(entry):
                        rel_ranked.append(entry_id)
                ranked_lists.append(("relation", rel_ranked))

        # ── 4. RRF Fusion ──
        if not ranked_lists:
            return []

        fused = self._rrf_fuse(
            ranked_lists,
            entry_cache,
            k=k,
            mode_weights=mode_weights,
            is_code=any(t == "code" for t in (content_types or [])),
        )

        # ── 5. Update access_count for returned results ──
        if fused:
            ids = [r["id"] for r in fused]
            self._exec(
                f"UPDATE entries SET access_count = access_count + 1 WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            self._get_conn().commit()

        return fused

    def update(
        self,
        entry_id: str,
        new_content: str,
        new_embedding: np.ndarray,
        new_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Update entry with versioning

        Creates new version and marks old as inactive

        Returns:
            new_entry_id
        """
        # Load old entry
        old_entry = self._load_entry(entry_id)
        if not old_entry:
            raise ValueError(f"Entry {entry_id} not found")

        with self._lock:
            # Soft-delete old entry
            with self._tx() as conn:
                conn.execute("UPDATE entries SET is_active = 0 WHERE id = ?", (entry_id,))

            # Create new entry with version increment
            new_meta = new_meta or self._extract_meta(old_entry)
            new_meta["version"] = old_entry["version"] + 1

            new_id = self.add(
                content=new_content,
                embedding=new_embedding,
                content_type=old_entry["content_type"],
                meta=new_meta,
                ttl=old_entry["ttl"],
            )

            # Prüfen ob add() einen existierenden Eintrag zurückgegeben hat
            if new_id != entry_id:  # Nicht der alte Eintrag selbst
                existing_entry = self._load_entry(new_id)
                if existing_entry and existing_entry.get("supersedes") is None and existing_entry.get("version",
                                                                                                      1) == 1:
                    # Nur updaten wenn es wirklich unser neuer Eintrag ist, nicht ein Fremd-Eintrag
                    # Sicherste Lösung: content_hash erneut prüfen
                    new_hash = hashlib.sha256(new_content.encode()).hexdigest()
                    if existing_entry["content_hash"] == new_hash and existing_entry["created_at"] == existing_entry[
                        "updated_at"]:
                        with self._tx() as conn:
                            conn.execute(
                                "UPDATE entries SET supersedes = ?, version = ? WHERE id = ?",
                                (entry_id, old_entry["version"] + 1, new_id),
                            )

            # Set supersedes relationship and version
            with self._tx() as conn:
                conn.execute(
                    "UPDATE entries SET supersedes = ?, version = ? WHERE id = ?",
                    (entry_id, old_entry["version"] + 1, new_id),
                )

        return new_id

    def delete(self, entry_id: str, hard: bool = False):
        """
        Delete entry (soft by default, hard if specified)

        Args:
            entry_id: Entry to delete
            hard: If True, permanently remove from all indexes
        """
        with self._lock:
            if hard:
                # Remove from all indexes
                with self._tx() as conn:
                    # Remove from entries
                    conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))

                    # Remove from FTS
                    conn.execute(
                        "DELETE FROM entries_fts WHERE entry_id = ?", (entry_id,)
                    )

                    # Remove concepts
                    conn.execute(
                        "DELETE FROM concept_index WHERE entry_id = ?", (entry_id,)
                    )

                # Remove from FAISS (mark as deleted, don't rebuild index)
                if entry_id in self._id_map:
                    faiss_idx = self._id_map[entry_id]
                    del self._id_map[entry_id]
                    del self._idx_map[faiss_idx]
            else:
                # Soft delete
                with self._tx() as conn:
                    conn.execute(
                        "UPDATE entries SET is_active = 0 WHERE id = ?", (entry_id,)
                    )
                    # Remove concepts to prevent soft-deleted entries from appearing in concept-based searches
                    conn.execute(
                        "DELETE FROM concept_index WHERE entry_id = ?", (entry_id,)
                    )

                # Remove from FAISS maps to prevent soft-deleted entries from appearing in vector search
                if entry_id in self._id_map:
                    faiss_idx = self._id_map[entry_id]
                    del self._id_map[entry_id]
                    del self._idx_map[faiss_idx]

    # ════════════════════ Entity-Relation Graph ════════════════════

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Add entity to knowledge graph

        Args:
            entity_id: Unique entity ID (e.g., "company:spacex")
            entity_type: person|company|project|location|module
            name: Human-readable name
            meta: Additional metadata
        """
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO entities (id, entity_type, name, meta, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    entity_id,
                    entity_type,
                    name,
                    json.dumps(meta) if meta else None,
                    time.time(),
                ),
            )
        return entity_id

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        weight: float = 1.0,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Add relation between entities

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: WORKS_AT|DEPENDS_ON|LOCATED_IN|PART_OF
            weight: Relation strength
            meta: Additional metadata
        """
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO relations
                (source_id, target_id, rel_type, weight, meta, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    source_id,
                    target_id,
                    rel_type,
                    weight,
                    json.dumps(meta) if meta else None,
                    time.time(),
                ),
            )

    def get_related(
        self,
        entity_id: str,
        depth: int = 1,
        rel_types: Optional[List[str]] = None,
        direction: str = "outgoing",
    ) -> List[Dict[str, Any]]:
        """
        Get related entities via graph traversal (CTE)

        Args:
            entity_id: Starting entity
            depth: Traversal depth (1 = direct relations)
            rel_types: Filter by relation types

        Returns:
            List of related entities with relation info
        """
        # Build CTE query for graph traversal
        # Determine direction: outgoing (source->target), incoming (target->source), or both
        if direction == "incoming":
            # Reverse direction: find entities that point TO this entity
            base_case = """
                SELECT
                    r.source_id as target_id,
                    r.rel_type,
                    r.weight,
                    1 as depth
                FROM relations r
                WHERE r.target_id = ?
            """
            recursive_case = """
                SELECT
                    r.source_id as target_id,
                    r.rel_type,
                    r.weight,
                    related.depth + 1
                FROM relations r
                INNER JOIN related ON r.target_id = related.target_id
                WHERE related.depth < ?
            """
        else:
            # Outgoing direction (default): find entities this entity points TO
            base_case = """
                SELECT
                    r.target_id,
                    r.rel_type,
                    r.weight,
                    1 as depth
                FROM relations r
                WHERE r.source_id = ?
            """
            recursive_case = """
                SELECT
                    r.target_id,
                    r.rel_type,
                    r.weight,
                    related.depth + 1
                FROM relations r
                INNER JOIN related ON r.source_id = related.target_id
                WHERE related.depth < ?
            """

        cte_sql = f"""
            WITH RECURSIVE related AS (
                -- Base case: direct relations
                {base_case}

                UNION ALL

                -- Recursive case: follow relations
                {recursive_case}
            )
            SELECT
                related.target_id,
                related.rel_type,
                related.weight,
                related.depth,
                e.name,
                e.entity_type,
                e.meta
            FROM related
            INNER JOIN entities e ON e.id = related.target_id
            WHERE 1=1
        """

        params = [entity_id, depth]

        if rel_types:
            cte_sql += " AND related.rel_type IN ({})".format(
                ",".join("?" * len(rel_types))
            )
            params.extend(rel_types)

        cte_sql += " ORDER BY related.depth, related.weight DESC"

        results = []
        for row in self._exec(cte_sql, tuple(params)):
            results.append(
                {
                    "id": row["target_id"],
                    "name": row["name"],
                    "type": row["entity_type"],
                    "rel_type": row["rel_type"],
                    "weight": row["weight"],
                    "depth": row["depth"],
                    "meta": json.loads(row["meta"]) if row["meta"] else {},
                }
            )

        return results

    # ════════════════════ Persistence ════════════════════

    def save(self, target_dir: Optional[str] = None) -> bool:
        """
        Save complete state to directory (NO pickle!)

        Layout:
            <target_dir>/
            ├── entries.db       (SQLite — already on disk via WAL, just checkpoint)
            ├── vectors.faiss    (native FAISS binary)
            └── vector_ids.json  (ID maps + metadata)

        Args:
            target_dir: Target directory (default: self.db_dir)

        Returns:
            True on success
        """
        import shutil

        target = Path(target_dir) if target_dir else self.db_dir
        target.mkdir(parents=True, exist_ok=True)

        # 1. SQLite: WAL checkpoint flushes all WAL data into the main .db file
        self._exec("PRAGMA wal_checkpoint(TRUNCATE)")

        # 2. If saving to a different directory, copy the SQLite db
        if target != self.db_dir:
            src_db = self.db_dir / "entries.db"
            dst_db = target / "entries.db"
            if src_db.exists():
                shutil.copy2(str(src_db), str(dst_db))
            # Also copy WAL/SHM if they exist (shouldn't after TRUNCATE, but defensive)
            for suffix in ["-wal", "-shm"]:
                src_extra = self.db_dir / f"entries.db{suffix}"
                if src_extra.exists():
                    shutil.copy2(str(src_extra), str(target / f"entries.db{suffix}"))

        # 3. FAISS: native save via faiss.serialize_index -> bytes -> file
        import faiss

        faiss_path = target / "vectors.faiss"
        index_bytes = faiss.serialize_index(self._faiss.index)
        faiss_path.write_bytes(index_bytes)

        # 4. ID Maps + metadata as JSON (NO pickle!)
        maps_path = target / "vector_ids.json"
        # JSON keys must be strings — convert int keys to str
        maps_data = {
            "version": 3,
            "id_map": {str(k): int(v) for k, v in self._id_map.items()},
            "next_idx": self._next_idx,
            "dim": self.dim,
            "space": self.space,
        }
        maps_path.write_text(json.dumps(maps_data), encoding="utf-8")

        return True

    def save_to_bytes(self) -> bytes:
        """
        Legacy-compatible: save to bytes (for MinIO upload, network transfer)

        Uses the directory save internally, then bundles into a single bytes blob.
        NOT pickle — uses a simple structured format.
        """
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # SQLite checkpoint first
            self._exec("PRAGMA wal_checkpoint(TRUNCATE)")
            db_path = self.db_dir / "entries.db"
            if db_path.exists():
                zf.write(str(db_path), "entries.db")

            # FAISS
            import faiss

            index_bytes = faiss.serialize_index(self._faiss.index)
            zf.writestr("vectors.faiss", index_bytes)

            # ID maps as JSON
            maps_data = {
                "version": 3,
                "id_map": {str(k): int(v) for k, v in self._id_map.items()},
                "next_idx": self._next_idx,
                "dim": self.dim,
                "space": self.space,
            }
            zf.writestr("vector_ids.json", json.dumps(maps_data))

        return buf.getvalue()

    def load(self, data: Optional[bytes] = None):
        """
        Load state from bytes or from the existing directory.

        Supports:
        - None/empty: load from self.db_dir (directory format)
        - ZIP bytes: v3 format (save_to_bytes output)
        - Pickle bytes: v2 legacy format (auto-detected and migrated)

        Args:
            data: Optional bytes to load from. If None, loads from self.db_dir.
        """
        if data is None or not isinstance(data, (bytes, bytearray)) or len(data) == 0:
            # Load from directory (already on disk from __init__)
            self._load_faiss_from_dir()
            return

        # Detect format: ZIP (v3) vs Pickle (v2/legacy)
        if data[:4] == b"PK\x03\x04":
            # ZIP format (v3)
            self._load_from_zip(data)
        else:
            # Try pickle (legacy v2)
            self._load_from_pickle_legacy(data)

    def _load_from_zip(self, data: bytes):
        """Load from ZIP-based v3 format"""
        import io
        import zipfile

        buf = io.BytesIO(data)
        with zipfile.ZipFile(buf, "r") as zf:
            # 1. Extract SQLite database
            if "entries.db" in zf.namelist():
                db_bytes = zf.read("entries.db")
                db_path = self.db_dir / "entries.db"
                # Close existing connection before overwriting
                self.close()
                db_path.write_bytes(db_bytes)
                # Re-init connection (schema already exists in the loaded db)
                # Just reinitialize the connection, don't re-run schema
                self._local = threading.local()

            # 2. Load FAISS index
            if "vectors.faiss" in zf.namelist():
                import faiss

                faiss_bytes = zf.read("vectors.faiss")
                self._faiss.index = faiss.deserialize_index(
                    np.frombuffer(faiss_bytes, dtype=np.uint8)
                )

            # 3. Load ID maps from JSON
            if "vector_ids.json" in zf.namelist():
                maps_data = json.loads(zf.read("vector_ids.json").decode("utf-8"))
                self._id_map = {k: int(v) for k, v in maps_data["id_map"].items()}
                self._idx_map = {int(v): k for k, v in maps_data["id_map"].items()}
                self._next_idx = maps_data["next_idx"]
                self.dim = maps_data.get("dim", self.dim)
                self.space = maps_data.get("space", self.space)

        # Rebuild FTS5 index from loaded entries
        self._rebuild_fts5()

    def _load_from_pickle_legacy(self, data: bytes):
        """Load from legacy pickle v2 format (backward compatibility)"""
        try:
            loaded = pickle.loads(data)
        except Exception:
            return

        if isinstance(loaded, dict) and loaded.get("version") == 2:
            # V2 pickle format: has sql_dump + faiss_bytes
            # Clear existing data
            with self._tx() as conn:
                conn.execute("DELETE FROM concept_index")
                conn.execute("DELETE FROM relations")
                conn.execute("DELETE FROM entities")
                conn.execute("DELETE FROM entries")
                conn.execute("DELETE FROM entries_fts")

            # Re-execute SQL dump (careful filtering)
            sql_dump = loaded.get("sql_dump", "")
            if sql_dump:
                self._exec_filtered_sql_dump(sql_dump)

            # Rebuild FTS5
            self._rebuild_fts5()

            # Restore FAISS
            self._faiss.load(loaded["faiss_bytes"])
            self._id_map = loaded.get("id_map", {})
            self._idx_map = loaded.get("idx_map", {})
            self._next_idx = loaded.get("next_idx", 0)
            self.dim = loaded.get("dim", self.dim)
            self.space = loaded.get("space", self.space)
        else:
            # V1 / KnowledgeBase pickle format
            self._migrate_from_pickle(data)

    def _exec_filtered_sql_dump(self, sql_dump: str):
        """Execute a SQL dump, filtering out schema statements safely"""
        lines = sql_dump.split("\n")
        filtered = []
        skip_depth = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip schema creation statements
            upper = stripped.upper()
            if (
                upper.startswith("CREATE ")
                or upper.startswith("BEGIN ")
                or upper == "COMMIT;"
            ):
                if "(" in stripped and ")" not in stripped:
                    skip_depth = 1
                continue
            if skip_depth > 0:
                if ")" in stripped:
                    skip_depth = 0
                continue
            # Skip FTS5 internal tables
            if "entries_fts" in stripped:
                continue
            filtered.append(line)

        if filtered:
            with self._tx() as conn:
                conn.executescript("\n".join(filtered))

    def _rebuild_fts5(self):
        """Rebuild FTS5 index from entries table"""
        try:
            with self._tx() as conn:
                conn.execute("DELETE FROM entries_fts")
                conn.execute("""
                    INSERT INTO entries_fts(rowid, content, entry_id, space)
                    SELECT rowid, content, id, space FROM entries WHERE is_active = 1
                """)
        except Exception:
            pass  # FTS5 rebuild failure is non-fatal

    def _load_faiss_from_dir(self):
        """Load FAISS index + ID maps from directory (v3 native format)"""
        faiss_path = self.db_dir / "vectors.faiss"
        maps_path = self.db_dir / "vector_ids.json"

        if faiss_path.exists() and maps_path.exists():
            try:
                import faiss

                index_bytes = faiss_path.read_bytes()
                self._faiss.index = faiss.deserialize_index(
                    np.frombuffer(index_bytes, dtype=np.uint8)
                )

                maps_text = maps_path.read_text(encoding="utf-8")
                maps_data = json.loads(maps_text)
                self._id_map = {k: int(v) for k, v in maps_data["id_map"].items()}
                self._idx_map = {int(v): k for k, v in maps_data["id_map"].items()}
                self._next_idx = maps_data["next_idx"]
                self.dim = maps_data.get("dim", self.dim)
                self.space = maps_data.get("space", self.space)
            except Exception:
                pass  # Start fresh if loading fails

        # Legacy fallback: try old pickle-based format
        elif (self.db_dir / "faiss.index").exists() and (
            self.db_dir / "id_maps.pkl"
        ).exists():
            try:
                faiss_bytes = (self.db_dir / "faiss.index").read_bytes()
                self._faiss.load(faiss_bytes)

                with open(self.db_dir / "id_maps.pkl", "rb") as f:
                    maps = pickle.load(f)
                    self._id_map = maps["id_map"]
                    self._idx_map = maps["idx_map"]
                    self._next_idx = maps["next_idx"]
            except Exception:
                pass

    def _migrate_from_pickle(self, data: bytes):
        """
        Migrate from legacy pickle format (V1)

        Handles migration from KnowledgeBase pickle format
        """
        try:
            loaded = pickle.loads(data)

            # Detect old format
            if "chunks" in loaded or "entries" not in loaded:
                # Old format from KnowledgeBase
                chunks = loaded.get("chunks", [])
                embeddings = loaded.get("embeddings", [])  # Check for parallel array

                # Re-import FAISS
                from toolboxv2.mods.isaa.base.VectorStores.types import Chunk

                for i, chunk in enumerate(chunks):
                    if isinstance(chunk, dict):
                        # Old chunk format
                        content = chunk.get("content", "")

                        # Get embedding from chunk OR from parallel array
                        embedding = chunk.get("embedding")
                        if embedding is None and i < len(embeddings):
                            embedding = embeddings[i]

                        metadata = chunk.get("metadata", {})

                        if embedding is not None:
                            self.add(
                                content=content,
                                embedding=embedding,
                                content_type=metadata.get("type", "text"),
                                meta=metadata,
                                concepts=metadata.get("concepts", []),
                            )
                # Migration successful
                return True
        except Exception as e:
            # Migration failed, return False instead of raising
            return False

    def _init_minio(self):
        """Lazy init MinIO client (reuse from blob_instance.py)"""
        if self._minio is not None:
            return

        if MinIOClient is None:
            raise RuntimeError("MinIO not available - minio package not installed")

        try:
            # Load config from environment (pattern from blob_instance.py:55-76)
            endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
            access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
            secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
            secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

            self._minio = MinIOClient(
                endpoint, access_key=access_key, secret_key=secret_key, secure=secure
            )

            # Ensure bucket exists
            if not self._minio.bucket_exists(self._minio_bucket):
                self._minio.make_bucket(self._minio_bucket)
        except Exception as e:
            # Fallback to local-only mode
            self._minio = None
            raise RuntimeError(f"MinIO initialization failed: {e}")

    def save_to_minio(self, key: Optional[str] = None):
        """
        Backup to MinIO

        Args:
            key: Optional key (default: space name)
        """
        if self._minio is None:
            self._init_minio()

        if self._minio is None:
            raise RuntimeError("MinIO not available")

        key = key or f"{self.space}/memory_backup.zip"

        # Get data as ZIP bytes (network-safe format)
        data = self.save_to_bytes()

        # Upload to MinIO (pattern from scoped_storage.py:836-868)
        from io import BytesIO

        self._minio.put_object(self._minio_bucket, key, BytesIO(data), length=len(data))

        return True

    def load_from_minio(self, key: Optional[str] = None):
        """
        Restore from MinIO

        Args:
            key: Optional key (default: space name)
        """
        if self._minio is None:
            self._init_minio()

        if self._minio is None:
            raise RuntimeError("MinIO not available")

        key = key or f"{self.space}/memory_backup.zip"

        # Download from MinIO (pattern from scoped_storage.py:870-883)
        response = self._minio.get_object(self._minio_bucket, key)
        data = response.read()

        # Load data
        self.load(data)

    # ════════════════════ Lifecycle Management ════════════════════

    def invalidate_by_source(self, source: str) -> int:
        with self._lock:
            # IDs sammeln bevor wir deaktivieren
            rows = self._exec(
                "SELECT id FROM entries WHERE meta_source = ? AND space = ? AND is_active = 1",
                (source, self.space),
            ).fetchall()
            ids = [row[0] for row in rows]

            if not ids:
                return 0

            with self._tx() as conn:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE entries SET is_active = 0 WHERE id IN ({placeholders})", ids
                )
                conn.execute(
                    f"DELETE FROM concept_index WHERE entry_id IN ({placeholders})", ids
                )

            # FAISS-Maps aufräumen
            for eid in ids:
                if eid in self._id_map:
                    fidx = self._id_map.pop(eid)
                    self._idx_map.pop(fidx, None)

        return len(ids)

    def cleanup_expired(self, rebuild_threshold: int = 100) -> int:
        now = time.time()

        with self._lock:
            with self._tx() as conn:
                # Erst IDs sammeln, dann aus allen Indizes löschen
                expired_rows = conn.execute(
                    "SELECT id FROM entries WHERE ttl IS NOT NULL AND created_at + ttl < ?",
                    (now,),
                ).fetchall()
                expired_ids = [row[0] for row in expired_rows]

                if not expired_ids:
                    return 0

                placeholders = ",".join("?" * len(expired_ids))
                conn.execute(f"DELETE FROM entries_fts WHERE entry_id IN ({placeholders})", expired_ids)
                conn.execute(f"DELETE FROM concept_index WHERE entry_id IN ({placeholders})", expired_ids)
                conn.execute(f"DELETE FROM entries WHERE id IN ({placeholders})", expired_ids)

            # FAISS-Maps aufräumen
            for eid in expired_ids:
                if eid in self._id_map:
                    fidx = self._id_map.pop(eid)
                    self._idx_map.pop(fidx, None)

        deleted_count = len(expired_ids)
        if deleted_count >= rebuild_threshold:
            self.rebuild_faiss_index()

        return deleted_count

    def rebuild_faiss_index(self) -> int:
        """
        Rebuild FAISS index from active entries only.

        This removes all embeddings from deleted entries and compacts the index.
        Returns the number of entries in the rebuilt index.
        """
        with self._lock:
            # Get all active entry IDs from SQLite
            active_entries = self._exec(
                "SELECT id FROM entries WHERE is_active = 1 AND space = ?",
                (self.space,),
            ).fetchall()

            if not active_entries:
                # No active entries, reset everything
                self._faiss = FaissVectorStore(self.dim)
                self._id_map = {}
                self._idx_map = {}
                self._next_idx = 0
                return 0

            # Create set of active IDs for O(1) lookup
            active_ids = {row["id"] for row in active_entries}

            # Create new index
            new_faiss = FaissVectorStore(self.dim)
            new_id_map = {}
            new_idx_map = {}
            new_next_idx = 0

            # Iterate through existing chunks and add only active ones
            for faiss_idx, chunk in enumerate(self._faiss.chunks):
                entry_id = chunk.metadata.get("id")
                if entry_id and entry_id in active_ids:
                    # Get embedding from old index
                    embedding = self._faiss.index.reconstruct(faiss_idx)

                    # Add to new index
                    new_faiss.add_embeddings(embedding.reshape(1, -1), [chunk])
                    new_id_map[entry_id] = new_next_idx
                    new_idx_map[new_next_idx] = entry_id
                    new_next_idx += 1

            # Replace old index and mappings
            self._faiss = new_faiss
            self._id_map = new_id_map
            self._idx_map = new_idx_map
            self._next_idx = new_next_idx

            return new_next_idx

    def stats(self) -> Dict[str, Any]:
        """
        Get memory statistics

        Returns:
            Dict with active entries, total entries, entities, relations, etc.
        """
        active = self._exec(
            "SELECT COUNT(*) FROM entries WHERE is_active = 1 AND space = ?",
            (self.space,),
        ).fetchone()[0]

        total = self._exec(
            "SELECT COUNT(*) FROM entries WHERE space = ?", (self.space,)
        ).fetchone()[0]

        entities = self._exec("SELECT COUNT(*) FROM entities").fetchone()[0]
        relations = self._exec("SELECT COUNT(*) FROM relations").fetchone()[0]
        concepts = self._exec(
            "SELECT COUNT(DISTINCT concept) FROM concept_index WHERE entry_id IN (SELECT id FROM entries WHERE space = ?)",
            (self.space,),
        ).fetchone()[0]

        with_ttl = self._exec(
            "SELECT COUNT(*) FROM entries WHERE is_active = 1 AND space = ? AND ttl IS NOT NULL",
            (self.space,),
        ).fetchone()[0]

        return {
            "active": active,
            "total": total,
            "entities": entities,
            "relations": relations,
            "concepts": concepts,
            "faiss_size": len(self._id_map),
            "space": self.space,
            "dim": self.dim,
            "with_ttl": with_ttl,
        }

    def close(self):
        """Close database connection"""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ════════════════════ Helper Methods ════════════════════

    def _load_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Load entry by ID"""
        row = self._exec("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()

        if not row:
            return None

        return dict(row)

    def _extract_meta(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from entry"""
        meta = {
            "role": entry.get("meta_role"),
            "source": entry.get("meta_source"),
            "language": entry.get("meta_language"),
            "category": entry.get("meta_category"),
            "importance": entry.get("meta_importance", 0.5),
        }

        if entry.get("meta_custom"):
            try:
                custom = json.loads(entry["meta_custom"])
                meta.update(custom)
            except:
                pass

        return meta

    def _load_concepts(self, entry_id: str) -> List[str]:
        """Load concepts for an entry from concept_index table"""
        rows = self._exec(
            "SELECT concept FROM concept_index WHERE entry_id = ?", (entry_id,)
        ).fetchall()
        return [row[0] for row in rows]

    def _fts_escape(self, query: str) -> str:
        """Escape FTS5 special characters"""
        # Remove or escape special FTS5 characters (Fix #4: added apostrophe)
        special_chars = ["*", "^", "-", '"', "(", ")", "{", "}", "[", "]", "'", ":", "~", "+"]
        for char in special_chars:
            query = query.replace(char, " ")
        # FTS5-Keywords neutralisieren
        tokens = query.split()
        tokens = [t for t in tokens if t.upper() not in ("AND", "OR", "NOT", "NEAR")]
        return " ".join(tokens).strip()

    def _rrf_fuse(
        self,
        ranked_lists: List[Tuple[str, List[str]]],
        entry_cache: Dict[str, Dict],
        k: int = 5,
        rrf_k: int = 60,
        mode_weights: Optional[Dict[str, float]] = None,
        is_code: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion: score(d) = sum( w_i / (rrf_k + rank_i(d)) )

        Args:
            ranked_lists: List of (mode_name, [entry_id, ...]) — ranked results per mode
            entry_cache: Pre-loaded entry dicts by entry_id
            k: Number of results to return
            rrf_k: RRF constant (default 60, higher = less weight to top ranks)
            mode_weights: Dict of mode -> weight. If None, uses defaults
            is_code: If True, boost BM25 weight (exact term matching for code)

        Returns:
            Fused and ranked result dicts
        """
        # Default weights: code mode boosts BM25 for exact term matching
        if mode_weights is None:
            if is_code:
                mode_weights = {"vector": 0.30, "bm25": 0.50, "relation": 0.20}
            else:
                mode_weights = {"vector": 0.40, "bm25": 0.35, "relation": 0.25}

        scores = {}  # entry_id -> rrf_score

        for mode_name, ranked_ids in ranked_lists:
            w = mode_weights.get(mode_name, 0.25)
            for rank, entry_id in enumerate(ranked_ids):
                scores[entry_id] = scores.get(entry_id, 0.0) + w / (rrf_k + rank + 1)

        # Sort by fused score descending, take top-k
        top_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Build result dicts
        results = []
        for entry_id, rrf_score in top_ids:
            entry = entry_cache.get(entry_id)
            if not entry:
                continue
            results.append(
                {
                    "id": entry_id,
                    "content": entry.get("content", ""),
                    "score": rrf_score,
                    "content_type": entry.get("content_type", "text"),
                    "meta": self._extract_meta(entry),
                    "source": "rrf",
                    "concepts": self._load_concepts(entry_id),
                }
            )

        return results
