"""
AISemanticMemory V2 — Drop-in replacement for AgentUtils.AISemanticMemory

Uses HybridMemoryStore (SQLite + FAISS + FTS5) as backend instead of KnowledgeBase.
Preserves the complete V1 API surface:
  - Singleton metaclass
  - Multi-space management (dict of HybridMemoryStores)
  - Internal embedding generation via litellm_embed
  - save/load per-memory and all-memories
  - query() with to_str formatting

Feature flag:  set  AISemanticMemory._use_v2 = False  before first instantiation
               to fall back to the old KnowledgeBase path.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from toolboxv2 import Singleton

from .hybrid_memory import HybridMemoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag — set to False to revert to the old KnowledgeBase backend
# ---------------------------------------------------------------------------
_USE_V2 = True


class AISemanticMemory(metaclass=Singleton):
    """
    Singleton semantic-memory manager backed by HybridMemoryStore (V2).

    100% API-compatible with the V1 class in AgentUtils.py.
    Each *memory_name* maps to its own HybridMemoryStore instance (one SQLite
    DB + FAISS index per space).
    """

    # ── class-level config ────────────────────────────────────────────
    _use_v2: bool = _USE_V2

    # Map of known embedding models → dimension
    EMBEDDING_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "nomic-embed-text": 768,
        "default": 768,
    }

    # ── __init__ ──────────────────────────────────────────────────────

    def __init__(
        self,
        base_path: str = "/semantic_memory",
        default_model: str | None = None,
        default_embedding_model: str | None = None,
        default_similarity_threshold: float = 0.61,
        default_batch_size: int = 64,
        default_n_clusters: int = 2,
        default_deduplication_threshold: float = 0.85,
    ):
        if default_model is None:
            default_model = os.getenv("BLITZMODEL", "")
        if default_embedding_model is None:
            default_embedding_model = os.getenv("DEFAULTMODELEMBEDDING", "")

        self.base_path: str = os.path.join(os.getcwd(), ".data", base_path)
        os.makedirs(self.base_path, exist_ok=True)

        self.memories: dict[str, HybridMemoryStore] = {}

        self.embedding_dims = dict(self.EMBEDDING_DIMS)

        self.default_config: dict[str, Any] = {
            "embedding_model": default_embedding_model,
            "embedding_dim": self._get_embedding_dim(default_embedding_model),
            "similarity_threshold": default_similarity_threshold,
            "batch_size": default_batch_size,
            "n_clusters": default_n_clusters,
            "deduplication_threshold": default_deduplication_threshold,
            "model_name": default_model,
        }

    # ── helpers ────────────────────────────────────────────────────────

    def _get_embedding_dim(self, model_name: str | None) -> int:
        if not model_name:
            return 768
        return self.embedding_dims.get(model_name, 768)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize memory name for filesystem safety (identical to V1)."""
        name = (
            re.sub(r"[^a-zA-Z0-9_-]", "-", name)[:63]
            .strip("-")
            .replace(":", "_")
            .replace(" ", "_")
        )
        if not name:
            raise ValueError("Invalid memory name")
        if len(name) < 3:
            name += "Z" * (3 - len(name))
        return name

    def _get_or_create_store(self, sanitized_name: str) -> HybridMemoryStore:
        """Return an existing store or lazily create one."""
        if sanitized_name not in self.memories:
            self.create_memory(sanitized_name)
        return self.memories[sanitized_name]

    def _get_target_memories(
        self, memory_names: str | list[str] | None
    ) -> list[tuple[str, HybridMemoryStore]]:
        """Get target memories for query — identical semantics to V1."""
        if not memory_names:
            return list(self.memories.items())

        names = [memory_names] if isinstance(memory_names, str) else memory_names
        targets = []
        for name in names:
            sanitized = self._sanitize_name(name)
            if store := self.memories.get(sanitized):
                targets.append((sanitized, store))
        return targets

    # ── embedding ──────────────────────────────────────────────────────

    async def get_embeddings(self, text: str | list[str]) -> np.ndarray:
        """Generate embeddings via litellm (identical to V1)."""
        from toolboxv2.mods.isaa.extras.adapter import litellm_embed

        texts = [text] if isinstance(text, str) else text
        return (
            await litellm_embed(
                texts=texts,
                model=self.default_config["embedding_model"],
                dimensions=self.default_config["embedding_dim"],
            )
        )[0]

    # ── CRUD ───────────────────────────────────────────────────────────

    def create_memory(
        self,
        name: str,
        model_config: dict | None = None,
        storage_config: dict | None = None,
    ) -> HybridMemoryStore:
        """
        Create a new memory space (HybridMemoryStore instance).

        Compatible with V1 signature — model_config / storage_config are
        consumed for embedding_dim determination; extra KnowledgeBase params
        are silently accepted for backward-compat.
        """
        sanitized = self._sanitize_name(name)
        if sanitized in self.memories:
            raise ValueError(f"Memory '{name}' already exists")

        # Determine embedding dimension
        embedding_model = self.default_config["embedding_model"]
        if model_config:
            embedding_model = model_config.get("embedding_model", embedding_model)
        embedding_dim = self._get_embedding_dim(embedding_model)

        if storage_config and "embedding_model" in storage_config:
            embedding_dim = self._get_embedding_dim(
                storage_config["embedding_model"]
            )

        db_dir = os.path.join(self.base_path, sanitized)
        os.makedirs(db_dir, exist_ok=True)

        store = HybridMemoryStore(
            db_dir=db_dir,
            embedding_dim=embedding_dim,
            space=sanitized,
        )
        self.memories[sanitized] = store
        return store

    async def add_data(
        self,
        memory_name: str,
        data: str | list[str] | bytes | dict,
        metadata: dict | None = None,
        direct: bool = False,
    ) -> bool:
        """
        Add data to a memory space.

        Matches V1 signature exactly:
          - Auto-creates memory if it doesn't exist
          - Handles str, list[str], bytes, dict
          - Generates embeddings internally
        """
        sanitized = self._sanitize_name(memory_name)
        store = self._get_or_create_store(sanitized)

        # ── Process input ──
        texts: list[str] = []
        if isinstance(data, bytes):
            try:
                from toolboxv2.mods.isaa.base.AgentUtils import extract_text_natively

                filename = (metadata or {}).get("filename", "")
                text = extract_text_natively(data, filename=filename)
                texts = [text.replace("\\t", "").replace("\t", "")]
            except Exception as e:
                raise ValueError(f"File processing failed: {e}")
        elif isinstance(data, str):
            texts = [data.replace("\\t", "").replace("\t", "")]
        elif isinstance(data, list):
            texts = [d.replace("\\t", "").replace("\t", "") for d in data]
        elif isinstance(data, dict):
            raise NotImplementedError("Custom knowledge graph insertion not supported")
        else:
            raise ValueError("Unsupported data type")

        # ── Add each text ──
        added = 0
        for text in texts:
            if not text.strip():
                continue
            try:
                emb = await self.get_embeddings(text)
                # Determine content_type heuristic
                content_type = "code" if _looks_like_code(text) else "text"

                # Extract simple concepts from text (words > 3 chars)
                concepts = _extract_concepts(text)

                entry_id = store.add(
                    content=text,
                    embedding=emb,
                    content_type=content_type,
                    meta=metadata,
                    concepts=concepts,
                )
                if entry_id:
                    added += 1
            except Exception as e:
                logger.error(f"add_data failed for chunk: {e}")
                import traceback
                traceback.print_exc()

        return added > 0

    async def query(
        self,
        query: str,
        memory_names: str | list[str] | None = None,
        query_params: dict | None = None,
        to_str: bool = False,
        unified_retrieve: bool = False,
        **kwargs,
    ) -> str | list[dict]:
        """
        Query memories — V1-compatible signature.

        Uses HybridMemoryStore's vector + BM25 + relation search with RRF
        instead of KnowledgeBase.retrieve_with_overview / unified_retrieve.
        """
        if query_params is None:
            query_params = {}

        # Compat: map legacy kwargs
        if "max_entries" in kwargs:
            query_params["k"] = kwargs["max_entries"]

        targets = self._get_target_memories(memory_names)
        if not targets:
            return "" if to_str else []

        k = int(query_params.get("k", 3))
        min_sim = float(query_params.get("min_similarity", 0.2))

        # Generate query embedding once
        try:
            query_emb = await self.get_embeddings(query)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return "Error generating query embedding." if to_str else []

        # Decide search modes
        search_modes = ("vector", "bm25", "relation")

        results: list[dict] = []
        for name, store in targets:
            try:
                hits = store.query(
                    query_text=query,
                    query_embedding=query_emb,
                    k=k,
                    search_modes=search_modes,
                    min_similarity=min_sim,
                )
                if hits:
                    results.append({"memory": name, "type": "standard", "hits": hits})
            except Exception as e:
                logger.error(f"Query failed on {name}: {e}")

        if to_str:
            return _format_results_as_str(results)

        return results

    # ── accessors ──────────────────────────────────────────────────────

    def get(self, names) -> list[HybridMemoryStore]:
        """Return list of store objects for given names."""
        return [m for _, m in self._get_target_memories(names)]

    def list_memories(self) -> list[str]:
        """List all memory space names."""
        return list(self.memories.keys())

    def get_memory_size(self, name: str | None) -> int:
        """Return number of active entries in a memory space."""
        if name is None:
            return 0
        sanitized = self._sanitize_name(name)
        if store := self.memories.get(sanitized):
            stats = store.stats()
            return stats.get("active", 0)
        return 0

    async def delete_memory(self, name: str) -> bool:
        """Delete a memory space and close its store."""
        sanitized = self._sanitize_name(name)
        if sanitized in self.memories:
            try:
                self.memories[sanitized].close()
            except Exception:
                pass
            del self.memories[sanitized]
            return True
        return False

    # ── persistence ────────────────────────────────────────────────────

    def save_memory(self, name: str, path: str) -> bool | bytes:
        """
        Save a single memory space.

        If *path* is a directory: saves as directory format (SQLite + FAISS + JSON).
        If *path* ends with .zip: saves as ZIP bytes to that file.
        Otherwise: saves as ZIP bytes to the given file path.

        Returns True on success, False on failure.
        """
        sanitized = self._sanitize_name(name)
        store = self.memories.get(sanitized)
        if not store:
            return False
        try:
            p = None
            if path is not None:
                p = Path(path)
                if p.is_dir() or not p.suffix:
                    # Directory-based save
                    p.mkdir(parents=True, exist_ok=True)
                    return store.save(target_dir=str(p))
            # File-based save (ZIP bytes)
            data = store.save_to_bytes()
            if p:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            return data
        except Exception as e:
            logger.error(f"Error saving memory '{name}': {e}")

            import traceback
            traceback.print_exc()
            return False

    def load_memory(self, name: str, path: str | bytes) -> bool:
        """
        Load a single memory space.

        - *path* as str: file path (.zip / .pkl) or directory
        - *path* as bytes: raw serialized data
        """
        sanitized = self._sanitize_name(name)
        if sanitized in self.memories:
            # Already loaded — V1 returns True in this case
            return True
        try:
            if isinstance(path, bytes):
                store = self._make_store(sanitized)
                store.load(path)
                self.memories[sanitized] = store
                return True

            p = Path(path)
            if p.is_dir():
                # Directory-based load
                store = HybridMemoryStore(
                    db_dir=str(p),
                    embedding_dim=self.default_config["embedding_dim"],
                    space=sanitized,
                )
                self.memories[sanitized] = store
                return True

            if p.exists():
                data = p.read_bytes()
                store = self._make_store(sanitized)
                store.load(data)
                self.memories[sanitized] = store
                return True

            return False
        except Exception as e:
            logger.error(f"Error loading memory '{name}': {e}")
            return False

    def save_all_memories(self, path: str) -> bool:
        """Save every memory space into *path* directory (one sub-dir each)."""
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)
        for name, store in self.memories.items():
            try:
                mem_dir = base / name
                mem_dir.mkdir(parents=True, exist_ok=True)
                store.save(target_dir=str(mem_dir))
            except Exception as e:
                logger.error(f"Error saving memory '{name}': {e}")

                import traceback
                traceback.print_exc()
                return False
        # Also write a manifest so load_all_memories knows what exists
        manifest = {n: {"embedding_dim": s.dim} for n, s in self.memories.items()}
        (base / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return True

    def load_all_memories(self, path: str) -> bool:
        """Load all memory spaces from *path* directory."""
        base = Path(path)
        if not base.is_dir():
            return False

        # Try manifest first
        manifest_path = base / "_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for name, info in manifest.items():
                    mem_dir = base / name
                    if mem_dir.is_dir() and name not in self.memories:
                        dim = info.get("embedding_dim", self.default_config["embedding_dim"])
                        store = HybridMemoryStore(
                            db_dir=str(mem_dir), embedding_dim=dim, space=name
                        )
                        self.memories[name] = store
                return True
            except Exception as e:
                logger.error(f"Error loading manifest: {e}")

        # Fallback: scan for .pkl / .zip files (legacy V1 compat)
        for file in sorted(base.iterdir()):
            if file.is_dir() and file.name != "__pycache__" and not file.name.startswith("_"):
                # Assume directory-format store
                name = file.name
                if name not in self.memories:
                    try:
                        store = HybridMemoryStore(
                            db_dir=str(file),
                            embedding_dim=self.default_config["embedding_dim"],
                            space=name,
                        )
                        self.memories[name] = store
                    except Exception as e:
                        logger.error(f"Error loading memory dir '{name}': {e}")
                        return False
            elif file.suffix in (".pkl", ".zip"):
                name = file.stem
                if name not in self.memories:
                    try:
                        data = file.read_bytes()
                        store = self._make_store(name)
                        store.load(data)
                        self.memories[name] = store
                    except Exception as e:
                        logger.error(f"Error loading memory file '{name}': {e}")
                        return False
        return True

    def _make_store(self, sanitized_name: str) -> HybridMemoryStore:
        """Create a new HybridMemoryStore in our base_path."""
        db_dir = os.path.join(self.base_path, sanitized_name)
        os.makedirs(db_dir, exist_ok=True)
        return HybridMemoryStore(
            db_dir=db_dir,
            embedding_dim=self.default_config["embedding_dim"],
            space=sanitized_name,
        )

    # ── lifecycle ──────────────────────────────────────────────────────

    def close(self):
        """Close all stores."""
        for store in self.memories.values():
            try:
                store.close()
            except Exception:
                pass
        self.memories.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ═══════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════

def _looks_like_code(text: str) -> bool:
    """Simple heuristic: does the text look like source code?"""
    code_indicators = [
        "def ", "class ", "import ", "from ", "return ",
        "function ", "const ", "let ", "var ",
        "if (", "for (", "while (",
        "=>", "->", "::", "//", "/*",
    ]
    lines = text.split("\n")[:20]
    score = sum(1 for line in lines for ind in code_indicators if ind in line)
    return score >= 3


def _extract_concepts(text: str, max_concepts: int = 20) -> list[str]:
    """Extract simple concepts from text (unique words > 3 chars)."""
    words = set()
    for word in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower()):
        if len(word) > 3 and word not in _STOP_WORDS:
            words.add(word)
            if len(words) >= max_concepts:
                break
    return list(words)


_STOP_WORDS = frozenset({
    "this", "that", "with", "from", "have", "been", "were", "will",
    "would", "could", "should", "their", "there", "then", "than",
    "these", "those", "they", "them", "what", "when", "where",
    "which", "while", "about", "after", "before", "between",
    "into", "through", "during", "above", "below", "each",
    "some", "such", "only", "also", "just", "more", "most",
    "other", "very", "much", "many", "well", "even", "back",
    "over", "down", "still", "here", "does", "done", "make",
    "made", "like", "long", "look", "come", "came", "said",
    "know", "take", "took", "give", "gave", "tell", "told",
    "work", "call", "need", "want", "seem", "help", "talk",
    "turn", "start", "show", "hear", "play", "keep", "move",
    "live", "mean", "find", "found", "left", "part", "same",
    "true", "false", "none", "null", "self",
})


def _format_results_as_str(results: list[dict]) -> str:
    """
    Format query results into the V1 string format:
      Source [memory_name]: chunk_text
    """
    if not results:
        return "No relevant information found in memory."

    formatted: list[str] = []
    seen: set[str] = set()

    for item in results:
        mem_name = item["memory"]
        for hit in item.get("hits", []):
            text = hit.get("content", "").strip()
            if text and text not in seen:
                seen.add(text)
                formatted.append(f"Source [{mem_name}]: {text}")

    if not formatted:
        return "No relevant information found in memory."

    return "\n\n".join(formatted)
