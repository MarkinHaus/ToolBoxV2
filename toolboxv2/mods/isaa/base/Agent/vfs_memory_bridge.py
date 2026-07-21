"""
VFS ↔ Memory Bridge — indexes native session-VFS files into HybridMemoryStore.

Design:
- Hooks into VFSv2's `on_change` delta stream (chained, does NOT steal the
  observability slot). Write path stays sync + O(1): only a dict entry is
  queued. Embeddings happen in a debounced background worker.
- Indexed entries carry meta_source = "vfs:{path}" so a rewrite is a clean
  `invalidate_by_source()` + re-add — solves staleness structurally.
- Scope: ONLY native session files. Skipped: /global, /shared, any mounted
  path (ShadowMount), shared-store paths, readonly/system files, vfs_guide.md.
  vfs_shell writes go through vfs.write()/create() → covered automatically.
- Retroactive: `backfill()` scans existing files at session init, skips
  content whose hash is already indexed (idempotent, cheap on restart).

Space name: VFSIndex/{agent}.{session_id}
Author: Markin / ToolBoxV2
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

from toolboxv2 import get_logger

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.agent_session_v2 import AgentSessionV2
    from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VFSDelta

logger = get_logger()

_SKIP_PREFIXES = ("/shared",)
GLOBAL_PREFIX = "/global"
GLOBAL_INDEX_SPACE = "VFSIndex/global"
_SKIP_FILENAMES = {"vfs_guide.md", "active_rules.md"}
_CHUNK_SIZE = 1400
_CHUNK_OVERLAP = 120
_MAX_CHUNKS_PER_FILE = 24
_DEBOUNCE_S = 3.0


def vfs_index_space(agent_name: str, session_id: str) -> str:
    return f"VFSIndex/{agent_name}.{session_id}"


def _chunk(text: str) -> list[str]:
    """Structure-aware chunking: split on markdown headers, then paragraphs,
    then merge greedily up to _CHUNK_SIZE. Falls back to fixed windows for
    unstructured blobs. Keeps semantic units intact instead of cutting
    mid-sentence at byte offsets."""
    if len(text) <= _CHUNK_SIZE:
        return [text]

    import re as _re

    # split into semantic units: headers start a new unit, blank lines split
    units: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if _re.match(r"^#{1,6}\s", line) and current:
            units.append("\n".join(current))
            current = [line]
        elif line.strip() == "" and current and len("\n".join(current)) > 200:
            units.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        units.append("\n".join(current))

    # greedy merge into chunks
    chunks: list[str] = []
    buf = ""
    for u in units:
        if len(u) > _CHUNK_SIZE:
            # oversized single unit → fixed windows
            if buf:
                chunks.append(buf)
                buf = ""
            i = 0
            while i < len(u) and len(chunks) < _MAX_CHUNKS_PER_FILE:
                chunks.append(u[i : i + _CHUNK_SIZE])
                i += _CHUNK_SIZE - _CHUNK_OVERLAP
            continue
        if buf and len(buf) + len(u) + 1 > _CHUNK_SIZE:
            chunks.append(buf)
            buf = u
        else:
            buf = f"{buf}\n{u}" if buf else u
        if len(chunks) >= _MAX_CHUNKS_PER_FILE:
            break
    if buf and len(chunks) < _MAX_CHUNKS_PER_FILE:
        chunks.append(buf)
    return chunks or [text[:_CHUNK_SIZE]]


class VFSMemoryIndexer:
    """Background indexer wiring one AgentSessionV2's VFS into semantic memory."""

    def __init__(self, session: "AgentSessionV2"):
        self.session = session
        self.vfs = session.vfs
        self.memory = session._memory
        self.space = vfs_index_space(session.agent_name, session.session_id)
        self._pending: dict[str, float] = {}  # path → not-before timestamp
        self._deletes: set[str] = set()
        self._hashes: dict[str, str] = {}  # path → content sha1 (indexed state)
        self._worker: asyncio.Task | None = None
        self._stopped = False
        self._prev_on_change = None

    # ── hook ──────────────────────────────────────────────────────────────

    def attach(self):
        """Chain into vfs.on_change without displacing existing consumers."""
        self._prev_on_change = self.vfs.on_change

        def _chained(delta: "VFSDelta"):
            if self._prev_on_change is not None:
                try:
                    self._prev_on_change(delta)
                except Exception:
                    pass
            try:
                self._on_delta(delta)
            except Exception:
                pass

        self.vfs.on_change = _chained

    def stop(self):
        self._stopped = True
        if self._worker and not self._worker.done():
            self._worker.cancel()
        # restore chain
        if self.vfs.on_change is not None and self._prev_on_change is not None:
            self.vfs.on_change = self._prev_on_change

    # ── eligibility ───────────────────────────────────────────────────────

    def _space_for(self, path: str) -> str:
        """/global/* is indexed into a single shared space (all agents/sessions
        see the same files, so they share one index — no per-session dupes)."""
        if path.startswith(GLOBAL_PREFIX):
            return GLOBAL_INDEX_SPACE
        return self.space

    def _eligible(self, path: str) -> bool:
        if any(path.startswith(p) for p in _SKIP_PREFIXES):
            return False
        # /global IS a mount + shared store — but explicitly wanted in memory.
        # Allow it before the mount/shared checks below would reject it.
        if path.startswith(GLOBAL_PREFIX):
            f = self.vfs.files.get(path)
            if f is not None and getattr(f, "filename", "") in _SKIP_FILENAMES:
                return False
            return True
        f = self.vfs.files.get(path)
        if f is not None:
            if getattr(f, "readonly", False):
                return False
            if getattr(f, "filename", "") in _SKIP_FILENAMES:
                return False
        # mounted (disk-backed) paths → belong to their own store, not ours
        try:
            if self.vfs._get_mount_for_path(path) is not None:
                return False
        except Exception:
            pass
        try:
            if self.vfs._get_shared_store_info(path) is not None:
                return False
        except Exception:
            pass
        if path in getattr(self.vfs, "_shadow_index", {}):
            return False
        return True

    # ── delta intake (sync, zero-latency) ────────────────────────────────

    def _on_delta(self, delta: "VFSDelta"):
        if self._stopped:
            return
        action = delta.action
        path = delta.path
        if action in ("create", "write", "append", "edit"):
            if self._eligible(path):
                self._pending[path] = time.time() + _DEBOUNCE_S
                self._ensure_worker()
        elif action == "delete":
            self._pending.pop(path, None)
            self._deletes.add(path)
            self._ensure_worker()
        elif action == "mv":
            old = delta.old_path or ""
            if old:
                self._pending.pop(old, None)
                self._deletes.add(old)
            if self._eligible(path):
                self._pending[path] = time.time() + _DEBOUNCE_S
            self._ensure_worker()

    def _ensure_worker(self):
        if self._worker is None or self._worker.done():
            try:
                self._worker = asyncio.get_running_loop().create_task(
                    self._worker_loop()
                )
            except RuntimeError:
                pass  # no loop (sync test context) — backfill/flush will catch up

    # ── background worker ─────────────────────────────────────────────────

    async def _worker_loop(self):
        try:
            while not self._stopped and (self._pending or self._deletes):
                now = time.time()
                due = [p for p, t in self._pending.items() if t <= now]
                deletes = list(self._deletes)
                self._deletes.clear()
                for p in deletes:
                    await self._unindex(p)
                for p in due:
                    self._pending.pop(p, None)
                    await self._index_path(p)
                if self._pending or self._deletes:
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[VFSMemoryIndexer:{self.space}] worker error: {e}")

    async def flush(self):
        """Index everything pending immediately (tests / shutdown)."""
        for p in list(self._deletes):
            await self._unindex(p)
        self._deletes.clear()
        for p in list(self._pending):
            self._pending.pop(p, None)
            await self._index_path(p)

    # ── index ops ─────────────────────────────────────────────────────────

    def _get_content(self, path: str) -> str | None:
        # /global files are lazy shadow files — read through the manager so
        # unloaded shadows still index correctly.
        if path.startswith(GLOBAL_PREFIX):
            try:
                from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs

                rel = path[len(GLOBAL_PREFIX):].lstrip("/")
                res = get_global_vfs().read_file(rel)
                if res.get("success"):
                    c = res.get("content")
                    return c if isinstance(c, str) else None
            except Exception:
                pass
        f = self.vfs.files.get(path)
        if f is None:
            return None
        content = getattr(f, "_content", None)
        if content is None:
            content = getattr(f, "content", None)
        return content if isinstance(content, str) else None

    def _store(self, space: str | None = None):
        stores = self.memory.get(space or self.space)
        return stores[0] if stores else None

    async def _unindex(self, path: str):
        store = self._store(self._space_for(path))
        if store is not None:
            try:
                store.invalidate_by_source(f"vfs:{path}")
            except Exception as e:
                logger.debug(f"[VFSMemoryIndexer] unindex {path}: {e}")
        self._hashes.pop(path, None)

    async def _index_path(self, path: str):
        if not self._eligible(path):
            return
        content = self._get_content(path)
        if not content or not content.strip():
            return
        h = hashlib.sha1(content.encode("utf-8", "replace")).hexdigest()
        if self._hashes.get(path) == h:
            return  # unchanged
        space = self._space_for(path)
        source = f"vfs:{path}"
        store = self._store(space)
        if store is not None:
            if space == GLOBAL_INDEX_SPACE and _global_hash_current(store, source, h):
                self._hashes[path] = h
                return  # another session already indexed this exact content
            try:
                store.invalidate_by_source(source)
            except Exception:
                pass
        try:
            ok = await self.memory.add_data(
                space,
                _chunk(content),
                metadata={
                    "source": source,
                    "vfs_path": path,
                    "category": "vfs",
                    "content_hash": h,
                },
            )
            if ok:
                self._hashes[path] = h
        except Exception as e:
            logger.error(f"[VFSMemoryIndexer] index {path} failed: {e}")

    # ── retroactive backfill ──────────────────────────────────────────────

    def _load_indexed_hashes(self, space: str | None = None) -> dict[str, str]:
        """Read already-indexed vfs sources + hashes from the store (cheap SQL)."""
        out: dict[str, str] = {}
        store = self._store(space)
        if store is None:
            return out
        try:
            rows = store._exec(
                "SELECT meta_source, meta_custom FROM entries "
                "WHERE space = ? AND is_active = 1 AND meta_source LIKE 'vfs:%'",
                (store.space,),
            ).fetchall()
            for row in rows:
                src = row["meta_source"]
                path = src[4:]
                try:
                    custom = json.loads(row["meta_custom"] or "{}")
                    if "content_hash" in custom:
                        out[path] = custom["content_hash"]
                except Exception:
                    out.setdefault(path, "")
        except Exception as e:
            logger.debug(f"[VFSMemoryIndexer] hash preload failed: {e}")
        return out

    async def backfill(self):
        """Index all eligible existing files. Idempotent via content hashes."""
        # ensure spaces/stores exist so hash preload has something to read
        for sp in (self.space, GLOBAL_INDEX_SPACE):
            try:
                self.memory._get_or_create_store(self.memory._sanitize_name(sp))
            except Exception:
                pass
        self._hashes.update(self._load_indexed_hashes())
        self._hashes.update(self._load_indexed_hashes(GLOBAL_INDEX_SPACE))
        count = 0
        for path in list(self.vfs.files.keys()):
            if not self._eligible(path):
                continue
            await self._index_path(path)
            count += 1
        logger.info(
            f"[VFSMemoryIndexer:{self.space}] backfill scanned {count} files"
        )


# =============================================================================
# GLOBAL INDEX TOOL — one-shot / repeatable indexing of the /global folder
# =============================================================================


def _global_hash_current(store, source: str, h: str) -> bool:
    """True if the store already holds ACTIVE entries for source with this hash."""
    try:
        rows = store._exec(
            "SELECT meta_custom FROM entries "
            "WHERE space = ? AND is_active = 1 AND meta_source = ? LIMIT 1",
            (store.space, source),
        ).fetchall()
        for row in rows:
            custom = json.loads(row["meta_custom"] or "{}")
            if custom.get("content_hash") == h:
                return True
    except Exception:
        pass
    return False


async def index_global_memory(memory, only_new: bool = True) -> dict:
    """Index the ENTIRE /global folder (from disk, recursively) into the
    shared GLOBAL_INDEX_SPACE. Skips content that is already indexed and
    unchanged (hash check) when only_new=True. Uses structure-aware chunking.

    Safe to call repeatedly — idempotent. Registered as an agent tool so the
    agent (or the user) can trigger a full /global refresh on demand.
    """
    from pathlib import Path

    from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs

    gvfs = get_global_vfs()
    root = Path(gvfs.data_dir)
    stats = {"scanned": 0, "indexed": 0, "skipped_unchanged": 0,
             "skipped_binary": 0, "errors": []}

    try:
        memory._get_or_create_store(memory._sanitize_name(GLOBAL_INDEX_SPACE))
    except Exception:
        pass
    stores = memory.get(GLOBAL_INDEX_SPACE)
    store = stores[0] if stores else None

    for fp in sorted(root.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(root).as_posix()
        vpath = f"{GLOBAL_PREFIX}/{rel}"
        stats["scanned"] += 1
        try:
            raw = fp.read_bytes()
            if b"\x00" in raw[:1024]:
                stats["skipped_binary"] += 1
                continue
            content = raw.decode("utf-8", "replace")
            if not content.strip():
                continue
            h = hashlib.sha1(content.encode("utf-8", "replace")).hexdigest()
            source = f"vfs:{vpath}"
            if only_new and store is not None and _global_hash_current(store, source, h):
                stats["skipped_unchanged"] += 1
                continue
            if store is not None:
                try:
                    store.invalidate_by_source(source)
                except Exception:
                    pass
            ok = await memory.add_data(
                GLOBAL_INDEX_SPACE,
                _chunk(content),
                metadata={
                    "source": source,
                    "vfs_path": vpath,
                    "category": "vfs",
                    "content_hash": h,
                },
            )
            if ok:
                stats["indexed"] += 1
        except Exception as e:
            stats["errors"].append(f"{vpath}: {e}")
    return stats
