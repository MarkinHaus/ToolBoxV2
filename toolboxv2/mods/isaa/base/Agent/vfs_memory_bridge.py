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
import gc
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
# memfix: .memory/* holds archived tool outputs (shell/think dumps) - indexing
# them poisons recall and balloons VFSIndex/global. Excluded at any depth.
_GLOBAL_SKIP_SEGMENTS = ("/.memory/",)


def _global_path_eligible(rel: str) -> bool:
    """True if a /global-relative path should be memory-indexed.

    Gates BOTH indexing paths (live on_change + index_global_memory bulk scan).
    Excludes system/noise trees like .memory at any depth; normal knowledge
    content stays eligible.
    """
    norm = "/" + rel.replace("\\", "/")
    return not any(seg in norm for seg in _GLOBAL_SKIP_SEGMENTS)
_CHUNK_SIZE = 1400
_CHUNK_OVERLAP = 120
_MAX_CHUNKS_PER_FILE = 24
# memfix: _chunk() never emits more than _MAX_CHUNKS_PER_FILE * _CHUNK_SIZE
# (~34 KB) of text, so reading anything substantially larger only burns RAM.
_MAX_INDEX_FILE_BYTES = 256 * 1024
_DEBOUNCE_S = 3.0


def vfs_index_space(agent_name: str, session_id: str) -> str:
    return f"VFSIndex/{agent_name}.{session_id}"


def agent_index_space(agent_name: str) -> str:
    """Agent-global VFS index — backfilled ONCE per agent and shared across all
    that agent's sessions (bulk existing files live here). Session spaces hold
    only the few session-specific files, written live via on_change."""
    return f"VFSIndex/{agent_name}"


# per-agent backfill guard: ensures the (expensive) initial scan runs at most
# once per agent process, no matter how many sessions/sub-agents spin up.
_AGENT_BACKFILL_LOCK = __import__("asyncio").Lock()
_AGENT_BACKFILL_DONE: set[str] = set()


class _IndexProgressHub:
    """Process-wide singleton that coalesces ALL concurrent VFS/global index
    jobs into ONE shared spinner.

    THE BUG THIS FIXES: the icli spins up several agent sessions; each ran its
    own backfill concurrently, and each created its own Spinner. Under the
    prompt-toolkit patch the render loop draws the primary line PLUS every
    secondary spinner's info — so N concurrent index jobs produced
    "… 279/668 …  [… 279/668 … | … 279/668 … | …]" stacked in the bottom
    toolbar instead of one line.

    THE FIX: there is only ever ONE Spinner for all VFS indexing. Each job
    registers with a key and pushes its (done, total); the hub renders a single
    aggregated line (summed done/total across live jobs). When the last job
    finishes the spinner is closed exactly once. Thread-safe (backfills run in
    different tasks/threads)."""

    _instance: "_IndexProgressHub | None" = None
    _cls_lock = __import__("threading").Lock()

    def __new__(cls):
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self):
        import threading

        self._lock = threading.Lock()
        self._jobs: dict[str, tuple[int, int]] = {}  # key → (done, total)
        self._spinner = None

    def _label(self) -> str:
        done = sum(d for d, _ in self._jobs.values())
        total = sum(t for _, t in self._jobs.values())
        remaining = max(0, total - done)
        pct = int(done * 100 / total) if total else 0
        n = len(self._jobs)
        scope = "memory index" if n == 1 else f"memory index ×{n}"
        return f"Indexing into {scope} {done}/{total} ({pct}%, {remaining} left)"

    def start(self, key: str, total: int, enabled: bool = True):
        with self._lock:
            self._jobs[key] = (0, total)
            if enabled and self._spinner is None:
                try:
                    from toolboxv2 import Spinner

                    self._spinner = Spinner(message=self._label(), symbols="d")
                    self._spinner.__enter__()
                except Exception:
                    self._spinner = None
            self._refresh()

    def update(self, key: str, done: int):
        with self._lock:
            if key in self._jobs:
                self._jobs[key] = (done, self._jobs[key][1])
                self._refresh()

    def finish(self, key: str):
        with self._lock:
            self._jobs.pop(key, None)
            if self._jobs:
                self._refresh()
            elif self._spinner is not None:
                # last job → close the single spinner exactly once
                try:
                    self._spinner.__exit__(None, None, None)
                except Exception:
                    pass
                finally:
                    self._spinner = None

    def _refresh(self):
        if self._spinner is not None:
            try:
                self._spinner.message = self._label()
            except Exception:
                pass


class _ProgressReporter:
    """Thin per-job handle over the shared _IndexProgressHub. A `progress_cb`,
    when supplied, bypasses the hub entirely and routes to that callback (UI)."""

    _counter = 0
    _counter_lock = __import__("threading").Lock()

    def __init__(self, label: str, total: int, cb=None, enabled: bool = True):
        self.total = max(0, int(total))
        self.cb = cb
        self._closed = False
        self._key = None
        if cb is None and enabled and self.total > 0:
            with _ProgressReporter._counter_lock:
                _ProgressReporter._counter += 1
                self._key = f"{label}#{_ProgressReporter._counter}"
            _IndexProgressHub().start(self._key, self.total, enabled=True)

    def update(self, done: int, path: str = "", suffix: str = ""):
        if self.cb is not None:
            try:
                self.cb(done, self.total, path)
            except Exception:
                pass
        elif self._key is not None:
            _IndexProgressHub().update(self._key, done)

    def close(self, final_message: str = ""):
        if self._closed:
            return
        self._closed = True
        if self._key is not None:
            _IndexProgressHub().finish(self._key)
            self._key = None


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
        self.agent_space = agent_index_space(session.agent_name)
        self.agent_name = session.agent_name
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

    def _space_for(self, path: str, session_scoped: bool = False) -> str:
        """/global → shared global space. Otherwise: the initial backfill writes
        into the agent-global space (once per agent, reused by all sessions);
        live on_change writes go to this session's space (the few
        session-specific files)."""
        if path.startswith(GLOBAL_PREFIX):
            return GLOBAL_INDEX_SPACE
        return self.space if session_scoped else self.agent_space

    def _eligible(self, path: str) -> bool:
        if any(path.startswith(p) for p in _SKIP_PREFIXES):
            return False
        # memfix: skip oversized files before anything reads them into RAM.
        f_size = self.vfs.files.get(path)
        if f_size is not None:
            try:
                if int(getattr(f_size, "size_bytes", 0) or 0) > _MAX_INDEX_FILE_BYTES:
                    return False
            except (TypeError, ValueError):
                pass
        # /global IS a mount + shared store — but explicitly wanted in memory.
        # Allow it before the mount/shared checks below would reject it.
        if path.startswith(GLOBAL_PREFIX):
            if not _global_path_eligible(path[len(GLOBAL_PREFIX):]):
                return False
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
                    await self._index_path(p, session_scoped=True)
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
            await self._index_path(p, session_scoped=True)

    def save_index(self):
        """Persist FAISS vectors for all VFSIndex spaces to disk.

        Critical: without this call, FAISS vectors live in RAM only and are
        lost on every process restart. The SQLite entries survive (WAL flush),
        but vector search returns nothing → recall degrades to BM25-only.
        """
        for sp in (self.space, self.agent_space, GLOBAL_INDEX_SPACE):
            store = self._store(sp)
            if store is None:
                continue
            try:
                if store._faiss.index.ntotal > 0:
                    store.save()
            except Exception as e:
                logger.debug(f"[VFSMemoryIndexer] save_index {sp}: {e}")

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
        try:
            content = getattr(f, "_content", None)
            if content is None:
                content = getattr(f, "content", None)
        except Exception:
            return None
        return content if isinstance(content, str) else None

    def _store(self, space: str | None = None):
        stores = self.memory.get(space or self.space)
        return stores[0] if stores else None

    async def _unindex(self, path: str):
        # a path may have been indexed in the agent space (backfill) or this
        # session's space (live) — invalidate both; global stays separate.
        if path.startswith(GLOBAL_PREFIX):
            spaces = [GLOBAL_INDEX_SPACE]
        else:
            spaces = [self.space, self.agent_space]
        for sp in spaces:
            store = self._store(sp)
            if store is not None:
                try:
                    store.invalidate_by_source(f"vfs:{path}")
                except Exception as e:
                    logger.debug(f"[VFSMemoryIndexer] unindex {path} in {sp}: {e}")
        self._hashes.pop(path, None)

    async def _index_path(self, path: str, session_scoped: bool = False):
        if not self._eligible(path):
            return
        content = self._get_content(path)
        if not content or not content.strip():
            return
        h = hashlib.sha1(content.encode("utf-8", "replace")).hexdigest()
        if self._hashes.get(path) == h:
            return  # unchanged
        space = self._space_for(path, session_scoped)
        source = f"vfs:{path}"
        store = self._store(space)
        if store is not None:
            if space in (GLOBAL_INDEX_SPACE, self.agent_space) and \
                    _global_hash_current(store, source, h):
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
        """Read already-indexed vfs sources + hashes from the store (cheap SQL).

        Embedding BLOBs in SQLite are the source of truth (Option C): rows
        without a BLOB (embedding IS NULL, e.g. migrated legacy entries) are
        not considered indexed, so backfill re-embeds exactly the files whose
        vectors are missing. An empty FAISS RAM index is irrelevant - it is
        rebuilt from the BLOBs on load.
        """
        out: dict[str, str] = {}
        store = self._store(space)
        if store is None:
            return out
        try:
            rows = store._exec(
                "SELECT meta_source, meta_custom FROM entries "
                "WHERE space = ? AND is_active = 1 AND meta_source LIKE 'vfs:%' "
                "AND embedding IS NOT NULL",
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

    async def backfill(self, show_progress: bool = True, progress_cb=None):
        """Index existing VFS files ONCE PER AGENT into the agent-global space,
        shared by all of that agent's sessions. Guarded so concurrent sessions
        / sub-agents don't each re-scan (previously produced ×80 stacked jobs).
        Idempotent via content hashes.

        `progress_cb(done, total, path)` overrides the default inline progress."""
        # run at most once per agent process
        async with _AGENT_BACKFILL_LOCK:
            if self.agent_name in _AGENT_BACKFILL_DONE:
                return
            _AGENT_BACKFILL_DONE.add(self.agent_name)

        # ensure spaces/stores exist so hash preload has something to read
        for sp in (self.agent_space, GLOBAL_INDEX_SPACE):
            try:
                self.memory._get_or_create_store(self.memory._sanitize_name(sp))
            except Exception:
                pass
        self._hashes.update(self._load_indexed_hashes(self.agent_space))
        self._hashes.update(self._load_indexed_hashes(GLOBAL_INDEX_SPACE))

        # count total up front so the bar has a denominator
        todo = [p for p in list(self.vfs.files.keys()) if self._eligible(p)]
        total = len(todo)
        if total == 0:
            logger.info(f"[VFSMemoryIndexer:{self.agent_space}] backfill: nothing to index")
            return

        bar = _ProgressReporter(
            "Indexing VFS into memory", total,
            cb=progress_cb, enabled=show_progress)
        done = 0
        try:
            for path in todo:
                await self._index_path(path)  # agent-scoped (session_scoped=False)
                done += 1
                bar.update(done, path)
                if done % 50 == 0:
                    self.save_index()
                    gc.collect()
            # final save after backfill complete
            self.save_index()
        finally:
            bar.close(f"VFS indexed: {done}/{total} files")
        logger.info(
            f"[VFSMemoryIndexer:{self.agent_space}] backfill indexed {done}/{total} files"
        )


# =============================================================================
# GLOBAL INDEX TOOL — one-shot / repeatable indexing of the /global folder
# =============================================================================


def _global_hash_current(store, source: str, h: str) -> bool:
    """True if the store already holds ACTIVE entries for source with this hash.

    Since we store embeddings in SQLite BLOBs, we no longer force a re-index
    if FAISS is empty (it will be rebuilt from SQLite on load).
    """
    try:
        rows = store._exec(
            "SELECT meta_custom FROM entries "
            "WHERE space = ? AND is_active = 1 AND meta_source = ? "
            "AND embedding IS NOT NULL LIMIT 1",
            (store.space, source),
        ).fetchall()
        for row in rows:
            custom = json.loads(row["meta_custom"] or "{}")
            if custom.get("content_hash") == h:
                return True
    except Exception:
        pass
    return False


async def index_global_memory(memory, only_new: bool = True,
                              show_progress: bool = True, progress_cb=None) -> dict:
    """Index the ENTIRE /global folder (from disk, recursively) into the
    shared GLOBAL_INDEX_SPACE. Skips content that is already indexed and
    unchanged (hash check) when only_new=True. Uses structure-aware chunking.

    Safe to call repeatedly — idempotent. Registered as an agent tool so the
    agent (or the user) can trigger a full /global refresh on demand.
    Shows a progress bar (total files + how many remain) unless disabled;
    `progress_cb(done, total, path)` overrides the default Spinner."""
    from pathlib import Path

    from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs

    gvfs = get_global_vfs()
    root = Path(gvfs.data_dir)
    stats = {"scanned": 0, "indexed": 0, "skipped_unchanged": 0,
             "skipped_binary": 0, "skipped_excluded": 0, "errors": []}

    try:
        memory._get_or_create_store(memory._sanitize_name(GLOBAL_INDEX_SPACE))
    except Exception:
        pass
    stores = memory.get(GLOBAL_INDEX_SPACE)
    store = stores[0] if stores else None

    # materialize + count up front so the bar has a denominator
    files = [fp for fp in sorted(root.rglob("*")) if fp.is_file()]
    total = len(files)

    bar = _ProgressReporter(
        "Indexing /global into memory", total,
        cb=progress_cb, enabled=show_progress)
    done = 0
    try:
        for fp in files:
            rel = fp.relative_to(root).as_posix()
            vpath = f"{GLOBAL_PREFIX}/{rel}"
            stats["scanned"] += 1
            if not _global_path_eligible(rel):
                stats["skipped_excluded"] += 1
                done += 1
                continue
            try:
                raw = fp.read_bytes()
                if b"\x00" in raw[:1024]:
                    stats["skipped_binary"] += 1
                else:
                    content = raw.decode("utf-8", "replace")
                    if content.strip():
                        h = hashlib.sha1(content.encode("utf-8", "replace")).hexdigest()
                        source = f"vfs:{vpath}"
                        if only_new and store is not None and _global_hash_current(store, source, h):
                            stats["skipped_unchanged"] += 1
                        else:
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
            done += 1
            bar.update(done, vpath,
                       suffix=f"{stats['indexed']} new")
    finally:
        bar.close(
            f"/global indexed: {stats['indexed']} new, "
            f"{stats['skipped_unchanged']} unchanged, {total} scanned")
    return stats
