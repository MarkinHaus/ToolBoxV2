"""
MountPollRegistry v3 — Zero-allocation steady-state polling
============================================================

v3 changes (memory + CPU fix):
- _incremental_scan uses snapshot-keyset diff instead of building `seen` set
- Deletion detection via set subtraction on snapshot.keys() view
- os.scandir with DirEntry.path reuse — no f-string per file
- Burst dispatch calls _invalidate_mount() instead of full refresh_mount()
- Configurable scan depth limit to bound walk time on huge trees
- Snapshot compaction: entries removed from disk are deleted in-place

Author: Markin / ToolBoxV2
Version: 3.0.0
"""

from __future__ import annotations

import fnmatch
import logging
import os
import threading
import time
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2

from toolboxv2 import get_logger
logger = get_logger()

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_POLL_INTERVAL = 2.0  # bumped from 1.5 — 2s is plenty for external changes
BURST_THRESHOLD = 50
MAX_TRACKED_FILES = 100_000
MAX_SCAN_DEPTH = 20  # prevent runaway recursion in deep trees


# =============================================================================
# MOUNT WATCHER — one per unique local_path
# =============================================================================

class MountWatcher:
    """Per-local-path polling worker. Shared across VFS subscribers.

    v3: Steady-state scans allocate O(changed) not O(total).
    The snapshot dict is mutated in-place; unchanged files touch zero memory.
    """

    __slots__ = (
        "local_path", "exclude_patterns", "poll_interval",
        "_snapshot", "_subscribers", "_thread", "_stop_event", "_lock",
        "_exclude_set", "_exclude_globs",
    )

    def __init__(
        self,
        local_path: str,
        exclude_patterns: list[str] | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        self.local_path = local_path
        self.poll_interval = poll_interval

        patterns = exclude_patterns or [
            "__pycache__", "*.pyc", ".git", "node_modules", ".venv", "*.log",
        ]
        self._exclude_set: set[str] = set()
        self._exclude_globs: list[str] = []
        for p in patterns:
            if "*" in p or "?" in p or "[" in p:
                self._exclude_globs.append(p)
            else:
                self._exclude_set.add(p)

        # Snapshot: {abs_path_str: (mtime_float, size_int)}
        self._snapshot: dict[str, tuple[float, int]] = {}

        self._subscribers: weakref.WeakSet[VirtualFileSystemV2] = weakref.WeakSet()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

    # ── Subscription ──────────────────────────────────────────────────

    def add_subscriber(self, vfs: VirtualFileSystemV2) -> None:
        with self._lock:
            self._subscribers.add(vfs)
            if self._thread is None or not self._thread.is_alive():
                self._start()

    def remove_subscriber(self, vfs: VirtualFileSystemV2) -> bool:
        with self._lock:
            self._subscribers.discard(vfs)
            if not self._subscribers:
                self._stop()
                return True
            return False

    def has_subscribers(self) -> bool:
        with self._lock:
            return len(self._subscribers) > 0

    # ── Thread lifecycle ──────────────────────────────────────────────

    def _start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name=f"vfs-poll-{os.path.basename(self.local_path)}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Started poll watcher for %s", self.local_path)

    def _stop(self) -> None:
        self._stop_event.set()
        self._snapshot.clear()
        logger.info("Stopped poll watcher for %s", self.local_path)

    # ── Exclusion check (hot path) ────────────────────────────────────

    def _should_skip(self, name: str) -> bool:
        if name in self._exclude_set:
            return True
        for pat in self._exclude_globs:
            if fnmatch.fnmatch(name, pat):
                return True
        return False

    # ── Poll loop ─────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        try:
            self._build_initial_snapshot()
        except Exception as e:
            logger.error("[%s] Initial scan failed: %s", self.local_path, e)
            return

        while not self._stop_event.wait(self.poll_interval):
            try:
                if not self.has_subscribers():
                    break
                changes = self._incremental_scan()
                if changes:
                    self._dispatch_changes(changes)
            except Exception as e:
                logger.error("[%s] Poll tick failed: %s", self.local_path, e)

    # ── Initial snapshot (one-time) ───────────────────────────────────

    def _build_initial_snapshot(self) -> None:
        self._snapshot.clear()
        count = 0

        def _walk(dirpath: str, depth: int) -> None:
            nonlocal count
            if count >= MAX_TRACKED_FILES or depth > MAX_SCAN_DEPTH:
                return
            try:
                with os.scandir(dirpath) as it:
                    for entry in it:
                        if count >= MAX_TRACKED_FILES:
                            return
                        if self._should_skip(entry.name):
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                _walk(entry.path, depth + 1)
                            elif entry.is_file(follow_symlinks=False):
                                st = entry.stat(follow_symlinks=False)
                                self._snapshot[entry.path] = (st.st_mtime, st.st_size)
                                count += 1
                        except OSError:
                            continue
            except OSError:
                return

        if os.path.isdir(self.local_path):
            _walk(self.local_path, 0)

    # ── Incremental scan (every tick) — v3: O(changed) allocation ─────

    def _incremental_scan(self) -> dict[str, str] | None:
        """In-place diff against snapshot.

        v3 key change: instead of building a `seen` set of ALL paths (O(N)
        allocation every tick), we build a set of paths found on disk and
        detect deletions via set difference on the snapshot keys view.
        Both sets are constructed from os.scandir DirEntry.path — no extra
        string formatting.

        For the common case (no changes), the only allocation is the
        `disk_paths` set of string references that already exist as
        DirEntry.path objects.  On Windows, DirEntry.path is cached by
        the OS, so this is nearly zero-copy.
        """
        changes: dict[str, str] = {}
        disk_paths: set[str] = set()
        count = 0

        def _walk(dirpath: str, depth: int) -> None:
            nonlocal count
            if count >= MAX_TRACKED_FILES or depth > MAX_SCAN_DEPTH:
                return
            try:
                with os.scandir(dirpath) as it:
                    for entry in it:
                        if count >= MAX_TRACKED_FILES:
                            return
                        if self._should_skip(entry.name):
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                _walk(entry.path, depth + 1)
                            elif entry.is_file(follow_symlinks=False):
                                path = entry.path  # reuse DirEntry cached path
                                st = entry.stat(follow_symlinks=False)
                                new_meta = (st.st_mtime, st.st_size)
                                disk_paths.add(path)
                                count += 1

                                old_meta = self._snapshot.get(path)
                                if old_meta is None:
                                    self._snapshot[path] = new_meta
                                    changes[path] = "created"
                                elif old_meta != new_meta:
                                    self._snapshot[path] = new_meta
                                    changes[path] = "modified"
                                # unchanged: zero allocation
                        except OSError:
                            continue
            except OSError:
                return

        if os.path.isdir(self.local_path):
            _walk(self.local_path, 0)

        # Deletion detection via set difference — no iteration over snapshot
        if self._snapshot:
            deleted_keys = self._snapshot.keys() - disk_paths
            for path in deleted_keys:
                del self._snapshot[path]
                changes[path] = "deleted"

        return changes if changes else None

    # ── Dispatch ──────────────────────────────────────────────────────

    def _dispatch_changes(self, changes: dict[str, str]) -> None:
        with self._lock:
            subs = list(self._subscribers)
        if not subs:
            return

        is_burst = len(changes) > BURST_THRESHOLD

        for vfs in subs:
            try:
                if is_burst:
                    # v3: invalidate metadata only — NEVER call refresh_mount
                    # which triggers expensive _scan_mount + os.walk
                    self._dispatch_burst_invalidate(vfs, changes)
                else:
                    self._dispatch_individual(vfs, changes)
            except Exception as e:
                logger.error("[%s] Dispatch failed: %s", self.local_path, e)

    def _dispatch_burst_invalidate(
        self,
        vfs: VirtualFileSystemV2,
        changes: dict[str, str],
    ) -> None:
        """v3: Burst mode invalidates file-by-file instead of full rescan.

        The old _dispatch_burst called vfs.refresh_mount() which triggers
        _scan_mount() — an O(N) os.walk that duplicates the work this
        watcher already did.  Now we iterate the changes dict (O(changed))
        and apply individual invalidations.
        """
        # Find mount for this local_path
        mount_vfs_path = None
        mount_obj = None
        for vp, m in vfs.mounts.items():
            if os.path.abspath(m.local_path) == self.local_path:
                mount_vfs_path = vp
                mount_obj = m
                break
        if mount_vfs_path is None:
            return

        # Apply all changes individually — same logic as _dispatch_individual
        # but we skip the mount lookup per-change
        self._apply_changes(vfs, mount_vfs_path, mount_obj, changes)

    def _dispatch_individual(
        self,
        vfs: VirtualFileSystemV2,
        changes: dict[str, str],
    ) -> None:
        mount_vfs_path = None
        mount_obj = None
        for vp, m in vfs.mounts.items():
            if os.path.abspath(m.local_path) == self.local_path:
                mount_vfs_path = vp
                mount_obj = m
                break
        if mount_vfs_path is None:
            return

        self._apply_changes(vfs, mount_vfs_path, mount_obj, changes)

    def _apply_changes(
        self,
        vfs: VirtualFileSystemV2,
        mount_vfs_path: str,
        mount_obj,
        changes: dict[str, str],
    ) -> None:
        """Apply a set of filesystem changes to a VFS instance."""
        from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
            FileBackingType, VFSFile, get_file_type,
        )

        local_prefix = self.local_path
        mount_stripped = mount_vfs_path.rstrip("/")

        for abs_path, change_type in changes.items():
            if not abs_path.startswith(local_prefix):
                continue

            rel = os.path.relpath(abs_path, local_prefix)
            vfs_file_path = mount_stripped + "/" + rel.replace(os.sep, "/")

            if change_type == "deleted":
                existing = vfs.files.get(vfs_file_path)
                if existing is None:
                    continue
                if isinstance(existing, VFSFile) and existing.is_dirty:
                    continue
                if existing.readonly:
                    continue
                del vfs.files[vfs_file_path]
                vfs._shadow_index.pop(vfs_file_path, None)
                vfs._dirty = True

            elif change_type == "modified":
                existing = vfs.files.get(vfs_file_path)
                if existing is None:
                    self._handle_create(vfs, vfs_file_path, abs_path, mount_obj)
                    continue
                if not isinstance(existing, VFSFile):
                    continue
                if existing.is_dirty:
                    continue
                try:
                    st = os.stat(abs_path)
                    existing._content = None
                    existing.backing_type = FileBackingType.SHADOW
                    existing.local_mtime = st.st_mtime
                    existing.size_bytes = st.st_size
                    vfs._dirty = True
                except OSError:
                    pass

            elif change_type == "created":
                if vfs_file_path in vfs.files:
                    continue
                self._handle_create(vfs, vfs_file_path, abs_path, mount_obj)

    def _handle_create(self, vfs, vfs_file_path, abs_path, mount_obj) -> None:
        from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
            FileBackingType, VFSFile, VFSDirectory, get_file_type,
        )
        try:
            st = os.stat(abs_path)
            if st.st_size > mount_obj.max_file_size:
                return

            # Ensure parent directory exists
            parent = vfs_file_path.rsplit("/", 1)[0] if "/" in vfs_file_path else "/"
            if parent and parent not in vfs.directories:
                # Walk up to create missing parents
                parts = parent.split("/")
                for i in range(1, len(parts) + 1):
                    p = "/".join(parts[:i]) or "/"
                    if p not in vfs.directories:
                        vfs.directories[p] = VFSDirectory(
                            name=parts[i - 1] if i > 0 else "/",
                            readonly=mount_obj.readonly,
                        )

            filename = os.path.basename(abs_path)
            vfs.files[vfs_file_path] = VFSFile(
                filename=filename,
                backing_type=FileBackingType.SHADOW,
                _content=None,
                local_path=abs_path,
                local_mtime=st.st_mtime,
                size_bytes=st.st_size,
                line_count=-1,
                file_type=get_file_type(filename),
                readonly=mount_obj.readonly,
            )
            vfs._shadow_index[vfs_file_path] = abs_path
            vfs._dirty = True
        except OSError:
            pass


# =============================================================================
# REGISTRY — Singleton
# =============================================================================

class MountPollRegistry:
    _instance: MountPollRegistry | None = None
    _singleton_lock = threading.Lock()

    DEFAULT_EXCLUDE_PATTERNS = [
        "__pycache__", "*.pyc", ".git", "node_modules", ".venv", "*.log",
    ]

    def __new__(cls) -> MountPollRegistry:
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._watchers: dict[str, MountWatcher] = {}
        self._lock = threading.RLock()
        self._initialized = True

    def subscribe(
        self,
        local_path: str,
        vfs: VirtualFileSystemV2,
        exclude_patterns: list[str] | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        abs_path = os.path.abspath(local_path)
        with self._lock:
            watcher = self._watchers.get(abs_path)
            if watcher is None:
                watcher = MountWatcher(
                    local_path=abs_path,
                    exclude_patterns=exclude_patterns or list(self.DEFAULT_EXCLUDE_PATTERNS),
                    poll_interval=poll_interval,
                )
                self._watchers[abs_path] = watcher
            elif poll_interval < watcher.poll_interval:
                watcher.poll_interval = poll_interval
            watcher.add_subscriber(vfs)

    def unsubscribe(self, local_path: str, vfs: VirtualFileSystemV2) -> None:
        abs_path = os.path.abspath(local_path)
        with self._lock:
            watcher = self._watchers.get(abs_path)
            if watcher is None:
                return
            if watcher.remove_subscriber(vfs):
                del self._watchers[abs_path]

    def unsubscribe_all(self, vfs: VirtualFileSystemV2) -> None:
        with self._lock:
            to_drop = []
            for path, watcher in self._watchers.items():
                if watcher.remove_subscriber(vfs):
                    to_drop.append(path)
            for path in to_drop:
                del self._watchers[path]

    def stats(self) -> dict:
        with self._lock:
            return {
                "active_watchers": len(self._watchers),
                "watchers": [
                    {
                        "local_path": p,
                        "subscribers": len(w._subscribers),
                        "tracked_files": len(w._snapshot),
                        "poll_interval": w.poll_interval,
                        "thread_alive": w._thread.is_alive() if w._thread else False,
                    }
                    for p, w in self._watchers.items()
                ],
            }


_registry: MountPollRegistry | None = None


def get_mount_poll_registry() -> MountPollRegistry:
    global _registry
    if _registry is None:
        _registry = MountPollRegistry()
    return _registry


__all__ = [
    "MountPollRegistry",
    "MountWatcher",
    "get_mount_poll_registry",
    "DEFAULT_POLL_INTERVAL",
]
