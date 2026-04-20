"""
MountPollRegistry - Background polling for multi-agent VFS mount sync
======================================================================

Watches local mount paths for external changes (other agents, editors, git)
and invalidates the corresponding entries in subscribed VFS instances.

Design principles:
- One poll thread per unique local_path (shared across VFS subscribers)
- Refcounted subscriptions — last unsubscribe stops the poller
- Lazy invalidation: mark file._content = None, don't reload proactively
- Respects is_dirty: never clobbers unsynced agent work
- Debounced: multi-rapid changes coalesce into one invalidation
- Burst-safe: threshold triggers full _scan_mount instead of per-file handling

Integration:
- VirtualFileSystemV2.mount()   → registry.subscribe(local_path, vfs)
- VirtualFileSystemV2.unmount() → registry.unsubscribe(local_path, vfs)
- VirtualFileSystemV2.__del__   → registry.unsubscribe_all(vfs)  (safety net)

Author: Markin / ToolBoxV2
Version: 1.0.0
"""

from __future__ import annotations

import fnmatch
import logging
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2

logger = logging.getLogger("vfs.poll")


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_POLL_INTERVAL = 1.5      # seconds between disk scans
DEFAULT_DEBOUNCE_WINDOW = 0.3    # coalesce rapid changes within this window
BURST_THRESHOLD = 50             # >N changed files in one tick → full rescan
MAX_TRACKED_FILES_PER_MOUNT = 100_000  # safety cap


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FileSnapshot:
    """Lightweight disk metadata for change detection."""
    mtime: float
    size: int


@dataclass
class MountWatcher:
    """
    Per-local-path polling worker.

    Shared across all VFS instances that mount the same local_path.
    Refcounted via subscribers set.
    """
    local_path: str
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "__pycache__", "*.pyc", ".git", "node_modules", ".venv", "*.log",
    ])
    poll_interval: float = DEFAULT_POLL_INTERVAL

    # Runtime state
    _snapshot: dict[str, FileSnapshot] = field(default_factory=dict)
    _subscribers: "weakref.WeakSet[VirtualFileSystemV2]" = field(
        default_factory=weakref.WeakSet
    )
    _thread: threading.Thread | None = None
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _pending_changes: dict[str, str] = field(default_factory=dict)  # path -> change_type
    _last_flush: float = 0.0

    def add_subscriber(self, vfs: "VirtualFileSystemV2") -> None:
        with self._lock:
            self._subscribers.add(vfs)
            if self._thread is None or not self._thread.is_alive():
                self._start()

    def remove_subscriber(self, vfs: "VirtualFileSystemV2") -> bool:
        """Returns True if the watcher has no more subscribers (caller can drop it)."""
        with self._lock:
            self._subscribers.discard(vfs)
            if not self._subscribers:
                self._stop()
                return True
            return False

    def has_subscribers(self) -> bool:
        with self._lock:
            # WeakSet auto-cleans, but force a len() check
            return len(self._subscribers) > 0

    def _start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name=f"vfs-poll-{os.path.basename(self.local_path)}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Started poll watcher for {self.local_path}")

    def _stop(self) -> None:
        self._stop_event.set()
        # Do not join — daemon thread exits on next tick
        logger.info(f"Stopped poll watcher for {self.local_path}")

    # ──────────────────────────────────────────────────────────────────────
    # POLL LOOP
    # ──────────────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Main loop — runs until stop_event is set or no subscribers remain."""
        # Initial snapshot
        try:
            self._snapshot = self._scan_disk()
        except Exception as e:
            logger.error(f"[{self.local_path}] Initial scan failed: {e}")
            return

        while not self._stop_event.wait(self.poll_interval):
            try:
                if not self.has_subscribers():
                    break

                new_snapshot = self._scan_disk()
                changes = self._diff_snapshots(self._snapshot, new_snapshot)
                self._snapshot = new_snapshot

                if changes:
                    self._dispatch_changes(changes)

            except Exception as e:
                logger.error(f"[{self.local_path}] Poll iteration failed: {e}")
                # Continue — one bad tick shouldn't kill the watcher

    def _scan_disk(self) -> dict[str, FileSnapshot]:
        """
        Walk mount dir, return {abs_path: FileSnapshot}.

        Uses os.scandir() for speed. Respects exclude_patterns.
        Caps at MAX_TRACKED_FILES_PER_MOUNT (safety).
        """
        snapshot: dict[str, FileSnapshot] = {}
        count = 0

        def should_skip(name: str) -> bool:
            for pat in self.exclude_patterns:
                if fnmatch.fnmatch(name, pat):
                    return True
            return False

        def walk(dirpath: str) -> None:
            nonlocal count
            if count >= MAX_TRACKED_FILES_PER_MOUNT:
                return
            try:
                with os.scandir(dirpath) as it:
                    for entry in it:
                        if count >= MAX_TRACKED_FILES_PER_MOUNT:
                            return
                        if should_skip(entry.name):
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                walk(entry.path)
                            elif entry.is_file(follow_symlinks=False):
                                st = entry.stat(follow_symlinks=False)
                                snapshot[entry.path] = FileSnapshot(
                                    mtime=st.st_mtime,
                                    size=st.st_size,
                                )
                                count += 1
                        except OSError:
                            continue
            except OSError:
                return

        if os.path.isdir(self.local_path):
            walk(self.local_path)

        return snapshot

    def _diff_snapshots(
        self,
        old: dict[str, FileSnapshot],
        new: dict[str, FileSnapshot],
    ) -> dict[str, str]:
        """
        Compute changes between snapshots.

        Returns: {abs_path: change_type}  where change_type ∈ {'created', 'modified', 'deleted'}
        """
        changes: dict[str, str] = {}

        for path, snap in new.items():
            if path not in old:
                changes[path] = "created"
            else:
                old_snap = old[path]
                if snap.mtime != old_snap.mtime or snap.size != old_snap.size:
                    changes[path] = "modified"

        for path in old:
            if path not in new:
                changes[path] = "deleted"

        return changes

    # ──────────────────────────────────────────────────────────────────────
    # DISPATCH
    # ──────────────────────────────────────────────────────────────────────

    def _dispatch_changes(self, changes: dict[str, str]) -> None:
        """
        Notify all subscribers about changes.

        For bursts (>BURST_THRESHOLD), trigger a full refresh_mount instead
        of per-file handling to avoid O(n) per-subscriber work.
        """
        # Grab a strong-ref list of live subscribers under lock
        with self._lock:
            subs = list(self._subscribers)

        if not subs:
            return

        is_burst = len(changes) > BURST_THRESHOLD

        for vfs in subs:
            try:
                if is_burst:
                    self._dispatch_burst(vfs)
                else:
                    self._dispatch_individual(vfs, changes)
            except Exception as e:
                logger.error(f"[{self.local_path}] Dispatch to VFS failed: {e}")

    def _dispatch_burst(self, vfs: "VirtualFileSystemV2") -> None:
        """
        Burst mode — many changes at once. Trigger a full refresh_mount on
        the VFS, which handles zombies and MODIFIED preservation (Ebene 1).
        """
        # Find the vfs_path this VFS uses for our local_path
        for vfs_path, mount in vfs.mounts.items():
            if mount.local_path == self.local_path:
                try:
                    vfs.refresh_mount(vfs_path)
                except Exception as e:
                    logger.error(
                        f"[{self.local_path}] Burst refresh_mount failed: {e}"
                    )
                break

    def _dispatch_individual(
        self,
        vfs: "VirtualFileSystemV2",
        changes: dict[str, str],
    ) -> None:
        """
        Fine-grained mode — invalidate per file.

        Strategy:
        - created  → add shadow entry
        - modified → invalidate cached content (set _content = None)
                     so next read() reloads from disk
        - deleted  → remove from VFS (unless dirty — then preserve and warn)
        """
        # Find the vfs_path for this local_path
        mount_vfs_path = None
        mount_obj = None
        for vp, m in vfs.mounts.items():
            if m.local_path == self.local_path:
                mount_vfs_path = vp
                mount_obj = m
                break

        if mount_vfs_path is None:
            return  # Mount was removed between dispatch and handling

        # Import here to avoid circular
        from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
            FileBackingType,
            VFSFile,
            VFSDirectory,
            get_file_type,
        )

        for abs_path, change_type in changes.items():
            # Skip if outside our mount (shouldn't happen but defensive)
            if not abs_path.startswith(self.local_path):
                continue

            # Map local abs_path → vfs_path
            rel = os.path.relpath(abs_path, self.local_path)
            vfs_file_path = (
                mount_vfs_path.rstrip("/") + "/" + rel.replace(os.sep, "/")
            )
            vfs_file_path = vfs.__class__._normalize_path(vfs, vfs_file_path)

            if change_type == "deleted":
                existing = vfs.files.get(vfs_file_path)
                if existing is None:
                    continue
                if isinstance(existing, VFSFile) and existing.is_dirty:
                    # Preserve — agent has unsynced work
                    logger.warning(
                        f"[{self.local_path}] External delete of dirty file: "
                        f"{vfs_file_path} — keeping in-memory copy"
                    )
                    continue
                if existing.readonly:
                    continue
                del vfs.files[vfs_file_path]
                vfs._shadow_index.pop(vfs_file_path, None)
                vfs._dirty = True

            elif change_type == "modified":
                existing = vfs.files.get(vfs_file_path)
                if existing is None:
                    # File exists on disk but not in VFS — treat as created
                    self._handle_create(
                        vfs, vfs_file_path, abs_path, mount_obj
                    )
                    continue
                if not isinstance(existing, VFSFile):
                    continue
                if existing.is_dirty:
                    # Agent has unsynced work. Don't clobber. Next write()
                    # will detect the conflict (Fix #4 from Ebene 1).
                    continue
                # Invalidate cached content — next read() reloads
                try:
                    st = os.stat(abs_path)
                    existing._content = None
                    existing.backing_type = FileBackingType.SHADOW
                    existing.local_mtime = st.st_mtime
                    existing.size_bytes = st.st_size
                    vfs._dirty = True
                except OSError:
                    pass  # Race: file disappeared between poll and dispatch

            elif change_type == "created":
                if vfs_file_path in vfs.files:
                    continue  # Already known
                self._handle_create(vfs, vfs_file_path, abs_path, mount_obj)

    def _handle_create(
        self,
        vfs: "VirtualFileSystemV2",
        vfs_file_path: str,
        abs_path: str,
        mount_obj,
    ) -> None:
        """Add a new shadow entry for an externally-created file."""
        from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
            FileBackingType,
            VFSFile,
            VFSDirectory,
            get_file_type,
        )

        try:
            st = os.stat(abs_path)
            if st.st_size > mount_obj.max_file_size:
                return

            # Ensure parent dirs exist in VFS
            parent = vfs._get_parent_path(vfs_file_path)
            if not vfs._is_directory(parent):
                vfs.mkdir(parent, parents=True)

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
            pass  # Race: file gone again


# =============================================================================
# REGISTRY (Singleton)
# =============================================================================


class MountPollRegistry:
    """
    Central registry of mount watchers.

    One MountWatcher per unique local_path. Multiple VFS instances sharing
    the same mount all subscribe to the same watcher (refcounted).

    Thread-safe.
    """

    _instance: "MountPollRegistry | None" = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "MountPollRegistry":
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

    # Module-level default: falls kein exclude_patterns übergeben wird
    DEFAULT_EXCLUDE_PATTERNS = [
        "__pycache__", "*.pyc", ".git", "node_modules", ".venv", "*.log",
    ]

    def subscribe(
        self,
        local_path: str,
        vfs: "VirtualFileSystemV2",
        exclude_patterns: list[str] | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        """Subscribe a VFS to disk-change events on local_path.

        If a watcher already exists for local_path, the new subscriber joins it.
        The existing watcher's poll_interval is NOT changed — first subscriber wins.
        """
        abs_path = os.path.abspath(local_path)
        with self._lock:
            watcher = self._watchers.get(abs_path)
            if watcher is None:
                watcher = MountWatcher(
                    local_path=abs_path,
                    exclude_patterns=list(exclude_patterns) if exclude_patterns
                    else list(self.DEFAULT_EXCLUDE_PATTERNS),
                    poll_interval=poll_interval,
                )
                self._watchers[abs_path] = watcher
            else:
                # Existing watcher — if caller requests faster interval, adopt it.
                # This lets tests override the default 1.5s by subscribing with a
                # smaller interval after mount() already auto-subscribed.
                if poll_interval < watcher.poll_interval:
                    watcher.poll_interval = poll_interval
            watcher.add_subscriber(vfs)
            logger.debug(
                f"VFS subscribed to {abs_path} "
                f"(now {len(watcher._subscribers)} subs, interval={watcher.poll_interval}s)"
            )

    def unsubscribe(
        self,
        local_path: str,
        vfs: "VirtualFileSystemV2",
    ) -> None:
        """Unsubscribe a VFS. Drops the watcher if refcount hits zero."""
        abs_path = os.path.abspath(local_path)
        with self._lock:
            watcher = self._watchers.get(abs_path)
            if watcher is None:
                return
            dropped = watcher.remove_subscriber(vfs)
            if dropped:
                del self._watchers[abs_path]
                logger.debug(f"Dropped watcher for {abs_path}")

    def unsubscribe_all(self, vfs: "VirtualFileSystemV2") -> None:
        """
        Safety net — remove VFS from every watcher it might be in.

        Called on VFS teardown. Cheap enough: we walk all watchers.
        """
        with self._lock:
            to_drop: list[str] = []
            for path, watcher in self._watchers.items():
                dropped = watcher.remove_subscriber(vfs)
                if dropped:
                    to_drop.append(path)
            for path in to_drop:
                del self._watchers[path]

    def stats(self) -> dict:
        """Diagnostic info."""
        with self._lock:
            return {
                "active_watchers": len(self._watchers),
                "watchers": [
                    {
                        "local_path": p,
                        "subscribers": len(w._subscribers),
                        "tracked_files": len(w._snapshot),
                        "poll_interval": w.poll_interval,
                        "thread_alive": (
                            w._thread.is_alive() if w._thread else False
                        ),
                    }
                    for p, w in self._watchers.items()
                ],
            }


# Module-level singleton accessor
_registry: MountPollRegistry | None = None


def get_mount_poll_registry() -> MountPollRegistry:
    """Get the global MountPollRegistry singleton."""
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
