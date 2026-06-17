"""
VFS ↔ LiveSync Adapter
======================
Keep a VFS share folder live-synced across machines via the existing
CloudM/LiveSync engine. GENERIC: works for /global and ANY VFS share folder.

Design ("VFS as view, not storage"):
    - LiveSync runs on the REAL backing disk dir (watchdog + MinIO + WS).
    - The VFS path is an overlay mounted on that dir.
    - On a remote change, on_remote_change fires → vfs.refresh_mount(vfs_path)
      so every reader sees one logical folder ("as if it exists only once").

Usage (icli):
    mgr = VFSSyncManager(session.vfs)
    await mgr.connect("/global", local_dir, token=share_token)   # join existing
    await mgr.connect("/team/notes", local_dir2, token=tok2)     # any folder
    ...
    await mgr.disconnect("/global")
    await mgr.stop_all()
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

from .config import SyncConfig, ShareToken
from .client import SyncClient

logger = logging.getLogger("LiveSync")


class VFSSyncAdapter:
    """One synced VFS folder = one SyncClient on its backing disk dir."""

    def __init__(
        self,
        vfs,
        vfs_path: str,
        local_dir: str,
        *,
        token: Optional[str] = None,
        sync_config: Optional[SyncConfig] = None,
        readonly: bool = False,
        auto_sync: bool = True,
    ):
        if not (token or sync_config):
            raise ValueError("provide either token or sync_config")
        self.vfs = vfs
        self.vfs_path = vfs_path
        self.local_dir = local_dir
        self.readonly = readonly
        self.auto_sync = auto_sync

        if sync_config is None:
            sync_config = ShareToken.decode(token).to_sync_config(local_dir)
        self.config = sync_config
        self.share_id = self.config.share_id

        self._client: Optional[SyncClient] = None
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── lifecycle ──

    async def start(self):
        """Mount the folder into VFS and launch the sync client."""
        self._loop = asyncio.get_event_loop()

        # Mount the backing dir as a VFS overlay (no-op if vfs is None).
        if self.vfs is not None:
            try:
                self.vfs.mount(
                    self.local_dir, self.vfs_path,
                    readonly=self.readonly, auto_sync=self.auto_sync,
                )
            except Exception as e:
                logger.error(f"[VFSSync] mount failed {self.vfs_path}: {e}")

        self._client = SyncClient(self.config)
        self._client.on_remote_change = self._on_remote_change
        self._task = asyncio.create_task(self._client.run())
        logger.info(f"[VFSSync] started {self.vfs_path} ← share {self.share_id}")

    async def stop(self):
        """Stop the sync client and unmount the VFS overlay."""
        if self._client:
            await self._client.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self.vfs is not None:
            try:
                unmount = getattr(self.vfs, "unmount", None)
                if unmount:
                    unmount(self.vfs_path)
            except Exception as e:
                logger.error(f"[VFSSync] unmount failed {self.vfs_path}: {e}")
        logger.info(f"[VFSSync] stopped {self.vfs_path}")

    # ── remote-change bridge ──

    def _on_remote_change(self, event_type: str, payload: dict):
        """Fired (in the loop) after a remote mutation is applied to disk.

        Refresh the VFS overlay so readers see the new state immediately.
        Handles both sync and async refresh_mount implementations.
        """
        if self.vfs is None:
            return
        refresh = getattr(self.vfs, "refresh_mount", None)
        if refresh is None:
            return
        try:
            res = refresh(self.vfs_path)
            if asyncio.iscoroutine(res):
                # schedule without blocking the WS handler
                asyncio.create_task(res)
        except Exception as e:
            logger.error(f"[VFSSync] refresh_mount failed {self.vfs_path}: {e}")


class VFSSyncManager:
    """Manage many synced VFS folders at once (one adapter per folder)."""

    def __init__(self, vfs):
        self.vfs = vfs
        self._adapters: Dict[str, VFSSyncAdapter] = {}

    async def connect(
        self,
        vfs_path: str,
        local_dir: str,
        *,
        token: Optional[str] = None,
        sync_config: Optional[SyncConfig] = None,
        readonly: bool = False,
    ) -> VFSSyncAdapter:
        if vfs_path in self._adapters:
            return self._adapters[vfs_path]
        adapter = VFSSyncAdapter(
            self.vfs, vfs_path, local_dir,
            token=token, sync_config=sync_config, readonly=readonly,
        )
        await adapter.start()
        self._adapters[vfs_path] = adapter
        return adapter

    async def disconnect(self, vfs_path: str) -> bool:
        adapter = self._adapters.pop(vfs_path, None)
        if not adapter:
            return False
        await adapter.stop()
        return True

    def list(self) -> Dict[str, str]:
        """{vfs_path: share_id} for all active folders."""
        return {p: a.share_id for p, a in self._adapters.items()}

    async def stop_all(self):
        for path in list(self._adapters):
            await self.disconnect(path)
