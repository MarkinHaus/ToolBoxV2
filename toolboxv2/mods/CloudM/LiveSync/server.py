"""
LiveSync Server (SyncService)
=============================
Central coordination service running on the remote server.

Responsibilities:
  - WebSocket server for client notifications (NEVER file content)
  - Auth + MinIO credential provisioning via CredentialBroker
  - Server-side SQLite index (source of truth for checksums)
  - File-change broadcast to all connected clients
  - Conflict detection on incoming changes
  - Full-state export for new clients (gzipped DB → MinIO)
  - Watchdog on server vault (thread-safe queue → asyncio)
  - Ping/pong keepalive

Run standalone:
    python -m toolboxv2.mods.CloudM.LiveSync.server --vault /path --port 8765
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .protocol import (
    MsgType, SyncMessage, FileType, classify_file, should_ignore, MAX_FILE_SIZE,
)
from .config import SyncConfig, load_env_config
from .crypto import compute_checksum_file
from .index import LocalIndex
from .minio_helper import (
    create_minio_client, ensure_bucket, healthcheck,
    upload_bytes, download_bytes, make_object_key,
    list_remote_files, vend_credentials_for_share,
)
from .conflict import detect_conflict

try:
    import websockets
    from websockets.server import serve as ws_serve
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object  # type: ignore

logger = logging.getLogger("LiveSync")


# ── Thread-safe Watchdog → asyncio Queue (BUG FIX from spec) ──

class AsyncWatchdogHandler(FileSystemEventHandler):
    """
    Bridge between synchronous Watchdog callbacks and asyncio.

    CRITICAL: Never call asyncio.create_task() from Watchdog threads.
    Instead, use loop.call_soon_threadsafe() to enqueue events.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, vault_path: str):
        self.loop = loop
        self.queue = queue
        self.vault_path = Path(vault_path)

    def _enqueue(self, event_type: str, src_path: str):
        """Thread-safe enqueue into asyncio queue."""
        try:
            rel_path = str(Path(src_path).relative_to(self.vault_path)).replace("\\", "/")
        except ValueError:
            return

        if should_ignore(rel_path):
            return

        self.loop.call_soon_threadsafe(self.queue.put_nowait, (event_type, rel_path))

    def on_modified(self, event):
        if event.is_directory:
            return
        self._enqueue("modified", event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        self._enqueue("created", event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        self._enqueue("deleted", event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        try:
            old = str(Path(event.src_path).relative_to(self.vault_path)).replace("\\", "/")
            new = str(Path(event.dest_path).relative_to(self.vault_path)).replace("\\", "/")
            if should_ignore(new):
                return
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait, ("renamed", old, new)
            )
        except ValueError:
            pass


# ── SyncServer ──

class SyncServer:
    """
    Central sync coordination server.

    Manages:
    - WebSocket connections + auth
    - Server-side file index
    - Change broadcast
    - Conflict detection
    - Watchdog on vault directory
    """

    def __init__(
        self,
        vault_path: str,
        share_id: str,
        env_config: Optional[dict] = None,
    ):
        self.vault_path = Path(vault_path)
        self.share_id = share_id
        self.env_config = env_config or load_env_config()

        # Index
        index_path = self.vault_path / ".livesync_server.db"
        self.index = LocalIndex(str(index_path))

        # Connected clients: {client_id: {ws, client_id, device_type}}
        self.clients: Dict[str, Dict[str, Any]] = {}

        # Pending broadcasts (batched)
        self._pending_broadcasts: List[SyncMessage] = []

        # Watchdog → asyncio queue
        self._watch_queue: asyncio.Queue = asyncio.Queue()
        self._observer: Optional[Any] = None

        # Debounce state: {rel_path: last_event_time}
        self._debounce: Dict[str, float] = {}
        self._debounce_delay = 2.0  # seconds

        # MinIO admin client (for full-state export, healthcheck)
        self._minio_admin = None

        self._running = False

    # ── Lifecycle ──

    async def start(self, host: str = "0.0.0.0", port: int = 8765):
        """Start the sync server (WS + watchdog + index)."""
        if not WS_AVAILABLE:
            raise RuntimeError("websockets library required: pip install websockets")

        self._running = True

        # Init index
        await self._init_index()

        # Init MinIO admin client
        try:
            self._minio_admin = create_minio_client(self.env_config)
            bucket = self.env_config.get("bucket", "livesync")
            ensure_bucket(self._minio_admin, bucket)
            ok, msg = healthcheck(self._minio_admin)
            if ok:
                logger.info(f"[LiveSync] MinIO connected: {self.env_config['endpoint']}")
            else:
                logger.error(f"[LiveSync] MinIO healthcheck failed: {msg}")
        except Exception as e:
            logger.error(f"[LiveSync] MinIO init failed: {e}")
            self._minio_admin = None

        # Start watchdog
        if WATCHDOG_AVAILABLE:
            self._start_watchdog()
            logger.info(f"[LiveSync] File watcher started for {self.vault_path}")

        # Start WS server
        logger.info(f"[LiveSync] Starting server on ws://{host}:{port}")

        async with ws_serve(self._handle_client, host, port):
            logger.info(f"[LiveSync] Server running on ws://{host}:{port}")
            # Main loop: process watchdog events + pending broadcasts
            while self._running:
                await self._process_watch_events()
                await self._flush_broadcasts()
                await asyncio.sleep(0.1)

    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join()
        for cid, client in list(self.clients.items()):
            try:
                await client["ws"].close()
            except Exception:
                pass
        await self.index.close()
        logger.info("[LiveSync] Server stopped")

    # ── Index Init ──

    async def _init_index(self):
        """Build initial checksum index from vault files."""
        await self.index.init()
        count = 0
        for root, dirs, files in os.walk(self.vault_path):
            # Prune ignored dirs in-place
            dirs[:] = [d for d in dirs if d not in (".obsidian", ".git", ".sync-trash", "__pycache__", ".livesync_server.db")]
            for fname in files:
                full = os.path.join(root, fname)
                rel = str(Path(full).relative_to(self.vault_path)).replace("\\", "/")
                if should_ignore(rel):
                    continue
                try:
                    stat = os.stat(full)
                    if stat.st_size > MAX_FILE_SIZE:
                        continue
                    checksum = compute_checksum_file(full)
                    await self.index.upsert_file(
                        rel, stat.st_mtime, stat.st_size, checksum, "synced",
                        make_object_key(self.share_id, rel),
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"[LiveSync] Index build skip {rel}: {e}")
        logger.info(f"[LiveSync] Index built: {count} files")

    # ── Watchdog ──

    def _start_watchdog(self):
        loop = asyncio.get_event_loop()
        handler = AsyncWatchdogHandler(loop, self._watch_queue, str(self.vault_path))
        self._observer = Observer()
        self._observer.schedule(handler, str(self.vault_path), recursive=True)
        self._observer.start()

    async def _process_watch_events(self):
        """
        Drain watchdog queue with debouncing.

        Collect events for 2s, deduplicate by path, then process.
        """
        batch: Dict[str, tuple] = {}
        deadline = time.time() + 0.5  # check every 0.5s

        while not self._watch_queue.empty() and time.time() < deadline:
            try:
                event = self._watch_queue.get_nowait()
                if event[0] == "renamed":
                    _, old, new = event
                    batch[new] = ("renamed", old, new)
                else:
                    event_type, rel_path = event
                    # Debounce: only keep latest event per path
                    now = time.time()
                    last = self._debounce.get(rel_path, 0)
                    if now - last >= self._debounce_delay:
                        batch[rel_path] = (event_type, rel_path)
                        self._debounce[rel_path] = now
            except asyncio.QueueEmpty:
                break

        # Process batch
        for key, event in batch.items():
            try:
                if event[0] == "renamed":
                    await self._on_server_file_renamed(event[1], event[2])
                elif event[0] == "deleted":
                    await self._on_server_file_deleted(event[1])
                else:
                    await self._on_server_file_changed(event[1])
            except Exception as e:
                logger.error(f"[LiveSync] Watchdog event error {event}: {e}")

    async def _on_server_file_changed(self, rel_path: str):
        """Handle a server-side file change (from agent, script, etc.)."""
        full = self.vault_path / rel_path
        if not full.exists():
            return
        try:
            stat = full.stat()
            if stat.st_size > MAX_FILE_SIZE:
                logger.warning(f"[LiveSync] File too large, skipping: {rel_path}")
                return
            checksum = compute_checksum_file(str(full))
            minio_key = make_object_key(self.share_id, rel_path)

            # Check if actually changed
            existing = await self.index.get_file(rel_path)
            if existing and existing["checksum"] == checksum:
                return

            await self.index.upsert_file(
                rel_path, stat.st_mtime, stat.st_size, checksum, "synced", minio_key,
            )

            # Upload to MinIO if admin client available
            if self._minio_admin:
                from .crypto import encrypt_file
                # We need the encryption key — for server-side changes we use
                # the share's key. This requires the key to be available.
                # For now, broadcast the notification; clients pull from MinIO.
                pass

            # Broadcast
            msg = SyncMessage.file_changed(
                rel_path, checksum, minio_key,
                file_type=classify_file(rel_path).value,
                source_client="server",
            )
            self._pending_broadcasts.append(msg)
            await self.index.log_sync_event(rel_path, "server_change", checksum, "server")
            logger.info(f"[LiveSync] Server file changed: {rel_path}")
        except Exception as e:
            logger.error(f"[LiveSync] Server change error {rel_path}: {e}")

    async def _on_server_file_deleted(self, rel_path: str):
        """Handle server-side file deletion."""
        await self.index.delete_file(rel_path)
        msg = SyncMessage.file_deleted(rel_path, source_client="server")
        self._pending_broadcasts.append(msg)
        await self.index.log_sync_event(rel_path, "delete", "", "server")
        logger.info(f"[LiveSync] Server file deleted: {rel_path}")

    async def _on_server_file_renamed(self, old_path: str, new_path: str):
        """Handle server-side file rename."""
        full = self.vault_path / new_path
        checksum = ""
        if full.exists():
            checksum = compute_checksum_file(str(full))
        minio_key = make_object_key(self.share_id, new_path)

        await self.index.delete_file(old_path)
        if full.exists():
            stat = full.stat()
            await self.index.upsert_file(
                new_path, stat.st_mtime, stat.st_size, checksum, "synced", minio_key,
            )

        msg = SyncMessage.file_renamed(
            old_path, new_path, checksum, minio_key, source_client="server",
        )
        self._pending_broadcasts.append(msg)
        logger.info(f"[LiveSync] Server file renamed: {old_path} → {new_path}")

    # ── Client Handling ──

    async def _handle_client(self, websocket):
        """Handle a single client WebSocket connection."""
        client_id = None
        try:
            # Wait for auth
            raw = await asyncio.wait_for(websocket.recv(), timeout=30)
            msg = SyncMessage.from_json(raw)

            if msg.type != MsgType.AUTH:
                await websocket.send(
                    SyncMessage.error("First message must be auth").to_json()
                )
                return

            # Authenticate + provision credentials
            client_id = msg.payload.get("client_id", f"client-{int(time.time())}")
            device_type = msg.payload.get("device_type", "unknown")
            share_id = msg.payload.get("share_id", self.share_id)

            # Mint scoped MinIO credentials
            minio_creds = vend_credentials_for_share(share_id, self.env_config)

            # Get current checksums for initial sync
            checksums = await self.index.get_all_checksums()

            # Register client
            self.clients[client_id] = {
                "ws": websocket,
                "client_id": client_id,
                "device_type": device_type,
            }

            # Send auth success
            await websocket.send(
                SyncMessage.auth_success(client_id, minio_creds, checksums).to_json()
            )
            logger.info(f"[LiveSync] Client connected: {client_id} ({device_type})")

            # Handle messages
            async for raw_msg in websocket:
                await self._handle_message(client_id, raw_msg)

        except asyncio.TimeoutError:
            logger.warning("[LiveSync] Client auth timeout")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[LiveSync] Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"[LiveSync] Client error ({client_id}): {e}")
        finally:
            if client_id and client_id in self.clients:
                del self.clients[client_id]

    async def _handle_message(self, client_id: str, raw: str):
        """Route incoming client message."""
        try:
            msg = SyncMessage.from_json(raw)

            if msg.type == MsgType.PING:
                ws = self.clients[client_id]["ws"]
                await ws.send(SyncMessage.pong().to_json())

            elif msg.type == MsgType.FILE_CHANGED:
                p = msg.payload
                await self._process_file_changed(
                    client_id, p["path"], p["checksum"],
                    p["minio_key"], p.get("file_type", "other"),
                )

            elif msg.type == MsgType.FILE_DELETED:
                await self._process_file_deleted(client_id, msg.payload["path"])

            elif msg.type == MsgType.FILE_RENAMED:
                p = msg.payload
                await self._process_file_renamed(
                    client_id, p["old_path"], p["new_path"],
                    p.get("checksum", ""), p.get("minio_key", ""),
                )

            elif msg.type == MsgType.REQUEST_SYNC:
                await self._send_sync_state(client_id)

            elif msg.type == MsgType.REQUEST_FULL:
                await self._send_full_state(client_id)

            else:
                logger.warning(f"[LiveSync] Unknown message type from {client_id}: {msg.type}")

        except Exception as e:
            logger.error(f"[LiveSync] Message error from {client_id}: {e}")
            if client_id in self.clients:
                ws = self.clients[client_id]["ws"]
                try:
                    await ws.send(SyncMessage.error(str(e)).to_json())
                except Exception:
                    pass

    # ── Core Processing ──

    async def _process_file_changed(
        self,
        client_id: str,
        path: str,
        checksum: str,
        minio_key: str,
        file_type: str,
    ):
        """Process file_changed from a client."""
        # Conflict check
        has_conflict = await self._check_conflict(path, checksum)

        if has_conflict:
            existing = await self.index.get_file(path)
            server_checksum = existing["checksum"] if existing else ""

            logger.warning(
                f"[LiveSync] Conflict detected: {path} "
                f"(server={server_checksum}, client {client_id}={checksum})"
            )

            ft = classify_file(path)
            if ft == FileType.TEXT and path.endswith(".md"):
                resolution = "merge_markers"
            else:
                resolution = "latest_wins"

            # Notify ALL clients
            conflict_msg = SyncMessage.conflict(
                path=path,
                local_checksum=server_checksum,
                remote_checksum=checksum,
                resolution=resolution,
                message=f"Conflict on {path}: server={server_checksum}, {client_id}={checksum}",
            )
            await self._broadcast(conflict_msg)
            await self.index.log_sync_event(path, "conflict", checksum, client_id)

        # Update index (latest writer wins at index level)
        await self.index.upsert_file(
            path, time.time(), 0, checksum, "synced", minio_key,
        )

        # ACK to sender
        if client_id in self.clients:
            ws = self.clients[client_id]["ws"]
            await ws.send(SyncMessage.ack(path, checksum).to_json())

        # Broadcast to others
        broadcast_msg = SyncMessage.file_changed(
            path, checksum, minio_key, file_type, source_client=client_id,
        )
        await self._broadcast(broadcast_msg, skip_client=client_id)

        await self.index.log_sync_event(path, "upload", checksum, client_id)
        logger.info(f"[LiveSync] File synced: {path} from {client_id}")

    async def _process_file_deleted(self, client_id: str, path: str):
        """Process file_deleted from a client."""
        await self.index.delete_file(path)

        # Broadcast
        msg = SyncMessage.file_deleted(path, source_client=client_id)
        await self._broadcast(msg, skip_client=client_id)

        await self.index.log_sync_event(path, "delete", "", client_id)
        logger.info(f"[LiveSync] File deleted: {path} by {client_id}")

    async def _process_file_renamed(
        self, client_id: str, old_path: str, new_path: str,
        checksum: str, minio_key: str,
    ):
        """Process file_renamed from a client."""
        await self.index.delete_file(old_path)
        if checksum:
            await self.index.upsert_file(
                new_path, time.time(), 0, checksum, "synced", minio_key,
            )

        msg = SyncMessage.file_renamed(
            old_path, new_path, checksum, minio_key, source_client=client_id,
        )
        await self._broadcast(msg, skip_client=client_id)
        logger.info(f"[LiveSync] File renamed: {old_path} → {new_path} by {client_id}")

    async def _check_conflict(self, path: str, incoming_checksum: str) -> bool:
        """Check if incoming change conflicts with server state."""
        existing = await self.index.get_file(path)
        if not existing:
            return False  # New file
        return detect_conflict(existing["checksum"], incoming_checksum)

    # ── Broadcast ──

    async def _broadcast(self, msg: SyncMessage, skip_client: Optional[str] = None):
        """Send message to all connected clients (except skip_client)."""
        raw = msg.to_json()
        for cid, client in list(self.clients.items()):
            if cid == skip_client:
                continue
            try:
                await client["ws"].send(raw)
            except Exception as e:
                logger.warning(f"[LiveSync] Broadcast to {cid} failed: {e}")

    async def _flush_broadcasts(self):
        """Send all pending broadcast messages."""
        if not self._pending_broadcasts:
            return
        msgs = self._pending_broadcasts[:]
        self._pending_broadcasts.clear()
        for msg in msgs:
            await self._broadcast(msg)

    # ── Full State (Scenario S5) ──

    async def _send_sync_state(self, client_id: str):
        """Send current checksums to a client."""
        if client_id not in self.clients:
            return
        checksums = await self.index.get_all_checksums()
        ws = self.clients[client_id]["ws"]
        await ws.send(SyncMessage.auth_success(
            client_id, {}, checksums,
        ).to_json())

    async def _send_full_state(self, client_id: str):
        """
        Export full index as gzipped DB → upload to MinIO → notify client.

        For large vaults (100k files), sending checksums as JSON over WS
        would be too large. Instead: gzipped SQLite dump via MinIO.
        """
        if client_id not in self.clients:
            return

        try:
            data = await self.index.export_gzipped()
            minio_key = f"{self.share_id}/.meta/index.db.gz"

            if self._minio_admin:
                bucket = self.env_config.get("bucket", "livesync")
                upload_bytes(self._minio_admin, bucket, minio_key, data)

            checksums = await self.index.get_all_checksums()
            file_count = len(checksums)

            ws = self.clients[client_id]["ws"]
            await ws.send(
                SyncMessage.full_state_ready(minio_key, file_count).to_json()
            )
            logger.info(
                f"[LiveSync] Full state exported for {client_id}: "
                f"{file_count} files, {len(data)} bytes compressed"
            )
        except Exception as e:
            logger.error(f"[LiveSync] Full state export failed: {e}")
            if client_id in self.clients:
                ws = self.clients[client_id]["ws"]
                await ws.send(SyncMessage.error(f"Full state export failed: {e}").to_json())


# ── Standalone entry point ──

async def _run_standalone():
    parser = argparse.ArgumentParser(description="LiveSync Server")
    parser.add_argument("--vault", "-v", required=True, help="Path to vault")
    parser.add_argument("--share-id", "-s", default="default", help="Share ID")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", "-p", type=int, default=8765, help="WS port")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    env_config = load_env_config()
    env_config["ws_host"] = args.host
    env_config["ws_port"] = args.port

    server = SyncServer(
        vault_path=args.vault,
        share_id=args.share_id,
        env_config=env_config,
    )

    try:
        await server.start(host=args.host, port=args.port)
    except KeyboardInterrupt:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(_run_standalone())
