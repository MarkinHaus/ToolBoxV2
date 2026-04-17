"""
LiveSync Client
===============
Runs on each device (Desktop, Termux, Laptop).

Responsibilities:
  - WS client with auto-reconnect (exponential backoff)
  - Watchdog → debounced batch → encrypt → upload to MinIO → WS notify
  - WS notification → download from MinIO → decrypt → atomic write
  - Reconnect catchup: compare server checksums, pull missing
  - Offline buffer: queue changes while disconnected, push on reconnect

Data path:
  Upload:  local file → zlib+AES → MinIO (direct) → WS notification (metadata only)
  Download: WS notification → MinIO (direct) → AES+zlib → local file

Run standalone:
    python -m toolboxv2.mods.CloudM.LiveSync.client --token <base64> --vault /path
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .protocol import (
    MsgType, SyncMessage, classify_file, should_ignore, MAX_FILE_SIZE,
)
from .config import SyncConfig, ShareToken
from .crypto import (
    encrypt_file, decrypt_bytes, compute_checksum, compute_checksum_file,
)
from .index import LocalIndex
from .minio_helper import (
    create_minio_client, make_object_key, upload_bytes, upload_metadata,
)
from .conflict import (
    create_backup, move_to_sync_trash, detect_conflict,
)

try:
    import websockets
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

from platform import node
node_ = node()

if 'localhost' in node_ and (host := os.getenv('HOSTNAME', 'localhost')) != 'localhost':
    node_ = node_.replace('localhost', host)


# ── Backoff ──

def _backoff_delay(attempt: int, base: float = 1.0, maximum: float = 60.0) -> float:
    """Exponential backoff with cap."""
    delay = min(base * (2 ** attempt), maximum)
    return delay


# ── Debounce Batch ──

class DebounceBatch:
    """
    Collect filesystem events, deduplicate by path, flush after delay.

    Spec rule: 2s debounce, dedup, then batch upload.
    """

    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.pending: Dict[str, str] = {}  # {rel_path: event_type}
        self._last_add: float = 0

    def add(self, rel_path: str, event_type: str):
        """Add or update an event. Latest event type wins per path."""
        self.pending[rel_path] = event_type
        self._last_add = time.time()

    def is_ready(self) -> bool:
        """True if delay has elapsed since last add and there are pending items."""
        if not self.pending:
            return False
        return (time.time() - self._last_add) >= self.delay

    def flush(self) -> Dict[str, str]:
        """Return and clear all pending events."""
        items = dict(self.pending)
        self.pending.clear()
        return items


# ── Watchdog Handler (thread-safe → asyncio) ──

class ClientWatchdogHandler(FileSystemEventHandler):
    """
    Thread-safe bridge: Watchdog thread → asyncio queue.
    NEVER calls asyncio.create_task from sync context.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, vault_path: str):
        self.loop = loop
        self.queue = queue
        self.vault_path = Path(vault_path)

    def _enqueue(self, event_type: str, src_path: str):
        try:
            rel = str(Path(src_path).relative_to(self.vault_path)).replace("\\", "/")
        except ValueError:
            return
        if should_ignore(rel):
            return
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (event_type, rel))

    def on_modified(self, event):
        if not event.is_directory:
            self._enqueue("modified", event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._enqueue("created", event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._enqueue("deleted", event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            try:
                old = str(Path(event.src_path).relative_to(self.vault_path)).replace("\\", "/")
                new = str(Path(event.dest_path).relative_to(self.vault_path)).replace("\\", "/")
                if not should_ignore(new):
                    self.loop.call_soon_threadsafe(
                        self.queue.put_nowait, ("renamed", old, new)
                    )
            except ValueError:
                pass


# ── SyncClient ──

class SyncClient:
    """
    LiveSync client running on a user device.

    Lifecycle:
        client = SyncClient(config)
        await client.run()  # blocks until stopped
    """

    def __init__(self, config: SyncConfig):
        self.config = config
        self.vault = Path(config.vault_path)
        self.vault.mkdir(parents=True, exist_ok=True)

        # Index
        db_path = self.vault / ".livesync_client.db"
        self.index = LocalIndex(str(db_path))

        # MinIO client (set after auth)
        self._minio = None

        # WebSocket
        self._ws = None
        self._running = False
        self._reconnect_attempt = 0

        # Watchdog
        self._watch_queue: asyncio.Queue = asyncio.Queue()
        self._debounce = DebounceBatch(delay=config.debounce_seconds)
        self._observer = None

        # Offline buffer: changes made while disconnected
        self._offline_buffer: List[Tuple[str, str]] = []  # [(event_type, rel_path), ...]

        # Concurrency limiter
        self._transfer_sem = asyncio.Semaphore(config.max_concurrent_transfers)

        # Suppress watchdog events for files we're currently writing
        self._writing_paths: set = set()

    # ── Lifecycle ──

    async def run(self):
        """Main entry point — connect, sync, watch. Reconnects on failure."""
        if not WS_AVAILABLE:
            raise RuntimeError("websockets required: pip install websockets")

        self._running = True
        await self.index.init()

        # Start watchdog
        if WATCHDOG_AVAILABLE:
            self._start_watchdog()
            logger.info(f"[LiveSync] File watcher started: {self.vault}")

        while self._running:
            try:
                await self._connect_and_sync()
            except Exception as e:
                logger.error(f"[LiveSync] Connection error: {e}")

            if not self._running:
                break

            delay = _backoff_delay(
                self._reconnect_attempt,
                self.config.reconnect_base_delay,
                self.config.reconnect_max_delay,
            )
            logger.info(f"[LiveSync] Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempt + 1})")
            await asyncio.sleep(delay)
            self._reconnect_attempt += 1

        await self._cleanup()

    async def stop(self):
        """Graceful shutdown."""
        self._running = False

    async def _cleanup(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        await self.index.close()
        logger.info("[LiveSync] Client stopped")

    # ── Connection ──

    async def _connect_and_sync(self):
        """Connect to server, authenticate, sync, then run event loop."""
        logger.info(f"[LiveSync] Connecting to {self.config.ws_endpoint}")

        async with websockets.connect(self.config.ws_endpoint) as ws:
            self._ws = ws
            self._reconnect_attempt = 0

            # Auth
            client_id = f"{node_}-{os.getpid()}"
            await ws.send(SyncMessage.auth(
                client_id, "desktop", self.config.share_id,
            ).to_json())

            # Wait for auth_success
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = SyncMessage.from_json(raw)

            if msg.type == MsgType.ERROR:
                raise ConnectionError(f"Auth failed: {msg.payload.get('message')}")
            if msg.type != MsgType.AUTH_SUCCESS:
                raise ConnectionError(f"Unexpected response: {msg.type}")

            # Extract MinIO credentials
            minio_creds = msg.payload.get("minio_credentials", {})
            if minio_creds.get("endpoint"):
                self._minio = create_minio_client(minio_creds)
            else:
                logger.warning("[LiveSync] No MinIO credentials received, using config")
                self._minio = create_minio_client({
                    "endpoint": self.config.minio_endpoint,
                    "access_key": minio_creds.get("access_key", ""),
                    "secret_key": minio_creds.get("secret_key", ""),
                    "secure": minio_creds.get("secure", False),
                })

            server_checksums = msg.payload.get("checksums", {})
            logger.info(f"[LiveSync] Authenticated. Server has {len(server_checksums)} files")

            # Catchup: compare server state with local
            await self._catchup_sync(server_checksums)

            # Flush offline buffer
            await self._flush_offline_buffer()

            # Main event loop
            await self._event_loop(ws)

    async def _event_loop(self, ws):
        """Process WS messages and local watchdog events concurrently."""
        recv_task = asyncio.create_task(self._ws_recv_loop(ws))
        watch_task = asyncio.create_task(self._watch_loop())
        ping_task = asyncio.create_task(self._ping_loop(ws))

        try:
            done, pending = await asyncio.wait(
                [recv_task, watch_task, ping_task],
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in done:
                if task.exception():
                    raise task.exception()
        finally:
            for task in [recv_task, watch_task, ping_task]:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    # ── WS Receive Loop ──

    async def _ws_recv_loop(self, ws):
        """Listen for server messages."""
        async for raw in ws:
            try:
                msg = SyncMessage.from_json(raw)
                await self._handle_server_message(msg)
            except Exception as e:
                logger.error(f"[LiveSync] Message handling error: {e}")

    async def _handle_server_message(self, msg: SyncMessage):
        """Route incoming server message."""
        if msg.type == MsgType.PONG:
            return

        elif msg.type == MsgType.FILE_CHANGED:
            p = msg.payload
            await self._download_file(
                p["path"], p["minio_key"],
                expected_checksum=p.get("checksum"),
            )

        elif msg.type == MsgType.FILE_DELETED:
            await self._handle_remote_delete(msg.payload["path"])

        elif msg.type == MsgType.FILE_RENAMED:
            p = msg.payload
            await self._handle_remote_rename(
                p["old_path"], p["new_path"], p.get("minio_key", ""),
            )

        elif msg.type == MsgType.CONFLICT:
            p = msg.payload
            logger.warning(
                f"[LiveSync] Conflict on {p['path']}: {p.get('resolution', 'unknown')} — {p.get('message', '')}"
            )
            await self.index.log_sync_event(
                p["path"], "conflict", p.get("remote_checksum", ""), "server",
            )

        elif msg.type == MsgType.FULL_STATE_READY:
            await self._handle_full_state(msg.payload)

        elif msg.type == MsgType.ACK:
            pass  # logged on upload side already

        elif msg.type == MsgType.ERROR:
            logger.error(f"[LiveSync] Server error: {msg.payload.get('message')}")

    # ── Ping/Pong ──

    async def _ping_loop(self, ws):
        """Send periodic pings for keepalive."""
        while self._running:
            await asyncio.sleep(30)
            try:
                await ws.send(SyncMessage.ping().to_json())
            except Exception:
                break

    # ── Watchdog Loop ──

    def _start_watchdog(self):
        loop = asyncio.get_event_loop()
        handler = ClientWatchdogHandler(loop, self._watch_queue, str(self.vault))
        self._observer = Observer()
        self._observer.schedule(handler, str(self.vault), recursive=True)
        self._observer.start()

    async def _watch_loop(self):
        """Drain watchdog queue → debounce → batch process."""
        while self._running:
            # Drain queue into debouncer
            while not self._watch_queue.empty():
                try:
                    event = self._watch_queue.get_nowait()
                    if event[0] == "renamed":
                        # Handle renames immediately (no debounce)
                        await self._handle_local_rename(event[1], event[2])
                    else:
                        event_type, rel_path = event
                        # Skip files we're currently writing (download in progress)
                        if rel_path not in self._writing_paths:
                            self._debounce.add(rel_path, event_type)
                except asyncio.QueueEmpty:
                    break

            # Flush debounce batch if ready
            if self._debounce.is_ready():
                batch = self._debounce.flush()
                for rel_path, event_type in batch.items():
                    try:
                        if event_type == "deleted":
                            await self._handle_local_delete(rel_path)
                        else:
                            await self._upload_file(rel_path)
                    except Exception as e:
                        logger.error(f"[LiveSync] Batch process error {rel_path}: {e}")

            await asyncio.sleep(0.2)

    # ── Upload ──

    async def _upload_file(self, rel_path: str):
        """
        Upload a local file: read → checksum check → encrypt → MinIO → WS notify.
        """
        full = self.vault / rel_path
        if not full.exists():
            return
        if full.stat().st_size > MAX_FILE_SIZE:
            logger.warning(f"[LiveSync] File too large, skipping: {rel_path}")
            return

        async with self._transfer_sem:
            try:
                # Checksum first — skip if unchanged
                checksum = compute_checksum_file(str(full))
                existing = await self.index.get_file(rel_path)
                if existing and existing["checksum"] == checksum:
                    return  # No actual change

                # Encrypt
                encrypted = encrypt_file(str(full), self.config.encryption_key)

                # Upload to MinIO
                minio_key = make_object_key(self.config.prefix, rel_path)
                bucket = self.config.bucket

                if self._minio:
                    upload_bytes(self._minio, bucket, minio_key, encrypted)

                    # Upload metadata
                    upload_metadata(self._minio, bucket, self.config.prefix, rel_path, {
                        "checksum": checksum,
                        "mtime": full.stat().st_mtime,
                        "size": full.stat().st_size,
                        "source_client": f"{node_}",
                        "file_type": classify_file(rel_path).value,
                    })

                # Update index
                stat = full.stat()
                await self.index.upsert_file(
                    rel_path, stat.st_mtime, stat.st_size, checksum,
                    "synced", minio_key,
                )

                # Notify server via WS
                if self._ws:
                    await self._ws.send(SyncMessage.file_changed(
                        rel_path, checksum, minio_key,
                        file_type=classify_file(rel_path).value,
                    ).to_json())

                await self.index.log_sync_event(rel_path, "upload", checksum)
                logger.info(f"[LiveSync] Uploaded: {rel_path}")

            except Exception as e:
                logger.error(f"[LiveSync] Upload failed {rel_path}: {e}")
                await self.index.set_sync_state(rel_path, "pending_upload")

    # ── Download ──

    async def _download_file(
        self,
        rel_path: str,
        minio_key: str,
        expected_checksum: Optional[str] = None,
        retries: int = 3,
    ):
        """
        Download from MinIO → decrypt → verify checksum → atomic write.

        Scenario S4: partial download → .sync-tmp stays, retry on reconnect.
        """
        async with self._transfer_sem:
            full = self.vault / rel_path

            for attempt in range(retries):
                try:
                    # Backup existing file before overwrite
                    if full.exists():
                        create_backup(str(full))

                    # Download
                    if not self._minio:
                        logger.error(f"[LiveSync] No MinIO client for download: {rel_path}")
                        await self.index.set_sync_state(rel_path, "pending_download")
                        return

                    resp = self._minio.get_object(self.config.bucket, minio_key)
                    try:
                        encrypted = resp.read()
                    finally:
                        resp.close()
                        resp.release_conn()

                    # Decrypt
                    data = decrypt_bytes(encrypted, self.config.encryption_key)

                    # Verify checksum
                    actual_cs = compute_checksum(data)
                    if expected_checksum and actual_cs != expected_checksum:
                        if attempt < retries - 1:
                            logger.warning(
                                f"[LiveSync] Checksum mismatch {rel_path} "
                                f"(got {actual_cs}, expected {expected_checksum}), retry {attempt + 1}"
                            )
                            continue
                        else:
                            logger.error(
                                f"[LiveSync] Checksum mismatch after {retries} retries: "
                                f"{rel_path} — manual intervention needed"
                            )
                            await self.index.set_sync_state(rel_path, "conflict")
                            return

                    # Atomic write (suppress watchdog for this path)
                    self._writing_paths.add(rel_path)
                    try:
                        full.parent.mkdir(parents=True, exist_ok=True)
                        tmp = full.with_suffix(full.suffix + ".sync-tmp")
                        with open(tmp, "wb") as f:
                            f.write(data)
                        if full.exists():
                            full.unlink()
                        os.rename(tmp, full)
                    finally:
                        # Small delay so watchdog event for our write gets suppressed
                        await asyncio.sleep(0.1)
                        self._writing_paths.discard(rel_path)

                    # Update index
                    stat = full.stat()
                    await self.index.upsert_file(
                        rel_path, stat.st_mtime, stat.st_size, actual_cs,
                        "synced", minio_key,
                    )
                    await self.index.log_sync_event(rel_path, "download", actual_cs)
                    logger.info(f"[LiveSync] Downloaded: {rel_path}")
                    return  # success

                except Exception as e:
                    if attempt < retries - 1:
                        logger.warning(
                            f"[LiveSync] Download error {rel_path} (attempt {attempt + 1}): {e}"
                        )
                        await asyncio.sleep(1)
                    else:
                        logger.error(
                            f"[LiveSync] Download failed after {retries} retries: {rel_path}: {e}"
                        )
                        await self.index.set_sync_state(rel_path, "pending_download")

    # ── Delete ──

    async def _handle_remote_delete(self, rel_path: str):
        """
        Handle remote deletion (Scenario S6).
        Move to .sync-trash — NEVER permanently delete.
        """
        full = self.vault / rel_path
        if full.exists():
            move_to_sync_trash(str(self.vault), rel_path)
            logger.info(f"[LiveSync] File deleted remotely: {rel_path} → moved to .sync-trash/")

        await self.index.delete_file(rel_path)
        await self.index.log_sync_event(rel_path, "delete", "", "remote")

    async def _handle_local_delete(self, rel_path: str):
        """Handle local file deletion → notify server."""
        await self.index.delete_file(rel_path)
        if self._ws:
            await self._ws.send(SyncMessage.file_deleted(rel_path).to_json())
        await self.index.log_sync_event(rel_path, "delete", "")
        logger.info(f"[LiveSync] Local delete: {rel_path}")

    # ── Rename ──

    async def _handle_remote_rename(self, old_path: str, new_path: str, minio_key: str):
        """Handle remote rename."""
        old_full = self.vault / old_path
        new_full = self.vault / new_path

        if old_full.exists():
            new_full.parent.mkdir(parents=True, exist_ok=True)
            os.rename(old_full, new_full)

        await self.index.delete_file(old_path)
        if new_full.exists():
            stat = new_full.stat()
            cs = compute_checksum_file(str(new_full))
            await self.index.upsert_file(new_path, stat.st_mtime, stat.st_size, cs, "synced", minio_key)

        logger.info(f"[LiveSync] Remote rename: {old_path} → {new_path}")

    async def _handle_local_rename(self, old_path: str, new_path: str):
        """Handle local rename → notify server."""
        new_full = self.vault / new_path
        checksum = ""
        minio_key = ""
        if new_full.exists():
            checksum = compute_checksum_file(str(new_full))
            minio_key = make_object_key(self.config.prefix, new_path)
            # Upload renamed file to MinIO
            await self._upload_file(new_path)

        await self.index.delete_file(old_path)
        if self._ws:
            await self._ws.send(SyncMessage.file_renamed(
                old_path, new_path, checksum, minio_key,
            ).to_json())
        logger.info(f"[LiveSync] Local rename: {old_path} → {new_path}")

    # ── Catchup (Scenario S1 + S5) ──

    async def _catchup_sync(self, server_checksums: Dict[str, str]):
        """
        Compare server state with local index → download missing, upload new.

        Scenario S1: client reconnects → pulls all changes from offline period.
        Scenario S5: new client → full download.
        """
        to_download, to_upload = await self._compute_diff(server_checksums)

        if to_download or to_upload:
            logger.info(
                f"[LiveSync] Catchup: {len(to_download)} to download, "
                f"{len(to_upload)} to upload"
            )

        # Downloads first (server is source of truth on reconnect)
        for rel_path, minio_key in to_download:
            cs = server_checksums.get(rel_path)
            await self._download_file(rel_path, minio_key, expected_checksum=cs)

        # Then uploads (local changes made while offline)
        for rel_path in to_upload:
            await self._upload_file(rel_path)

        if to_download or to_upload:
            total = len(to_download) + len(to_upload)
            logger.info(f"[LiveSync] Catchup complete: synced {total} files")

    async def _compute_diff(
        self, server_checksums: Dict[str, str]
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Compute what needs to be downloaded vs uploaded.

        Returns:
            (to_download: [(rel_path, minio_key)], to_upload: [rel_path])
        """
        to_download: List[Tuple[str, str]] = []
        to_upload: List[str] = []

        local_checksums = await self.index.get_all_checksums()

        # Scan local filesystem for files not in index
        local_fs_files: Dict[str, str] = {}
        for root, dirs, files in os.walk(self.vault):
            dirs[:] = [d for d in dirs if d not in (".obsidian", ".git", ".sync-trash", "__pycache__")]
            for fname in files:
                full = os.path.join(root, fname)
                rel = str(Path(full).relative_to(self.vault)).replace("\\", "/")
                if should_ignore(rel):
                    continue
                try:
                    cs = compute_checksum_file(full)
                    local_fs_files[rel] = cs
                except Exception:
                    pass

        # Files on server we don't have (or differ)
        for rel_path, server_cs in server_checksums.items():
            local_cs = local_fs_files.get(rel_path) or local_checksums.get(rel_path)
            if not local_cs or local_cs != server_cs:
                minio_key = make_object_key(self.config.prefix, rel_path)
                to_download.append((rel_path, minio_key))

        # Files we have locally that server doesn't
        for rel_path, local_cs in local_fs_files.items():
            if rel_path not in server_checksums:
                to_upload.append(rel_path)

        return to_download, to_upload

    # ── Full State (Scenario S5) ──

    async def _handle_full_state(self, payload: dict):
        """Handle full_state_ready: download gzipped index from MinIO."""
        minio_key = payload.get("minio_key", "")
        file_count = payload.get("file_count", 0)

        if not self._minio or not minio_key:
            logger.error("[LiveSync] Cannot download full state: no MinIO or key")
            return

        try:
            logger.info(f"[LiveSync] Downloading full state: {file_count} files")
            resp = self._minio.get_object(self.config.bucket, minio_key)
            try:
                data = resp.read()
            finally:
                resp.close()
                resp.release_conn()

            await self.index.import_gzipped(data)
            checksums = await self.index.get_all_checksums()

            # Download all files we're missing
            missing = []
            for rel_path, cs in checksums.items():
                local = self.vault / rel_path
                if not local.exists():
                    mk = make_object_key(self.config.prefix, rel_path)
                    missing.append((rel_path, mk, cs))

            logger.info(f"[LiveSync] Full state: {len(missing)} files to download")

            for rel_path, mk, cs in missing:
                await self._download_file(rel_path, mk, expected_checksum=cs)

            logger.info(f"[LiveSync] Full state sync complete")

        except Exception as e:
            logger.error(f"[LiveSync] Full state download failed: {e}")

    # ── Offline Buffer ──

    async def _flush_offline_buffer(self):
        """Push any changes made while disconnected."""
        if not self._offline_buffer:
            return

        logger.info(f"[LiveSync] Flushing {len(self._offline_buffer)} offline changes")
        buf = list(self._offline_buffer)
        self._offline_buffer.clear()

        for event_type, rel_path in buf:
            try:
                if event_type == "deleted":
                    await self._handle_local_delete(rel_path)
                else:
                    await self._upload_file(rel_path)
            except Exception as e:
                logger.error(f"[LiveSync] Offline flush error {rel_path}: {e}")


# ── Standalone entry point ──

async def _run_standalone():
    import argparse
    parser = argparse.ArgumentParser(description="LiveSync Client")
    parser.add_argument("--token", "-t", required=True, help="Share token (base64)")
    parser.add_argument("--vault", "-v", required=True, help="Local vault path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    token = ShareToken.decode(args.token)
    config = token.to_sync_config(args.vault)

    client = SyncClient(config)
    try:
        await client.run()
    except KeyboardInterrupt:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(_run_standalone())
