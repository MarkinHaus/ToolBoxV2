"""
LiveSync Client
===============
Runs on each device (Desktop, Termux, Laptop).

Responsibilities:
  - WS client with auto-reconnect (exponential backoff)
  - Watchdog → debounced batch → encrypt → upload → WS notify
  - WS notification → download → decrypt → atomic write
  - Reconnect catchup: compare server checksums, pull missing
    (changes made while offline are picked up by this catchup, not by a
    separate buffer)

The client holds NO object-storage credentials. Every transfer runs against a
short-lived presigned URL that the server issues for exactly one object.

Data path:
  Upload:   local file → zlib+AES → request PUT url → HTTP PUT → WS notify
  Download: WS notify → request GET url → HTTP GET → AES+zlib → local file

Run standalone:
    python -m toolboxv2.mods.CloudM.LiveSync.client --token <base64> --vault /path
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
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
    make_object_key, http_download, http_upload, TransferError,
)
from .conflict import (
    create_backup, move_to_sync_trash, detect_conflict, save_conflict_copy,
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

from toolboxv2 import get_logger
logger = get_logger()

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
                if not should_ignore(new) and not should_ignore(old):
                    self.loop.call_soon_threadsafe(
                        self.queue.put_nowait, ("renamed", old, new)
                    )
            except ValueError:
                pass


class AuthenticationError(Exception):
    """The server rejected our share token. Reconnecting will not help."""


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

        # Presigned-URL plumbing. req_id → Future resolved by the recv loop.
        self._url_ttl: int = 900
        self._pending_url_requests: Dict[str, asyncio.Future] = {}

        # Background tasks spawned from the recv loop (downloads). Kept in a
        # set so they are not garbage collected mid-flight and can be
        # cancelled on shutdown.
        self._bg_tasks: set = set()

        # One lock per path: two downloads of the same file must not
        # interleave their atomic writes.
        self._path_locks: Dict[str, asyncio.Lock] = {}

        # {rel_path: {checksums already parked as a conflict copy}}. The
        # server broadcasts its verdict to every client, so the same collision
        # arrives more than once; without this each copy would be written again.
        self._preserved: Dict[str, set] = {}

        # Paths the server has declared collided, plus ones our own catchup
        # found edited on both sides. Needed because an upload writes our
        # checksum into the index, so a moment later the index no longer shows
        # that this file was edited here — and the incoming winner would
        # overwrite it as if we were merely one revision behind.
        self._contested: set = set()

        # WebSocket
        self._ws = None
        self._running = False
        self._cleaned = False
        self._reconnect_attempt = 0

        # Watchdog
        self._watch_queue: asyncio.Queue = asyncio.Queue()
        self._debounce = DebounceBatch(delay=config.debounce_seconds)
        self._observer = None

        # Concurrency limiter
        self._transfer_sem = asyncio.Semaphore(config.max_concurrent_transfers)

        # Suppress watchdog events caused by our own writes.
        # {rel_path: unix_ts until which self-inflicted events are ignored}
        self._writing_until: Dict[str, float] = {}

        # Optional VFS-bridge hook: callback(event_type:str, payload:dict)
        # fired after a remote change is applied locally. Set by VFSSyncAdapter.
        self.on_remote_change = None

        # Optional status hook: callback(status:str, detail:str) with status in
        # {"connecting", "live", "auth_failed", "stopped"}. Set by
        # VFSSyncAdapter so callers can tell a live share from a dead one.
        self.on_status_change = None
        self.status: str = "connecting"

    # ── Status ──

    def _set_status(self, status: str, detail: str = "") -> None:
        """Record the connection state and notify the optional hook."""
        self.status = status
        if self.on_status_change:
            try:
                self.on_status_change(status, detail)
            except Exception as e:
                logger.error(f"[LiveSync] on_status_change hook error: {e}")

    @property
    def device_id(self) -> str:
        """Stable name of this machine, used in conflict-copy filenames."""
        return node_

    async def _preserve_local_version(
        self, rel_path: str, reason: str, checksum: str = ""
    ) -> Optional[str]:
        """
        Move the local version aside so the remote one can land.

        Same rule for every file type: nothing is merged, nothing is lost. The
        copy is suppressed from the watcher because we created it ourselves,
        and should_ignore keeps it from ever syncing.

        checksum: content hash of the local version. Passing it makes the call
        idempotent — the same bytes are never parked twice, no matter how many
        conflict broadcasts arrive for one collision.
        """
        if checksum:
            done = self._preserved.setdefault(rel_path, set())
            if checksum in done:
                return None
            done.add(checksum)

        self._suppress(rel_path)
        try:
            saved = await asyncio.to_thread(
                save_conflict_copy, str(self.vault), rel_path, self.device_id)
        except Exception as e:
            logger.error(f"[LiveSync] Could not preserve local {rel_path}: {e}")
            return None

        if saved:
            name = Path(saved).name
            self._suppress(str(Path(saved).relative_to(self.vault)).replace("\\", "/"))
            logger.warning(
                f"[LiveSync] Conflict on {rel_path} ({reason}) — "
                f"local version kept as {name}")
            await self.index.log_sync_event(rel_path, "conflict", "", self.device_id)
        return saved

    # ── Concurrency helpers ──

    def _path_lock(self, rel_path: str) -> asyncio.Lock:
        """Return (and memoize) the per-path serialization lock."""
        lock = self._path_locks.get(rel_path)
        if lock is None:
            lock = asyncio.Lock()
            self._path_locks[rel_path] = lock
        return lock

    def _spawn(self, coro) -> None:
        """Run a coroutine detached from the receive loop."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    # ── Self-write suppression ──

    def _suppress(self, *rel_paths: str) -> None:
        """
        Mark paths as "we are about to touch this ourselves".

        The window must outlive the debounce delay: a watchdog event is only
        evaluated when the debounce batch is flushed, which happens at least
        ``debounce_seconds`` after the event was recorded. A window shorter
        than that expires before the event is ever looked at.
        """
        until = time.time() + max(3.0, self.config.debounce_seconds * 3)
        for rel_path in rel_paths:
            self._writing_until[rel_path] = until

    def _is_suppressed(self, rel_path: str) -> bool:
        """True while a path is inside its self-write window."""
        until = self._writing_until.get(rel_path)
        if until is None:
            return False
        if time.time() >= until:
            del self._writing_until[rel_path]
            return False
        return True

    def _prune_suppressions(self) -> None:
        """Drop expired suppression entries so the map cannot grow forever."""
        now = time.time()
        for rel_path in [p for p, t in self._writing_until.items() if now >= t]:
            del self._writing_until[rel_path]

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
            except AuthenticationError as e:
                # The server rejected our token. Retrying cannot fix that.
                logger.error(f"[LiveSync] Authentication failed: {e}")
                self._set_status("auth_failed", str(e))
                self._running = False
            except Exception as e:
                logger.error(f"[LiveSync] Connection error: {e}")
                self._set_status("connecting", str(e))

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
            if self._reconnect_attempt > 10:
                self._running = False

        await self._cleanup()

    async def stop(self):
        """
        Graceful shutdown.

        Tears down here rather than leaving it to run(): callers cancel the
        run() task right after this, and a cancelled task never reaches its
        own cleanup. That leaked the watchdog observer and the aiosqlite
        worker — both non-daemon threads — on every disconnect.
        """
        self._running = False
        await self._cleanup()

    async def _cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        for task in list(self._bg_tasks):
            task.cancel()
        for task in list(self._bg_tasks):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._fail_pending_urls("client stopped")
        if self._observer:
            self._observer.stop()
            self._observer.join()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        await self.index.close()
        if self.status != "auth_failed":
            self._set_status("stopped")
        logger.info("[LiveSync] Client stopped")

    # ── Connection ──

    async def _connect_and_sync(self):
        """Connect to server, authenticate, sync, then run event loop."""
        logger.info(f"[LiveSync] Connecting to {self.config.ws_endpoint}")

        async with websockets.connect(self.config.ws_endpoint) as ws:
            self._ws = ws

            # Auth (token required by server since FIX 3)
            client_id = f"{node_}-{os.getpid()}"
            await ws.send(SyncMessage.auth(
                client_id, "desktop", self.config.share_id,
                token=self.config.share_token,
            ).to_json())

            # Wait for auth_success
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = SyncMessage.from_json(raw)

            if msg.type == MsgType.ERROR:
                raise AuthenticationError(
                    f"Auth failed: {msg.payload.get('message')}")
            if msg.type != MsgType.AUTH_SUCCESS:
                raise ConnectionError(f"Unexpected response: {msg.type}")

            # Auth succeeded — only now reset the reconnect backoff.
            # (Resetting right after TCP connect made auth failures retry
            # every 1.0s forever: "attempt 1" loop.)
            self._reconnect_attempt = 0

            self._url_ttl = int(msg.payload.get("url_ttl", 900) or 900)
            server_checksums = msg.payload.get("checksums", {})
            logger.info(
                f"[LiveSync] Authenticated. Server has {len(server_checksums)} files")
            self._set_status("live")

            # The receive loop must run BEFORE the catchup: catchup asks the
            # server for presigned URLs and the answers arrive on this loop.
            recv_task = asyncio.create_task(self._ws_recv_loop(ws))
            try:
                await self._catchup_sync(server_checksums)
                await self._event_loop(ws, recv_task)
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._fail_pending_urls("connection closed")

    async def _event_loop(self, ws, recv_task):
        """Process WS messages and local watchdog events concurrently."""
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
            for task in [watch_task, ping_task]:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    # ── Presigned URLs ──

    def _fail_pending_urls(self, reason: str) -> None:
        """Reject every in-flight URL request so no caller hangs."""
        for req_id, fut in list(self._pending_url_requests.items()):
            if not fut.done():
                fut.set_exception(TransferError(f"url request aborted: {reason}"))
            self._pending_url_requests.pop(req_id, None)

    async def _request_urls(self, op: str, paths: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Ask the server for presigned URLs.

        Args:
            op: "get" for downloads, "put" for uploads
            paths: relative paths inside the vault

        Returns:
            {rel_path: {"file": url, "meta": url}} — "meta" only for op "put".
            Paths the server refused or has no object for are absent.

        Raises:
            TransferError: no connection, timeout, or a server-side error.
        """
        if not paths:
            return {}
        if not self._ws:
            raise TransferError("no server connection for URL request")

        req_id = uuid.uuid4().hex[:12]
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_url_requests[req_id] = fut
        try:
            await self._ws.send(
                SyncMessage.request_urls(req_id, op, list(paths)).to_json())
            payload = await asyncio.wait_for(fut, timeout=60)
        except asyncio.TimeoutError as exc:
            raise TransferError(f"URL request timed out ({op}, {len(paths)} paths)") from exc
        finally:
            self._pending_url_requests.pop(req_id, None)

        if payload.get("error"):
            raise TransferError(f"server refused URL request: {payload['error']}")

        missing = payload.get("missing") or []
        if missing:
            logger.warning(
                f"[LiveSync] No {op} URL for {len(missing)} path(s): "
                f"{', '.join(missing[:5])}")
        return payload.get("urls") or {}

    # ── WS Receive Loop ──

    async def _ws_recv_loop(self, ws):
        """Listen for server messages."""
        try:
            async for raw in ws:
                try:
                    msg = SyncMessage.from_json(raw)
                    await self._handle_server_message(msg)
                except Exception as e:
                    logger.error(f"[LiveSync] Message handling error: {e}")
        finally:
            self._fail_pending_urls("receive loop ended")

    async def _handle_server_message(self, msg: SyncMessage):
        """Route incoming server message."""
        if msg.type == MsgType.PONG:
            return

        elif msg.type == MsgType.URLS_GRANTED:
            fut = self._pending_url_requests.get(msg.payload.get("req_id", ""))
            if fut is not None and not fut.done():
                fut.set_result(msg.payload)
            return

        elif msg.type == MsgType.FILE_CHANGED:
            p = msg.payload
            # Detached: the download needs presigned URLs, whose answer
            # arrives on THIS loop. Awaiting it here would deadlock.
            self._spawn(self._download_file(
                p["path"], p["minio_key"],
                expected_checksum=p.get("checksum"),
            ))

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
                f"[LiveSync] Conflict on {p['path']}: "
                f"{p.get('resolution', 'unknown')} — {p.get('message', '')}"
            )
            await self.index.log_sync_event(
                p["path"], "conflict", p.get("remote_checksum", ""), "server",
            )
            # The server only raises this for a real collision, so act on it:
            # mark the path so the winning version — which arrives right after
            # as a normal file_changed — parks our copy instead of erasing it.
            # No download from here: remote_checksum names the winner, but the
            # bytes are fetched by that file_changed, and doing it twice raced
            # with itself.
            await self.index.set_sync_state(p["path"], "conflict")
            self._contested.add(p["path"])

        elif msg.type == MsgType.FULL_STATE_READY:
            self._spawn(self._handle_full_state(msg.payload))

        elif msg.type == MsgType.ACK:
            pass  # logged on upload side already

        elif msg.type == MsgType.ERROR:
            logger.error(f"[LiveSync] Server error: {msg.payload.get('message')}")

        # VFS-bridge hook — notify after a remote mutation is applied locally
        if self.on_remote_change and msg.type in (
            MsgType.FILE_CHANGED, MsgType.FILE_DELETED, MsgType.FILE_RENAMED,
        ):
            try:
                self.on_remote_change(msg.type.value, msg.payload)
            except Exception as e:
                logger.error(f"[LiveSync] on_remote_change hook error: {e}")

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
                        # Handle renames immediately (no debounce). A rename we
                        # performed ourselves (remote rename applied locally)
                        # must not be echoed back to the server.
                        if self._is_suppressed(event[1]) or self._is_suppressed(event[2]):
                            continue
                        await self._handle_local_rename(event[1], event[2])
                    else:
                        event_type, rel_path = event
                        self._debounce.add(rel_path, event_type)
                except asyncio.QueueEmpty:
                    break

            # Flush debounce batch if ready
            if self._debounce.is_ready():
                batch = self._debounce.flush()
                for rel_path, event_type in batch.items():
                    try:
                        if event_type == "deleted":
                            # A "deleted" event inside our own write window is
                            # self-inflicted (atomic replace, trash move) and
                            # must never become a delete broadcast.
                            if self._is_suppressed(rel_path):
                                continue
                            await self._handle_local_delete(rel_path)
                        else:
                            # created/modified are safe to re-evaluate: the
                            # checksum guard in _upload_file turns a
                            # self-inflicted event into a no-op.
                            await self._upload_file(rel_path)
                    except Exception as e:
                        logger.error(f"[LiveSync] Batch process error {rel_path}: {e}")

            self._prune_suppressions()
            await asyncio.sleep(0.2)

    # ── Upload ──

    async def _upload_file(
        self, rel_path: str, urls: Optional[Dict[str, str]] = None
    ):
        """
        Upload a local file: checksum check → encrypt → presigned PUT → notify.

        Args:
            rel_path: path relative to the vault root
            urls: pre-fetched {"file": url, "meta": url} from a batch request;
                requested on demand when omitted.
        """
        full = self.vault / rel_path
        if not full.exists():
            return
        if full.stat().st_size > MAX_FILE_SIZE:
            logger.warning(f"[LiveSync] File too large, skipping: {rel_path}")
            return

        async with self._transfer_sem, self._path_lock(rel_path):
            try:
                # Checksum first — skip if unchanged
                checksum = compute_checksum_file(str(full))
                existing = await self.index.get_file(rel_path)
                if existing and existing["checksum"] == checksum:
                    return  # No actual change

                # What we were in sync with before this edit. The server needs
                # it to tell a normal next revision from a real collision.
                base_checksum = (existing or {}).get("checksum", "")

                # Encrypt
                encrypted = encrypt_file(str(full), self.config.encryption_key)
                minio_key = make_object_key(self.config.prefix, rel_path)

                if not urls:
                    granted = await self._request_urls("put", [rel_path])
                    urls = granted.get(rel_path)
                if not urls or not urls.get("file"):
                    raise TransferError(f"no upload URL granted for {rel_path}")

                await asyncio.to_thread(http_upload, urls["file"], encrypted)

                if urls.get("meta"):
                    stat_now = full.stat()
                    meta = json.dumps({
                        "checksum": checksum,
                        "mtime": stat_now.st_mtime,
                        "size": stat_now.st_size,
                        "source_client": f"{node_}",
                        "file_type": classify_file(rel_path).value,
                    }).encode("utf-8")
                    await asyncio.to_thread(http_upload, urls["meta"], meta)

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
                        base_checksum=base_checksum,
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
        url: Optional[str] = None,
    ):
        """
        Presigned GET → decrypt → verify checksum → atomic write.

        Args:
            rel_path: path relative to the vault root
            minio_key: object key, recorded in the index
            expected_checksum: plaintext checksum announced by the server
            retries: attempts before giving up
            url: pre-fetched download URL from a batch request; requested on
                demand when omitted, and re-requested on every retry because a
                presigned URL can expire mid-flight.
        """
        async with self._transfer_sem, self._path_lock(rel_path):
            full = self.vault / rel_path

            for attempt in range(retries):
                try:
                    # Backup existing file before overwrite
                    if full.exists():
                        create_backup(str(full))

                    if not url:
                        granted = await self._request_urls("get", [rel_path])
                        entry = granted.get(rel_path)
                        if not entry or not entry.get("file"):
                            raise TransferError(
                                f"no download URL granted for {rel_path}")
                        url = entry["file"]

                    encrypted = await asyncio.to_thread(http_download, url)

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
                            url = None
                            continue
                        else:
                            logger.error(
                                f"[LiveSync] Checksum mismatch after {retries} retries: "
                                f"{rel_path} — manual intervention needed"
                            )
                            await self.index.set_sync_state(rel_path, "conflict")
                            return

                    # Never overwrite bytes that exist nowhere else.
                    #
                    # Two conditions, and both are needed. Different from the
                    # incoming version alone is not a conflict — that is the
                    # ordinary case of being one revision behind, and treating
                    # it as one would file a conflict copy on every update.
                    # It is a conflict only when the file on disk also differs
                    # from the index, because the index holds the version we
                    # last agreed on with the server: a third, unsynced state
                    # means somebody edited here. A path the server has already
                    # declared contested counts too — by then our own upload
                    # has written our checksum into the index, so the index
                    # alone can no longer tell the two cases apart.
                    #
                    # Checked at this point rather than before the download:
                    # the incoming version is decrypted and checksum-verified
                    # by now, so the replacement is certain to happen. Moving
                    # the file aside any earlier left the original name empty
                    # whenever a download went on to fail.
                    if full.exists():
                        try:
                            local_cs = await asyncio.to_thread(
                                compute_checksum_file, str(full))
                            indexed = await self.index.get_file(rel_path)
                            indexed_cs = (indexed or {}).get("checksum", "")
                            contested = rel_path in self._contested
                            if detect_conflict(local_cs, actual_cs) and (
                                    contested
                                    or detect_conflict(local_cs, indexed_cs)):
                                await self._preserve_local_version(
                                    rel_path,
                                    "collided with server version" if contested
                                    else "unsynced local changes",
                                    checksum=local_cs)
                            self._contested.discard(rel_path)
                        except Exception as e:
                            logger.error(
                                f"[LiveSync] Conflict check failed {rel_path}: {e}")

                    # Atomic write. os.replace() overwrites in place — no
                    # unlink first, so the watchdog never sees a "deleted"
                    # event for a file we are only updating.
                    self._suppress(rel_path)
                    full.parent.mkdir(parents=True, exist_ok=True)
                    tmp = full.with_suffix(full.suffix + ".sync-tmp")
                    try:
                        with open(tmp, "wb") as f:
                            f.write(data)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp, full)
                    except BaseException:
                        try:
                            tmp.unlink()
                        except OSError:
                            pass
                        raise

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
                    url = None  # force a fresh URL on the next attempt
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
            # The move makes the source vanish, which the watchdog reports as a
            # local deletion. Without suppression that bounces straight back to
            # the server as a second delete.
            self._suppress(rel_path)
            move_to_sync_trash(str(self.vault), rel_path)
            logger.info(f"[LiveSync] File deleted remotely: {rel_path} → moved to .sync-trash/")

        await self.index.delete_file(rel_path)
        await self.index.log_sync_event(rel_path, "delete", "", "remote")

    async def _handle_local_delete(self, rel_path: str):
        """Handle local file deletion → notify server."""
        # Reality check before destroying remote state: a delete broadcast for
        # a file that is still on disk is always wrong.
        if (self.vault / rel_path).exists():
            logger.debug(
                f"[LiveSync] Ignoring stale delete event, file still present: {rel_path}"
            )
            return

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
            # Our own rename produces a watchdog "renamed" event plus a
            # "deleted" for the old path — neither may be echoed back.
            self._suppress(old_path, new_path)
            new_full.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old_full, new_full)

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

        # Downloads first (server is source of truth on reconnect).
        # One batched URL request instead of one round trip per file.
        if to_download:
            try:
                get_urls = await self._request_urls(
                    "get", [rel for rel, _ in to_download])
            except TransferError as e:
                logger.error(f"[LiveSync] Catchup URL batch failed: {e}")
                get_urls = {}
            for rel_path, minio_key in to_download:
                entry = get_urls.get(rel_path) or {}
                await self._download_file(
                    rel_path, minio_key,
                    expected_checksum=server_checksums.get(rel_path),
                    url=entry.get("file") or None,
                )

        # Then uploads (local changes made while offline)
        if to_upload:
            try:
                put_urls = await self._request_urls("put", list(to_upload))
            except TransferError as e:
                logger.error(f"[LiveSync] Catchup URL batch failed: {e}")
                put_urls = {}
            for rel_path in to_upload:
                await self._upload_file(rel_path, urls=put_urls.get(rel_path))

        if to_download or to_upload:
            total = len(to_download) + len(to_upload)
            logger.info(f"[LiveSync] Catchup complete: synced {total} files")

    async def _compute_diff(
        self, server_checksums: Dict[str, str]
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Compute what needs to be downloaded vs uploaded.

        Three cases per path, decided against the INDEX rather than against
        the server alone. The index records the state we last had in sync, so
        it is what tells a stale local file apart from a locally edited one:

          - on disk == index, differs from server → we are behind, download
          - on disk != index, differs from server → we edited while offline.
            Upload ours and keep the server's version as a conflict copy;
            never overwrite the local edit silently.
          - not on the server at all → upload

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

        # Files on server we don't have (or that differ)
        for rel_path, server_cs in server_checksums.items():
            disk_cs = local_fs_files.get(rel_path)
            indexed_cs = local_checksums.get(rel_path)
            local_cs = disk_cs or indexed_cs

            if local_cs == server_cs:
                continue

            locally_edited = bool(disk_cs and indexed_cs and disk_cs != indexed_cs)
            if locally_edited:
                # Both sides moved since we last agreed. Push our version and
                # let the server rule on it; whichever side loses ends up as a
                # conflict copy through the normal download path, so there is
                # exactly one place that resolves collisions.
                logger.warning(
                    f"[LiveSync] Offline edit collides with server version: "
                    f"{rel_path}")
                await self.index.set_sync_state(rel_path, "conflict")
                self._contested.add(rel_path)
                to_upload.append(rel_path)
                continue

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
        url = payload.get("url", "")

        if not url:
            logger.error("[LiveSync] Cannot download full state: no URL from server")
            return

        try:
            logger.info(f"[LiveSync] Downloading full state: {file_count} files")
            data = await asyncio.to_thread(http_download, url)

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

            if missing:
                try:
                    get_urls = await self._request_urls(
                        "get", [rel for rel, _, _ in missing])
                except TransferError as e:
                    logger.error(f"[LiveSync] Full state URL batch failed: {e}")
                    get_urls = {}
                for rel_path, mk, cs in missing:
                    entry = get_urls.get(rel_path) or {}
                    await self._download_file(
                        rel_path, mk, expected_checksum=cs,
                        url=entry.get("file") or None,
                    )

            logger.info(f"[LiveSync] Full state sync complete")

        except Exception as e:
            logger.error(f"[LiveSync] Full state download failed: {e}")


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
    config = token.to_sync_config(args.vault, raw_token=args.token)

    client = SyncClient(config)
    try:
        await client.run()
    except KeyboardInterrupt:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(_run_standalone())
