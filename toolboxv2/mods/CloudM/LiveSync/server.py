"""
LiveSync Server (SyncService)
=============================
Central coordination service running on the remote server.

Responsibilities:
  - WebSocket server for client notifications (NEVER file content)
  - Auth via HMAC-signed share tokens
  - Presigned per-object URLs (clients never receive S3 credentials)
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
    MsgType, SyncMessage, FileType, classify_file, should_ignore,
)
from .config import SyncConfig, ShareToken, load_env_config
from .share_store import VALID_MODES, get_share
from .index import LocalIndex
from .minio_helper import (
    create_minio_client, ensure_bucket, healthcheck,
    upload_bytes, download_bytes, make_object_key, make_meta_key,
    list_remote_files, presign_get, presign_put, object_exists,
)
from .conflict import detect_conflict


try:
    import websockets
    from websockets.server import serve as ws_serve
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

from toolboxv2 import get_logger
logger = get_logger()


# ── Thread-safe Watchdog → asyncio Queue (BUG FIX from spec) ──

class SyncServer:
    """
    Central sync coordination server.

    Manages:
    - WebSocket connections + auth
    - Server-side file index (fed by client messages only)
    - Change broadcast
    - Conflict detection

    Modes
    -----
    relay (default)
        The server is a pure broker. It never reads or writes share content
        and never watches its own directory; the vault folder holds nothing
        but ``.livesync_server.db``.

    replica
        Same broker, plus an ordinary SyncClient of its own on the same
        folder, so this machine also carries a copy. All file handling goes
        through that client — the identical code path every other node uses,
        with the encryption key on the client side. The broker half still
        never sees plaintext.

    The old design had the server watch its own vault and broadcast changes it
    could not upload (no key), which announced files no client could ever
    fetch. That path is gone.
    """

    def __init__(
        self,
        vault_path: str,
        share_id: str,
        env_config: Optional[dict] = None,
        mode: str = "relay",
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.vault_path = Path(vault_path)
        self.share_id = share_id
        self.mode = mode
        self.env_config = env_config or load_env_config()

        # One authoritative bucket name for the whole server. The old code
        # defaulted to "livesync" in two places and "tb-shared" in a third,
        # so a missing env key silently split reads from writes.
        self.bucket = self.env_config.get("bucket") or "tb-shared"
        self.url_ttl = int(self.env_config.get("url_ttl") or 900)

        # Index
        index_path = self.vault_path / ".livesync_server.db"
        self.index = LocalIndex(str(index_path))

        # Connected clients: {client_id: {ws, client_id, device_type}}
        self.clients: Dict[str, Dict[str, Any]] = {}

        # Pending broadcasts (batched)
        self._pending_broadcasts: List[SyncMessage] = []

        # MinIO admin client (for presigning, full-state export, healthcheck)
        self._minio_admin = None

        # replica mode: our own SyncClient on the same folder
        self._replica_client = None
        self._replica_task: Optional[asyncio.Task] = None

        self._running = False

    # ── Lifecycle ──

    def _build_ssl_context(self):
        """FIX 4: Build SSL context for WSS if configured."""
        ssl_mode = os.getenv("LIVESYNC_WSS", "false").lower()
        if ssl_mode not in ("true", "1", "yes"):
            return None

        import ssl
        cert_path = os.getenv("LIVESYNC_SSL_CERT", "")
        key_path = os.getenv("LIVESYNC_SSL_KEY", "")

        if not cert_path or not key_path:
            logger.warning("[LiveSync] LIVESYNC_WSS=true but no cert/key set, falling back to ws://")
            return None

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert_path, key_path)
        logger.info(f"[LiveSync] SSL enabled: cert={cert_path}")
        return ctx

    async def start(self, host: str = "0.0.0.0", port: int = 8765):
        """Start the sync server (WS/WSS + watchdog + index)."""
        if not WS_AVAILABLE:
            raise RuntimeError("websockets library required: pip install websockets")

        self._running = True

        # Init index
        await self._init_index()

        # Init MinIO admin client
        try:
            self._minio_admin = create_minio_client(self.env_config)
            ensure_bucket(self._minio_admin, self.bucket)
            ok, msg = healthcheck(self._minio_admin)
            if ok:
                logger.info(f"[LiveSync] MinIO connected: {self.env_config['endpoint']}")
            else:
                logger.error(f"[LiveSync] MinIO healthcheck failed: {msg}")
        except Exception as e:
            logger.error(f"[LiveSync] MinIO init failed: {e}")
            self._minio_admin = None

        # FIX 4: Build SSL context for WSS (None = plain ws://)
        ssl_ctx = self._build_ssl_context()
        scheme = "wss" if ssl_ctx else "ws"
        logger.info(f"[LiveSync] Starting server on {scheme}://{host}:{port}")

        async with ws_serve(self._handle_client, host, port, ssl=ssl_ctx):
            logger.info(
                f"[LiveSync] Server running on {scheme}://{host}:{port} "
                f"[mode={self.mode}]")

            if self.mode == "replica":
                await self._start_replica_client(port)

            # Main loop: flush pending broadcasts
            while self._running:
                await self._flush_broadcasts()
                await asyncio.sleep(0.1)

    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        await self._stop_replica_client()
        for cid, client in list(self.clients.items()):
            try:
                await client["ws"].close()
            except Exception:
                pass
        await self.index.close()
        logger.info("[LiveSync] Server stopped")

    # ── Index Init ──

    async def _init_index(self):
        """
        Open the index. Contents come from clients, never from a disk scan.

        Scanning the vault used to add local files with a checksum but no
        object in storage, so the server advertised files that nobody could
        download — and the entry stayed in the index for good. In replica mode
        the embedded SyncClient uploads those same files properly and reports
        them like any other client.
        """
        await self.index.init()
        existing = len(await self.index.get_all_checksums())
        logger.info(f"[LiveSync] Index opened: {existing} known files")

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
                logger.warning("[LiveSync] Rejected: non-auth first message")
                return

            # ── FIX 3: Token-Validierung vor Credential-Vending ──
            client_id = msg.payload.get("client_id", f"client-{int(time.time())}")
            device_type = msg.payload.get("device_type", "unknown")
            share_id = msg.payload.get("share_id", self.share_id)
            client_token = msg.payload.get("token", "")

            # Validate share_id against this server's active share
            if share_id != self.share_id:
                await websocket.send(
                    SyncMessage.error(f"Invalid share_id for this server").to_json()
                )
                logger.warning(
                    f"[LiveSync] AUTH_DENIED: share_id mismatch. "
                    f"Client={client_id} requested={share_id} expected={self.share_id}"
                )
                return

            # Verify the token: HMAC signature (this node's device key),
            # expiry, and share binding. A token we did not sign is a forgery
            # and there is no unsigned fallback.
            reason = ""
            if not client_token:
                reason = "missing_token"
            else:
                try:
                    tok = ShareToken.verify(client_token)
                    if tok.share_id != share_id:
                        reason = "share_id_mismatch"
                except ValueError as exc:
                    reason = str(exc)
                except RuntimeError as exc:
                    # Device key unavailable — the server cannot authenticate
                    # anyone. Fail closed and say so.
                    logger.error(f"[LiveSync] AUTH unavailable: {exc}")
                    await websocket.send(
                        SyncMessage.error("Server cannot verify tokens").to_json()
                    )
                    return

            if reason:
                await websocket.send(
                    SyncMessage.error("Invalid or missing share token").to_json()
                )
                logger.warning(
                    f"[LiveSync] AUTH_DENIED: {reason}. "
                    f"Client={client_id} share={share_id}",
                    extra={"audit_action": "AUTH_DENIED", "client_id": client_id,
                           "share_id": share_id, "reason": reason}
                )
                return

            # Log client IP for audit
            peer = websocket.remote_address[0] if hasattr(websocket, 'remote_address') else "unknown"
            logger.info(
                f"[LiveSync] AUTH_SUCCESS: client={client_id} share={share_id} "
                f"device={device_type} ip={peer}"
            )

            # No credentials are handed out. The client asks for a presigned
            # URL per object via REQUEST_URLS when it needs one.
            checksums = await self.index.get_all_checksums()

            # Register client
            self.clients[client_id] = {
                "ws": websocket,
                "client_id": client_id,
                "device_type": device_type,
                "peer_ip": peer,
                "auth_at": time.time(),
            }

            # Send auth success
            await websocket.send(
                SyncMessage.auth_success(
                    client_id, checksums, url_ttl=self.url_ttl).to_json()
            )
            logger.info(f"[LiveSync] Client connected: {client_id} ({device_type}) from {peer}")

            # Handle messages
            async for raw_msg in websocket:
                await self._handle_message(client_id, raw_msg)

        except asyncio.TimeoutError:
            logger.warning("[LiveSync] Client auth timeout")
        except websockets.exceptions.ConnectionClosed:
            logger.info(
                f"[LiveSync] Client disconnected: {client_id}",
                extra={"audit_action": "CLIENT_DISCONNECT", "client_id": client_id}
            )
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

            elif msg.type == MsgType.REQUEST_URLS:
                p = msg.payload
                await self._grant_urls(
                    client_id, p.get("req_id", ""), p.get("op", ""),
                    p.get("paths", []),
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
        logger.info(
            f"[LiveSync] File synced: {path} from {client_id}",
            extra={"audit_action": "FILE_UPLOAD", "client_id": client_id,
                   "share_id": self.share_id, "path": path, "checksum": checksum}
        )
    async def _process_file_deleted(self, client_id: str, path: str):
        """Process file_deleted from a client."""
        await self.index.delete_file(path)
        # Delete encrypted object from MinIO to prevent orphaned data
        if self._minio_admin:
            try:
                from .minio_helper import delete_file_and_meta
                delete_file_and_meta(
                    self._minio_admin, self.bucket, self.share_id, path)
                logger.info(f"[LiveSync] MinIO object deleted: {self.share_id}/{path}")
            except Exception as e:
                logger.warning(f"[LiveSync] MinIO delete failed for {path}: {e}")

        # Broadcast
        msg = SyncMessage.file_deleted(path, source_client=client_id)
        await self._broadcast(msg, skip_client=client_id)

        await self.index.log_sync_event(path, "delete", "", client_id)
        logger.warning(
            f"[LiveSync] File deleted: {path} by {client_id}",
            extra={"audit_action": "FILE_DELETE", "client_id": client_id,
                   "share_id": self.share_id, "path": path}
        )

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
            client_id, checksums, url_ttl=self.url_ttl,
        ).to_json())

    # ── Replica mode ──

    async def _start_replica_client(self, port: int):
        """
        Run an ordinary SyncClient against our own vault folder.

        The token comes from the encrypted share store, never from the command
        line or the environment, so it does not leak through ``ps`` or
        ``/proc/<pid>/environ``. The client connects to loopback regardless of
        the endpoint baked into the token: it is on this machine.
        """
        record = get_share(self.share_id)
        if not record or not record.get("token"):
            logger.error(
                f"[LiveSync] replica mode needs a stored token for share "
                f"{self.share_id}, none found — running as relay instead"
            )
            self.mode = "relay"
            return

        try:
            from .client import SyncClient
            config = ShareToken.decode(record["token"]).to_sync_config(
                str(self.vault_path), raw_token=record["token"])
        except ValueError as e:
            logger.error(
                f"[LiveSync] replica mode: stored token unusable ({e}) — "
                "running as relay instead"
            )
            self.mode = "relay"
            return

        config.ws_endpoint = f"ws://127.0.0.1:{port}"
        self._replica_client = SyncClient(config)
        self._replica_task = asyncio.create_task(self._replica_client.run())
        logger.info(
            f"[LiveSync] Replica client started on {self.vault_path}")

    async def _stop_replica_client(self):
        """Stop the embedded client and release its threads."""
        if self._replica_client:
            try:
                await self._replica_client.stop()
            except Exception as e:
                logger.error(f"[LiveSync] Replica client stop failed: {e}")
        if self._replica_task:
            self._replica_task.cancel()
            try:
                await self._replica_task
            except (asyncio.CancelledError, Exception):
                pass
        self._replica_client = None
        self._replica_task = None

    # ── Presigned URL vending ──

    MAX_URL_BATCH = 500

    @staticmethod
    def _is_safe_rel_path(rel_path: str) -> bool:
        """
        Reject anything that could escape the share prefix.

        A client controls this string, so it decides which object key the
        server signs. Without this check ``../<other-share>/x`` would produce
        a valid presigned URL for a foreign share.
        """
        if not rel_path or not isinstance(rel_path, str):
            return False
        if len(rel_path) > 1024:
            return False
        normalized = rel_path.replace("\\", "/")
        if normalized.startswith("/") or normalized.startswith("~"):
            return False
        if ":" in normalized.split("/")[0] and len(normalized.split("/")[0]) == 2:
            return False  # drive letter, e.g. C:/
        parts = normalized.split("/")
        if any(part in ("", ".", "..") for part in parts):
            return False
        if should_ignore(normalized):
            return False
        return True

    async def _grant_urls(
        self, client_id: str, req_id: str, op: str, paths: list
    ):
        """
        Hand out short-lived presigned URLs for specific objects.

        op "get" → one URL per existing object.
        op "put" → one URL for the encrypted object plus one for its metadata
        sidecar.

        Every path is validated against the share prefix first: the client
        never gets a URL for anything outside its own share.
        """
        if client_id not in self.clients:
            return
        ws = self.clients[client_id]["ws"]

        if op not in ("get", "put"):
            await ws.send(SyncMessage.urls_granted(
                req_id, op, {}, error=f"unsupported op '{op}'").to_json())
            return

        if not self._minio_admin:
            logger.error(
                "[LiveSync] URL request but no object storage connection")
            await ws.send(SyncMessage.urls_granted(
                req_id, op, {}, missing=list(paths),
                error="object storage unavailable").to_json())
            return

        if len(paths) > self.MAX_URL_BATCH:
            await ws.send(SyncMessage.urls_granted(
                req_id, op, {}, missing=list(paths),
                error=f"too many paths (max {self.MAX_URL_BATCH})").to_json())
            return

        urls: dict = {}
        missing: list = []

        for rel_path in paths:
            if not self._is_safe_rel_path(rel_path):
                missing.append(rel_path)
                logger.warning(
                    f"[LiveSync] URL_DENIED: unsafe path from {client_id}: "
                    f"{rel_path!r}",
                    extra={"audit_action": "URL_DENIED",
                           "client_id": client_id, "share_id": self.share_id}
                )
                continue

            file_key = make_object_key(self.share_id, rel_path)
            try:
                if op == "get":
                    if not object_exists(self._minio_admin, self.bucket, file_key):
                        missing.append(rel_path)
                        continue
                    urls[rel_path] = {
                        "file": presign_get(
                            self._minio_admin, self.bucket, file_key, self.url_ttl),
                    }
                else:
                    meta_key = make_meta_key(self.share_id, rel_path)
                    urls[rel_path] = {
                        "file": presign_put(
                            self._minio_admin, self.bucket, file_key, self.url_ttl),
                        "meta": presign_put(
                            self._minio_admin, self.bucket, meta_key, self.url_ttl),
                    }
            except Exception as e:
                missing.append(rel_path)
                logger.error(f"[LiveSync] Presign failed for {rel_path}: {e}")

        await ws.send(SyncMessage.urls_granted(
            req_id, op, urls, missing=missing, expires_in=self.url_ttl).to_json())

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
                upload_bytes(self._minio_admin, self.bucket, minio_key, data)

            checksums = await self.index.get_all_checksums()
            file_count = len(checksums)

            url = ""
            if self._minio_admin:
                url = presign_get(
                    self._minio_admin, self.bucket, minio_key, self.url_ttl)

            ws = self.clients[client_id]["ws"]
            await ws.send(
                SyncMessage.full_state_ready(minio_key, file_count, url).to_json()
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
    parser.add_argument(
        "--mode", choices=list(VALID_MODES), default="relay",
        help="relay: broker only. replica: also keep a local copy of the folder "
             "(needs the share token in the encrypted share store)",
    )
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
        mode=args.mode,
    )

    try:
        await server.start(host=args.host, port=args.port)
    except KeyboardInterrupt:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(_run_standalone())
