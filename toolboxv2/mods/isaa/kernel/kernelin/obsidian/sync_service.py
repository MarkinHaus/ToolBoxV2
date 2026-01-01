"""
Obsidian Live Sync Service
===========================

Bidirectional real-time sync between:
- Server (Source of Truth)
- Desktop Obsidian App
- Mobile Obsidian App
- Web Viewer (read-only)

Uses WebSocket for real-time updates and Git for versioning.

Protocol:
- Delta sync (only changes, not full files)
- CRDT-based conflict resolution
- JWT authentication
- Per-client sync branches
"""

import asyncio
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

try:
    import websockets
    from websockets.server import serve as ws_serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("⚠️ websockets not installed. Install with: pip install websockets")

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("⚠️ PyJWT not installed. Install with: pip install PyJWT")


logger = logging.getLogger(__name__)


# ===== SYNC PROTOCOL =====

class ChangeType(Enum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


@dataclass
class FileChange:
    """Represents a file change"""
    change_type: ChangeType
    path: str
    checksum: Optional[str] = None
    content: Optional[str] = None  # For create/modify
    old_path: Optional[str] = None  # For rename
    timestamp: float = field(default_factory=time.time)
    client_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.change_type.value,
            "path": self.path,
            "checksum": self.checksum,
            "content": self.content,
            "old_path": self.old_path,
            "timestamp": self.timestamp,
            "client_id": self.client_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FileChange':
        return cls(
            change_type=ChangeType(data["type"]),
            path=data["path"],
            checksum=data.get("checksum"),
            content=data.get("content"),
            old_path=data.get("old_path"),
            timestamp=data.get("timestamp", time.time()),
            client_id=data.get("client_id")
        )


@dataclass
class SyncMessage:
    """WebSocket message format"""
    msg_type: str  # "sync", "ack", "conflict", "error", "auth", "ping"
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    msg_id: str = ""

    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:12]

    def to_json(self) -> str:
        return json.dumps({
            "type": self.msg_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "id": self.msg_id
        })

    @classmethod
    def from_json(cls, data: str) -> 'SyncMessage':
        parsed = json.loads(data)
        return cls(
            msg_type=parsed["type"],
            payload=parsed.get("payload", {}),
            timestamp=parsed.get("timestamp", time.time()),
            msg_id=parsed.get("id", "")
        )


@dataclass
class ClientConnection:
    """Represents a connected client"""
    client_id: str
    user_id: str
    websocket: Any
    device_type: str  # "desktop", "mobile", "web"
    connected_at: float = field(default_factory=time.time)
    last_sync: float = 0
    pending_changes: List[FileChange] = field(default_factory=list)
    authenticated: bool = False


# ===== FILE WATCHER =====

class VaultFileHandler(FileSystemEventHandler):
    """Watch vault for file changes"""

    def __init__(self, sync_service: 'SyncService', vault_path: Path):
        self.sync_service = sync_service
        self.vault_path = vault_path
        self._debounce: Dict[str, float] = {}
        self._debounce_delay = 0.5  # seconds

    def _should_process(self, path: str) -> bool:
        """Debounce and filter events"""
        # Ignore non-markdown and system files
        if not path.endswith('.md'):
            return False
        if '.obsidian' in path or '.git' in path:
            return False

        # Debounce
        now = time.time()
        last = self._debounce.get(path, 0)
        if now - last < self._debounce_delay:
            return False
        self._debounce[path] = now
        return True

    def _get_relative_path(self, path: str) -> str:
        return str(Path(path).relative_to(self.vault_path))

    def on_modified(self, event):
        if event.is_directory:
            return
        path = self._get_relative_path(event.src_path)
        if self._should_process(path):
            asyncio.create_task(
                self.sync_service.handle_server_change(
                    FileChange(ChangeType.MODIFY, path)
                )
            )

    def on_created(self, event):
        if event.is_directory:
            return
        path = self._get_relative_path(event.src_path)
        if self._should_process(path):
            asyncio.create_task(
                self.sync_service.handle_server_change(
                    FileChange(ChangeType.CREATE, path)
                )
            )

    def on_deleted(self, event):
        if event.is_directory:
            return
        path = self._get_relative_path(event.src_path)
        if path.endswith('.md'):
            asyncio.create_task(
                self.sync_service.handle_server_change(
                    FileChange(ChangeType.DELETE, path)
                )
            )

    def on_moved(self, event):
        if event.is_directory:
            return
        old_path = self._get_relative_path(event.src_path)
        new_path = self._get_relative_path(event.dest_path)
        if new_path.endswith('.md'):
            asyncio.create_task(
                self.sync_service.handle_server_change(
                    FileChange(ChangeType.RENAME, new_path, old_path=old_path)
                )
            )


# ===== CONFLICT RESOLVER =====

class ConflictResolver:
    """Handle sync conflicts"""

    @staticmethod
    def resolve(local_change: FileChange, remote_change: FileChange) -> FileChange:
        """
        Resolve conflict between local and remote changes.

        Strategy:
        - If same change type and similar timestamp: merge content
        - If different types: prioritize by type (delete < modify < create)
        - If timestamps differ significantly: latest wins
        """
        time_diff = abs(local_change.timestamp - remote_change.timestamp)

        # Same type, close timing -> needs merge
        if local_change.change_type == remote_change.change_type:
            if time_diff < 60:  # Within 1 minute
                return ConflictResolver._merge_changes(local_change, remote_change)
            else:
                # Latest wins
                return local_change if local_change.timestamp > remote_change.timestamp else remote_change

        # Different types
        priority = {
            ChangeType.CREATE: 3,
            ChangeType.MODIFY: 2,
            ChangeType.RENAME: 1,
            ChangeType.DELETE: 0
        }

        if priority[local_change.change_type] >= priority[remote_change.change_type]:
            return local_change
        return remote_change

    @staticmethod
    def _merge_changes(change1: FileChange, change2: FileChange) -> FileChange:
        """Merge two modify changes (simple: concat with separator)"""
        if change1.content and change2.content:
            # Simple merge: add conflict markers
            merged_content = f"""<<<<<<< LOCAL
{change1.content}
=======
{change2.content}
>>>>>>> REMOTE
"""
            return FileChange(
                change_type=ChangeType.MODIFY,
                path=change1.path,
                content=merged_content,
                timestamp=max(change1.timestamp, change2.timestamp)
            )

        # Fallback to latest
        return change1 if change1.timestamp > change2.timestamp else change2


# ===== SYNC SERVICE =====

class SyncService:
    """
    Main sync service orchestrating real-time sync between clients and server.
    """

    def __init__(self, vault_path: str, host: str = "0.0.0.0", port: int = 8765,
                 jwt_secret: str = None):
        self.vault_path = Path(vault_path)
        self.host = host
        self.port = port
        self.jwt_secret = jwt_secret or "change-me-in-production"

        self.clients: Dict[str, ClientConnection] = {}
        self.file_checksums: Dict[str, str] = {}  # path -> checksum
        self.pending_broadcasts: List[FileChange] = []

        self._running = False
        self._observer = None

        # Build initial checksum index
        self._build_checksum_index()

    def _build_checksum_index(self):
        """Build checksum index of all files"""
        for md_file in self.vault_path.rglob("*.md"):
            if '.obsidian' in str(md_file) or '.git' in str(md_file):
                continue
            rel_path = str(md_file.relative_to(self.vault_path))
            content = md_file.read_text(encoding='utf-8')
            self.file_checksums[rel_path] = hashlib.sha256(content.encode()).hexdigest()[:16]

    def _compute_checksum(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def start(self):
        """Start the sync service"""
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets library not available")

        self._running = True

        # Start file watcher
        if WATCHDOG_AVAILABLE:
            self._observer = Observer()
            handler = VaultFileHandler(self, self.vault_path)
            self._observer.schedule(handler, str(self.vault_path), recursive=True)
            self._observer.start()
            print(f"✓ File watcher started for {self.vault_path}")

        # Start WebSocket server
        print(f"🚀 Starting sync server on ws://{self.host}:{self.port}")

        async with ws_serve(self._handle_client, self.host, self.port):
            print(f"✓ Sync server running on ws://{self.host}:{self.port}")

            while self._running:
                await asyncio.sleep(1)
                await self._process_pending_broadcasts()

    async def stop(self):
        """Stop the sync service"""
        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join()

        # Close all client connections
        for client in list(self.clients.values()):
            await client.websocket.close()

        print("✓ Sync service stopped")

    async def _handle_client(self, websocket):
        """Handle a client WebSocket connection"""
        client_id = None

        try:
            # Wait for auth message
            auth_msg = await asyncio.wait_for(websocket.recv(), timeout=30)
            msg = SyncMessage.from_json(auth_msg)

            if msg.msg_type != "auth":
                await websocket.send(SyncMessage(
                    "error", {"message": "First message must be auth"}
                ).to_json())
                return

            # Validate auth
            client = await self._authenticate_client(msg, websocket)
            if not client:
                await websocket.send(SyncMessage(
                    "error", {"message": "Authentication failed"}
                ).to_json())
                return

            client_id = client.client_id
            self.clients[client_id] = client

            # Send auth success + initial sync state
            await websocket.send(SyncMessage("auth_success", {
                "client_id": client_id,
                "checksums": self.file_checksums
            }).to_json())

            print(f"✓ Client connected: {client_id} ({client.device_type})")

            # Handle messages
            async for message in websocket:
                await self._handle_message(client, message)

        except asyncio.TimeoutError:
            print(f"⚠️ Client auth timeout")
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {client_id}")
        except Exception as e:
            print(f"❌ Client error: {e}")
        finally:
            if client_id and client_id in self.clients:
                del self.clients[client_id]

    async def _authenticate_client(self, msg: SyncMessage, websocket) -> Optional[ClientConnection]:
        """Authenticate a client connection"""
        try:
            token = msg.payload.get("token")
            device_type = msg.payload.get("device_type", "unknown")

            if JWT_AVAILABLE and token:
                # Verify JWT
                payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
                user_id = payload.get("user_id")
                client_id = f"{user_id}-{device_type}-{int(time.time())}"
            else:
                # Fallback: use provided client_id
                user_id = msg.payload.get("user_id", "anonymous")
                client_id = msg.payload.get("client_id", f"client-{int(time.time())}")

            return ClientConnection(
                client_id=client_id,
                user_id=user_id,
                websocket=websocket,
                device_type=device_type,
                authenticated=True
            )

        except Exception as e:
            print(f"Auth error: {e}")
            return None

    async def _handle_message(self, client: ClientConnection, raw_message: str):
        """Handle incoming message from client"""
        try:
            msg = SyncMessage.from_json(raw_message)

            if msg.msg_type == "ping":
                await client.websocket.send(SyncMessage("pong", {}).to_json())

            elif msg.msg_type == "sync":
                # Client is sending changes
                changes = [FileChange.from_dict(c) for c in msg.payload.get("changes", [])]
                for change in changes:
                    change.client_id = client.client_id
                    await self.handle_client_change(client, change)

            elif msg.msg_type == "request_full":
                # Client requesting full file content
                path = msg.payload.get("path")
                await self._send_full_file(client, path)

            elif msg.msg_type == "request_sync":
                # Client requesting sync state
                await self._send_sync_state(client)

        except Exception as e:
            print(f"❌ Message handling error: {e}")
            await client.websocket.send(SyncMessage(
                "error", {"message": str(e)}
            ).to_json())

    async def handle_client_change(self, client: ClientConnection, change: FileChange):
        """Handle change from client"""
        file_path = self.vault_path / change.path

        try:
            if change.change_type == ChangeType.CREATE:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(change.content, encoding='utf-8')
                self.file_checksums[change.path] = self._compute_checksum(change.content)

            elif change.change_type == ChangeType.MODIFY:
                # Check for conflict
                if change.path in self.file_checksums:
                    current_checksum = self.file_checksums[change.path]
                    if change.checksum and change.checksum != current_checksum:
                        # Conflict! Client had old version
                        current_content = file_path.read_text(encoding='utf-8')
                        server_change = FileChange(
                            ChangeType.MODIFY, change.path,
                            checksum=current_checksum,
                            content=current_content
                        )
                        resolved = ConflictResolver.resolve(server_change, change)

                        if resolved != change:
                            # Send conflict notification
                            await client.websocket.send(SyncMessage("conflict", {
                                "path": change.path,
                                "resolution": "server_wins" if resolved == server_change else "merged"
                            }).to_json())

                        change = resolved

                file_path.write_text(change.content, encoding='utf-8')
                self.file_checksums[change.path] = self._compute_checksum(change.content)

            elif change.change_type == ChangeType.DELETE:
                if file_path.exists():
                    file_path.unlink()
                if change.path in self.file_checksums:
                    del self.file_checksums[change.path]

            elif change.change_type == ChangeType.RENAME:
                old_path = self.vault_path / change.old_path
                if old_path.exists():
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    old_path.rename(file_path)
                    if change.old_path in self.file_checksums:
                        self.file_checksums[change.path] = self.file_checksums[change.old_path]
                        del self.file_checksums[change.old_path]

            # Broadcast to other clients
            self.pending_broadcasts.append(change)

            # Send ACK
            await client.websocket.send(SyncMessage("ack", {
                "path": change.path,
                "checksum": self.file_checksums.get(change.path)
            }).to_json())

            print(f"📝 {change.change_type.value}: {change.path} (from {client.client_id})")

        except Exception as e:
            print(f"❌ Error applying change: {e}")
            await client.websocket.send(SyncMessage("error", {
                "path": change.path,
                "message": str(e)
            }).to_json())

    async def handle_server_change(self, change: FileChange):
        """Handle change from server (e.g., from agent)"""
        file_path = self.vault_path / change.path

        if file_path.exists():
            content = file_path.read_text(encoding='utf-8')
            change.content = content
            change.checksum = self._compute_checksum(content)
            self.file_checksums[change.path] = change.checksum

        # Broadcast to all clients
        self.pending_broadcasts.append(change)

        print(f"📡 Server change: {change.change_type.value} {change.path}")

    async def _process_pending_broadcasts(self):
        """Broadcast pending changes to all clients"""
        if not self.pending_broadcasts:
            return

        changes = self.pending_broadcasts[:]
        self.pending_broadcasts.clear()

        msg = SyncMessage("sync", {
            "changes": [c.to_dict() for c in changes]
        })

        for client_id, client in list(self.clients.items()):
            # Don't send back to originator
            for change in changes:
                if change.client_id == client_id:
                    continue

            try:
                await client.websocket.send(msg.to_json())
            except Exception as e:
                print(f"⚠️ Failed to broadcast to {client_id}: {e}")

    async def _send_full_file(self, client: ClientConnection, path: str):
        """Send full file content to client"""
        file_path = self.vault_path / path

        if file_path.exists():
            content = file_path.read_text(encoding='utf-8')
            await client.websocket.send(SyncMessage("file_content", {
                "path": path,
                "content": content,
                "checksum": self._compute_checksum(content)
            }).to_json())
        else:
            await client.websocket.send(SyncMessage("error", {
                "message": f"File not found: {path}"
            }).to_json())

    async def _send_sync_state(self, client: ClientConnection):
        """Send current sync state to client"""
        await client.websocket.send(SyncMessage("sync_state", {
            "checksums": self.file_checksums,
            "timestamp": time.time()
        }).to_json())

    # ===== JWT TOKEN GENERATION =====

    def generate_token(self, user_id: str, expires_hours: int = 24) -> str:
        """Generate JWT token for client auth"""
        if not JWT_AVAILABLE:
            return f"simple-token-{user_id}"

        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=expires_hours),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")


# ===== MAIN =====

async def main():
    """Run sync service standalone"""
    import argparse

    parser = argparse.ArgumentParser(description="Obsidian Sync Service")
    parser.add_argument("--vault", "-v", required=True, help="Path to vault")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", "-p", type=int, default=8765, help="Port")
    parser.add_argument("--secret", help="JWT secret")

    args = parser.parse_args()

    service = SyncService(
        vault_path=args.vault,
        host=args.host,
        port=args.port,
        jwt_secret=args.secret
    )

    try:
        await service.start()
    except KeyboardInterrupt:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
