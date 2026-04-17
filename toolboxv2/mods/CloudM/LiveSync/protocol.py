"""
LiveSync Protocol
=================
Pydantic models for all WebSocket message types.
WebSocket transports ONLY metadata + MinIO keys — NEVER file content.

Message flow:
  Client → Server:  auth, file_changed, file_deleted, file_renamed, request_full, request_sync, ping
  Server → Client:  auth_success, file_changed, file_deleted, file_renamed,
                     full_state_ready, conflict, ack, pong, error
"""

from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# ── Message Types ──

class MsgType(str, Enum):
    """All valid WebSocket message types."""

    # Auth
    AUTH = "auth"
    AUTH_SUCCESS = "auth_success"

    # File operations (notification-only, no content)
    FILE_CHANGED = "file_changed"
    FILE_DELETED = "file_deleted"
    FILE_RENAMED = "file_renamed"

    # Sync state
    REQUEST_FULL = "request_full"
    REQUEST_SYNC = "request_sync"
    FULL_STATE_READY = "full_state_ready"

    # Conflict
    CONFLICT = "conflict"

    # Acknowledgement
    ACK = "ack"

    # Keepalive
    PING = "ping"
    PONG = "pong"

    # Error
    ERROR = "error"


class FileType(str, Enum):
    """File type categories for conflict resolution strategy."""
    TEXT = "text"      # .md, .txt, .json, .csv → merge-marker for .md
    IMAGE = "image"    # .png, .jpg, .jpeg, .gif, .webp → latest-wins
    DOCUMENT = "doc"   # .pdf → latest-wins
    OTHER = "other"    # fallback → latest-wins


# Map extensions → FileType
_EXT_MAP: Dict[str, FileType] = {}
for _ext in (".md", ".txt", ".json", ".csv"):
    _EXT_MAP[_ext] = FileType.TEXT
for _ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
    _EXT_MAP[_ext] = FileType.IMAGE
_EXT_MAP[".pdf"] = FileType.DOCUMENT


def classify_file(path: str) -> FileType:
    """Classify a file path into a FileType for conflict resolution."""
    for ext, ft in _EXT_MAP.items():
        if path.lower().endswith(ext):
            return ft
    return FileType.OTHER


# ── Ignore Rules ──

# Directories to never sync
IGNORE_DIRS = frozenset({".obsidian", ".git", ".sync-trash", "__pycache__"})

# File suffixes to never sync
IGNORE_SUFFIXES = (
    ".tmp", ".sync-tmp", ".backup", ".conflict",
    ".tb_sync_index.db", ".livesync_client.db", ".livesync_server.db",
    ".livesync_client.db-journal", ".livesync_server.db-journal",
    ".livesync_client.db-wal", ".livesync_server.db-wal",
)

# Max file size: 50 MB
MAX_FILE_SIZE = 50 * 1024 * 1024


def should_ignore(rel_path: str) -> bool:
    """Return True if this path should be excluded from sync."""
    parts = rel_path.replace("\\", "/").split("/")
    for part in parts:
        if part in IGNORE_DIRS:
            return True
    for suffix in IGNORE_SUFFIXES:
        if rel_path.endswith(suffix):
            return True
    return False


# ── Message ID ──

def _make_msg_id() -> str:
    """Generate a short unique message ID."""
    return hashlib.sha256(f"{time.time()}-{id(object())}".encode()).hexdigest()[:12]


# ── Payload Models ──

class AuthPayload(BaseModel):
    """Client → Server: initial authentication."""
    client_id: str
    device_type: str = "desktop"  # "desktop", "mobile", "termux"
    share_id: str = ""

class AuthSuccessPayload(BaseModel):
    """Server → Client: auth OK + initial state."""
    client_id: str
    minio_credentials: Dict[str, Any]
    checksums: Dict[str, str] = Field(default_factory=dict)

class FileChangedPayload(BaseModel):
    """Bidirectional: a file was created or modified."""
    path: str
    checksum: str
    minio_key: str
    file_type: str = "other"
    source_client: Optional[str] = None  # set by server on broadcast

class FileDeletedPayload(BaseModel):
    """Bidirectional: a file was deleted."""
    path: str
    source_client: Optional[str] = None

class FileRenamedPayload(BaseModel):
    """Bidirectional: a file was renamed/moved."""
    old_path: str
    new_path: str
    checksum: str = ""
    minio_key: str = ""
    source_client: Optional[str] = None

class RequestFullPayload(BaseModel):
    """Client → Server: request full file content."""
    path: str

class RequestSyncPayload(BaseModel):
    """Client → Server: request current sync state."""
    pass

class FullStateReadyPayload(BaseModel):
    """Server → Client: full index DB available in MinIO."""
    minio_key: str
    file_count: int = 0

class ConflictPayload(BaseModel):
    """Server → Client: conflict detected."""
    path: str
    local_checksum: str = ""
    remote_checksum: str = ""
    resolution: str = ""          # "merge_markers", "latest_wins"
    winner: Optional[str] = None  # client_id of winner (latest-wins)
    loser_backup: Optional[str] = None  # backup key/path
    message: str = ""

class AckPayload(BaseModel):
    """Server → Client: change acknowledged."""
    path: str
    checksum: str = ""

class ErrorPayload(BaseModel):
    """Server → Client: error notification."""
    message: str
    path: Optional[str] = None


# ── Envelope ──

class SyncMessage(BaseModel):
    """
    Top-level WebSocket message envelope.

    Wire format (JSON):
      {"type": "file_changed", "payload": {...}, "timestamp": 1713..., "msg_id": "a1b2c3"}
    """
    type: MsgType
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    msg_id: str = Field(default_factory=_make_msg_id)

    def to_json(self) -> str:
        """Serialize to JSON string for WebSocket send."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> SyncMessage:
        """Deserialize from JSON string."""
        return cls.model_validate_json(raw)

    # ── Factory helpers ──

    @classmethod
    def auth(cls, client_id: str, device_type: str, share_id: str) -> SyncMessage:
        return cls(
            type=MsgType.AUTH,
            payload=AuthPayload(
                client_id=client_id,
                device_type=device_type,
                share_id=share_id,
            ).model_dump(),
        )

    @classmethod
    def auth_success(cls, client_id: str, minio_creds: dict, checksums: dict) -> SyncMessage:
        return cls(
            type=MsgType.AUTH_SUCCESS,
            payload=AuthSuccessPayload(
                client_id=client_id,
                minio_credentials=minio_creds,
                checksums=checksums,
            ).model_dump(),
        )

    @classmethod
    def file_changed(
        cls,
        path: str,
        checksum: str,
        minio_key: str,
        file_type: str = "",
        source_client: Optional[str] = None,
    ) -> SyncMessage:
        if not file_type:
            file_type = classify_file(path).value
        return cls(
            type=MsgType.FILE_CHANGED,
            payload=FileChangedPayload(
                path=path,
                checksum=checksum,
                minio_key=minio_key,
                file_type=file_type,
                source_client=source_client,
            ).model_dump(),
        )

    @classmethod
    def file_deleted(cls, path: str, source_client: Optional[str] = None) -> SyncMessage:
        return cls(
            type=MsgType.FILE_DELETED,
            payload=FileDeletedPayload(path=path, source_client=source_client).model_dump(),
        )

    @classmethod
    def file_renamed(
        cls,
        old_path: str,
        new_path: str,
        checksum: str = "",
        minio_key: str = "",
        source_client: Optional[str] = None,
    ) -> SyncMessage:
        return cls(
            type=MsgType.FILE_RENAMED,
            payload=FileRenamedPayload(
                old_path=old_path,
                new_path=new_path,
                checksum=checksum,
                minio_key=minio_key,
                source_client=source_client,
            ).model_dump(),
        )

    @classmethod
    def request_full(cls, path: str) -> SyncMessage:
        return cls(
            type=MsgType.REQUEST_FULL,
            payload=RequestFullPayload(path=path).model_dump(),
        )

    @classmethod
    def request_sync(cls) -> SyncMessage:
        return cls(
            type=MsgType.REQUEST_SYNC,
            payload={},
        )

    @classmethod
    def full_state_ready(cls, minio_key: str, file_count: int) -> SyncMessage:
        return cls(
            type=MsgType.FULL_STATE_READY,
            payload=FullStateReadyPayload(minio_key=minio_key, file_count=file_count).model_dump(),
        )

    @classmethod
    def conflict(
        cls,
        path: str,
        local_checksum: str = "",
        remote_checksum: str = "",
        resolution: str = "",
        winner: Optional[str] = None,
        loser_backup: Optional[str] = None,
        message: str = "",
    ) -> SyncMessage:
        return cls(
            type=MsgType.CONFLICT,
            payload=ConflictPayload(
                path=path,
                local_checksum=local_checksum,
                remote_checksum=remote_checksum,
                resolution=resolution,
                winner=winner,
                loser_backup=loser_backup,
                message=message,
            ).model_dump(),
        )

    @classmethod
    def ack(cls, path: str, checksum: str = "") -> SyncMessage:
        return cls(
            type=MsgType.ACK,
            payload=AckPayload(path=path, checksum=checksum).model_dump(),
        )

    @classmethod
    def ping(cls) -> SyncMessage:
        return cls(type=MsgType.PING)

    @classmethod
    def pong(cls) -> SyncMessage:
        return cls(type=MsgType.PONG)

    @classmethod
    def error(cls, message: str, path: Optional[str] = None) -> SyncMessage:
        return cls(
            type=MsgType.ERROR,
            payload=ErrorPayload(message=message, path=path).model_dump(),
        )
