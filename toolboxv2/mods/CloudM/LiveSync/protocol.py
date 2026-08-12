"""
LiveSync Protocol
=================
Pydantic models for all WebSocket message types.
WebSocket transports ONLY metadata + MinIO keys — NEVER file content.

Message flow:
  Client → Server:  auth, file_changed, file_deleted, file_renamed,
                     request_urls, request_full, request_sync, ping
  Server → Client:  auth_success, urls_granted, file_changed, file_deleted,
                     file_renamed, full_state_ready, conflict, ack, pong, error

Object storage is reached with short-lived presigned URLs handed out by the
server per object. Clients never hold S3 access keys.
"""

from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Any, Dict, List, Optional

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

    # Presigned object-storage URLs
    REQUEST_URLS = "request_urls"
    URLS_GRANTED = "urls_granted"

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
    """
    File type category, reported with every change.

    The category is informational — every type syncs the same way and every
    conflict is resolved the same way (the local version is kept beside the
    remote one, see conflict.make_conflict_name). What the category does drive
    is how a file is presented and logged, and it is the honest answer to
    "what kind of thing is this".

    ``is_text`` is the one distinction that matters mechanically: text files
    can be shown, diffed and merged by hand, binaries cannot.
    """
    TEXT = "text"          # .md .txt .rst .log
    CODE = "code"          # .py .js .rs .go .c …
    DATA = "data"          # .json .yaml .toml .csv .xml …
    IMAGE = "image"        # .png .jpg .gif .webp …
    DOCUMENT = "doc"       # .pdf .docx .xlsx .pptx …
    AUDIO = "audio"        # .mp3 .wav .flac …
    VIDEO = "video"        # .mp4 .mkv .mov …
    ARCHIVE = "archive"    # .zip .tar.gz .7z …
    BINARY = "binary"      # .exe .so .db .bin …
    OTHER = "other"        # unknown extension


# Extension → FileType. Longest match wins, so ".tar.gz" beats ".gz".
_EXT_MAP: Dict[str, FileType] = {}


def _register(file_type: FileType, *extensions: str) -> None:
    for ext in extensions:
        _EXT_MAP[ext] = file_type


_register(
    FileType.TEXT,
    ".md", ".markdown", ".txt", ".text", ".rst", ".log", ".tex", ".adoc",
    ".org", ".rtf", ".srt", ".vtt",
)
_register(
    FileType.CODE,
    ".py", ".pyi", ".ipynb", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
    ".rs", ".go", ".java", ".kt", ".kts", ".scala", ".c", ".h", ".cc",
    ".cpp", ".hpp", ".cs", ".swift", ".rb", ".php", ".pl", ".lua", ".r",
    ".jl", ".dart", ".ex", ".exs", ".erl", ".hs", ".clj", ".vim", ".el",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".html", ".htm", ".css", ".scss", ".sass", ".less", ".svg",
    ".vue", ".svelte", ".astro", ".proto", ".graphql", ".gql",
    ".dockerfile", ".makefile", ".cmake", ".gradle", ".tf", ".tfvars",
)
_register(
    FileType.DATA,
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".properties", ".env", ".csv", ".tsv", ".xml", ".plist",
    ".geojson", ".gpx", ".ics", ".bib",
)
_register(
    FileType.IMAGE,
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
    ".ico", ".heic", ".heif", ".avif", ".psd", ".xcf", ".raw", ".cr2",
    ".nef", ".dng",
)
_register(
    FileType.DOCUMENT,
    ".pdf", ".doc", ".docx", ".dot", ".dotx", ".odt", ".ott",
    ".xls", ".xlsx", ".xlsm", ".ods", ".ppt", ".pptx", ".odp",
    ".pages", ".numbers", ".key", ".epub", ".mobi", ".azw3", ".djvu",
)
_register(
    FileType.AUDIO,
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".oga", ".opus", ".m4a",
    ".wma", ".aiff", ".mid", ".midi",
)
_register(
    FileType.VIDEO,
    ".mp4", ".m4v", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm",
    ".mpeg", ".mpg", ".3gp",
)
_register(
    FileType.ARCHIVE,
    ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz",
    ".txz", ".gz", ".bz2", ".xz", ".zst", ".7z", ".rar", ".iso", ".dmg",
    ".jar", ".war", ".whl", ".deb", ".rpm", ".apk",
)
_register(
    FileType.BINARY,
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib", ".obj",
    ".class", ".pyc", ".pyo", ".wasm", ".db", ".sqlite", ".sqlite3",
    ".mdb", ".pak", ".dat", ".pt", ".pth", ".onnx", ".safetensors",
    ".gguf", ".ggml", ".npy", ".npz", ".parquet", ".pickle", ".pkl",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
)

# Categories whose bytes are not human-readable.
_BINARY_TYPES = frozenset({
    FileType.IMAGE, FileType.DOCUMENT, FileType.AUDIO, FileType.VIDEO,
    FileType.ARCHIVE, FileType.BINARY,
})

# Extensionless files that are text by convention.
_TEXT_BASENAMES = frozenset({
    "makefile", "dockerfile", "license", "licence", "readme", "changelog",
    "authors", "contributing", "notice", "copying", "codeowners",
    "procfile", "gemfile", "rakefile", "vagrantfile", "jenkinsfile",
})


def classify_file(path: str) -> FileType:
    """
    Classify a file by its name.

    Compound extensions win over their tail (``.tar.gz`` is an archive, not
    just ``.gz``), and a handful of extensionless conventions (Makefile,
    Dockerfile, LICENSE) are recognised as text.

    Returns FileType.OTHER for anything unknown — which still syncs, it is
    just handled as an opaque blob.
    """
    name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()

    best: Optional[FileType] = None
    best_len = 0
    for ext, file_type in _EXT_MAP.items():
        if name.endswith(ext) and len(ext) > best_len:
            best, best_len = file_type, len(ext)
    if best is not None:
        return best

    if name in _TEXT_BASENAMES or name.lstrip(".") in _TEXT_BASENAMES:
        return FileType.TEXT
    if name.startswith(".") and "." not in name[1:]:
        # dotfiles like .gitignore, .editorconfig, .env.local
        return FileType.DATA
    return FileType.OTHER


def is_text_type(file_type: FileType) -> bool:
    """True for categories whose content is human-readable."""
    return file_type not in _BINARY_TYPES


def is_binary_file(path: str, sample: Optional[bytes] = None) -> bool:
    """
    Decide whether a file should be treated as binary.

    Extension first, because it is cheap and right nearly always. For unknown
    extensions a content sample decides: a NUL byte in the first few KiB means
    binary, and so does content that is not valid UTF-8. Without a sample an
    unknown extension is assumed binary — the safer guess, since treating a
    binary as text is what corrupts files.
    """
    file_type = classify_file(path)
    if file_type is not FileType.OTHER:
        return file_type in _BINARY_TYPES

    if sample is None:
        return True
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


# ── Ignore Rules ──

# Directories to never sync
IGNORE_DIRS = frozenset({
    ".obsidian", ".git", ".sync-trash", ".sync-backups", "__pycache__",
})

# Marker that identifies a conflict copy: "notes.conflict.<device>.<ts>.md".
# It sits in the middle of the name, so a plain suffix check cannot see it.
CONFLICT_MARKER = ".conflict."

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
    """
    Return True if this path should be excluded from sync.

    Conflict copies stay local on purpose: they are one node's answer to a
    collision, and pushing them would hand every other node a file it never
    made — and start the next collision.
    """
    normalized = rel_path.replace("\\", "/")
    parts = normalized.split("/")
    for part in parts:
        if part in IGNORE_DIRS:
            return True
    for suffix in IGNORE_SUFFIXES:
        if normalized.endswith(suffix):
            return True
    if CONFLICT_MARKER in parts[-1]:
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
    token: str = ""

class AuthSuccessPayload(BaseModel):
    """
    Server → Client: auth OK + initial state.

    Carries no credentials. Object access happens exclusively through
    presigned URLs requested via ``request_urls``.
    """
    client_id: str
    checksums: Dict[str, str] = Field(default_factory=dict)
    url_ttl: int = 900  # seconds a granted URL stays valid


class RequestUrlsPayload(BaseModel):
    """
    Client → Server: ask for presigned URLs.

    op "get" → one download URL per path.
    op "put" → one upload URL for the encrypted object and one for its
    metadata sidecar, per path.
    """
    req_id: str
    op: str                      # "get" | "put"
    paths: List[str] = Field(default_factory=list)


class UrlsGrantedPayload(BaseModel):
    """
    Server → Client: presigned URLs, keyed by relative path.

    urls[path] = {"file": <url>, "meta": <url>}   (meta only for op "put")
    missing    = paths with no object (op "get") or rejected paths
    """
    req_id: str
    op: str
    urls: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    missing: List[str] = Field(default_factory=list)
    expires_in: int = 900
    error: str = ""

class FileChangedPayload(BaseModel):
    """
    Bidirectional: a file was created or modified.

    base_checksum is the version the sender started from — the checksum it had
    in sync with the server before this edit. It is what makes real conflict
    detection possible: the server compares the BASE against what it holds, not
    the new checksum. Comparing the new one flags every ordinary update as a
    conflict, because a new revision differs from the old one by definition.

    Empty base_checksum means "I believe this file is new here".
    Never set on a server broadcast; it is a client-to-server field.
    """
    path: str
    checksum: str
    minio_key: str
    file_type: str = "other"
    base_checksum: str = ""
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
    """Server → Client: full index DB available, with its download URL."""
    minio_key: str
    file_count: int = 0
    url: str = ""

# NOTE on ConflictPayload semantics (both sides rely on this):
#   local_checksum  — the version the server held and that just lost
#   remote_checksum — the version that won and is now the object in storage
# A conflict message is only ever sent for a genuine collision: two clients
# edited from the same base. The winning file_changed follows immediately, so
# a client marks the path and lets that download do the work.
class ConflictPayload(BaseModel):
    """Server → Client: conflict detected."""
    path: str
    local_checksum: str = ""
    remote_checksum: str = ""
    resolution: str = ""          # "keep_both"
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
    def auth(cls, client_id: str, device_type: str, share_id: str, token: str = "") -> SyncMessage:
        return cls(
            type=MsgType.AUTH,
            payload=AuthPayload(
                client_id=client_id,
                device_type=device_type,
                share_id=share_id,
                token=token,
            ).model_dump(),
        )

    @classmethod
    def auth_success(
        cls, client_id: str, checksums: dict, url_ttl: int = 900
    ) -> SyncMessage:
        return cls(
            type=MsgType.AUTH_SUCCESS,
            payload=AuthSuccessPayload(
                client_id=client_id,
                checksums=checksums,
                url_ttl=url_ttl,
            ).model_dump(),
        )

    @classmethod
    def request_urls(cls, req_id: str, op: str, paths: List[str]) -> SyncMessage:
        return cls(
            type=MsgType.REQUEST_URLS,
            payload=RequestUrlsPayload(req_id=req_id, op=op, paths=paths).model_dump(),
        )

    @classmethod
    def urls_granted(
        cls,
        req_id: str,
        op: str,
        urls: Dict[str, Dict[str, str]],
        missing: Optional[List[str]] = None,
        expires_in: int = 900,
        error: str = "",
    ) -> SyncMessage:
        return cls(
            type=MsgType.URLS_GRANTED,
            payload=UrlsGrantedPayload(
                req_id=req_id,
                op=op,
                urls=urls,
                missing=missing or [],
                expires_in=expires_in,
                error=error,
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
        base_checksum: str = "",
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
                base_checksum=base_checksum,
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
    def full_state_ready(
        cls, minio_key: str, file_count: int, url: str = ""
    ) -> SyncMessage:
        return cls(
            type=MsgType.FULL_STATE_READY,
            payload=FullStateReadyPayload(
                minio_key=minio_key, file_count=file_count, url=url,
            ).model_dump(),
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
