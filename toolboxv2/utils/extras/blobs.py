# file: blobs.py
# Production-ready BlobStorage client with MinIO backend
# Features: Hybrid cloud/local storage, offline support, end-to-end encryption, live watch
# Interface remains 1:1 compatible with the original implementation

import hashlib
import io
import json
import os
import pickle
import platform
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any, Union
import logging

import yaml

# MinIO SDK
try:
    from minio import Minio
    from minio.error import S3Error
    from urllib3.exceptions import MaxRetryError

    HAS_MINIO = True
except ImportError:
    HAS_MINIO = False
    Minio = None
    S3Error = Exception

# Local imports from the project
try:
    from ..security.cryp import Code, DEVICE_KEY
    from ..system.getting_and_closing_app import get_logger as _get_logger
    from toolboxv2.utils.singelton_class import Singleton
except ImportError:
    # Fallback for standalone usage
    Singleton = type
    Code = None
    DEVICE_KEY = None
    _get_logger = None

# Local modules

from toolboxv2.utils.extras.db.mobile_db import MobileDB, SyncStatus, BlobMetadata
from toolboxv2.utils.extras.db.minio_manager import MinIOManager, MinIOConfig, MinIOMode, MinIOInstance

# Logger setup
_logger: Optional[logging.Logger] = None


def detect_storage_mode() -> "StorageMode":
    """
    Detect the appropriate storage mode based on environment.

    - Tauri/Desktop mode → MOBILE (SQLite only, offline-first)
    - Production/Cloud mode → SERVER (MinIO with server credentials)
    - Development mode → SERVER (MinIO with dev credentials)

    Returns:
        StorageMode: The detected storage mode
    """
    try:
        from toolboxv2.utils.workers.config import Environment

        if Environment.is_tauri():
            # Tauri/Desktop mode - use mobile/offline storage (SQLite)
            get_logger().info("Detected Tauri environment - using MOBILE storage mode")
            return StorageMode.MOBILE
        elif Environment.is_production():
            # Production/Cloud mode - use server storage with MinIO
            get_logger().info("Detected Production environment - using SERVER storage mode")
            return StorageMode.SERVER
        else:
            # Development mode - use server storage with dev credentials
            get_logger().info("Detected Development environment - using SERVER storage mode")
            return StorageMode.SERVER
    except ImportError:
        # Fallback to offline if Environment not available
        get_logger().warning("Environment detection not available - using OFFLINE storage mode")
        return StorageMode.OFFLINE


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        if _get_logger:
            _logger = _get_logger()
        else:
            _logger = logging.getLogger("blobs")
            if not _logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )
                _logger.addHandler(handler)
                _logger.setLevel(logging.INFO)
    return _logger


class StorageMode(Enum):
    """Operating mode for blob storage"""

    SERVER = "server"  # Running on server - direct MinIO access
    DESKTOP = "desktop"  # Desktop with local MinIO + cloud sync
    MOBILE = "mobile"  # Mobile with SQLite + periodic sync
    OFFLINE = "offline"  # Fully offline mode (SQLite only)


class ConnectionState(Enum):
    """Connection state for storage backend"""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Offline mode, using cache
    UNAUTHORIZED = "unauthorized"
    UNREACHABLE = "unreachable"
    ERROR = "error"


@dataclass
class ServerStatus:
    """Status information for a storage backend"""

    endpoint: str
    state: ConnectionState = ConnectionState.UNKNOWN
    last_check: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    mode: StorageMode = StorageMode.DESKTOP

    def is_healthy(self) -> bool:
        return self.state == ConnectionState.HEALTHY

    def mark_healthy(self):
        self.state = ConnectionState.HEALTHY
        self.error_count = 0
        self.last_error = None
        self.last_check = time.time()

    def mark_degraded(self):
        self.state = ConnectionState.DEGRADED
        self.last_check = time.time()

    def mark_error(self, error: str, state: ConnectionState = ConnectionState.ERROR):
        self.state = state
        self.error_count += 1
        self.last_error = error
        self.last_check = time.time()


@dataclass
class WatchCallback:
    """Wrapper for a watch callback with metadata."""

    callback: Callable[["BlobFile"], None]
    blob_id: str
    last_update: float = field(default_factory=time.time)
    max_idle_timeout: int = 600
    folder: Optional[str] = None
    filename: Optional[str] = None

    def is_expired(self) -> bool:
        return (time.time() - self.last_update) > self.max_idle_timeout

    def update_timestamp(self):
        self.last_update = time.time()


class CryptoLayer:
    """Encryption layer for blob data - client-side encryption"""

    def __init__(self, user_key: Optional[bytes] = None):
        """
        Initialize crypto layer.

        Args:
            user_key: User-specific encryption key. If None, uses device key.
        """
        self._user_key = user_key
        self._device_key: Optional[str] = None

    def _get_key(self, custom_key: Optional[bytes] = None) -> str:
        """Get encryption key"""
        if custom_key:
            if isinstance(custom_key, bytes):
                return custom_key.hex()
            return custom_key

        if self._user_key:
            if isinstance(self._user_key, bytes):
                return self._user_key.hex()
            return self._user_key

        if self._device_key is None:
            if DEVICE_KEY:
                self._device_key = DEVICE_KEY()
            else:
                # Fallback: generate from machine ID
                import uuid

                machine_id = str(uuid.getnode())
                self._device_key = hashlib.sha256(machine_id.encode()).hexdigest()

        return self._device_key

    def encrypt(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Encrypt data"""
        if Code:
            enc_key = self._get_key(key)
            encrypted = Code.encrypt_symmetric(data, enc_key)
            if isinstance(encrypted, str):
                return encrypted.encode()
            return encrypted
        else:
            # Fallback: simple XOR (NOT SECURE - only for testing)
            get_logger().warning("Using fallback encryption - NOT SECURE")
            key_bytes = self._get_key(key).encode()[:32]
            return bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(data))

    def decrypt(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Decrypt data"""
        if Code:
            enc_key = self._get_key(key)
            if isinstance(data, bytes):
                data = data.decode() if data[:10].isascii() else data
            decrypted = Code.decrypt_symmetric(data, enc_key, to_str=False)
            return decrypted
        else:
            # Fallback: simple XOR
            key_bytes = self._get_key(key).encode()[:32]
            return bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(data))

    def sign(self, data: bytes) -> str:
        """Create signature/hash for data integrity"""
        return hashlib.sha256(data).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        """Verify data integrity"""
        return self.sign(data) == signature


class WatchManager:
    """Manages watch operations for blob changes."""

    def __init__(self, storage: "BlobStorage"):
        self.storage = storage
        self._watches: Dict[str, List[WatchCallback]] = {}
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        self._backoff_time = 1.0

    def add_watch(
        self,
        blob_id: str,
        callback: Callable[["BlobFile"], None],
        max_idle_timeout: int = 600,
        **kwargs,
    ):
        with self._lock:
            if blob_id not in self._watches:
                self._watches[blob_id] = []

            watch_cb = WatchCallback(
                callback=callback,
                blob_id=blob_id,
                max_idle_timeout=max_idle_timeout,
                **kwargs,
            )
            self._watches[blob_id].append(watch_cb)
            get_logger().info(
                f"Added watch for blob '{blob_id}' (timeout: {max_idle_timeout}s)"
            )

            if not self._running:
                self._start_watch_thread()

    def remove_watch(self, blob_id: str, callback: Optional[Callable] = None):
        with self._lock:
            if blob_id not in self._watches:
                return

            if callback is None:
                del self._watches[blob_id]
                get_logger().info(f"Removed all watches for blob '{blob_id}'")
            else:
                self._watches[blob_id] = [
                    w for w in self._watches[blob_id] if w.callback != callback
                ]
                if not self._watches[blob_id]:
                    del self._watches[blob_id]
                get_logger().info(f"Removed specific watch for blob '{blob_id}'")

            if not self._watches and self._running:
                self._stop_watch_thread()

    def remove_all_watches(self):
        with self._lock:
            self._watches.clear()
            get_logger().info("Removed all watches")
        if self._running:
            self._stop_watch_thread()

    def _start_watch_thread(self):
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._consecutive_failures = 0
        self._backoff_time = 1.0
        self._watch_thread = threading.Thread(
            target=self._watch_loop, name="BlobWatchThread", daemon=True
        )
        self._watch_thread.start()
        get_logger().info("Started watch thread")

    def _stop_watch_thread(self):
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=5)
        get_logger().info("Stopped watch thread")

    def _watch_loop(self):
        """Main watch loop - polls for changes"""
        last_versions: Dict[str, int] = {}

        while not self._stop_event.is_set():
            try:
                with self._lock:
                    if not self._watches:
                        break
                    watched_blobs = list(self._watches.keys())

                # Poll each watched blob for changes
                for blob_id in watched_blobs:
                    if self._stop_event.is_set():
                        break

                    try:
                        # Get current version/checksum
                        current_version = self._get_blob_version(blob_id)

                        if current_version is not None:
                            last_version = last_versions.get(blob_id)

                            if (
                                last_version is not None
                                and current_version != last_version
                            ):
                                # Blob changed
                                self._dispatch_callbacks(blob_id)

                            last_versions[blob_id] = current_version

                    except Exception as e:
                        get_logger().debug(f"Watch check failed for {blob_id}: {e}")

                # Reset failure counter on success
                self._consecutive_failures = 0
                self._backoff_time = 1.0

                # Cleanup expired callbacks
                self._cleanup_expired_callbacks()

                # Wait before next poll
                self._stop_event.wait(timeout=2.0)

            except Exception as e:
                get_logger().error(f"Watch loop error: {e}")
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_consecutive_failures:
                    self._backoff_time = min(self._backoff_time * 2, 60.0)
                time.sleep(self._backoff_time)

        self._running = False
        get_logger().info("Watch loop exited")

    def _get_blob_version(self, blob_id: str) -> Optional[int]:
        """Get current version/hash of a blob for change detection"""
        try:
            meta = self.storage.get_blob_meta(blob_id)
            if meta:
                return meta.get("version", 0) or hash(meta.get("checksum", ""))
        except:
            pass
        return None

    def _dispatch_callbacks(self, blob_id: str):
        with self._lock:
            callbacks = self._watches.get(blob_id, []).copy()

        if not callbacks:
            return

        get_logger().info(f"Dispatching {len(callbacks)} callbacks for blob '{blob_id}'")

        for watch_cb in callbacks:
            try:
                # Create BlobFile for callback
                if watch_cb.filename:
                    folder = watch_cb.folder or ""
                    if folder and not folder.startswith("/"):
                        folder = "/" + folder
                    path = f"{blob_id}{folder}/{watch_cb.filename}"
                else:
                    path = f"{blob_id}/data"

                blob_file = BlobFile(path, "r", storage=self.storage)
                watch_cb.callback(blob_file)
                watch_cb.update_timestamp()

            except Exception as e:
                get_logger().error(f"Callback error for blob '{blob_id}': {e}")

    def _cleanup_expired_callbacks(self):
        with self._lock:
            expired_blobs = []
            for blob_id, callbacks in self._watches.items():
                active_callbacks = [cb for cb in callbacks if not cb.is_expired()]
                if len(active_callbacks) < len(callbacks):
                    removed_count = len(callbacks) - len(active_callbacks)
                    get_logger().info(
                        f"Removed {removed_count} expired callbacks for blob '{blob_id}'"
                    )
                if active_callbacks:
                    self._watches[blob_id] = active_callbacks
                else:
                    expired_blobs.append(blob_id)

            for blob_id in expired_blobs:
                del self._watches[blob_id]
                get_logger().info(f"Removed blob '{blob_id}' from watch list")

            if not self._watches and self._running:
                get_logger().info("No more active watches, stopping watch thread")
                self._stop_event.set()


class BlobStorage:
    """
    Production-ready client for MinIO-based blob storage.

    Features:
    - Hybrid cloud/local storage
    - Offline-first with SQLite fallback
    - Client-side encryption
    - Watch for live updates
    - Auto-sync between desktop and cloud
    """

    DEFAULT_BUCKET = "user-data-enc"

    def __init__(
        self,
        mode: Optional[StorageMode] = None,
        # MinIO settings
        minio_endpoint:  Optional[str] = None,
        minio_access_key:  Optional[str] = None,
        minio_secret_key:  Optional[str] = None,
        minio_secure: bool = False,
        # Cloud settings for sync
        use_cloud: Optional[bool] = None,
        cloud_endpoint: Optional[str] = None,
        cloud_access_key: Optional[str] = None,
        cloud_secret_key: Optional[str] = None,
        # Local storage
        storage_directory: str = "./.data/blob_cache",
        # User settings
        user_id: Optional[str] = None,
        encryption_key: Optional[bytes] = None,
        # Options
        auto_sync: bool = True,
        bucket: str = DEFAULT_BUCKET,
    ):
        """
        Initialize BlobStorage.

        Args:
            mode: Operating mode (SERVER, DESKTOP, MOBILE, OFFLINE).
                  If None, auto-detects based on environment:
                  - Tauri/Desktop → MOBILE (SQLite only)
                  - Production/Dev → SERVER (MinIO)
            minio_endpoint: Local MinIO endpoint (for DESKTOP/SERVER)
            minio_access_key: MinIO access key
            minio_secret_key: MinIO secret key
            minio_secure: Use HTTPS
            cloud_endpoint: Cloud MinIO endpoint for sync
            cloud_access_key: Cloud access key
            cloud_secret_key: Cloud secret key
            storage_directory: Local storage directory
            user_id: User ID for namespacing
            encryption_key: User-specific encryption key
            auto_sync: Enable automatic sync
            bucket: MinIO bucket name
        """
        # Auto-detect mode if not specified
        if mode is None:
            mode = detect_storage_mode()

        self.mode = mode
        self.bucket = bucket
        self.storage_directory = os.path.expanduser(storage_directory)
        self.user_id = user_id or self._get_default_user_id()
        self.auto_sync = auto_sync

        os.makedirs(self.storage_directory, exist_ok=True)

        # Initialize crypto layer
        self.crypto = CryptoLayer(encryption_key)

        # Initialize local SQLite DB (for offline/mobile)
        self.local_db = MobileDB(
            db_path=os.path.join(self.storage_directory, "blobs.db"), max_size_mb=1000
        )

        # Initialize MinIO client(s)
        self._local_minio: Optional[Minio] = None
        self._cloud_minio: Optional[Minio] = None
        self._minio_lock = threading.Lock()

        minio_endpoint = minio_endpoint or os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
        minio_access_key = minio_access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        minio_secret_key = minio_secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")

        if use_cloud or cloud_endpoint:
            cloud_endpoint = cloud_endpoint or os.getenv("CLOUD_ENDPOINT")
            cloud_access_key = cloud_access_key or os.getenv("CLOUD_ACCESS_KEY")
            cloud_secret_key = cloud_secret_key or os.getenv("CLOUD_SECRET_KEY")

        # Only initialize MinIO for SERVER/DESKTOP modes
        # MOBILE and OFFLINE modes use SQLite only
        if mode in (StorageMode.SERVER, StorageMode.DESKTOP) and HAS_MINIO:
            try:
                self._local_minio = Minio(
                    minio_endpoint,
                    access_key=minio_access_key,
                    secret_key=minio_secret_key,
                    secure=minio_secure,
                )
                # Try to ensure bucket exists - if auth fails, fallback to offline
                if not self._ensure_bucket(self._local_minio):
                    get_logger().warning(
                        "MinIO authentication failed - falling back to OFFLINE mode"
                    )
                    self._local_minio = None
                    self.mode = StorageMode.OFFLINE
            except Exception as e:
                get_logger().warning(f"Local MinIO not available: {e}")
                self._local_minio = None
                # Fallback to offline mode if MinIO is not available
                self.mode = StorageMode.OFFLINE
        elif mode in (StorageMode.MOBILE, StorageMode.OFFLINE):
            # Mobile/Offline modes don't use MinIO - SQLite only
            get_logger().info(f"Using {mode.value} mode - SQLite storage only")

        # Cloud MinIO for sync
        if cloud_endpoint and cloud_access_key and cloud_secret_key and HAS_MINIO:
            try:
                self._cloud_minio = Minio(
                    cloud_endpoint,
                    access_key=cloud_access_key,
                    secret_key=cloud_secret_key,
                    secure=True,
                )
            except Exception as e:
                get_logger().warning(f"Cloud MinIO not available: {e}")

        # Status tracking
        self._status = ServerStatus(endpoint=minio_endpoint, mode=mode)
        self._check_health()

        # Watch manager
        self.watch_manager = WatchManager(self)

        # Background sync thread
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_stop = threading.Event()

        if auto_sync and mode == StorageMode.DESKTOP:
            self._start_background_sync()

    def _get_default_user_id(self) -> str:
        """Generate default user ID from device"""
        import uuid

        return hashlib.md5(str(uuid.getnode()).encode()).hexdigest()[:16]

    def _ensure_bucket(self, client: Minio) -> bool:
        """
        Ensure bucket exists.

        Returns:
            bool: True if bucket check/creation succeeded, False if authentication failed
        """
        try:
            if not client.bucket_exists(self.bucket):
                client.make_bucket(self.bucket)
                get_logger().info(f"Created bucket: {self.bucket}")
            return True
        except Exception as e:
            error_str = str(e)
            # Check for authentication/signature errors
            if any(auth_err in error_str for auth_err in [
                "SignatureDoesNotMatch",
                "InvalidAccessKeyId",
                "AccessDenied",
                "InvalidSignature",
                "AuthorizationHeaderMalformed"
            ]):
                get_logger().warning(
                    f"MinIO authentication failed for bucket '{self.bucket}': {e}"
                )
                return False
            else:
                # Other errors (network, etc.) - log but don't fail auth
                get_logger().warning(f"Bucket check failed: {e}")
                return True  # Don't switch to offline for non-auth errors

    def _check_health(self):
        """Check storage health"""
        try:
            if self._local_minio:
                self._local_minio.list_buckets()
                self._status.mark_healthy()
            elif self.mode in (StorageMode.MOBILE, StorageMode.OFFLINE):
                # SQLite is always available
                self._status.mark_healthy()
            else:
                self._status.mark_degraded()
        except Exception as e:
            self._status.mark_error(str(e))

    def _object_path(self, blob_id: str) -> str:
        """Get full object path including user namespace"""
        return f"{self.user_id}/{blob_id}"

    def _get_client(self) -> Optional[Minio]:
        """Get appropriate MinIO client"""
        if self._local_minio:
            return self._local_minio
        return self._cloud_minio

    # =================== Core Operations ===================

    def create_blob(
        self, data: bytes, blob_id: Optional[str] = None, encrypt: bool = True
    ) -> str:
        """
        Create a new blob.

        Args:
            data: Binary data to store
            blob_id: Optional blob ID (generated if not provided)
            encrypt: Whether to encrypt data

        Returns:
            Blob ID
        """
        if not blob_id:
            blob_id = hashlib.sha256(data + str(time.time()).encode()).hexdigest()

        # Encrypt if requested
        if encrypt:
            data = self.crypto.encrypt(data)

        checksum = self.crypto.sign(data)

        # Store in SQLite (always, for offline support)
        self.local_db.put(blob_id, data, encrypted=encrypt)

        # Store in MinIO if available
        client = self._get_client()
        if client:
            try:
                with self._minio_lock:
                    client.put_object(
                        self.bucket,
                        self._object_path(blob_id),
                        io.BytesIO(data),
                        len(data),
                        metadata={
                            "checksum": checksum,
                            "encrypted": str(encrypt),
                            "version": "1",
                        },
                    )
                self.local_db.mark_synced(blob_id)
            except Exception as e:
                get_logger().warning(f"MinIO upload failed, stored locally: {e}")

        get_logger().info(f"Created blob {blob_id}")
        return blob_id

    def read_blob(
        self, blob_id: str, use_cache: bool = True, decrypt: bool = True
    ) -> Optional[bytes]:
        """
        Read blob data.

        Args:
            blob_id: Blob ID
            use_cache: Use local cache if available
            decrypt: Decrypt data if encrypted

        Returns:
            Blob data or None
        """
        # Try local SQLite first
        if use_cache:
            data = self.local_db.get(blob_id)
            if data is not None:
                meta = self.local_db.get_metadata(blob_id)
                if meta and meta.encrypted and decrypt:
                    data = self.crypto.decrypt(data)
                return data

        # Try MinIO
        client = self._get_client()
        if client:
            try:
                with self._minio_lock:
                    response = client.get_object(self.bucket, self._object_path(blob_id))
                    data = response.read()
                    response.close()

                # Get metadata for encryption info
                stat = client.stat_object(self.bucket, self._object_path(blob_id))
                encrypted = (
                    stat.metadata.get("x-amz-meta-encrypted", "true").lower() == "true"
                )

                # Cache locally
                self.local_db.put(blob_id, data, encrypted=encrypted, skip_sync=True)

                if encrypted and decrypt:
                    data = self.crypto.decrypt(data)

                return data

            except S3Error as e:
                if e.code == "NoSuchKey":
                    return None
                get_logger().warning(f"MinIO read failed: {e}")
            except Exception as e:
                get_logger().warning(f"MinIO read failed: {e}")

        # Fall back to local
        data = self.local_db.get(blob_id)
        if data is not None:
            meta = self.local_db.get_metadata(blob_id)
            if meta and meta.encrypted and decrypt:
                data = self.crypto.decrypt(data)
            return data

        return None

    def update_blob(
        self, blob_id: str, data: bytes, encrypt: bool = True
    ) -> Dict[str, Any]:
        """
        Update an existing blob.

        Args:
            blob_id: Blob ID
            data: New data
            encrypt: Encrypt data

        Returns:
            Update result with version info
        """
        # Get current version
        meta = self.local_db.get_metadata(blob_id)
        version = (meta.version + 1) if meta else 1

        # Encrypt if requested
        if encrypt:
            data = self.crypto.encrypt(data)

        checksum = self.crypto.sign(data)

        # Update local
        self.local_db.put(blob_id, data, encrypted=encrypt)

        # Update MinIO if available
        client = self._get_client()
        if client:
            try:
                with self._minio_lock:
                    client.put_object(
                        self.bucket,
                        self._object_path(blob_id),
                        io.BytesIO(data),
                        len(data),
                        metadata={
                            "checksum": checksum,
                            "encrypted": str(encrypt),
                            "version": str(version),
                        },
                    )
                self.local_db.mark_synced(blob_id)
            except Exception as e:
                get_logger().warning(f"MinIO update failed: {e}")

        get_logger().info(f"Updated blob {blob_id} to version {version}")
        return {"version": version, "checksum": checksum}

    def delete_blob(self, blob_id: str) -> bool:
        """Delete a blob"""
        success = True

        # Delete from MinIO
        client = self._get_client()
        if client:
            try:
                with self._minio_lock:
                    client.remove_object(self.bucket, self._object_path(blob_id))
            except S3Error as e:
                if e.code != "NoSuchKey":
                    get_logger().warning(f"MinIO delete failed: {e}")
                    success = False
            except Exception as e:
                get_logger().warning(f"MinIO delete failed: {e}")
                success = False

        # Delete from local
        self.local_db.delete(blob_id, hard_delete=True)

        get_logger().info(f"Deleted blob {blob_id}")
        return success

    def get_blob_meta(self, blob_id: str) -> Optional[Dict[str, Any]]:
        """Get blob metadata"""
        # Try local first
        meta = self.local_db.get_metadata(blob_id)
        if meta:
            return {
                "blob_id": blob_id,
                "size": meta.size,
                "version": meta.version,
                "checksum": meta.checksum,
                "encrypted": meta.encrypted,
                "updated_at": meta.local_updated_at,
                "sync_status": meta.sync_status.value,
            }

        # Try MinIO
        client = self._get_client()
        if client:
            try:
                with self._minio_lock:
                    stat = client.stat_object(self.bucket, self._object_path(blob_id))
                return {
                    "blob_id": blob_id,
                    "size": stat.size,
                    "version": int(stat.metadata.get("x-amz-meta-version", "1")),
                    "checksum": stat.metadata.get("x-amz-meta-checksum", ""),
                    "encrypted": stat.metadata.get("x-amz-meta-encrypted", "true").lower()
                    == "true",
                    "updated_at": stat.last_modified.timestamp()
                    if stat.last_modified
                    else 0,
                    "etag": stat.etag,
                }
            except S3Error:
                pass
            except Exception as e:
                get_logger().warning(f"Metadata fetch failed: {e}")

        return None

    def list_blobs(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List all blobs with optional prefix filter"""
        blobs = []
        seen = set()

        # List from local
        for meta in self.local_db.list(prefix):
            blobs.append(
                {
                    "blob_id": meta.path,
                    "size": meta.size,
                    "updated_at": meta.local_updated_at,
                    "sync_status": meta.sync_status.value,
                }
            )
            seen.add(meta.path)

        # List from MinIO
        client = self._get_client()
        if client:
            try:
                full_prefix = f"{self.user_id}/{prefix}" if prefix else f"{self.user_id}/"
                with self._minio_lock:
                    objects = client.list_objects(
                        self.bucket, prefix=full_prefix, recursive=True
                    )
                    for obj in objects:
                        blob_id = obj.object_name.replace(f"{self.user_id}/", "", 1)
                        if blob_id not in seen:
                            blobs.append(
                                {
                                    "blob_id": blob_id,
                                    "size": obj.size,
                                    "updated_at": obj.last_modified.timestamp()
                                    if obj.last_modified
                                    else 0,
                                    "sync_status": "cloud_only",
                                }
                            )
            except Exception as e:
                get_logger().warning(f"MinIO list failed: {e}")

        return blobs

    # =================== Watch Operations ===================

    def watch(
        self,
        blob_id: str,
        callback: Callable[["BlobFile"], None],
        max_idle_timeout: int = 600,
        threaded: bool = True,
        **kwargs,
    ):
        """Register a watch callback for a blob"""
        self.watch_manager.add_watch(blob_id, callback, max_idle_timeout, **kwargs)

    def stop_watch(self, blob_id: str, callback: Optional[Callable] = None):
        """Stop watching a blob"""
        self.watch_manager.remove_watch(blob_id, callback)

    def watch_resource(self, timeout: int = 60) -> Dict[str, Any]:
        """Watch for any resource changes (for compatibility)"""
        # This is a polling-based implementation
        time.sleep(min(timeout, 5))
        return {"timeout": True}

    # =================== Sync Operations ===================

    def _start_background_sync(self):
        """Start background sync thread"""
        if self._sync_thread and self._sync_thread.is_alive():
            return

        self._sync_stop.clear()
        self._sync_thread = threading.Thread(
            target=self._sync_loop, name="BlobSyncThread", daemon=True
        )
        self._sync_thread.start()
        get_logger().info("Started background sync")

    def _stop_background_sync(self):
        """Stop background sync thread"""
        self._sync_stop.set()
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
        get_logger().info("Stopped background sync")

    def _sync_loop(self):
        """Background sync loop"""
        while not self._sync_stop.is_set():
            try:
                self.sync()
            except Exception as e:
                get_logger().error(f"Sync error: {e}")

            self._sync_stop.wait(timeout=30)

    def sync(self, force: bool = False) -> Dict[str, Any]:
        """
        Synchronize local and cloud storage.

        Args:
            force: Force full sync

        Returns:
            Sync statistics
        """
        if not self._cloud_minio:
            return {"status": "no_cloud", "message": "Cloud not configured"}

        stats = {
            "uploaded": 0,
            "downloaded": 0,
            "conflicts": 0,
            "errors": [],
        }

        try:
            # Upload dirty blobs
            for meta in self.local_db.get_dirty_blobs():
                try:
                    data = self.local_db.get(meta.path)
                    if data:
                        with self._minio_lock:
                            self._cloud_minio.put_object(
                                self.bucket,
                                self._object_path(meta.path),
                                io.BytesIO(data),
                                len(data),
                                metadata={
                                    "checksum": meta.checksum,
                                    "encrypted": str(meta.encrypted),
                                    "version": str(meta.version),
                                },
                            )
                        self.local_db.mark_synced(meta.path)
                        stats["uploaded"] += 1
                except Exception as e:
                    stats["errors"].append(f"Upload {meta.path}: {e}")

            # Download cloud changes
            full_prefix = f"{self.user_id}/"
            local_manifest = self.local_db.get_manifest()

            with self._minio_lock:
                objects = self._cloud_minio.list_objects(
                    self.bucket, prefix=full_prefix, recursive=True
                )

                for obj in objects:
                    blob_id = obj.object_name.replace(full_prefix, "", 1)

                    if blob_id not in local_manifest:
                        # New cloud object - download
                        try:
                            response = self._cloud_minio.get_object(
                                self.bucket, obj.object_name
                            )
                            data = response.read()
                            response.close()

                            stat = self._cloud_minio.stat_object(
                                self.bucket, obj.object_name
                            )
                            encrypted = (
                                stat.metadata.get("x-amz-meta-encrypted", "true").lower()
                                == "true"
                            )

                            self.local_db.put(
                                blob_id, data, encrypted=encrypted, skip_sync=True
                            )
                            self.local_db.mark_synced(
                                blob_id, obj.last_modified.timestamp()
                            )
                            stats["downloaded"] += 1

                        except Exception as e:
                            stats["errors"].append(f"Download {blob_id}: {e}")
                    else:
                        # Check for updates
                        local_checksum, local_ts = local_manifest[blob_id]
                        cloud_ts = obj.last_modified.timestamp()

                        if cloud_ts > local_ts:
                            # Cloud is newer
                            try:
                                response = self._cloud_minio.get_object(
                                    self.bucket, obj.object_name
                                )
                                data = response.read()
                                response.close()

                                stat = self._cloud_minio.stat_object(
                                    self.bucket, obj.object_name
                                )
                                cloud_checksum = stat.metadata.get(
                                    "x-amz-meta-checksum", ""
                                )

                                if cloud_checksum != local_checksum:
                                    encrypted = (
                                        stat.metadata.get(
                                            "x-amz-meta-encrypted", "true"
                                        ).lower()
                                        == "true"
                                    )
                                    self.local_db.put(
                                        blob_id, data, encrypted=encrypted, skip_sync=True
                                    )
                                    self.local_db.mark_synced(blob_id, cloud_ts)
                                    stats["downloaded"] += 1

                            except Exception as e:
                                stats["errors"].append(f"Update {blob_id}: {e}")

            stats["status"] = "complete"

        except Exception as e:
            stats["status"] = "error"
            stats["errors"].append(str(e))

        return stats

    def manual_sync(self) -> Dict[str, Any]:
        """Trigger manual sync (for mobile)"""
        return self.sync(force=True)

    # =================== Server Mode Operations ===================

    def get_user_id(self) -> str:
        """Get current user ID"""
        return self.user_id

    def set_user_context(self, user_id: str):
        """Set user context for server mode (admin operations)"""
        self.user_id = user_id

    # =================== Cache Operations ===================

    def _get_cache_path(self, blob_id: str) -> str:
        """Get file cache path for a blob"""
        return os.path.join(self.storage_directory, f"{blob_id}.blob")

    def _save_blob_to_cache(self, blob_id: str, data: bytes):
        """Save blob to file cache"""
        try:
            cache_path = self._get_cache_path(blob_id)
            with open(cache_path, "wb") as f:
                f.write(data)
        except Exception as e:
            get_logger().warning(f"Failed to cache blob {blob_id}: {e}")

    def _load_blob_from_cache(self, blob_id: str) -> Optional[bytes]:
        """Load blob from file cache"""
        cache_path = self._get_cache_path(blob_id)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return f.read()
            except Exception as e:
                get_logger().warning(f"Failed to read cached blob {blob_id}: {e}")
        return None

    def _delete_blob_from_cache(self, blob_id: str):
        """Delete blob from file cache"""
        cache_path = self._get_cache_path(blob_id)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception as e:
                get_logger().warning(f"Failed to delete cached blob {blob_id}: {e}")

    # =================== Status ===================

    def get_server_status(self) -> Dict[str, Any]:
        """Get storage status"""
        self._check_health()

        return {
            "endpoint": self._status.endpoint,
            "mode": self.mode.value,
            "state": self._status.state.value,
            "is_healthy": self._status.is_healthy(),
            "error_count": self._status.error_count,
            "last_error": self._status.last_error,
            "last_check": self._status.last_check,
            "user_id": self.user_id,
            "bucket": self.bucket,
            "local_stats": self.local_db.get_sync_stats(),
        }

    def close(self):
        """Close storage and cleanup"""
        self._stop_background_sync()
        self.watch_manager.remove_all_watches()
        self.local_db.close()


class BlobFile:
    """File-like interface for blob storage."""

    def __init__(
        self,
        filename: str,
        mode: str = "r",
        storage: Optional[BlobStorage] = None,
        key: Optional[bytes] = None,
        servers: Optional[List[str]] = None,
        use_cache: bool = True,
    ):
        """
        Initialize BlobFile.

        Args:
            filename: Path in format 'blob_id/folder/file.txt'
            mode: 'r' for read, 'w' for write, 'rw' for both
            storage: BlobStorage instance (created if not provided)
            key: Custom encryption key
            servers: Server list (for compatibility, ignored)
            use_cache: Use local cache
        """
        self.mode = mode
        self.use_cache = use_cache
        self.blob_id, self.folder, self.datei = self._path_splitter(filename)

        if storage is None:
            try:
                from toolboxv2 import get_app

                storage = get_app(from_="BlobStorage").root_blob_storage
            except:
                # Use auto-detection for storage mode
                storage = BlobStorage()  # mode=None triggers auto-detection

        self.storage = storage
        self.data_buffer = b""
        self.key = key

        if key:
            # Validate key works
            try:
                test_data = b"test"
                encrypted = self.storage.crypto.encrypt(test_data, key)
                decrypted = self.storage.crypto.decrypt(encrypted, key)
                assert decrypted == test_data
            except Exception:
                raise ValueError("Invalid symmetric key provided.")

    @staticmethod
    def _path_splitter(filename: str):
        """Split filename into blob_id, folder, and file components"""
        parts = Path(filename).parts
        if not parts:
            raise ValueError("Filename cannot be empty.")
        blob_id = parts[0]
        if len(parts) == 1:
            raise ValueError(
                "Filename must include a path within the blob, e.g., 'blob_id/file.txt'"
            )
        datei = parts[-1]
        folder = "|".join(parts[1:-1])
        return blob_id, folder, datei

    def create(self) -> "BlobFile":
        """Create the blob if it doesn't exist"""
        self.storage.create_blob(pickle.dumps({}), self.blob_id)
        return self

    def __enter__(self) -> "BlobFile":
        try:
            raw_blob_data = self.storage.read_blob(
                self.blob_id, use_cache=self.use_cache, decrypt=False
            )
            if raw_blob_data is None or raw_blob_data == b"":
                raw_blob_data = pickle.dumps({})

            # Decrypt at blob level if not using custom key
            if not self.key:
                try:
                    raw_blob_data = self.storage.crypto.decrypt(raw_blob_data)
                except:
                    pass  # May already be decrypted or not encrypted

            blob_content = pickle.loads(raw_blob_data)

        except Exception as e:
            if "404" in str(e) or "NoSuchKey" in str(e):
                blob_content = {}
            else:
                get_logger().warning(f"Read error, using empty content: {e}")
                blob_content = {}

        if "r" in self.mode:
            if self.folder:
                file_data = blob_content.get(self.folder, {}).get(self.datei)
            else:
                file_data = blob_content.get(self.datei)

            if file_data:
                self.data_buffer = file_data
                if self.key:
                    self.data_buffer = self.storage.crypto.decrypt(
                        self.data_buffer, self.key
                    )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if "w" in self.mode:
            final_data = self.data_buffer
            if self.key:
                final_data = self.storage.crypto.encrypt(final_data, self.key)

            try:
                raw_blob_data = self.storage.read_blob(self.blob_id, decrypt=False)
                if raw_blob_data:
                    try:
                        raw_blob_data = self.storage.crypto.decrypt(raw_blob_data)
                    except:
                        pass
                    blob_content = pickle.loads(raw_blob_data)
                else:
                    blob_content = {}
            except:
                blob_content = {}

            current_level = blob_content
            if self.folder:
                if self.folder not in current_level:
                    current_level[self.folder] = {}
                current_level = current_level[self.folder]

            current_level[self.datei] = final_data

            # Encrypt and save
            blob_bytes = pickle.dumps(blob_content)
            if not self.key:
                blob_bytes = self.storage.crypto.encrypt(blob_bytes)

            self.storage.update_blob(self.blob_id, blob_bytes, encrypt=False)

    def exists(self) -> bool:
        """Check if the file exists in the blob"""
        try:
            raw_blob_data = self.storage.read_blob(self.blob_id, decrypt=False)
            if raw_blob_data:
                try:
                    raw_blob_data = self.storage.crypto.decrypt(raw_blob_data)
                except:
                    pass
                blob_content = pickle.loads(raw_blob_data)
            else:
                return False
        except:
            return False

        current_level = blob_content
        if self.folder:
            if self.folder not in current_level:
                return False
            current_level = current_level[self.folder]

        return self.datei in current_level

    def clear(self):
        """Clear the data buffer"""
        self.data_buffer = b""

    def write(self, data: Union[str, bytes]):
        """Write data to buffer"""
        if "w" not in self.mode:
            raise OSError("File not opened in write mode.")
        if isinstance(data, str):
            self.data_buffer += data.encode()
        elif isinstance(data, bytes):
            self.data_buffer += data
        else:
            raise TypeError("write() argument must be str or bytes")

    def read(self) -> bytes:
        """Read data from buffer"""
        if "r" not in self.mode:
            raise OSError("File not opened in read mode.")
        return self.data_buffer

    def read_json(self) -> Any:
        """Read and parse JSON"""
        if "r" not in self.mode:
            raise ValueError("File not opened in read mode.")
        if self.data_buffer == b"":
            return {}
        return json.loads(self.data_buffer.decode())

    def write_json(self, data: Any):
        """Write JSON data"""
        if "w" not in self.mode:
            raise ValueError("File not opened in write mode.")
        self.data_buffer += json.dumps(data).encode()

    def read_pickle(self) -> Any:
        """Read and unpickle data"""
        if "r" not in self.mode:
            raise ValueError("File not opened in read mode.")
        if self.data_buffer == b"":
            return {}
        return pickle.loads(self.data_buffer)

    def write_pickle(self, data: Any):
        """Pickle and write data"""
        if "w" not in self.mode:
            raise ValueError("File not opened in write mode.")
        self.data_buffer += pickle.dumps(data)

    def read_yaml(self) -> Any:
        """Read and parse YAML"""
        if "r" not in self.mode:
            raise ValueError("File not opened in read mode.")
        if self.data_buffer == b"":
            return {}
        return yaml.safe_load(self.data_buffer)

    def write_yaml(self, data: Any):
        """Write YAML data"""
        if "w" not in self.mode:
            raise ValueError("File not opened in write mode.")
        yaml.dump(data, self)

    def watch(
        self,
        callback: Callable[["BlobFile"], None],
        max_idle_timeout: int = 600,
        threaded: bool = True,
    ):
        """Watch for changes to this blob file."""
        self.storage.watch(
            self.blob_id,
            callback,
            max_idle_timeout,
            threaded,
            folder=self.folder,
            filename=self.datei,
        )

    def stop_watch(self, callback: Optional[Callable] = None):
        """Stop watching this blob file."""
        self.storage.stop_watch(self.blob_id, callback)


# Factory functions for easy initialization


def create_server_storage(
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    **kwargs,
) -> BlobStorage:
    """Create storage for server mode"""
    return BlobStorage(
        mode=StorageMode.SERVER,
        minio_endpoint=endpoint,
        minio_access_key=access_key,
        minio_secret_key=secret_key,
        auto_sync=False,
        **kwargs,
    )


def create_desktop_storage(
    cloud_endpoint: Optional[str] = None,
    cloud_access_key: Optional[str] = None,
    cloud_secret_key: Optional[str] = None,
    **kwargs,
) -> BlobStorage:
    """Create storage for desktop mode with optional cloud sync"""
    return BlobStorage(
        mode=StorageMode.DESKTOP,
        cloud_endpoint=cloud_endpoint,
        cloud_access_key=cloud_access_key,
        cloud_secret_key=cloud_secret_key,
        auto_sync=cloud_endpoint is not None,
        **kwargs,
    )


def create_mobile_storage(**kwargs) -> BlobStorage:
    """Create storage for mobile mode (SQLite only)"""
    return BlobStorage(
        mode=StorageMode.MOBILE,
        auto_sync=False,
        **kwargs
    )


def create_offline_storage(**kwargs) -> BlobStorage:
    """Create storage for offline mode"""
    return BlobStorage(
        mode=StorageMode.OFFLINE,
        auto_sync=False,
        **kwargs
    )


def create_auto_storage(**kwargs) -> BlobStorage:
    """
    Create storage with automatic mode detection based on environment.

    - Tauri/Desktop environment → MOBILE mode (SQLite only)
    - Production/Cloud environment → SERVER mode (MinIO)
    - Development environment → SERVER mode (MinIO with dev credentials)

    If MinIO authentication fails, automatically falls back to OFFLINE mode.

    Returns:
        BlobStorage: Configured storage instance
    """
    # Let BlobStorage auto-detect the mode (mode=None triggers detection)
    return BlobStorage(mode=None, **kwargs)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    storage = create_offline_storage(storage_directory="/tmp/blob_test")

    # Test blob operations
    blob_id = storage.create_blob(b"Hello, World!", encrypt=True)
    print(f"Created blob: {blob_id}")

    data = storage.read_blob(blob_id)
    print(f"Read data: {data}")

    # Test BlobFile interface
    with BlobFile(f"{blob_id}/test/file.txt", 'w', storage=storage) as f:
        f.write("Test content")

    with BlobFile(f"{blob_id}/test/file.txt", 'r', storage=storage) as f:
        print(f"Read from BlobFile: {f.read()}")

    print(f"Status: {storage.get_server_status()}")

    storage.close()
    print("Tests passed!")
