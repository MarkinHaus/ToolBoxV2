# file: mobile_db.py
# SQLite-based Local Storage for Mobile & Offline Capabilities
# Supports auto-sync with MinIO cloud, dirty tracking, and encryption

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, Iterator, Tuple
import logging

# Logger setup
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("mobile_db")
        if not _logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger


class SyncStatus(Enum):
    """Sync status for local objects"""
    SYNCED = "synced"           # In sync with cloud
    DIRTY = "dirty"             # Local changes need upload
    PENDING_DOWNLOAD = "pending_download"  # Cloud has newer version
    CONFLICT = "conflict"       # Both local and cloud changed
    DELETED = "deleted"         # Marked for deletion


@dataclass
class BlobMetadata:
    """Metadata for a stored blob"""
    path: str                   # Unique path/key
    size: int                   # Size in bytes
    checksum: str               # SHA256 hash
    local_updated_at: float     # Local timestamp
    cloud_updated_at: Optional[float] = None  # Cloud timestamp
    sync_status: SyncStatus = SyncStatus.DIRTY
    version: int = 1
    content_type: str = "application/octet-stream"
    encrypted: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "checksum": self.checksum,
            "local_updated_at": self.local_updated_at,
            "cloud_updated_at": self.cloud_updated_at,
            "sync_status": self.sync_status.value,
            "version": self.version,
            "content_type": self.content_type,
            "encrypted": self.encrypted,
        }
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> 'BlobMetadata':
        return cls(
            path=row["path"],
            size=row["size"],
            checksum=row["checksum"],
            local_updated_at=row["local_updated_at"],
            cloud_updated_at=row["cloud_updated_at"],
            sync_status=SyncStatus(row["sync_status"]),
            version=row["version"],
            content_type=row["content_type"],
            encrypted=bool(row["encrypted"]),
        )


class MobileDB:
    """
    SQLite-based local storage for mobile and offline scenarios.
    
    Features:
    - Offline-first design
    - Dirty tracking for sync
    - Auto-size management
    - Encryption-ready
    - Watch callbacks for changes
    """
    
    SCHEMA_VERSION = 1
    
    CREATE_TABLES_SQL = """
    -- Main blob storage table
    CREATE TABLE IF NOT EXISTS blobs (
        path TEXT PRIMARY KEY,
        data BLOB NOT NULL,
        size INTEGER NOT NULL,
        checksum TEXT NOT NULL,
        local_updated_at REAL NOT NULL,
        cloud_updated_at REAL,
        sync_status TEXT NOT NULL DEFAULT 'dirty',
        version INTEGER NOT NULL DEFAULT 1,
        content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
        encrypted INTEGER NOT NULL DEFAULT 1,
        created_at REAL NOT NULL DEFAULT (julianday('now'))
    );
    
    -- Index for sync operations
    CREATE INDEX IF NOT EXISTS idx_sync_status ON blobs(sync_status);
    CREATE INDEX IF NOT EXISTS idx_local_updated ON blobs(local_updated_at);
    
    -- Metadata table for database info
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    
    -- Sync log for debugging and conflict resolution
    CREATE TABLE IF NOT EXISTS sync_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL,
        action TEXT NOT NULL,
        timestamp REAL NOT NULL DEFAULT (julianday('now')),
        details TEXT
    );
    
    -- Watch subscriptions (for persistence across restarts)
    CREATE TABLE IF NOT EXISTS watch_subscriptions (
        path_pattern TEXT PRIMARY KEY,
        callback_id TEXT NOT NULL,
        created_at REAL NOT NULL DEFAULT (julianday('now'))
    );
    """
    
    def __init__(self, db_path: str = "mobile_data.db",
                 max_size_mb: int = 500,
                 auto_vacuum: bool = True):
        """
        Initialize MobileDB.
        
        Args:
            db_path: Path to SQLite database file
            max_size_mb: Maximum database size in MB (for auto-cleanup)
            auto_vacuum: Enable auto-vacuum
        """
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.auto_vacuum = auto_vacuum
        
        self._local = threading.local()
        self._lock = threading.RLock()
        self._watch_callbacks: Dict[str, List[Callable]] = {}
        self._closed = False
        
        # Initialize database
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            if self.auto_vacuum:
                self._local.connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
        return self._local.connection
    
    @contextmanager
    def _transaction(self):
        """Context manager for database transactions"""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def _init_db(self):
        """Initialize database schema"""
        with self._transaction() as conn:
            conn.executescript(self.CREATE_TABLES_SQL)
            
            # Store schema version
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("schema_version", str(self.SCHEMA_VERSION))
            )
    
    def close(self):
        """Close database connection"""
        self._closed = True
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # =================== Core CRUD Operations ===================
    
    def put(self, path: str, data: bytes, 
            content_type: str = "application/octet-stream",
            encrypted: bool = True,
            skip_sync: bool = False) -> BlobMetadata:
        """
        Store a blob.
        
        Args:
            path: Unique path/key for the blob
            data: Binary data to store
            content_type: MIME type
            encrypted: Whether data is encrypted
            skip_sync: If True, mark as synced (for cloud-pulled data)
        
        Returns:
            BlobMetadata for the stored blob
        """
        with self._lock:
            checksum = hashlib.sha256(data).hexdigest()
            now = time.time()
            
            # Check for existing blob
            existing = self.get_metadata(path)
            version = (existing.version + 1) if existing else 1
            
            sync_status = SyncStatus.SYNCED if skip_sync else SyncStatus.DIRTY
            
            with self._transaction() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO blobs 
                    (path, data, size, checksum, local_updated_at, 
                     sync_status, version, content_type, encrypted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (path, data, len(data), checksum, now,
                      sync_status.value, version, content_type, int(encrypted)))
                
                # Log the action
                conn.execute("""
                    INSERT INTO sync_log (path, action, details)
                    VALUES (?, ?, ?)
                """, (path, "put", json.dumps({"size": len(data), "version": version})))
            
            metadata = BlobMetadata(
                path=path,
                size=len(data),
                checksum=checksum,
                local_updated_at=now,
                sync_status=sync_status,
                version=version,
                content_type=content_type,
                encrypted=encrypted,
            )
            
            # Trigger watch callbacks
            self._notify_watchers(path, "put", metadata)
            
            # Check size limits
            self._check_size_limit()
            
            return metadata
    
    def get(self, path: str) -> Optional[bytes]:
        """
        Retrieve blob data.
        
        Args:
            path: Path/key of the blob
        
        Returns:
            Blob data or None if not found
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT data FROM blobs WHERE path = ? AND sync_status != 'deleted'",
            (path,)
        ).fetchone()
        
        return row["data"] if row else None
    
    def get_metadata(self, path: str) -> Optional[BlobMetadata]:
        """Get metadata for a blob"""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM blobs WHERE path = ?",
            (path,)
        ).fetchone()
        
        return BlobMetadata.from_row(row) if row else None
    
    def delete(self, path: str, hard_delete: bool = False) -> bool:
        """
        Delete a blob.
        
        Args:
            path: Path/key of the blob
            hard_delete: If True, remove immediately. If False, mark for sync deletion.
        
        Returns:
            True if blob was found and deleted
        """
        with self._lock:
            existing = self.get_metadata(path)
            if not existing:
                return False
            
            with self._transaction() as conn:
                if hard_delete:
                    conn.execute("DELETE FROM blobs WHERE path = ?", (path,))
                else:
                    # Mark as deleted for sync
                    conn.execute("""
                        UPDATE blobs SET sync_status = 'deleted', local_updated_at = ?
                        WHERE path = ?
                    """, (time.time(), path))
                
                conn.execute("""
                    INSERT INTO sync_log (path, action, details)
                    VALUES (?, ?, ?)
                """, (path, "delete", json.dumps({"hard": hard_delete})))
            
            self._notify_watchers(path, "delete", existing)
            return True
    
    def exists(self, path: str) -> bool:
        """Check if a blob exists"""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT 1 FROM blobs WHERE path = ? AND sync_status != 'deleted'",
            (path,)
        ).fetchone()
        return row is not None
    
    def list(self, prefix: str = "", 
             include_deleted: bool = False,
             sync_status: Optional[SyncStatus] = None) -> List[BlobMetadata]:
        """
        List blobs with optional filtering.
        
        Args:
            prefix: Path prefix to filter by
            include_deleted: Include deleted blobs
            sync_status: Filter by sync status
        
        Returns:
            List of BlobMetadata objects
        """
        conn = self._get_connection()
        
        query = "SELECT * FROM blobs WHERE path LIKE ?"
        params = [prefix + "%"]
        
        if not include_deleted:
            query += " AND sync_status != 'deleted'"
        
        if sync_status:
            query += " AND sync_status = ?"
            params.append(sync_status.value)
        
        query += " ORDER BY path"
        
        rows = conn.execute(query, params).fetchall()
        return [BlobMetadata.from_row(row) for row in rows]
    
    # =================== Sync Operations ===================
    
    def get_dirty_blobs(self) -> List[BlobMetadata]:
        """Get all blobs that need to be synced to cloud"""
        return self.list(sync_status=SyncStatus.DIRTY)
    
    def get_pending_deletes(self) -> List[BlobMetadata]:
        """Get blobs marked for deletion"""
        return self.list(sync_status=SyncStatus.DELETED, include_deleted=True)
    
    def mark_synced(self, path: str, cloud_timestamp: Optional[float] = None):
        """Mark a blob as synced with cloud"""
        with self._transaction() as conn:
            conn.execute("""
                UPDATE blobs 
                SET sync_status = 'synced', cloud_updated_at = ?
                WHERE path = ?
            """, (cloud_timestamp or time.time(), path))
    
    def mark_conflict(self, path: str):
        """Mark a blob as having a sync conflict"""
        with self._transaction() as conn:
            conn.execute("""
                UPDATE blobs SET sync_status = 'conflict'
                WHERE path = ?
            """, (path,))
    
    def resolve_conflict(self, path: str, use_local: bool = True):
        """
        Resolve a sync conflict.
        
        Args:
            path: Blob path
            use_local: If True, keep local version. If False, cloud wins.
        """
        with self._lock:
            if use_local:
                # Mark as dirty to re-upload
                with self._transaction() as conn:
                    conn.execute("""
                        UPDATE blobs SET sync_status = 'dirty'
                        WHERE path = ?
                    """, (path,))
            else:
                # Delete local, it will be re-downloaded
                self.delete(path, hard_delete=True)
    
    def get_sync_stats(self) -> Dict[str, int]:
        """Get statistics about sync status"""
        conn = self._get_connection()
        
        stats = {}
        for status in SyncStatus:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM blobs WHERE sync_status = ?",
                (status.value,)
            ).fetchone()
            stats[status.value] = row["count"]
        
        # Total size
        row = conn.execute(
            "SELECT SUM(size) as total FROM blobs WHERE sync_status != 'deleted'"
        ).fetchone()
        stats["total_size"] = row["total"] or 0
        
        return stats
    
    def needs_sync(self, cloud_manifest: Dict[str, Tuple[str, float]]) -> Dict[str, str]:
        """
        Compare local state with cloud manifest to determine sync actions.
        
        Args:
            cloud_manifest: Dict of {path: (checksum, timestamp)} from cloud
        
        Returns:
            Dict of {path: action} where action is 'upload', 'download', or 'conflict'
        """
        actions = {}
        
        local_blobs = {b.path: b for b in self.list()}
        
        # Check local blobs
        for path, blob in local_blobs.items():
            if path in cloud_manifest:
                cloud_checksum, cloud_ts = cloud_manifest[path]
                
                if blob.checksum != cloud_checksum:
                    # Content differs
                    if blob.local_updated_at > cloud_ts:
                        actions[path] = "upload"
                    elif cloud_ts > blob.local_updated_at:
                        actions[path] = "download"
                    else:
                        actions[path] = "conflict"
            else:
                # Not in cloud
                actions[path] = "upload"
        
        # Check cloud-only blobs
        for path in cloud_manifest:
            if path not in local_blobs:
                actions[path] = "download"
        
        return actions
    
    # =================== Watch/Subscription System ===================
    
    def watch(self, path_pattern: str, callback: Callable[[str, str, Any], None]) -> str:
        """
        Watch for changes to blobs matching a pattern.
        
        Args:
            path_pattern: Glob-like pattern (e.g., "user/*" or exact path)
            callback: Function(path, action, data) called on changes
        
        Returns:
            Subscription ID for unwatch
        """
        callback_id = hashlib.md5(
            f"{path_pattern}:{id(callback)}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        if path_pattern not in self._watch_callbacks:
            self._watch_callbacks[path_pattern] = []
        
        self._watch_callbacks[path_pattern].append((callback_id, callback))
        
        # Persist subscription
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO watch_subscriptions (path_pattern, callback_id)
                VALUES (?, ?)
            """, (path_pattern, callback_id))
        
        return callback_id
    
    def unwatch(self, callback_id: str):
        """Remove a watch subscription"""
        for pattern, callbacks in list(self._watch_callbacks.items()):
            self._watch_callbacks[pattern] = [
                (cid, cb) for cid, cb in callbacks if cid != callback_id
            ]
            if not self._watch_callbacks[pattern]:
                del self._watch_callbacks[pattern]
        
        with self._transaction() as conn:
            conn.execute(
                "DELETE FROM watch_subscriptions WHERE callback_id = ?",
                (callback_id,)
            )
    
    def _notify_watchers(self, path: str, action: str, data: Any):
        """Notify all matching watchers of a change"""
        for pattern, callbacks in self._watch_callbacks.items():
            if self._matches_pattern(path, pattern):
                for callback_id, callback in callbacks:
                    try:
                        callback(path, action, data)
                    except Exception as e:
                        get_logger().error(f"Watch callback error: {e}")
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a watch pattern"""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return path.startswith(pattern[:-1])
        return path == pattern
    
    # =================== Size Management ===================
    
    def get_db_size(self) -> int:
        """Get current database file size in bytes"""
        if self.db_path.exists():
            return self.db_path.stat().st_size
        return 0
    
    def _check_size_limit(self):
        """Check if we need to cleanup to stay under size limit"""
        current_size = self.get_db_size()
        
        if current_size > self.max_size_bytes:
            get_logger().warning(
                f"Database size ({current_size / 1024 / 1024:.1f}MB) exceeds limit "
                f"({self.max_size_bytes / 1024 / 1024:.1f}MB). Running cleanup..."
            )
            self.cleanup_old_synced(target_size=int(self.max_size_bytes * 0.8))
    
    def cleanup_old_synced(self, target_size: Optional[int] = None, 
                           max_age_days: int = 30) -> int:
        """
        Clean up old synced blobs to free space.
        
        Args:
            target_size: Target database size in bytes
            max_age_days: Remove synced blobs older than this
        
        Returns:
            Number of blobs removed
        """
        with self._lock:
            conn = self._get_connection()
            
            # First: remove hard-deleted entries
            conn.execute("DELETE FROM blobs WHERE sync_status = 'deleted'")
            
            # Get candidates for cleanup (synced blobs, oldest first)
            cutoff = time.time() - (max_age_days * 24 * 3600)
            
            candidates = conn.execute("""
                SELECT path, size FROM blobs 
                WHERE sync_status = 'synced' AND local_updated_at < ?
                ORDER BY local_updated_at ASC
            """, (cutoff,)).fetchall()
            
            removed = 0
            current_size = self.get_db_size()
            
            for row in candidates:
                if target_size and current_size <= target_size:
                    break
                
                conn.execute("DELETE FROM blobs WHERE path = ?", (row["path"],))
                current_size -= row["size"]
                removed += 1
            
            conn.commit()
            
            # Vacuum to reclaim space
            if removed > 0:
                conn.execute("PRAGMA incremental_vacuum")
            
            get_logger().info(f"Cleanup removed {removed} blobs")
            return removed
    
    def vacuum(self):
        """Run full vacuum to optimize database"""
        conn = self._get_connection()
        conn.execute("VACUUM")
    
    # =================== Import/Export ===================
    
    def export_for_sync(self) -> Iterator[Tuple[str, bytes, BlobMetadata]]:
        """
        Export dirty blobs for sync to cloud.
        
        Yields:
            Tuple of (path, data, metadata) for each dirty blob
        """
        for meta in self.get_dirty_blobs():
            data = self.get(meta.path)
            if data:
                yield meta.path, data, meta
    
    def import_from_cloud(self, path: str, data: bytes, 
                          cloud_timestamp: float,
                          checksum: str) -> bool:
        """
        Import a blob from cloud.
        
        Args:
            path: Blob path
            data: Blob data
            cloud_timestamp: Cloud modification timestamp
            checksum: Cloud checksum for verification
        
        Returns:
            True if imported successfully
        """
        # Verify checksum
        if hashlib.sha256(data).hexdigest() != checksum:
            get_logger().error(f"Checksum mismatch for {path}")
            return False
        
        existing = self.get_metadata(path)
        
        if existing:
            # Check for conflict
            if existing.sync_status == SyncStatus.DIRTY:
                if existing.local_updated_at > cloud_timestamp:
                    # Local is newer, keep it
                    return True
                elif existing.checksum != checksum:
                    # Both changed - conflict
                    self.mark_conflict(path)
                    return False
        
        # Import the blob
        self.put(path, data, skip_sync=True)
        self.mark_synced(path, cloud_timestamp)
        
        return True
    
    def get_manifest(self) -> Dict[str, Tuple[str, float]]:
        """
        Get local manifest for sync comparison.
        
        Returns:
            Dict of {path: (checksum, timestamp)}
        """
        return {
            meta.path: (meta.checksum, meta.local_updated_at)
            for meta in self.list()
        }


class MobileDBSyncManager:
    """
    Manages synchronization between MobileDB and MinIO cloud.
    """
    
    def __init__(self, mobile_db: MobileDB, 
                 minio_client: Any,  # MinIO client instance
                 bucket: str = "user-data-enc",
                 user_id: str = "default"):
        """
        Initialize sync manager.
        
        Args:
            mobile_db: MobileDB instance
            minio_client: MinIO client for cloud operations
            bucket: MinIO bucket name
            user_id: User ID for namespacing
        """
        self.db = mobile_db
        self.minio = minio_client
        self.bucket = bucket
        self.user_id = user_id
        self._sync_lock = threading.Lock()
        self._is_syncing = False
    
    def sync(self, force_full: bool = False) -> Dict[str, Any]:
        """
        Perform sync with cloud.
        
        Args:
            force_full: If True, do full sync instead of incremental
        
        Returns:
            Sync statistics
        """
        if self._is_syncing:
            get_logger().warning("Sync already in progress")
            return {"status": "already_syncing"}
        
        with self._sync_lock:
            self._is_syncing = True
            
            try:
                stats = {
                    "uploaded": 0,
                    "downloaded": 0,
                    "deleted": 0,
                    "conflicts": 0,
                    "errors": [],
                }
                
                # Phase 1: Push local changes
                for path, data, meta in self.db.export_for_sync():
                    try:
                        cloud_path = f"{self.user_id}/{path}"
                        self.minio.put_object(
                            self.bucket,
                            cloud_path,
                            data,
                            len(data),
                            metadata={
                                "checksum": meta.checksum,
                                "local_timestamp": str(meta.local_updated_at),
                                "version": str(meta.version),
                            }
                        )
                        self.db.mark_synced(path, time.time())
                        stats["uploaded"] += 1
                        
                    except Exception as e:
                        stats["errors"].append(f"Upload {path}: {e}")
                
                # Phase 2: Process deletes
                for meta in self.db.get_pending_deletes():
                    try:
                        cloud_path = f"{self.user_id}/{meta.path}"
                        self.minio.remove_object(self.bucket, cloud_path)
                        self.db.delete(meta.path, hard_delete=True)
                        stats["deleted"] += 1
                        
                    except Exception as e:
                        stats["errors"].append(f"Delete {meta.path}: {e}")
                
                # Phase 3: Pull cloud changes
                try:
                    cloud_objects = self.minio.list_objects(
                        self.bucket,
                        prefix=f"{self.user_id}/",
                        recursive=True
                    )
                    
                    local_manifest = self.db.get_manifest()
                    
                    for obj in cloud_objects:
                        path = obj.object_name.replace(f"{self.user_id}/", "", 1)
                        
                        # Check if we need to download
                        if path not in local_manifest:
                            # New cloud object
                            self._download_object(path, obj, stats)
                        else:
                            local_checksum, local_ts = local_manifest[path]
                            cloud_ts = obj.last_modified.timestamp()
                            
                            # Get cloud checksum from metadata
                            stat = self.minio.stat_object(self.bucket, obj.object_name)
                            cloud_checksum = stat.metadata.get("x-amz-meta-checksum", "")
                            
                            if cloud_checksum and cloud_checksum != local_checksum:
                                if cloud_ts > local_ts:
                                    self._download_object(path, obj, stats)
                
                except Exception as e:
                    stats["errors"].append(f"List objects: {e}")
                
                stats["status"] = "complete"
                return stats
                
            finally:
                self._is_syncing = False
    
    def _download_object(self, path: str, obj: Any, stats: Dict):
        """Download an object from cloud"""
        try:
            response = self.minio.get_object(self.bucket, obj.object_name)
            data = response.read()
            
            stat = self.minio.stat_object(self.bucket, obj.object_name)
            checksum = stat.metadata.get("x-amz-meta-checksum", 
                                         hashlib.sha256(data).hexdigest())
            
            if self.db.import_from_cloud(
                path, data,
                obj.last_modified.timestamp(),
                checksum
            ):
                stats["downloaded"] += 1
            else:
                stats["conflicts"] += 1
                
        except Exception as e:
            stats["errors"].append(f"Download {path}: {e}")
    
    def manual_sync_needed(self) -> bool:
        """Check if manual sync is needed (for mobile battery saving)"""
        stats = self.db.get_sync_stats()
        return stats.get("dirty", 0) > 0 or stats.get("deleted", 0) > 0


# Convenience functions

def create_mobile_db(path: str = "~/.toolboxv2/mobile_data.db",
                     max_size_mb: int = 500) -> MobileDB:
    """Create a MobileDB instance with sensible defaults"""
    return MobileDB(
        db_path=os.path.expanduser(path),
        max_size_mb=max_size_mb
    )


if __name__ == "__main__":
    # Quick test
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MobileDB(os.path.join(tmpdir, "test.db"))
        
        # Test basic operations
        meta = db.put("test/file.txt", b"Hello World")
        print(f"Created: {meta}")
        
        data = db.get("test/file.txt")
        print(f"Read: {data}")
        
        # Test listing
        db.put("test/file2.txt", b"Second file")
        db.put("other/file.txt", b"Other file")
        
        print(f"All blobs: {[b.path for b in db.list()]}")
        print(f"Test prefix: {[b.path for b in db.list('test/')]}")
        
        # Test sync stats
        print(f"Sync stats: {db.get_sync_stats()}")
        
        # Test watch
        def on_change(path, action, data):
            print(f"Change: {action} on {path}")
        
        watch_id = db.watch("test/*", on_change)
        db.put("test/file3.txt", b"Triggers watch")
        db.unwatch(watch_id)
        
        db.close()
        print("Tests passed!")
