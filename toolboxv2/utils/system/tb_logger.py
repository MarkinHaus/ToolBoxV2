# file: toolboxv2/utils/system/tb_logger.py
# Structured JSON Logging with MobileDB persistence (Local-First, Encrypted, Sync-Ready)
# Features:
#   - JSONL structured logging (Grafana/Loki/ELK compatible)
#   - Async batch persistence into MobileDB (non-blocking)
#   - User-specific audit log queries (by user, action, date range)
#   - Log sync to remote MinIO (manual / automatic / time-range filtered)

import datetime
import logging
import os
import queue
import time
import threading
import json
import io
from logging.handlers import SocketHandler
from typing import Dict, Any, Optional, List, Tuple, Iterator

from ..extras.Style import Style, remove_styles

loggerNameOfToolboxv2 = 'toolboxV2'


# ---------------------------------------------------------------------------
# JSON Formatter (stdlib only – no external dependency)
# ---------------------------------------------------------------------------

class JsonLogFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON (JSONL).
    Compatible with Grafana Loki, OpenSearch, ELK, Fluentd.
    """

    _SKIP_FIELDS = frozenset({
        "name", "msg", "args", "created", "relativeCreated",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "filename", "module", "levelno", "levelname", "pathname",
        "thread", "threadName", "process", "processName",
        "message", "msecs", "taskName",
    })

    def __init__(self, app_id: str = "", node_id: str = "", **kwargs):
        super().__init__()
        self.app_id = app_id
        self.node_id = node_id

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        log_dict: Dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "message": record.message,
        }

        if self.app_id:
            log_dict["app_id"] = self.app_id
        if self.node_id:
            log_dict["node_id"] = self.node_id

        # Merge extra fields (audit_action, user_id, details, …)
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in self._SKIP_FIELDS and not key.startswith("_"):
                    try:
                        json.dumps(value)
                        log_dict[key] = value
                    except (TypeError, ValueError):
                        log_dict[key] = str(value)

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            log_dict["exception"] = record.exc_text
        if record.stack_info:
            log_dict["stack"] = record.stack_info

        return json.dumps(log_dict, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# MobileDB Batch Handler (async, non-blocking)
# ---------------------------------------------------------------------------

class MobileDBLogHandler(logging.Handler):
    """
    Async batch handler that writes JSONL log chunks into MobileDB.
    The main thread is never blocked – logs go into a RAM queue first.
    """

    def __init__(self, db, node_id: str,
                 batch_size: int = 100, flush_interval: float = 5.0):
        super().__init__()
        self.db = db
        self.node_id = node_id
        self.log_queue: queue.Queue = queue.Queue()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._stop_event = threading.Event()

        self._worker = threading.Thread(
            target=self._process_queue, daemon=True, name="LogBatchWorker"
        )
        self._worker.start()

    def emit(self, record: logging.LogRecord):
        try:
            log_entry = self.format(record)
            self.log_queue.put_nowait(log_entry)
        except Exception:
            self.handleError(record)

    def _process_queue(self):
        batch: list = []
        last_flush = time.time()

        while not self._stop_event.is_set():
            try:
                entry = self.log_queue.get(timeout=1.0)
                batch.append(entry)
            except queue.Empty:
                pass

            now = time.time()
            if (len(batch) >= self.batch_size
                    or (batch and now - last_flush >= self.flush_interval)):
                self._flush_batch(batch)
                batch = []
                last_flush = now

        # Drain remaining on shutdown
        while not self.log_queue.empty():
            try:
                batch.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: list):
        if not batch:
            return

        chunk_data = "\n".join(batch).encode("utf-8")
        timestamp = int(time.time() * 1000)
        date_str = time.strftime("%Y-%m-%d")

        is_audit = '"audit_action"' in batch[0]
        prefix = "audit" if is_audit else "system"
        path = f"logs/{self.node_id}/{date_str}/{prefix}_{timestamp}.jsonl"

        try:
            self.db.put(
                path=path,
                data=chunk_data,
                content_type="application/jsonl",
                encrypted=True,
            )
        except Exception as e:
            print(f"CRITICAL: Failed to write log batch to MobileDB: {e}")

    def close(self):
        self._stop_event.set()
        self._worker.join(timeout=3.0)
        super().close()


# ---------------------------------------------------------------------------
# Observability Log Handler (direct push, live)
# ---------------------------------------------------------------------------

class ObservabilityLogHandler(logging.Handler):
    """
    Async batch handler that pushes JSONL logs directly to an ObservabilityAdapter.
    Same 5s / 100-entry batch pattern as MobileDBLogHandler, but bypasses
    MobileDB + sync — entries arrive in the dashboard within seconds.

    This handler runs ALONGSIDE MobileDBLogHandler (additive, not replacing).

    Usage:
        from .observability_adapter import OpenObserveAdapter

        adapter = OpenObserveAdapter(endpoint="http://localhost:5080", ...)
        handler = ObservabilityLogHandler(adapter)
        handler.setFormatter(JsonLogFormatter(app_id="myapp", node_id="desktop"))
        logger.addHandler(handler)

    Or use the convenience function:
        enable_live_observability(adapter)
    """

    def __init__(
        self,
        adapter,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        system_stream: str = "system_logs",
        audit_stream: str = "audit_logs",
    ):
        super().__init__()
        self.adapter = adapter
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.system_stream = system_stream
        self.audit_stream = audit_stream
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        self._worker = threading.Thread(
            target=self._process_queue, daemon=True, name="ObsLogWorker"
        )
        self._worker.start()

    def emit(self, record: logging.LogRecord):
        try:
            log_line = self.format(record)
            self._queue.put_nowait(log_line)
        except Exception:
            self.handleError(record)

    def _process_queue(self):
        batch: list = []
        last_flush = time.time()

        while not self._stop_event.is_set():
            try:
                entry = self._queue.get(timeout=1.0)
                batch.append(entry)
            except queue.Empty:
                pass

            now = time.time()
            if (len(batch) >= self.batch_size
                    or (batch and now - last_flush >= self.flush_interval)):
                self._flush_batch(batch)
                batch = []
                last_flush = now

        # Drain on shutdown
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: list):
        if not batch:
            return

        system_entries: list = []
        audit_entries: list = []

        for line in batch:
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            if "audit_action" in entry:
                audit_entries.append(entry)
            else:
                system_entries.append(entry)

        try:
            if system_entries:
                self.adapter.send_batch(system_entries, stream=self.system_stream)
        except Exception as e:
            print(f"ObsLogHandler: system push failed: {e}")

        try:
            if audit_entries:
                self.adapter.send_audit_batch(audit_entries)
        except Exception as e:
            print(f"ObsLogHandler: audit push failed: {e}")

    def close(self):
        self._stop_event.set()
        self._worker.join(timeout=3.0)
        super().close()

def setup_production_logging(
    level: int,
    app_id: str,
    node_id: str,
    local_db=None,
    interminal: bool = True,
) -> logging.Logger:
    """
    Initialize structured JSON logging with optional MobileDB persistence.

    Args:
        level:      Logging level (e.g. logging.DEBUG)
        app_id:     Application / instance identifier
        node_id:    Machine / node identifier
        local_db:   Optional MobileDB instance for encrypted local persistence
        interminal: Whether to also print to stderr (console)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(loggerNameOfToolboxv2)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.propagate = False

    formatter = JsonLogFormatter(app_id=app_id, node_id=node_id)

    if interminal:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(level)
        logger.addHandler(console)

    if local_db is not None:
        db_handler = MobileDBLogHandler(db=local_db, node_id=node_id)
        db_handler.setFormatter(formatter)
        db_handler.setLevel(level)
        logger.addHandler(db_handler)

    return logger


# ---------------------------------------------------------------------------
# Audit Logger (user-specific queries)
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Structured audit logger enforcing a strict schema.
    Supports querying stored audit events by user, action, and time range.

    Usage:
        audit = AuditLogger(logger, db=log_db, node_id="desktop-01")
        audit.log_action("user_42", "LOGIN", "/auth", status="SUCCESS")

        # Query last 20 actions of a user
        entries = audit.get_user_actions("user_42", last_n=20)

        # Summary per user for a week
        summary = audit.get_actions_summary(date_from="2026-02-16", date_to="2026-02-23")
    """

    def __init__(self, base_logger: logging.Logger, db=None, node_id: str = ""):
        """
        Args:
            base_logger: The configured Python logger
            db:          MobileDB instance (for query operations)
            node_id:     Node identifier (for path prefix filtering)
        """
        self.logger = base_logger
        self.db = db
        self.node_id = node_id

    def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        status: str = "SUCCESS",
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log a structured audit event."""
        audit_data = {
            "audit_action": action,
            "user_id": user_id,
            "resource": resource,
            "status": status,
            "details": details or {},
        }
        self.logger.info(
            f"AUDIT: {action} on {resource} by {user_id}",
            extra=audit_data,
        )

    # ---- Query API ----

    def query(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Query stored audit logs with filters.

        Args:
            user_id:    Filter by user (None = all users)
            action:     Filter by audit_action (e.g. "LOGIN", "DELETE")
            date_from:  Start date inclusive "YYYY-MM-DD" (None = no lower bound)
            date_to:    End date inclusive "YYYY-MM-DD" (None = no upper bound)
            status:     Filter by status ("SUCCESS", "FAILURE", …)
            limit:      Max entries to return

        Returns:
            List of parsed audit log dicts, newest first
        """
        if self.db is None:
            return []

        results: List[Dict[str, Any]] = []

        for _path, raw_lines in self._iter_audit_chunks(date_from, date_to):
            for line in raw_lines:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                if "audit_action" not in entry:
                    continue
                if user_id and entry.get("user_id") != user_id:
                    continue
                if action and entry.get("audit_action") != action:
                    continue
                if status and entry.get("status") != status:
                    continue

                results.append(entry)
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        results.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return results

    def get_user_actions(self, user_id: str, last_n: int = 50) -> List[Dict[str, Any]]:
        """Shortcut: get the last N audit entries for a specific user."""
        return self.query(user_id=user_id, limit=last_n)

    def get_actions_summary(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Aggregate audit actions grouped by user.

        Returns:
            {user_id: {action: count, …}, …}
        """
        entries = self.query(date_from=date_from, date_to=date_to, limit=10_000)
        summary: Dict[str, Dict[str, int]] = {}
        for e in entries:
            uid = e.get("user_id", "unknown")
            act = e.get("audit_action", "unknown")
            summary.setdefault(uid, {})
            summary[uid][act] = summary[uid].get(act, 0) + 1
        return summary

    # ---- Internal helpers ----

    def _iter_audit_chunks(
        self,
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> Iterator[Tuple[str, List[str]]]:
        """
        Yield (path, lines[]) for audit JSONL chunks within the date range.
        Path convention: logs/{node_id}/{YYYY-MM-DD}/audit_{ts}.jsonl
        """
        prefix = f"logs/{self.node_id}/" if self.node_id else "logs/"
        blobs = self.db.list(prefix=prefix)

        for meta in blobs:
            parts = meta.path.split("/")
            if len(parts) < 4:
                continue

            fname = parts[-1]
            if not fname.startswith("audit_"):
                continue

            date_folder = parts[-2]
            if date_from and date_folder < date_from:
                continue
            if date_to and date_folder > date_to:
                continue

            data = self.db.get(meta.path)
            if data is None:
                continue

            lines = data.decode("utf-8", errors="replace").split("\n")
            yield meta.path, lines


# ---------------------------------------------------------------------------
# Log Sync Manager (Local MobileDB → Remote MinIO)
# ---------------------------------------------------------------------------

class LogSyncManager:
    """
    Syncs log chunks from local MobileDB to a remote MinIO server.

    Supports:
      - Manual sync: sync_all(), sync_time_range(), sync_audit_only(), sync_system_only()
      - Automatic sync: start_auto_sync(interval_seconds=300)
      - Status: get_pending_stats()

    Usage:
        from minio import Minio

        sync = LogSyncManager(
            db=log_db,
            minio_client=Minio("ryzen.local:9000", ...),
            bucket="system-audit-logs",
            app_id=app.id,
            node_id="desktop-01",
        )
        sync.ensure_bucket()

        # Manual: push everything
        stats = sync.sync_all()

        # Manual: push only last week
        stats = sync.sync_time_range(date_from="2026-02-16", date_to="2026-02-23")

        # Manual: push only audit logs from today
        stats = sync.sync_audit_only(date_from="2026-02-23")

        # Automatic: every 5 minutes
        sync.start_auto_sync(interval_seconds=300)
        # ...
        sync.stop_auto_sync()
    """

    def __init__(
        self,
        db,
        minio_client,
        bucket: str = "system-audit-logs",
        app_id: str = "default",
        node_id: str = "",
    ):
        """
        Args:
            db:            MobileDB instance containing logs
            minio_client:  minio.Minio client connected to the remote/root server
            bucket:        Target MinIO bucket
            app_id:        App identifier (namespace in bucket)
            node_id:       Node identifier (sub-namespace)
        """
        self.db = db
        self.minio = minio_client
        self.bucket = bucket
        self.app_id = app_id
        self.node_id = node_id
        self._sync_lock = threading.Lock()
        self._auto_thread: Optional[threading.Thread] = None
        self._auto_stop = threading.Event()
        self._obs_adapter: Optional[Any] = None  # ObservabilityAdapter

    # ---- Observability Adapter ----

    def set_observability_adapter(self, adapter):
        """
        Attach an observability backend adapter (additive to MinIO sync).

        The adapter receives parsed log entries after each successful
        MinIO sync batch. It does NOT replace MinIO — it's a secondary target.

        Args:
            adapter: ObservabilityAdapter instance (or None to disable)

        Usage:
            from .observability_adapter import OpenObserveAdapter
            adapter = OpenObserveAdapter(
                endpoint="http://ryzen.local:5080",
                credentials=("admin@toolbox.local", "secret"),
            )
            sync_manager.set_observability_adapter(adapter)
        """
        self._obs_adapter = adapter

    def remove_observability_adapter(self):
        """Detach the current observability adapter."""
        if self._obs_adapter:
            try:
                self._obs_adapter.close()
            except Exception:
                pass
        self._obs_adapter = None

    # ---- Manual Sync ----

    def sync_all(self) -> Dict[str, Any]:
        """Push ALL dirty log chunks to remote MinIO."""
        return self._do_sync(prefix="logs/")

    def sync_time_range(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Push log chunks within a date range.

        Args:
            date_from: Start date inclusive "YYYY-MM-DD" (None = no lower bound)
            date_to:   End date inclusive "YYYY-MM-DD" (None = no upper bound)
        """
        return self._do_sync(prefix="logs/", date_from=date_from, date_to=date_to)

    def sync_audit_only(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Push only audit log chunks (optionally time-filtered)."""
        return self._do_sync(
            prefix="logs/", date_from=date_from, date_to=date_to,
            filename_filter="audit_",
        )

    def sync_system_only(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Push only system log chunks (optionally time-filtered)."""
        return self._do_sync(
            prefix="logs/", date_from=date_from, date_to=date_to,
            filename_filter="system_",
        )

    # ---- Automatic Sync ----

    def start_auto_sync(self, interval_seconds: float = 300.0):
        """
        Start background thread that syncs dirty logs every N seconds.

        Args:
            interval_seconds: Sync interval (default 5 min)
        """
        if self._auto_thread and self._auto_thread.is_alive():
            return

        self._auto_stop.clear()
        self._auto_thread = threading.Thread(
            target=self._auto_sync_loop,
            args=(interval_seconds,),
            daemon=True,
            name="LogAutoSync",
        )
        self._auto_thread.start()

    def stop_auto_sync(self):
        """Stop the automatic sync background thread."""
        self._auto_stop.set()
        if self._auto_thread:
            self._auto_thread.join(timeout=2.0)
            self._auto_thread = None

    @property
    def is_auto_syncing(self) -> bool:
        return self._auto_thread is not None and self._auto_thread.is_alive()

    def get_pending_stats(self) -> Dict[str, Any]:
        """Show how many log chunks are pending sync, broken down by type/date."""
        from ..extras.db.mobile_db import SyncStatus

        dirty = self.db.list(prefix="logs/", sync_status=SyncStatus.DIRTY)
        stats: Dict[str, Any] = {
            "total_dirty": len(dirty),
            "by_date": {},
            "audit": 0,
            "system": 0,
        }

        for meta in dirty:
            parts = meta.path.split("/")
            if len(parts) >= 4:
                date_folder = parts[-2]
                fname = parts[-1]
                stats["by_date"][date_folder] = stats["by_date"].get(date_folder, 0) + 1
                if fname.startswith("audit_"):
                    stats["audit"] += 1
                elif fname.startswith("system_"):
                    stats["system"] += 1

        return stats

    # ---- Internal ----

    def _forward_to_adapter(
        self,
        prefix: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        filename_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forward recently synced log entries to the observability adapter."""
        from ..extras.db.mobile_db import SyncStatus

        synced_blobs = self.db.list(prefix=prefix, sync_status=SyncStatus.SYNCED)

        system_entries: List[Dict[str, Any]] = []
        audit_entries: List[Dict[str, Any]] = []

        for meta in synced_blobs:
            parts = meta.path.split("/")
            if len(parts) < 4:
                continue

            fname = parts[-1]
            date_folder = parts[-2]

            if date_from and date_folder < date_from:
                continue
            if date_to and date_folder > date_to:
                continue
            if filename_filter and not fname.startswith(filename_filter):
                continue

            data = self.db.get(meta.path)
            if data is None:
                continue

            lines = data.decode("utf-8", errors="replace").split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                if "audit_action" in entry:
                    audit_entries.append(entry)
                else:
                    system_entries.append(entry)

        adapter_stats = {"system_sent": 0, "audit_sent": 0, "errors": []}
        batch_size = 500

        for i in range(0, len(system_entries), batch_size):
            batch = system_entries[i:i + batch_size]
            result = self._obs_adapter.send_batch(batch)
            adapter_stats["system_sent"] += result.get("sent", 0)
            if result.get("failed"):
                adapter_stats["errors"].extend(result.get("errors", [])[:3])

        for i in range(0, len(audit_entries), batch_size):
            batch = audit_entries[i:i + batch_size]
            result = self._obs_adapter.send_audit_batch(batch)
            adapter_stats["audit_sent"] += result.get("sent", 0)
            if result.get("failed"):
                adapter_stats["errors"].extend(result.get("errors", [])[:3])

        return adapter_stats

    def _auto_sync_loop(self, interval: float):
        while not self._auto_stop.is_set():
            try:
                self._do_sync(prefix="logs/")
            except Exception as e:
                print(f"LogAutoSync error: {e}")
            self._auto_stop.wait(timeout=interval)

    def _do_sync(
        self,
        prefix: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        filename_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Core sync: iterate dirty blobs matching filters, upload to MinIO."""
        from ..extras.db.mobile_db import SyncStatus

        if not self._sync_lock.acquire(blocking=False):
            return {"status": "already_syncing"}

        try:
            stats: Dict[str, Any] = {
                "uploaded": 0,
                "skipped": 0,
                "errors": [],
                "bytes_transferred": 0,
            }

            dirty_blobs = self.db.list(prefix=prefix, sync_status=SyncStatus.DIRTY)

            for meta in dirty_blobs:
                parts = meta.path.split("/")
                if len(parts) < 4:
                    stats["skipped"] += 1
                    continue

                fname = parts[-1]
                date_folder = parts[-2]

                # Date range filter
                if date_from and date_folder < date_from:
                    stats["skipped"] += 1
                    continue
                if date_to and date_folder > date_to:
                    stats["skipped"] += 1
                    continue

                # Filename prefix filter (audit_ / system_)
                if filename_filter and not fname.startswith(filename_filter):
                    stats["skipped"] += 1
                    continue

                data = self.db.get(meta.path)
                if data is None:
                    stats["errors"].append(f"read_failed:{meta.path}")
                    continue

                # Upload: bucket/{app_id}/logs/{node_id}/{date}/{chunk}.jsonl
                cloud_path = f"{self.app_id}/{meta.path}"
                try:
                    self.minio.put_object(
                        self.bucket,
                        cloud_path,
                        io.BytesIO(data),
                        len(data),
                        content_type="application/jsonl",
                        metadata={
                            "checksum": meta.checksum,
                            "node_id": self.node_id,
                            "local_timestamp": str(meta.local_updated_at),
                            "version": str(meta.version),
                        },
                    )
                    self.db.mark_synced(meta.path, time.time())
                    stats["uploaded"] += 1
                    stats["bytes_transferred"] += len(data)
                except Exception as e:
                    stats["errors"].append(f"upload:{meta.path}:{e}")

            # ── Forward to observability adapter (non-blocking, best-effort) ──
            if self._obs_adapter and stats["uploaded"] > 0:
                try:
                    adapter_stats = self._forward_to_adapter(
                        prefix=prefix,
                        date_from=date_from,
                        date_to=date_to,
                        filename_filter=filename_filter,
                    )
                    stats["adapter"] = adapter_stats
                except Exception as e:
                    stats.setdefault("adapter_errors", []).append(str(e))

            stats["status"] = "complete"
            return stats
        finally:
            self._sync_lock.release()

    def ensure_bucket(self, silent: bool = True, ping=0):
        """Create the target bucket if it doesn't exist.

        Args:
            silent: If True, fail silently when MinIO is not ready yet.
                    This prevents delays during first startup before MinIO is started.
        """
        if silent:
            # Quick check: is MinIO reachable at all? If not, skip silently.
            try:
                import urllib3
                # Check if we can reach the endpoint without waiting for retries
                endpoint = self.minio._base_url

                http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=1.0, read=1.0) if not ping else urllib3.Timeout(connect=ping*3, read=ping*1.2), retries=0)
                try:
                    res=http.request("GET", f"{endpoint}/minio/health/live", preload_content=False)
                except Exception as e:
                    # MinIO not ready yet - skip bucket creation
                    return
            except Exception as e:
                # urllib3 not available or other error - continue with normal flow
                pass

        try:
            if not self.minio.bucket_exists(self.bucket):
                self.minio.make_bucket(self.bucket)
        except Exception:
            # Silently fail - bucket will be created on first successful sync
            pass


# ---------------------------------------------------------------------------
# Module-level state (shared across setup_logging / get_logger)
# ---------------------------------------------------------------------------

_log_db = None          # MobileDB instance, set via set_log_db()
_app_id: str = ""       # Set via setup_logging app_name or setup_production_logging
_node_id: str = ""      # Set via setup_production_logging or auto-detected


def set_log_db(db, node_id: str = ""):
    """
    Register a MobileDB instance for structured log persistence.
    Call this BEFORE setup_logging() so the legacy API can attach the
    MobileDBLogHandler automatically.

    Args:
        db:       MobileDB instance (from create_mobile_db)
        node_id:  Machine / node identifier
    """
    global _log_db, _node_id
    _log_db = db
    if node_id:
        _node_id = node_id


def get_log_db():
    """Return the registered MobileDB instance (or None)."""
    return _log_db


# ---------------------------------------------------------------------------
# Unified setup_logging (bridges legacy signature → new system)
# ---------------------------------------------------------------------------

def setup_logging(
    level: int,
    name: str = loggerNameOfToolboxv2,
    online_level: Optional[int] = None,
    is_online: bool = False,
    file_level: Optional[int] = None,
    interminal: bool = False,
    logs_directory: str = "../logs",
    app_name: str = "main",
) -> Tuple[logging.Logger, str]:
    """
    Unified logger setup.

    Keeps the old (logger, filename) return signature so every existing
    call site stays untouched, but internally:
      - Console + File handlers use JsonLogFormatter (JSONL)
      - If a MobileDB was registered via set_log_db(), a
        MobileDBLogHandler is attached automatically.
      - SocketHandler still works for is_online=True.

    Returns:
        (logger, log_filename)
    """
    global loggerNameOfToolboxv2, _app_id

    if not online_level:
        online_level = level
    if not file_level:
        file_level = level

    loggerNameOfToolboxv2 = name
    _app_id = app_name

    # ---- Log file rotation (unchanged logic) ----

    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory, exist_ok=True)
    if not os.path.exists(logs_directory + "/Logs.info"):
        open(f"{logs_directory}/Logs.info", "a").close()

    available_log_levels = [
        logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING,
        logging.WARN, logging.INFO, logging.DEBUG, logging.NOTSET,
    ]
    for lbl, val in [("level", level), ("online_level", online_level), ("file_level", file_level)]:
        if val not in available_log_levels:
            raise ValueError(f"{lbl} must be one of {available_log_levels}, but is {val}")

    log_date = datetime.datetime.today().strftime('%Y-%m-%d')
    log_levels_names = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]
    log_level_index = log_levels_names.index(logging.getLevelName(level))

    filename = f"Logs-{name}-{log_date}-{log_levels_names[log_level_index]}"
    log_filename = f"{logs_directory}/{filename}.log"

    log_info_data: Dict[str, Any] = {filename: 0, "H": "localhost", "P": 62435}

    with open(f"{logs_directory}/Logs.info") as li:
        log_info_data_str = li.read()
        try:
            log_info_data = eval(log_info_data_str)
        except SyntaxError:
            if log_info_data_str:
                print(Style.RED(Style.Bold("Could not parse log info data")))

        if filename not in log_info_data:
            log_info_data[filename] = 0
        if not os.path.exists(log_filename):
            log_info_data[filename] = 0
        if os.path.exists(log_filename):
            log_info_data[filename] += 1
            while os.path.exists(f"{logs_directory}/{filename}#{log_info_data[filename]}.log"):
                log_info_data[filename] += 1
            try:
                os.rename(log_filename, f"{logs_directory}/{filename}#{log_info_data[filename]}.log")
            except PermissionError:
                pass

    with open(f"{logs_directory}/Logs.info", "w") as li:
        if len(log_info_data.keys()) >= 7:
            log_info_data = {
                filename: log_info_data[filename],
                "H": log_info_data["H"],
                "P": log_info_data["P"],
            }
        li.write(str(log_info_data))

    try:
        with open(log_filename, "a"):
            pass
    except OSError:
        log_filename = f"{logs_directory}/Logs-Test-{log_date}-{log_levels_names[log_level_index]}.log"
        with open(log_filename, "a"):
            pass

    # ---- Configure logger with new JSON formatters ----

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.propagate = False

    json_formatter = JsonLogFormatter(app_id=app_name, node_id=_node_id)

    # File handler – JSONL output (machine-parseable, replaces old plain text)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(file_level)
    logger.addHandler(file_handler)

    # Console handler
    if interminal:
        console_handler = logging.StreamHandler()
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s:%(lineno)d - %(message)s'
        console_handler.setFormatter(logging.Formatter(log_format))
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # Socket handler (legacy remote logging)
    if is_online:
        socket_handler = SocketHandler(log_info_data["H"], log_info_data["P"])
        socket_handler.setFormatter(json_formatter)
        socket_handler.setLevel(online_level)
        logger.addHandler(socket_handler)

    # MobileDB handler (encrypted, offline-first, sync-ready)
    if _log_db is not None:
        db_handler = MobileDBLogHandler(db=_log_db, node_id=_node_id)
        db_handler.setFormatter(json_formatter)
        db_handler.setLevel(level)
        logger.addHandler(db_handler)

    return logger, filename


# ---------------------------------------------------------------------------
# get_logger – returns the active production logger
# ---------------------------------------------------------------------------

def get_logger() -> logging.Logger:
    """
    Return the active toolboxv2 logger.
    Same API as before – but now backed by JSON formatting + MobileDB.
    """
    return logging.getLogger(loggerNameOfToolboxv2)


# ---------------------------------------------------------------------------
# Live Observability (attach/detach direct handler to active logger)
# ---------------------------------------------------------------------------

_obs_handler: Optional[ObservabilityLogHandler] = None


def enable_live_observability(
    adapter,
    level: int = logging.DEBUG,
    system_stream: str = "system_logs",
    audit_stream: str = "audit_logs",
    flush_interval: float = 5.0,
) -> ObservabilityLogHandler:
    """
    Attach a live ObservabilityLogHandler to the active logger.

    Every log entry from ANY code using get_logger() will be pushed
    to the adapter in near-realtime (5s batch window).

    Call this ONCE after setup_logging() — it's safe to call multiple
    times (previous handler is removed first).

    Args:
        adapter:         ObservabilityAdapter instance
        level:           Minimum log level to forward (default: DEBUG = everything)
        system_stream:   OpenObserve stream for system logs
        audit_stream:    OpenObserve stream for audit logs
        flush_interval:  Seconds between batch flushes (default: 5.0)

    Returns:
        The attached handler (for manual close if needed)

    Usage:
        from .observability_adapter import OpenObserveAdapter
        from .tb_logger import enable_live_observability

        adapter = OpenObserveAdapter(
            endpoint="http://localhost:5080",
            credentials=("admin@toolbox.local", "secret"),
        )
        enable_live_observability(adapter)

        # Done — all logs now appear in OpenObserve within 5s
    """
    global _obs_handler

    # Remove previous handler if exists
    if _obs_handler is not None:
        disable_live_observability()

    logger = get_logger()

    handler = ObservabilityLogHandler(
        adapter=adapter,
        system_stream=system_stream,
        audit_stream=audit_stream,
        flush_interval=flush_interval,
    )

    # Reuse the formatter from the first existing handler (has app_id + node_id)
    for h in logger.handlers:
        if isinstance(h.formatter, JsonLogFormatter):
            handler.setFormatter(h.formatter)
            break
    else:
        # Fallback: create a basic formatter
        handler.setFormatter(JsonLogFormatter(app_id=_app_id, node_id=_node_id))

    handler.setLevel(level)
    logger.addHandler(handler)
    _obs_handler = handler
    return handler


def disable_live_observability():
    """
    Remove the live observability handler from the active logger.
    Safe to call even if not enabled.
    """
    global _obs_handler

    if _obs_handler is None:
        return

    logger = get_logger()
    try:
        logger.removeHandler(_obs_handler)
        _obs_handler.close()
    except Exception as e:
        print(e)
        logger.error(f"Error disable_live_observability {e}")
        pass
    _obs_handler = None


def is_live_observability_enabled() -> bool:
    """Check if live observability is currently active."""
    return _obs_handler is not None and _obs_handler._worker.is_alive()


# ---------------------------------------------------------------------------
# File utilities (kept for backward compat)
# ---------------------------------------------------------------------------

def unstyle_log_files(filename):
    with open(filename) as f:
        text = f.read()
    text = remove_styles(text)
    text += "\n no-styles \n"
    with open(filename, "w") as f:
        f.write(text)


def edit_log_files(name: str, date: str, level: int, n=1, m=float('inf'), do=os.remove):
    year, month, day = date.split('-')
    if day.lower() == "xx":
        for i in range(1, 32):
            n_date = year + '-' + month + '-' + ('0' if i < 10 else '') + str(i)
            _edit_many_log_files(name, n_date, level, n, m, do)
    else:
        _edit_many_log_files(name, date, level, n, m, do)


def _edit_many_log_files(name, date, level, log_file_number, max_number, do):
    log_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]
    log_level_index = log_levels.index(logging.getLevelName(level))
    filename = f"Logs-{name}-{date}-{log_levels[log_level_index]}"
    if not log_file_number and os.path.exists(f"logs/{filename}.log"):
        print(f"editing {filename}.log")
        do(f"logs/{filename}.log")
    if not log_file_number:
        log_file_number += 1
    while os.path.exists(f"logs/{filename}#{log_file_number}.log"):
        if log_file_number >= max_number:
            break
        print(f"editing {filename}#{log_file_number}.log")
        do(f"logs/{filename}#{log_file_number}.log")
        log_file_number += 1
