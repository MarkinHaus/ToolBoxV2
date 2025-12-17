"""
ToolBoxV2 MCP Server - Managers
===============================
Stateful management with thread safety and atomic writes
Following ToolBox V2 Architecture Guidelines
"""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    APIKeyInfo,
    CacheEntry,
    FlowSession,
    FlowState,
    MCPConfig,
    PerformanceMetrics,
    PermissionLevel,
)

logger = logging.getLogger("mcp.managers")


# =============================================================================
# API KEY MANAGER
# =============================================================================


class APIKeyManager:
    """
    Thread-safe API key management with atomic persistence.

    Features:
    - Secure key generation with tb_mcp_ prefix
    - Hash-based storage (never stores raw keys)
    - Atomic writes to prevent corruption
    - Usage tracking
    """

    def __init__(self, keys_file: str):
        self.keys_file = Path(keys_file)
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="apikey")
        self._keys: Dict[str, APIKeyInfo] = {}
        self._loaded = False

    async def load(self) -> bool:
        """Load keys from encrypted storage."""
        async with self._lock:
            if self._loaded:
                return True

            try:
                loop = asyncio.get_running_loop()
                self._keys = await loop.run_in_executor(self._executor, self._sync_load)
                self._loaded = True
                logger.info(f"Loaded {len(self._keys)} API keys")
                return True
            except Exception as e:
                logger.warning(f"Could not load API keys: {e}")
                self._keys = {}
                self._loaded = True
                return False

    def _sync_load(self) -> Dict[str, APIKeyInfo]:
        """Synchronous load - runs in thread pool."""
        keys = {}

        # Try BlobFile first (encrypted)
        try:
            from toolboxv2.utils.extras.blobs import BlobFile
            from toolboxv2.utils.system.types import Code

            if BlobFile(str(self.keys_file), key=Code.DK()()).exists():
                with BlobFile(str(self.keys_file), key=Code.DK()(), mode="r") as f:
                    data = f.read_json()
                    for key_hash, info in data.items():
                        keys[key_hash] = APIKeyInfo(
                            name=info["name"],
                            permissions=info["permissions"],
                            created=info["created"],
                            last_used=info.get("last_used"),
                            usage_count=info.get("usage_count", 0),
                        )
                return keys
        except ImportError:
            pass

        # Fallback to plain JSON
        if self.keys_file.exists():
            with open(self.keys_file, "r") as f:
                data = json.load(f)
                for key_hash, info in data.items():
                    keys[key_hash] = APIKeyInfo(
                        name=info["name"],
                        permissions=info["permissions"],
                        created=info["created"],
                        last_used=info.get("last_used"),
                        usage_count=info.get("usage_count", 0),
                    )

        return keys

    async def save(self) -> bool:
        """Save keys with atomic write."""
        async with self._lock:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, self._sync_save)
                return True
            except Exception as e:
                logger.error(f"Failed to save API keys: {e}")
                return False

    def _sync_save(self):
        """Synchronous save with atomic write - runs in thread pool."""
        # Ensure directory exists
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data
        data = {key_hash: info.to_dict() for key_hash, info in self._keys.items()}

        # Try BlobFile first (encrypted)
        try:
            from toolboxv2.utils.extras.blobs import BlobFile
            from toolboxv2.utils.system.types import Code

            with BlobFile(str(self.keys_file), key=Code.DK()(), mode="w") as f:
                f.write_json(data)
            return
        except ImportError:
            pass

        # Fallback: Atomic write to plain JSON
        temp_file = self.keys_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)

        # Atomic replace
        os.replace(temp_file, self.keys_file)

    async def generate_key(
        self, name: str, permissions: Optional[List[str]] = None
    ) -> Tuple[str, APIKeyInfo]:
        """Generate a new API key."""
        await self.load()

        if permissions is None:
            permissions = [p.value for p in PermissionLevel]

        # Generate secure key
        api_key = f"tb_mcp_{uuid.uuid4().hex}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Create info
        info = APIKeyInfo(name=name, permissions=permissions, created=time.time())

        async with self._lock:
            self._keys[key_hash] = info

        await self.save()
        logger.info(f"Generated API key '{name}' with permissions: {permissions}")

        return api_key, info

    async def validate(self, api_key: str) -> Optional[APIKeyInfo]:
        """Validate an API key and update usage stats."""
        await self.load()

        if not api_key or not api_key.startswith("tb_mcp_"):
            return None

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        async with self._lock:
            if key_hash not in self._keys:
                return None

            info = self._keys[key_hash]
            info.last_used = time.time()
            info.usage_count += 1

        # Save in background (don't block validation)
        asyncio.create_task(self.save())

        return info

    async def revoke(self, name: str) -> bool:
        """Revoke an API key by name."""
        await self.load()

        async with self._lock:
            to_remove = None
            for key_hash, info in self._keys.items():
                if info.name == name:
                    to_remove = key_hash
                    break

            if to_remove:
                del self._keys[to_remove]
        if to_remove:
            await self.save()
            logger.info(f"Revoked API key '{name}'")
            return True

        return False

    async def list_keys(self) -> Dict[str, Dict]:
        """List all API keys (without raw keys)."""
        await self.load()

        return {
            key_hash[:8] + "...": info.to_dict() for key_hash, info in self._keys.items()
        }

    def close(self):
        """Cleanup executor."""
        self._executor.shutdown(wait=False)


# =============================================================================
# FLOW SESSION MANAGER
# =============================================================================


class FlowSessionManager:
    """
    Thread-safe flow session management with automatic cleanup.

    Features:
    - Session lifecycle management
    - Automatic expiration cleanup
    - Max session limit enforcement
    """

    def __init__(self, max_sessions: int = 100, timeout: int = 3600):
        self._sessions: Dict[str, FlowSession] = {}
        self._lock = asyncio.Lock()
        self._max_sessions = max_sessions
        self._timeout = timeout
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start_cleanup(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup(self):
        """Stop cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self):
        """Background cleanup every 5 minutes."""
        while True:
            try:
                await asyncio.sleep(300)
                count = await self.cleanup_expired()
                if count > 0:
                    logger.info(f"Cleaned up {count} expired flow sessions")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def create(
        self, flow_name: str, session_id: Optional[str] = None
    ) -> FlowSession:
        """Create a new flow session."""
        if session_id is None:
            session_id = f"flow_{uuid.uuid4().hex[:12]}"

        async with self._lock:
            # Enforce max sessions
            if len(self._sessions) >= self._max_sessions:
                # Remove oldest session
                oldest_id = min(
                    self._sessions.keys(), key=lambda k: self._sessions[k].last_activity
                )
                del self._sessions[oldest_id]
                logger.info(f"Removed oldest session {oldest_id} to make room")

            session = FlowSession(
                session_id=session_id,
                flow_name=flow_name,
                created=time.time(),
                last_activity=time.time(),
                state=FlowState.CREATED,
                context={},
                history=[],
            )

            self._sessions[session_id] = session

        logger.debug(f"Created flow session {session_id} for {flow_name}")
        return session

    async def get(self, session_id: str) -> Optional[FlowSession]:
        """Get session by ID."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.update_activity()
            return session

    async def update(
        self,
        session_id: str,
        state: Optional[FlowState] = None,
        context: Optional[Dict] = None,
        history_entry: Optional[str] = None,
    ) -> bool:
        """Update session state."""
        async with self._lock:
            if session_id not in self._sessions:
                return False

            session = self._sessions[session_id]
            session.update_activity()

            if state is not None:
                session.state = state
            if context is not None:
                session.context.update(context)
            if history_entry is not None:
                session.history.append(history_entry)

            return True

    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    async def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        async with self._lock:
            expired = [
                sid
                for sid, session in self._sessions.items()
                if session.is_expired(self._timeout)
            ]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    async def list_sessions(self) -> List[Dict]:
        """List all active sessions."""
        async with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "flow_name": s.flow_name,
                    "state": s.state.value,
                    "created": s.created,
                    "last_activity": s.last_activity,
                }
                for s in self._sessions.values()
            ]

    @property
    def count(self) -> int:
        return len(self._sessions)


# =============================================================================
# CACHE MANAGER
# =============================================================================


class CacheManager:
    """
    Thread-safe LRU cache for query results.

    Features:
    - TTL-based expiration
    - Size limit with LRU eviction
    - Hash-based keys
    """

    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(data: Any) -> str:
        """Generate cache key from data."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:16]

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        async with self._lock:
            # Enforce size limit
            if len(self._cache) >= self._max_size:
                # Remove oldest entry
                oldest_key = min(
                    self._cache.keys(), key=lambda k: self._cache[k].timestamp
                )
                del self._cache[oldest_key]

            self._cache[key] = CacheEntry(
                key=key, value=value, timestamp=time.time(), ttl=ttl or self._default_ttl
            )

    async def invalidate(self, key: str) -> bool:
        """Remove entry from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear entire cache."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1),
        }


# =============================================================================
# PYTHON CONTEXT MANAGER
# =============================================================================


class PythonContextManager:
    """
    Manages persistent Python execution state.

    Features:
    - Persistent globals across calls
    - ToolBox app injection
    - Safe state reset
    """

    def __init__(self):
        self._globals: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._execution_count = 0

    async def get_context(self, app: Any) -> Dict[str, Any]:
        """Get execution context with app injected."""
        async with self._lock:
            if "app" not in self._globals:
                self._globals["app"] = app
                self._globals["tb"] = app
                self._globals["__builtins__"] = __builtins__

            return self._globals.copy()

    async def update_context(self, new_vars: Dict[str, Any]) -> None:
        """Update context with new variables."""
        async with self._lock:
            # Filter out non-serializable system vars
            for key, value in new_vars.items():
                if not key.startswith("_"):
                    self._globals[key] = value

    async def reset(self) -> None:
        """Reset execution context."""
        async with self._lock:
            # Keep only the app reference
            app = self._globals.get("app")
            self._globals.clear()
            if app:
                self._globals["app"] = app
                self._globals["tb"] = app
                self._globals["__builtins__"] = __builtins__

    async def increment_count(self) -> int:
        """Increment and return execution count."""
        async with self._lock:
            self._execution_count += 1
            return self._execution_count

    @property
    def execution_count(self) -> int:
        return self._execution_count


# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================


class PerformanceTracker:
    """
    Thread-safe performance metrics tracking.
    """

    def __init__(self):
        self._metrics = PerformanceMetrics()
        self._lock = asyncio.Lock()

    async def record(
        self, response_time: float, cached: bool = False, error: bool = False
    ) -> None:
        """Record a request."""
        async with self._lock:
            self._metrics.record_request(response_time, cached, error)

    async def set_init_time(self, init_time: float) -> None:
        """Set initialization time."""
        async with self._lock:
            self._metrics.init_time = init_time

    @property
    def metrics(self) -> PerformanceMetrics:
        return self._metrics

    def to_dict(self) -> Dict[str, Any]:
        m = self._metrics
        return {
            "requests_handled": m.requests_handled,
            "avg_response_time": f"{m.avg_response_time:.3f}s",
            "cache_hit_rate": f"{m.cache_hit_rate:.1%}",
            "errors": m.errors,
            "init_time": f"{m.init_time:.2f}s",
        }
