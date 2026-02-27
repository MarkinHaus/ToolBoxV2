"""
FileHandler V2 - Unified Storage with UserDataAPI Integration
=============================================================

Modernized FileHandler that supports both .config and .data files
with automatic backend selection based on filename conventions.

Features:
- Backward compatible API with original FileHandler
- Automatic scope detection from filename
- UserDataAPI integration for cloud storage
- Both sync and async APIs
- Configurable backend selection

Filename Conventions for .data files:
- private.data    → USER_PRIVATE (encrypted, local + cloud sync)
- public.data     → USER_PUBLIC (others can read)
- shared.data     → PUBLIC_RW (everyone read/write)
- server.data     → SERVER_SCOPE
- *.data          → MOD_DATA (default, mod-specific)

For .config files:
- Always uses local storage with optional encryption
- Syncs with Manifest system

Author: ToolBoxV2
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

# Conditional imports for type checking
if TYPE_CHECKING:
    from toolboxv2.utils.system.types import RequestData, Session


# =============================================================================
# Enums and Constants
# =============================================================================


class StorageScope(str, Enum):
    """Storage scope for data files."""
    USER_PRIVATE = "user_private"      # Only user, encrypted, cloud sync
    USER_PUBLIC = "user_public"        # User owns, others can read
    PUBLIC_READ = "public_read"        # Admin writes, all read
    PUBLIC_RW = "public_rw"            # Everyone read/write
    SERVER_SCOPE = "server"            # Server-specific data
    MOD_DATA = "mod_data"              # Module-specific (default)
    CONFIG = "config"                  # Configuration files


class StorageBackend(str, Enum):
    """Storage backend type."""
    LOCAL = "local"          # Local JSON files
    USER_DATA_API = "cloud"  # UserDataAPI (MinIO/Redis)
    AUTO = "auto"            # Auto-detect from manifest


# Filename patterns for scope detection
SCOPE_PATTERNS: Dict[str, StorageScope] = {
    r"^private[\._]": StorageScope.USER_PRIVATE,
    r"^public[\._]": StorageScope.USER_PUBLIC,
    r"^shared[\._]": StorageScope.PUBLIC_RW,
    r"^server[\._]": StorageScope.SERVER_SCOPE,
    r"^global[\._]": StorageScope.PUBLIC_READ,
}


# =============================================================================
# User Context
# =============================================================================


@dataclass
class UserContext:
    """
    User context for storage operations.

    Can be created from:
    - RequestData object
    - Session object
    - Direct user_id
    - Anonymous (system context)
    """
    user_id: str = "system"
    session_id: str = ""
    cloudm_user_id: str = ""
    is_authenticated: bool = False
    is_admin: bool = False
    roles: List[str] = field(default_factory=list)

    @classmethod
    def from_request(cls, request: "RequestData") -> "UserContext":
        """Create context from RequestData."""
        session = request.session
        return cls(
            user_id=session.user_id or session.user_name or "anonymous",
            session_id=session.session_id or session.SiID,
            cloudm_user_id=session.cloudm_user_id,
            is_authenticated=session.is_authenticated,
            is_admin=session.spec == "admin" or int(session.level or 0) >= 10,
            roles=[session.spec] if session.spec else [],
        )

    @classmethod
    def from_session(cls, session: "Session") -> "UserContext":
        """Create context from Session object."""
        return cls(
            user_id=session.user_id or session.user_name or "anonymous",
            session_id=session.session_id or session.SiID,
            cloudm_user_id=getattr(session, 'cloudm_user_id', ''),
            is_authenticated=getattr(session, 'is_authenticated', False),
            is_admin=session.spec == "admin" or int(session.level or 0) >= 10,
            roles=[session.spec] if session.spec else [],
        )

    @classmethod
    def system(cls) -> "UserContext":
        """Create system/server context."""
        return cls(
            user_id="system",
            is_authenticated=True,
            is_admin=True,
            roles=["system"],
        )

    @classmethod
    def anonymous(cls) -> "UserContext":
        """Create anonymous context."""
        return cls(
            user_id="anonymous",
            is_authenticated=False,
        )


# Global context holder (set via App.set_context)
_current_context: Optional[UserContext] = None


def set_current_context(ctx: Union[UserContext, "RequestData", "Session", None]):
    """
    Set the current user context globally.

    Can be called with:
    - UserContext object
    - RequestData object
    - Session object
    - None (clears context)
    """
    global _current_context

    if ctx is None:
        _current_context = None
    elif isinstance(ctx, UserContext) or (hasattr(ctx, 'user_id') and hasattr(ctx, 'session_id')):
        _current_context = ctx
    elif hasattr(ctx, 'session'):  # RequestData
        _current_context = UserContext.from_request(ctx)
    elif hasattr(ctx, 'user_id') or hasattr(ctx, 'SiID'):  # Session
        _current_context = UserContext.from_session(ctx)
    else:
        raise TypeError(f"Cannot create UserContext from {type(ctx)}")


def get_current_context() -> UserContext:
    """Get current user context or system context if not set."""
    return _current_context or UserContext.system()


# =============================================================================
# Storage Backend Interface
# =============================================================================


class StorageBackendInterface(ABC):
    """Abstract interface for storage backends."""

    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """Load data for key (sync)."""
        pass

    @abstractmethod
    async def aload(self, key: str) -> Optional[Any]:
        """Load data for key (async)."""
        pass

    @abstractmethod
    def save(self, key: str, value: Any) -> bool:
        """Save data for key (sync)."""
        pass

    @abstractmethod
    async def asave(self, key: str, value: Any) -> bool:
        """Save data for key (async)."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data for key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix."""
        pass


# =============================================================================
# Local Storage Backend
# =============================================================================


class LocalStorageBackend(StorageBackendInterface):
    """
    Local file-based storage backend.

    Stores data as JSON files in the configured directory.
    Compatible with legacy FileHandler format.
    """

    def __init__(
        self,
        base_path: Path,
        encrypt: bool = False,
        encryption_key: Optional[str] = None
    ):
        self.base_path = Path(base_path)
        self.encrypt = encrypt
        self._encryption_key = encryption_key
        self._encoder: Optional[Any] = None

        # Initialize encryption if needed
        if encrypt:
            self._init_encryption()

    def _init_encryption(self):
        """Initialize encryption codec."""
        try:
            from toolboxv2.utils.security.cryp import Code
            self._encoder = Code()
        except ImportError:
            self.encrypt = False

    def _get_file_path(self, key: str) -> Path:
        """Get full file path for a key."""
        # Sanitize key for filesystem
        safe_key = re.sub(r'[^\w\-_.]', '_', key)
        return self.base_path / f"{safe_key}.json"

    def _encode(self, value: Any) -> str:
        """Encode value, optionally with encryption."""
        json_str = json.dumps(value, ensure_ascii=False, default=str)
        if self.encrypt and self._encoder:
            return self._encoder.encode_code(json_str)
        return json_str

    def _decode(self, data: str) -> Any:
        """Decode value, optionally with decryption."""
        if self.encrypt and self._encoder:
            data = self._encoder.decode_code(data)
        return json.loads(data)

    def load(self, key: str) -> Optional[Any]:
        """Load data from local file."""
        file_path = self._get_file_path(key)

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._decode(content)
        except (json.JSONDecodeError, IOError) as e:
            return None

    async def aload(self, key: str) -> Optional[Any]:
        """Async load (runs sync in executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, key)

    def save(self, key: str, value: Any) -> bool:
        """Save data to local file."""
        file_path = self._get_file_path(key)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content = self._encode(value)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except IOError:
            return False

    async def asave(self, key: str, value: Any) -> bool:
        """Async save (runs sync in executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.save, key, value)

    def delete(self, key: str) -> bool:
        """Delete local file."""
        file_path = self._get_file_path(key)
        try:
            if file_path.exists():
                file_path.unlink()
            return True
        except IOError:
            return False

    def exists(self, key: str) -> bool:
        """Check if file exists."""
        return self._get_file_path(key).exists()

    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys in storage."""
        if not self.base_path.exists():
            return []

        keys = []
        for file in self.base_path.glob("*.json"):
            key = file.stem
            if prefix and not key.startswith(prefix):
                continue
            keys.append(key)
        return keys


# =============================================================================
# UserDataAPI Storage Backend
# =============================================================================


class UserDataAPIBackend(StorageBackendInterface):
    """
    UserDataAPI-based storage backend.

    Integrates with the scoped storage system for cloud sync,
    encryption, and multi-user support.
    """

    def __init__(
        self,
        scope: StorageScope,
        mod_name: str,
        user_context: Optional[UserContext] = None,
        minio_endpoint: Optional[str] = None,
        minio_access_key: Optional[str] = None,
        minio_secret_key: Optional[str] = None,
    ):
        self.scope = scope
        self.mod_name = mod_name
        self._user_context = user_context
        self._minio_config = {
            'endpoint': minio_endpoint,
            'access_key': minio_access_key,
            'secret_key': minio_secret_key,
        }
        self._storage = None
        self._initialized = False

    @property
    def user_context(self) -> UserContext:
        """Get user context (from param or global)."""
        return self._user_context or get_current_context()

    def _get_storage(self):
        """Get or create storage instance."""
        if self._storage is None:
            try:
                from scoped_storage import ScopedBlobStorage, UserContext as ScopedUserContext

                # Convert our UserContext to scoped_storage UserContext
                scoped_ctx = ScopedUserContext(
                    user_id=self.user_context.user_id,
                    session_id=self.user_context.session_id,
                    is_admin=self.user_context.is_admin,
                    roles=self.user_context.roles,
                )

                self._storage = ScopedBlobStorage(
                    user_context=scoped_ctx,
                    minio_endpoint=self._minio_config.get('endpoint'),
                    minio_access_key=self._minio_config.get('access_key'),
                    minio_secret_key=self._minio_config.get('secret_key'),
                )
                self._initialized = True
            except ImportError:
                # Fallback to local storage if UserDataAPI not available
                self._storage = None

        return self._storage

    def _scope_to_api_scope(self):
        """Convert our scope to UserDataAPI scope."""
        try:
            from scoped_storage import Scope
            mapping = {
                StorageScope.USER_PRIVATE: Scope.USER_PRIVATE,
                StorageScope.USER_PUBLIC: Scope.USER_PUBLIC,
                StorageScope.PUBLIC_READ: Scope.PUBLIC_READ,
                StorageScope.PUBLIC_RW: Scope.PUBLIC_RW,
                StorageScope.SERVER_SCOPE: Scope.SERVER_SCOPE,
                StorageScope.MOD_DATA: Scope.MOD_DATA,
            }
            return mapping.get(self.scope, Scope.MOD_DATA)
        except ImportError:
            return None

    def _build_path(self, key: str) -> str:
        """Build storage path for key."""
        return f"{self.mod_name}/{key}.json"

    def load(self, key: str) -> Optional[Any]:
        """Load data via UserDataAPI."""
        storage = self._get_storage()
        if not storage:
            return None

        try:
            path = self._build_path(key)
            scope = self._scope_to_api_scope()

            data = storage.read(
                path=path,
                scope=scope,
                mod_name=self.mod_name if self.scope == StorageScope.MOD_DATA else None
            )

            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception:
            return None

    async def aload(self, key: str) -> Optional[Any]:
        """Async load via UserDataAPI."""
        # UserDataAPI is sync, run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, key)

    def save(self, key: str, value: Any) -> bool:
        """Save data via UserDataAPI."""
        storage = self._get_storage()
        if not storage:
            return False

        try:
            path = self._build_path(key)
            scope = self._scope_to_api_scope()
            data = json.dumps(value, ensure_ascii=False, default=str).encode('utf-8')

            storage.write(
                path=path,
                data=data,
                scope=scope,
                mod_name=self.mod_name if self.scope == StorageScope.MOD_DATA else None
            )
            return True
        except Exception:
            return False

    async def asave(self, key: str, value: Any) -> bool:
        """Async save via UserDataAPI."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.save, key, value)

    def delete(self, key: str) -> bool:
        """Delete via UserDataAPI."""
        storage = self._get_storage()
        if not storage:
            return False

        try:
            path = self._build_path(key)
            scope = self._scope_to_api_scope()

            storage.delete(
                path=path,
                scope=scope,
                mod_name=self.mod_name if self.scope == StorageScope.MOD_DATA else None
            )
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in UserDataAPI."""
        return self.load(key) is not None

    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys via UserDataAPI."""
        storage = self._get_storage()
        if not storage:
            return []

        try:
            scope = self._scope_to_api_scope()
            full_prefix = f"{self.mod_name}/{prefix}"

            paths = storage.list(
                prefix=full_prefix,
                scope=scope,
                mod_name=self.mod_name if self.scope == StorageScope.MOD_DATA else None
            )

            # Extract key names from paths
            keys = []
            for path in paths:
                # Remove mod_name prefix and .json suffix
                key = path.replace(f"{self.mod_name}/", "").replace(".json", "")
                keys.append(key)
            return keys
        except Exception:
            return []


# =============================================================================
# FileHandler V2 Main Class
# =============================================================================


class FileHandlerV2:
    """
    Unified FileHandler with UserDataAPI integration.

    Backward compatible with legacy FileHandler while adding:
    - Automatic scope detection from filename
    - UserDataAPI backend for cloud storage
    - Both sync and async APIs
    - Configurable encryption

    Usage:
        # Basic usage (auto-detects backend and scope)
        fh = FileHandlerV2("mymod.data", name="CloudM.MyMod")
        fh.load_file_handler()
        value = fh.get_file_handler("mykey")
        fh.add_to_save_file_handler("mykey", "myvalue")
        fh.save_file_handler()

        # With explicit scope
        fh = FileHandlerV2("private.data", name="MyMod", scope="user_private")

        # With explicit backend
        fh = FileHandlerV2("cache.data", name="MyMod", backend="local")

        # With user context
        fh = FileHandlerV2("user.data", name="MyMod", request=request)

        # Async API
        await fh.aload_file_handler()
        value = await fh.aget_file_handler("key")
    """

    def __init__(
        self,
        filename: str,
        name: str = "mainTool",
        keys: Optional[Dict[str, str]] = None,
        defaults: Optional[Dict[str, Any]] = None,
        # New V2 parameters
        scope: Optional[Union[str, StorageScope]] = None,
        backend: Union[str, StorageBackend] = StorageBackend.AUTO,
        encrypt: Optional[bool] = None,
        request: Optional["RequestData"] = None,
        user_context: Optional[UserContext] = None,
        base_path: Optional[Path] = None,
    ):
        """
        Initialize FileHandler V2.

        Args:
            filename: File name ending in .config or .data
            name: Module/tool name for namespacing
            keys: Key mapping (legacy compatibility)
            defaults: Default values (legacy compatibility)
            scope: Storage scope (auto-detected from filename if not specified)
            backend: Storage backend ('auto', 'local', 'cloud')
            encrypt: Enable encryption (auto for .config and USER_PRIVATE)
            request: RequestData for user context
            user_context: Direct user context
            base_path: Base path for local storage
        """
        # Validate filename
        if not (filename.endswith(".config") or filename.endswith(".data")):
            raise ValueError(f"Filename must end with .config or .data: {filename}")

        self.filename = filename
        self.name = name.replace('.', '-')
        self.is_config = filename.endswith(".config")

        # Determine scope
        self.scope = self._resolve_scope(scope, filename)

        # Determine backend
        self.backend_type = self._resolve_backend(backend)

        # Determine encryption
        if encrypt is None:
            # Auto-encrypt for .config and USER_PRIVATE
            encrypt = self.is_config or self.scope == StorageScope.USER_PRIVATE
        self.encrypt = encrypt

        # User context
        if user_context:
            self._user_context = user_context
        elif request:
            self._user_context = UserContext.from_request(request)
        else:
            self._user_context = None  # Will use global context

        # Storage paths
        self.base_path = base_path or self._get_default_base_path()

        # Initialize storage
        self._data: Dict[str, Any] = {}
        self._key_mapper: Dict[str, str] = {}
        self._backend: Optional[StorageBackendInterface] = None

        # Legacy compatibility
        self.file_handler_save: Dict[str, Any] = {}
        self.file_handler_load: Dict[str, Any] = {}
        self.file_handler_key_mapper: Dict[str, str] = {}
        self.file_handler_filename = filename
        self.file_handler_storage = None
        self.file_handler_max_loaded_index_ = 0
        self.file_handler_file_prefix = f".{filename.split('.')[1]}/{self.name}/"

        # Set defaults
        if keys or defaults:
            self.set_defaults_keys_file_handler(keys or {}, defaults or {})

    def _resolve_scope(
        self,
        scope: Optional[Union[str, StorageScope]],
        filename: str
    ) -> StorageScope:
        """Resolve storage scope from parameter or filename."""
        # Config files have their own scope
        if filename.endswith(".config"):
            return StorageScope.CONFIG

        # Explicit scope
        if scope:
            if isinstance(scope, str):
                try:
                    return StorageScope(scope)
                except ValueError:
                    pass
            elif isinstance(scope, StorageScope):
                return scope

        # Detect from filename
        basename = os.path.basename(filename).lower()
        for pattern, detected_scope in SCOPE_PATTERNS.items():
            if re.match(pattern, basename):
                return detected_scope

        # Default to MOD_DATA
        return StorageScope.MOD_DATA

    def _resolve_backend(
        self,
        backend: Union[str, StorageBackend]
    ) -> StorageBackend:
        """Resolve storage backend."""
        if isinstance(backend, str):
            try:
                return StorageBackend(backend)
            except ValueError:
                return StorageBackend.AUTO
        return backend

    def _get_default_base_path(self) -> Path:
        """Get default base path for local storage."""
        try:
            from toolboxv2 import tb_root_dir
            return tb_root_dir
        except ImportError:
            return Path.cwd()

    def _get_backend(self) -> StorageBackendInterface:
        """Get or create the storage backend."""
        if self._backend is not None:
            return self._backend

        # Determine which backend to use
        use_cloud = False

        if self.backend_type == StorageBackend.AUTO:
            # Auto-detect from manifest
            try:
                from toolboxv2.utils.manifest.loader import ManifestLoader
                loader = ManifestLoader(self.base_path)
                if loader.exists():
                    manifest = loader.load()
                    # Use cloud for CB (CLUSTER_BLOB) mode
                    use_cloud = manifest.database.mode.value == "CB"
            except Exception:
                pass

            # Config files always local
            if self.is_config:
                use_cloud = False

        elif self.backend_type == StorageBackend.USER_DATA_API:
            use_cloud = True
        else:
            use_cloud = False

        # Create backend
        if use_cloud and not self.is_config:
            # Try UserDataAPI backend
            try:
                self._backend = UserDataAPIBackend(
                    scope=self.scope,
                    mod_name=self.name,
                    user_context=self._user_context,
                )
            except Exception:
                use_cloud = False

        if not use_cloud or self._backend is None:
            # Use local backend
            local_path = self.base_path / self.file_handler_file_prefix
            self._backend = LocalStorageBackend(
                base_path=local_path,
                encrypt=self.encrypt,
            )

        return self._backend

    # =========================================================================
    # Legacy API (Sync) - Backward Compatible
    # =========================================================================

    def load_file_handler(self) -> "FileHandlerV2":
        """Load all data from storage (legacy API)."""
        backend = self._get_backend()

        # Load main data store
        data = backend.load("_main")
        if data and isinstance(data, dict):
            self._data = data
            self.file_handler_load = data.copy()
            self.file_handler_save = data.copy()

        return self

    def save_file_handler(self) -> "FileHandlerV2":
        """Save all data to storage (legacy API)."""
        backend = self._get_backend()

        # Merge load data into main data
        self._data.update(self.file_handler_load)

        # Save main data store
        backend.save("_main", self._data)

        # Update save cache
        self.file_handler_save = self._data.copy()

        return self

    def get_file_handler(self, key: str, default: Any = None) -> Any:
        """Get value for key (legacy API)."""
        # Check key mapper
        if key in self.file_handler_key_mapper:
            key = self.file_handler_key_mapper[key]

        # Try in-memory first
        if key in self.file_handler_load:
            return self.file_handler_load[key]

        if key in self._data:
            return self._data[key]

        return default

    def add_to_save_file_handler(self, key: str, value: Any) -> bool:
        """Add/update value for key (legacy API)."""
        # Map key if needed
        if key in self.file_handler_key_mapper:
            self._data[key] = value
            self.file_handler_load[key] = value
            key = self.file_handler_key_mapper[key]

        self._data[key] = value
        self.file_handler_load[key] = value
        self.file_handler_save[key] = value

        return True

    def remove_key_file_handler(self, key: str) -> None:
        """Remove a key (legacy API)."""
        if key in self.file_handler_key_mapper:
            key = self.file_handler_key_mapper[key]

        self._data.pop(key, None)
        self.file_handler_load.pop(key, None)
        self.file_handler_save.pop(key, None)

    def set_defaults_keys_file_handler(
        self,
        keys: Dict[str, str],
        defaults: Dict[str, Any]
    ) -> None:
        """Set key mappings and default values (legacy API)."""
        for short_key, full_key in keys.items():
            self.file_handler_key_mapper[short_key] = full_key
            self.file_handler_key_mapper[full_key] = short_key

            if short_key in defaults:
                # Store under both keys for easier access
                self._data[full_key] = defaults[short_key]
                self._data[short_key] = defaults[short_key]
                self.file_handler_load[full_key] = defaults[short_key]
                self.file_handler_load[short_key] = defaults[short_key]

    def delete_file(self) -> None:
        """Delete the storage file (legacy API)."""
        backend = self._get_backend()
        backend.delete("_main")
        self._data.clear()
        self.file_handler_load.clear()
        self.file_handler_save.clear()

    # =========================================================================
    # New V2 API (Sync)
    # =========================================================================

    def get(self, key: str, default: Any = None) -> Any:
        """Get value for key."""
        return self.get_file_handler(key, default)

    def set(self, key: str, value: Any) -> bool:
        """Set value for key."""
        return self.add_to_save_file_handler(key, value)

    def delete(self, key: str) -> None:
        """Delete a key."""
        self.remove_key_file_handler(key)

    def load(self) -> "FileHandlerV2":
        """Load data from storage."""
        return self.load_file_handler()

    def save(self) -> "FileHandlerV2":
        """Save data to storage."""
        return self.save_file_handler()

    def keys(self) -> List[str]:
        """Get all keys."""
        return list(self._data.keys())

    def list_keys(self) -> List[str]:
        """Get all keys (alias for keys(), avoids conflict with self.keys attribute)."""
        return list(self._data.keys())

    def items(self) -> List[tuple]:
        """Get all key-value pairs."""
        return list(self._data.items())

    def list_items(self) -> List[tuple]:
        """Get all key-value pairs (alias for items())."""
        return list(self._data.items())

    def to_dict(self) -> Dict[str, Any]:
        """Get all data as dictionary."""
        return self._data.copy()

    def update(self, data: Dict[str, Any]) -> "FileHandlerV2":
        """Update multiple values."""
        for key, value in data.items():
            self.set(key, value)
        return self

    # =========================================================================
    # Async API
    # =========================================================================

    async def aload_file_handler(self) -> "FileHandlerV2":
        """Load all data from storage (async)."""
        backend = self._get_backend()

        data = await backend.aload("_main")
        if data and isinstance(data, dict):
            self._data = data
            self.file_handler_load = data.copy()
            self.file_handler_save = data.copy()

        return self

    async def asave_file_handler(self) -> "FileHandlerV2":
        """Save all data to storage (async)."""
        backend = self._get_backend()

        self._data.update(self.file_handler_load)
        await backend.asave("_main", self._data)
        self.file_handler_save = self._data.copy()

        return self

    async def aget_file_handler(self, key: str, default: Any = None) -> Any:
        """Get value for key (async)."""
        # For in-memory data, no need for async
        return self.get_file_handler(key, default)

    async def aget(self, key: str, default: Any = None) -> Any:
        """Get value (async)."""
        return self.get(key, default)

    async def aset(self, key: str, value: Any) -> bool:
        """Set value (async)."""
        return self.set(key, value)

    async def aload(self) -> "FileHandlerV2":
        """Load data (async)."""
        return await self.aload_file_handler()

    async def asave(self) -> "FileHandlerV2":
        """Save data (async)."""
        return await self.asave_file_handler()

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def __enter__(self) -> "FileHandlerV2":
        """Enter context manager (loads data)."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager (saves data)."""
        if exc_type is None:
            self.save()

    async def __aenter__(self) -> "FileHandlerV2":
        """Enter async context manager."""
        await self.aload()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        if exc_type is None:
            await self.asave()

    # =========================================================================
    # Dict-like Access
    # =========================================================================

    def __getitem__(self, key: str) -> Any:
        """Get item via indexing."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item via indexing."""
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete item via indexing."""
        self.delete(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data

    def __len__(self) -> int:
        """Get number of keys."""
        if not hasattr(self, "_data"):
            return 0
        return len(self._data)

    def __iter__(self):
        """Iterate over keys."""
        return iter(self._data)


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_config_handler(
    name: str,
    filename: str = "settings.config",
    defaults: Optional[Dict[str, Any]] = None,
) -> FileHandlerV2:
    """
    Create a FileHandler for configuration data.

    Always uses local storage with encryption.
    """
    return FileHandlerV2(
        filename=filename,
        name=name,
        defaults=defaults,
        backend=StorageBackend.LOCAL,
        encrypt=True,
    )


def create_data_handler(
    name: str,
    scope: Union[str, StorageScope] = StorageScope.MOD_DATA,
    request: Optional["RequestData"] = None,
    backend: Union[str, StorageBackend] = StorageBackend.AUTO,
) -> FileHandlerV2:
    """
    Create a FileHandler for user/mod data.

    Uses UserDataAPI when available and configured.
    """
    # Determine filename from scope
    scope_filenames = {
        StorageScope.USER_PRIVATE: "private.data",
        StorageScope.USER_PUBLIC: "public.data",
        StorageScope.PUBLIC_RW: "shared.data",
        StorageScope.SERVER_SCOPE: "server.data",
        StorageScope.MOD_DATA: "mod.data",
    }

    if isinstance(scope, str):
        scope = StorageScope(scope)

    filename = scope_filenames.get(scope, "mod.data")

    return FileHandlerV2(
        filename=filename,
        name=name,
        scope=scope,
        backend=backend,
        request=request,
    )


# =============================================================================
# Legacy Compatibility Alias
# =============================================================================

# Allow drop-in replacement
FileHandler = FileHandlerV2


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    "FileHandlerV2",
    "FileHandler",  # Legacy alias

    # Enums
    "StorageScope",
    "StorageBackend",

    # Context
    "UserContext",
    "set_current_context",
    "get_current_context",

    # Factory functions
    "create_config_handler",
    "create_data_handler",

    # Backend interfaces (for extension)
    "StorageBackendInterface",
    "LocalStorageBackend",
    "UserDataAPIBackend",
]
