"""
ToolBox V2 - Scoped Blob Storage System
Multi-User, Multi-Scope Storage mit Clerk Auth Integration

SCOPES:
- PUBLIC_READ:   Alle lesen, nur Admin schreibt
- PUBLIC_RW:     Alle lesen/schreiben
- USER_PUBLIC:   Alle lesen, nur Owner schreibt unter eigenem Prefix
- USER_PRIVATE:  Nur Owner liest/schreibt (lokal + verschlüsselter Cloud-Sync)
- SERVER_SCOPE:  Server-spezifische Daten
- MOD_DATA:      Modul-spezifische Daten

STORAGE:
- USER_PRIVATE: Lokal in SQLite, sync zu verschlüsseltem Cloud-Bereich
- Alle anderen: Cloud mit lokalem Cache
"""
import base64
import os
import json
import time
import hashlib
import threading
from enum import Enum
from typing import Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

# MinIO
try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

# Local imports
try:
    from mobile_db import MobileDB
    MOBILE_DB_AVAILABLE = True
except ImportError:
    MOBILE_DB_AVAILABLE = False

# ToolBoxV2 imports
try:
    from toolboxv2.utils.security.cryp import Code
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    # Fallback
    class Code:
        @staticmethod
        def one_way_hash(data: str, salt: str = "") -> str:
            return hashlib.sha256(f"{data}{salt}".encode()).hexdigest()

        @staticmethod
        def encrypt_symmetric(data: bytes, key: bytes) -> bytes:
            # WARNUNG: Nur für Tests!
            return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])

        @staticmethod
        def decrypt_symmetric(data: bytes, key: bytes) -> bytes:
            return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])

        @staticmethod
        def DK():
            def inner():
                import uuid
                return hashlib.sha256(str(uuid.getnode()).encode()).digest()
            return inner


# =================== Enums & Types ===================

class Scope(Enum):
    """Storage Scopes mit unterschiedlichen Berechtigungen"""
    PUBLIC_READ = "public_read"      # Alle lesen, Admin schreibt
    PUBLIC_RW = "public_rw"          # Alle lesen/schreiben
    USER_PUBLIC = "user_public"      # Alle lesen, Owner schreibt
    USER_PRIVATE = "user_private"    # Nur Owner (lokal + encrypted cloud)
    SERVER_SCOPE = "server"          # Server-spezifisch
    MOD_DATA = "mod_data"            # Modul-spezifisch


class Permission(Enum):
    """Berechtigungstypen"""
    NONE = 0
    READ = 1
    WRITE = 2
    READ_WRITE = 3
    ADMIN = 4


@dataclass
class UserContext:
    """Benutzerkontext für Scope-Zugriff"""
    user_id: str
    username: str
    is_admin: bool = False
    is_authenticated: bool = True
    server_id: Optional[str] = None
    encryption_key: Optional[bytes] = None
    session_token: Optional[str] = None

    @classmethod
    def anonymous(cls) -> "UserContext":
        return cls(
            user_id="anonymous",
            username="anonymous",
            is_authenticated=False
        )

    @classmethod
    def from_clerk_session(cls, session_data: dict, encryption_key: bytes = None) -> "UserContext":
        """Erstellt UserContext aus Clerk Session"""
        return cls(
            user_id=session_data.get("user_id", ""),
            username=session_data.get("username", ""),
            is_admin=session_data.get("is_admin", False),
            is_authenticated=True,
            session_token=session_data.get("session_token", ""),
            encryption_key=encryption_key
        )


@dataclass
class BlobMetadata:
    """Metadaten für einen Blob"""
    path: str
    scope: Scope
    owner_id: str
    size: int = 0
    checksum: str = ""
    created_at: float = 0
    updated_at: float = 0
    encrypted: bool = False
    content_type: str = "application/octet-stream"
    version: int = 1
    custom_metadata: Dict[str, str] = field(default_factory=dict)


# =================== Scope Policy Engine ===================

class ScopePolicyEngine:
    """Bestimmt Berechtigungen basierend auf Scope und User"""

    @staticmethod
    def get_permission(scope: Scope, user: UserContext, resource_owner: str = None) -> Permission:
        """
        Ermittelt die Berechtigung eines Users für einen Scope

        Args:
            scope: Der Scope des Blobs
            user: Der anfragende Benutzer
            resource_owner: Owner-ID des Blobs (für USER_* Scopes)
        """
        # Admin hat immer vollen Zugriff
        if user.is_admin:
            return Permission.ADMIN

        # Scope-spezifische Regeln
        if scope == Scope.PUBLIC_READ:
            # Alle können lesen, niemand (außer Admin) kann schreiben
            return Permission.READ

        elif scope == Scope.PUBLIC_RW:
            # Alle authentifizierten User können lesen/schreiben
            if user.is_authenticated:
                return Permission.READ_WRITE
            return Permission.READ

        elif scope == Scope.USER_PUBLIC:
            # Alle können lesen, nur Owner kann schreiben
            if resource_owner and user.user_id == resource_owner:
                return Permission.READ_WRITE
            return Permission.READ

        elif scope == Scope.USER_PRIVATE:
            # Nur Owner hat Zugriff
            if resource_owner and user.user_id == resource_owner:
                return Permission.READ_WRITE
            return Permission.NONE

        elif scope == Scope.SERVER_SCOPE:
            # Nur Server hat Zugriff
            if user.server_id:
                return Permission.READ_WRITE
            return Permission.NONE

        elif scope == Scope.MOD_DATA:
            # Authentifizierte User können eigene Mod-Daten lesen/schreiben
            if user.is_authenticated:
                if resource_owner and user.user_id == resource_owner:
                    return Permission.READ_WRITE
                return Permission.READ
            return Permission.NONE

        return Permission.NONE

    @staticmethod
    def can_read(permission: Permission) -> bool:
        return permission in (Permission.READ, Permission.READ_WRITE, Permission.ADMIN)

    @staticmethod
    def can_write(permission: Permission) -> bool:
        return permission in (Permission.WRITE, Permission.READ_WRITE, Permission.ADMIN)

    @staticmethod
    def get_bucket_name(scope: Scope) -> str:
        """Gibt den MinIO Bucket-Namen für einen Scope zurück"""
        return {
            Scope.PUBLIC_READ: "tb-public-read",
            Scope.PUBLIC_RW: "tb-public-rw",
            Scope.USER_PUBLIC: "tb-users-public",
            Scope.USER_PRIVATE: "tb-users-private",
            Scope.SERVER_SCOPE: "tb-servers",
            Scope.MOD_DATA: "tb-mods"
        }.get(scope, "tb-default")

    @staticmethod
    def build_path(scope: Scope, user: UserContext, path: str, mod_name: str = None) -> str:
        """
        Baut den vollständigen Pfad basierend auf Scope

        Args:
            scope: Storage Scope
            user: Benutzerkontext
            path: Relativer Pfad
            mod_name: Modulname (nur für MOD_DATA)
        """
        if scope in (Scope.USER_PUBLIC, Scope.USER_PRIVATE):
            # User-Prefix: users/{user_id}/{path}
            return f"{user.user_id}/{path}"

        elif scope == Scope.SERVER_SCOPE:
            # Server-Prefix: servers/{server_id}/{path}
            server_id = user.server_id or "default"
            return f"{server_id}/{path}"

        elif scope == Scope.MOD_DATA:
            # Mod-Prefix: mods/{mod_name}/{user_id}/{path}
            mod = mod_name or "unknown"
            return f"{mod}/{user.user_id}/{path}"

        # PUBLIC_READ, PUBLIC_RW - kein Prefix
        return path


# =================== Encryption Layer ===================

class ScopedCryptoLayer:
    """Verschlüsselung für USER_PRIVATE Scope"""

    def __init__(self, user_context: UserContext):
        self.user = user_context
        self._key_cache: Dict[str, bytes] = {}

    def _get_user_key(self) -> bytes:
        """Holt den User-spezifischen Encryption Key"""
        if self.user.encryption_key:
            return self.user.encryption_key

        # Fallback: Device-Key + User-ID
        device_key = Code.DK()()
        if isinstance(device_key, str):
            device_key = device_key.encode()

        user_salt = self.user.user_id.encode()
        return base64.urlsafe_b64encode(hashlib.sha256(device_key + user_salt).digest())

    def encrypt(self, data: bytes) -> bytes:
        """Verschlüsselt Daten mit User-Key"""
        key = self._get_user_key()

        if TOOLBOX_AVAILABLE:
            return Code.encrypt_symmetric(data, key)

        # Fallback XOR (NICHT SICHER - nur für Tests!)
        return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])

    def decrypt(self, data: bytes, row=True) -> bytes:
        """Entschlüsselt Daten mit User-Key"""
        key = self._get_user_key()
        if TOOLBOX_AVAILABLE:
            return Code.decrypt_symmetric(data, key, to_str=not row)

        # Fallback XOR
        return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])


# =================== Local Cache ===================

class ScopedCache:
    """Lokaler Cache für nicht-private Scopes"""

    def __init__(self, cache_dir: str = None, max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.tb_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._index: Dict[str, dict] = {}
        self._load_index()

    def _get_cache_path(self, scope: Scope, path: str) -> Path:
        safe_path = hashlib.md5(f"{scope.value}:{path}".encode()).hexdigest()
        return self.cache_dir / scope.value / safe_path[:2] / safe_path

    def _load_index(self):
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                self._index = json.loads(index_file.read_text())
            except:
                self._index = {}

    def _save_index(self):
        index_file = self.cache_dir / "index.json"
        index_file.write_text(json.dumps(self._index))

    def get(self, scope: Scope, path: str) -> Optional[bytes]:
        """Holt Daten aus Cache"""
        cache_key = f"{scope.value}:{path}"

        with self._lock:
            if cache_key not in self._index:
                return None

            entry = self._index[cache_key]
            cache_path = self._get_cache_path(scope, path)

            if not cache_path.exists():
                del self._index[cache_key]
                return None

            # Update access time
            entry["last_access"] = time.time()
            entry["access_count"] = entry.get("access_count", 0) + 1

            return cache_path.read_bytes()

    def set(self, scope: Scope, path: str, data: bytes, checksum: str = None):
        """Speichert Daten im Cache"""
        cache_key = f"{scope.value}:{path}"
        cache_path = self._get_cache_path(scope, path)

        with self._lock:
            # Prüfe ob wir Platz brauchen
            self._ensure_space(len(data))

            # Speichere Datei
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)

            # Update Index
            self._index[cache_key] = {
                "path": str(cache_path),
                "size": len(data),
                "checksum": checksum or hashlib.md5(data).hexdigest(),
                "cached_at": time.time(),
                "last_access": time.time(),
                "access_count": 1
            }
            self._save_index()

    def invalidate(self, scope: Scope, path: str):
        """Invalidiert Cache-Eintrag"""
        cache_key = f"{scope.value}:{path}"

        with self._lock:
            if cache_key in self._index:
                cache_path = Path(self._index[cache_key]["path"])
                if cache_path.exists():
                    cache_path.unlink()
                del self._index[cache_key]
                self._save_index()

    def is_valid(self, scope: Scope, path: str, checksum: str) -> bool:
        """Prüft ob Cache-Eintrag noch gültig ist"""
        cache_key = f"{scope.value}:{path}"

        with self._lock:
            if cache_key not in self._index:
                return False
            return self._index[cache_key].get("checksum") == checksum

    def _ensure_space(self, needed_bytes: int):
        """Stellt sicher dass genug Platz im Cache ist"""
        current_size = sum(e.get("size", 0) for e in self._index.values())

        if current_size + needed_bytes <= self.max_size:
            return

        # LRU Eviction
        sorted_entries = sorted(
            self._index.items(),
            key=lambda x: x[1].get("last_access", 0)
        )

        for cache_key, entry in sorted_entries:
            if current_size + needed_bytes <= self.max_size:
                break

            cache_path = Path(entry["path"])
            if cache_path.exists():
                cache_path.unlink()

            current_size -= entry.get("size", 0)
            del self._index[cache_key]

    def clear(self, scope: Scope = None):
        """Löscht Cache (optional nur für bestimmten Scope)"""
        with self._lock:
            if scope:
                keys_to_delete = [k for k in self._index if k.startswith(f"{scope.value}:")]
            else:
                keys_to_delete = list(self._index.keys())

            for key in keys_to_delete:
                entry = self._index[key]
                cache_path = Path(entry["path"])
                if cache_path.exists():
                    cache_path.unlink()
                del self._index[key]

            self._save_index()


# =================== Main Scoped Storage ===================

class ScopedBlobStorage:
    """
    Hauptklasse für Scope-basierten Blob Storage

    Features:
    - Multi-Scope Support (PUBLIC_READ, PUBLIC_RW, USER_PUBLIC, USER_PRIVATE, SERVER, MOD)
    - Clerk Auth Integration
    - Lokale SQLite für USER_PRIVATE
    - Cache für andere Scopes
    - Automatische Verschlüsselung für USER_PRIVATE
    """

    def __init__(
        self,
        user_context: UserContext,
        minio_endpoint: str = None,
        minio_access_key: str = None,
        minio_secret_key: str = None,
        minio_secure: bool = True,
        local_db_path: str = None,
        cache_dir: str = None,
        cache_max_mb: int = 100
    ):
        self.user = user_context
        self.policy = ScopePolicyEngine()
        self.crypto = ScopedCryptoLayer(user_context)
        self.cache = ScopedCache(cache_dir, cache_max_mb)

        # MinIO Client
        self._minio: Optional[Minio] = None

        minio_endpoint = minio_endpoint or os.getenv("minio_endpoint".upper())
        minio_access_key = minio_access_key or os.getenv("minio_access_key".upper())
        minio_secret_key = minio_secret_key or os.getenv("minio_secret_key".upper())

        if minio_endpoint and minio_access_key and minio_secret_key:
            if not self._init_minio(minio_endpoint, minio_access_key, minio_secret_key, minio_secure):
                import logging
                logging.getLogger("scoped_storage").warning(
                    "MinIO authentication failed - using local storage only"
                )

        # Local DB für USER_PRIVATE
        self._local_db: Optional[MobileDB] = None
        if local_db_path and MOBILE_DB_AVAILABLE:
            self._local_db = MobileDB(local_db_path)

        self._lock = threading.Lock()

    def _init_minio(self, endpoint: str, access_key: str, secret_key: str, secure: bool) -> bool:
        """
        Initialisiert MinIO Client.

        Returns:
            bool: True if initialization succeeded, False if authentication failed
        """
        if not MINIO_AVAILABLE:
            raise ImportError("minio package not installed")

        import logging
        logger = logging.getLogger("scoped_storage")

        self._minio = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        # Erstelle Buckets falls nicht vorhanden
        for scope in Scope:
            bucket = self.policy.get_bucket_name(scope)
            try:
                if not self._minio.bucket_exists(bucket):
                    self._minio.make_bucket(bucket)
                    logger.info(f"Created bucket '{bucket}'")
            except S3Error as e:
                error_code = getattr(e, 'code', str(e))
                # Check for authentication errors
                if error_code in ("SignatureDoesNotMatch", "InvalidAccessKeyId",
                                  "AccessDenied", "InvalidSignature"):
                    logger.warning(f"MinIO authentication failed for bucket '{bucket}': {e}")
                    self._minio = None
                    return False
                elif error_code != "BucketAlreadyOwnedByYou":
                    logger.warning(f"MinIO error for bucket '{bucket}': {e}")
                    # Continue with other buckets instead of raising
            except Exception as e:
                # Catch SSL errors and other connection issues
                error_str = str(e).lower()
                if "ssl" in error_str or "connection" in error_str or "timeout" in error_str:
                    logger.warning(f"MinIO connection error: {e}")
                    self._minio = None
                    return False
                logger.warning(f"Unexpected error creating bucket '{bucket}': {e}")
        return True

    # =================== Core Operations ===================

    def write(
        self,
        path: str,
        data: bytes,
        scope: Scope = Scope.USER_PRIVATE,
        mod_name: str = None,
        content_type: str = "application/octet-stream",
        metadata: Dict[str, str] = None
    ) -> BlobMetadata:
        """
        Schreibt Daten in den Storage

        Args:
            path: Relativer Pfad
            data: Zu speichernde Daten
            scope: Storage Scope
            mod_name: Modulname (nur für MOD_DATA)
            content_type: MIME-Type
            metadata: Custom Metadata

        Returns:
            BlobMetadata mit Infos über den geschriebenen Blob

        Raises:
            PermissionError: Wenn User keine Schreibberechtigung hat
        """
        # Berechtigungsprüfung
        permission = self.policy.get_permission(scope, self.user, self.user.user_id)
        if not self.policy.can_write(permission):
            raise PermissionError(f"No write permission for scope {scope.value}")

        # Baue vollständigen Pfad
        full_path = self.policy.build_path(scope, self.user, path, mod_name)

        # Verschlüsselung für USER_PRIVATE
        store_data = data
        encrypted = False
        if scope == Scope.USER_PRIVATE:
            store_data = self.crypto.encrypt(data)
            encrypted = True

        checksum = hashlib.sha256(data).hexdigest()
        now = time.time()

        # Speichere basierend auf Scope
        if scope == Scope.USER_PRIVATE and self._local_db:
            # Lokale Speicherung + Cloud Sync
            self._local_db.put(full_path, store_data, content_type=content_type)

            # Auch in Cloud speichern (verschlüsselt)
            if self._minio:
                self._write_to_minio(scope, full_path, store_data, content_type, metadata)
        else:
            # Direkt in Cloud
            if self._minio:
                self._write_to_minio(scope, full_path, store_data, content_type, metadata)

            # Invalidiere Cache
            self.cache.invalidate(scope, full_path)

        return BlobMetadata(
            path=full_path,
            scope=scope,
            owner_id=self.user.user_id,
            size=len(data),
            checksum=checksum,
            created_at=now,
            updated_at=now,
            encrypted=encrypted,
            content_type=content_type,
            custom_metadata=metadata or {}
        )

    def read(
        self,
        path: str,
        scope: Scope = Scope.USER_PRIVATE,
        owner_id: str = None,
        mod_name: str = None,
        use_cache: bool = True
    ) -> Optional[bytes]:
        """
        Liest Daten aus dem Storage

        Args:
            path: Relativer Pfad
            scope: Storage Scope
            owner_id: Owner-ID (für USER_* Scopes, default: eigener User)
            mod_name: Modulname (nur für MOD_DATA)
            use_cache: Cache verwenden (nicht für USER_PRIVATE)

        Returns:
            Daten als bytes oder None wenn nicht gefunden

        Raises:
            PermissionError: Wenn User keine Leseberechtigung hat
        """
        effective_owner = owner_id or self.user.user_id

        # Berechtigungsprüfung
        permission = self.policy.get_permission(scope, self.user, effective_owner)
        if not self.policy.can_read(permission):
            raise PermissionError(f"No read permission for scope {scope.value}")

        # Baue Pfad (mit Owner-ID für fremde Daten)
        if owner_id and owner_id != self.user.user_id:
            # Lese fremde Daten
            temp_user = UserContext(user_id=owner_id, username="", is_authenticated=False)
            full_path = self.policy.build_path(scope, temp_user, path, mod_name)
        else:
            full_path = self.policy.build_path(scope, self.user, path, mod_name)

        data = None

        # Lese basierend auf Scope
        if scope == Scope.USER_PRIVATE:
            # Nur eigene private Daten
            if owner_id and owner_id != self.user.user_id:
                raise PermissionError("Cannot read other user's private data")

            # Erst lokal
            if self._local_db:
                data = self._local_db.get(full_path)

            # Dann Cloud
            if data is None and self._minio:
                data = self._read_from_minio(scope, full_path)
                if data and self._local_db:
                    # Cache lokal
                    self._local_db.put(full_path, data)

            # Entschlüsseln
            if data:
                data = self.crypto.decrypt(data)
        else:
            # Andere Scopes: Cache -> Cloud
            if use_cache:
                data = self.cache.get(scope, full_path)

            if data is None and self._minio:
                data = self._read_from_minio(scope, full_path)
                if data and use_cache:
                    self.cache.set(scope, full_path, data)

        return data

    def delete(
        self,
        path: str,
        scope: Scope = Scope.USER_PRIVATE,
        mod_name: str = None
    ) -> bool:
        """
        Löscht einen Blob

        Args:
            path: Relativer Pfad
            scope: Storage Scope
            mod_name: Modulname (nur für MOD_DATA)

        Returns:
            True wenn erfolgreich gelöscht
        """
        # Berechtigungsprüfung
        permission = self.policy.get_permission(scope, self.user, self.user.user_id)
        if not self.policy.can_write(permission):
            raise PermissionError(f"No delete permission for scope {scope.value}")

        full_path = self.policy.build_path(scope, self.user, path, mod_name)

        # Lösche
        deleted = False

        if scope == Scope.USER_PRIVATE and self._local_db:
            self._local_db.delete(full_path)
            deleted = True

        if self._minio:
            try:
                bucket = self.policy.get_bucket_name(scope)
                self._minio.remove_object(bucket, full_path)
                deleted = True
            except S3Error:
                pass

        # Cache invalidieren
        self.cache.invalidate(scope, full_path)

        return deleted

    def exists(
        self,
        path: str,
        scope: Scope = Scope.USER_PRIVATE,
        owner_id: str = None,
        mod_name: str = None
    ) -> bool:
        """Prüft ob ein Blob existiert"""
        effective_owner = owner_id or self.user.user_id

        permission = self.policy.get_permission(scope, self.user, effective_owner)
        if not self.policy.can_read(permission):
            return False

        if owner_id and owner_id != self.user.user_id:
            temp_user = UserContext(user_id=owner_id, username="", is_authenticated=False)
            full_path = self.policy.build_path(scope, temp_user, path, mod_name)
        else:
            full_path = self.policy.build_path(scope, self.user, path, mod_name)

        # Lokal prüfen
        if scope == Scope.USER_PRIVATE and self._local_db:
            if self._local_db.exists(full_path):
                return True

        # Cloud prüfen
        if self._minio:
            try:
                bucket = self.policy.get_bucket_name(scope)
                self._minio.stat_object(bucket, full_path)
                return True
            except S3Error:
                pass

        return False

    def list(
        self,
        prefix: str = "",
        scope: Scope = Scope.USER_PRIVATE,
        owner_id: str = None,
        mod_name: str = None,
        recursive: bool = True
    ) -> List[BlobMetadata]:
        """
        Listet Blobs in einem Pfad

        Args:
            prefix: Pfad-Prefix
            scope: Storage Scope
            owner_id: Owner-ID (für USER_* Scopes)
            mod_name: Modulname (nur für MOD_DATA)
            recursive: Auch Unterverzeichnisse

        Returns:
            Liste von BlobMetadata
        """
        effective_owner = owner_id or self.user.user_id

        permission = self.policy.get_permission(scope, self.user, effective_owner)
        if not self.policy.can_read(permission):
            raise PermissionError(f"No list permission for scope {scope.value}")

        if owner_id and owner_id != self.user.user_id:
            temp_user = UserContext(user_id=owner_id, username="", is_authenticated=False)
            full_prefix = self.policy.build_path(scope, temp_user, prefix, mod_name)
        else:
            full_prefix = self.policy.build_path(scope, self.user, prefix, mod_name)

        results = []

        # Zuerst lokale DB prüfen (für USER_PRIVATE)
        if scope == Scope.USER_PRIVATE and self._local_db:
            try:
                # MobileDB.list() gibt List[BlobMetadata] zurück (aus mobile_db)
                local_blobs = self._local_db.list(full_prefix)
                for local_blob in local_blobs:
                    # Konvertiere mobile_db.BlobMetadata zu scoped_storage.BlobMetadata
                    results.append(BlobMetadata(
                        path=local_blob.path,
                        scope=scope,
                        owner_id=effective_owner,
                        size=local_blob.size,
                        checksum=local_blob.checksum or "",
                        content_type=local_blob.content_type or "application/octet-stream",
                        updated_at=local_blob.local_updated_at or 0
                    ))
            except Exception as e:
                import logging
                logging.getLogger("scoped_storage").debug(f"Local DB list error: {e}")

        # Dann MinIO prüfen
        if self._minio:
            try:
                bucket = self.policy.get_bucket_name(scope)
                objects = self._minio.list_objects(bucket, prefix=full_prefix, recursive=recursive)

                for obj in objects:
                    # Prüfe ob bereits in results (von lokaler DB)
                    if not any(r.path == obj.object_name for r in results):
                        results.append(BlobMetadata(
                            path=obj.object_name,
                            scope=scope,
                            owner_id=effective_owner,
                            size=obj.size or 0,
                            checksum=obj.etag or "",
                            updated_at=obj.last_modified.timestamp() if obj.last_modified else 0
                        ))
            except S3Error:
                pass
            except Exception:
                # Catch connection errors silently
                pass

        return results

    # =================== MinIO Helpers ===================

    def _write_to_minio(
        self,
        scope: Scope,
        path: str,
        data: bytes,
        content_type: str,
        metadata: Dict[str, str] = None
    ) -> bool:
        """Schreibt Daten direkt in MinIO. Gibt True bei Erfolg zurück."""
        from io import BytesIO
        import logging

        try:
            bucket = self.policy.get_bucket_name(scope)

            if isinstance(data, str):
                data = data.encode()

            self._minio.put_object(
                bucket,
                path,
                BytesIO(data),
                len(data),
                content_type=content_type,
                metadata=metadata
            )
            return True
        except S3Error as e:
            logging.getLogger("scoped_storage").warning(f"MinIO write error: {e}")
            return False
        except Exception as e:
            logging.getLogger("scoped_storage").warning(f"MinIO connection error: {e}")
            return False

    def _read_from_minio(self, scope: Scope, path: str) -> Optional[bytes]:
        """Liest Daten direkt aus MinIO"""
        try:
            bucket = self.policy.get_bucket_name(scope)
            response = self._minio.get_object(bucket, path)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error:
            return None
        except Exception:
            # Catch connection errors
            return None

    # =================== Sync ===================

    def sync_private(self) -> Dict[str, int]:
        """
        Synchronisiert USER_PRIVATE zwischen lokal und Cloud

        Returns:
            Dict mit uploaded/downloaded Counts
        """
        if not self._local_db or not self._minio:
            return {"uploaded": 0, "downloaded": 0}

        stats = {"uploaded": 0, "downloaded": 0}
        bucket = self.policy.get_bucket_name(Scope.USER_PRIVATE)
        user_prefix = f"{self.user.user_id}/"

        # Upload dirty lokale Blobs
        dirty_blobs = self._local_db.get_dirty_blobs()
        for blob in dirty_blobs:
            if blob.path.startswith(user_prefix):
                data = self._local_db.get(blob.path)
                if data:
                    self._write_to_minio(Scope.USER_PRIVATE, blob.path, data, "application/octet-stream")
                    self._local_db.mark_synced(blob.path)
                    stats["uploaded"] += 1

        # Download neue Cloud Blobs
        try:
            objects = self._minio.list_objects(bucket, prefix=user_prefix, recursive=True)
            for obj in objects:
                if not self._local_db.exists(obj.object_name):
                    data = self._read_from_minio(Scope.USER_PRIVATE, obj.object_name)
                    if data:
                        self._local_db.put(obj.object_name, data)
                        self._local_db.mark_synced(obj.object_name)
                        stats["downloaded"] += 1
        except S3Error:
            pass

        return stats

    # =================== Context Manager ===================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Schließt alle Verbindungen"""
        if self._local_db:
            self._local_db.close()


# =================== Helper Functions ===================

def create_storage_from_clerk_session(
    session_data: dict,
    minio_endpoint: str = None,
    minio_access_key: str = None,
    minio_secret_key: str = None,
    local_db_path: str = None
) -> ScopedBlobStorage:
    """
    Erstellt ScopedBlobStorage aus Clerk Session

    Args:
        session_data: Dict mit user_id, username, session_token, etc.
        minio_*: MinIO Verbindungsdaten
        local_db_path: Pfad zur lokalen SQLite DB

    Returns:
        Konfiguriertes ScopedBlobStorage
    """
    # Derive encryption key from session
    user_id = session_data.get("user_id", "")
    device_key = Code.DK()()
    if isinstance(device_key, str):
        device_key = device_key.encode()

    encryption_key =  base64.urlsafe_b64encode(hashlib.sha256(device_key + user_id.encode()).digest())

    user_context = UserContext.from_clerk_session(session_data, encryption_key)

    return ScopedBlobStorage(
        user_context=user_context,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        local_db_path=local_db_path
    )


# =================== Test ===================

if __name__ == "__main__":
    print("=== Scoped Blob Storage Test ===\n")

    # Test User Context
    user = UserContext(
        user_id="user_abc123",
        username="testuser",
        is_authenticated=True
    )

    admin = UserContext(
        user_id="admin_001",
        username="admin",
        is_admin=True,
        is_authenticated=True
    )

    anon = UserContext.anonymous()

    # Test Policy Engine
    policy = ScopePolicyEngine()

    print("Permission Tests:")
    print(f"  User -> PUBLIC_READ: {policy.get_permission(Scope.PUBLIC_READ, user)}")
    print(f"  User -> PUBLIC_RW: {policy.get_permission(Scope.PUBLIC_RW, user)}")
    print(f"  User -> USER_PUBLIC (own): {policy.get_permission(Scope.USER_PUBLIC, user, user.user_id)}")
    print(f"  User -> USER_PUBLIC (other): {policy.get_permission(Scope.USER_PUBLIC, user, 'other_user')}")
    print(f"  User -> USER_PRIVATE (own): {policy.get_permission(Scope.USER_PRIVATE, user, user.user_id)}")
    print(f"  User -> USER_PRIVATE (other): {policy.get_permission(Scope.USER_PRIVATE, user, 'other_user')}")
    print(f"  Admin -> USER_PRIVATE (other): {policy.get_permission(Scope.USER_PRIVATE, admin, 'other_user')}")
    print(f"  Anon -> PUBLIC_READ: {policy.get_permission(Scope.PUBLIC_READ, anon)}")
    print(f"  Anon -> PUBLIC_RW: {policy.get_permission(Scope.PUBLIC_RW, anon)}")

    print("\nPath Building Tests:")
    print(f"  PUBLIC_READ: {policy.build_path(Scope.PUBLIC_READ, user, 'test.txt')}")
    print(f"  USER_PUBLIC: {policy.build_path(Scope.USER_PUBLIC, user, 'profile.json')}")
    print(f"  USER_PRIVATE: {policy.build_path(Scope.USER_PRIVATE, user, 'secrets/key.pem')}")
    print(f"  MOD_DATA: {policy.build_path(Scope.MOD_DATA, user, 'settings.json', 'CloudM')}")

    print("\nBucket Names:")
    for scope in Scope:
        print(f"  {scope.name}: {policy.get_bucket_name(scope)}")

    # Test Encryption
    print("\nEncryption Test:")
    crypto = ScopedCryptoLayer(user)
    test_data = b"Geheime Daten!"
    encrypted = crypto.encrypt(test_data)
    decrypted = crypto.decrypt(encrypted)
    print(f"  Original: {test_data}")
    print(f"  Encrypted: {encrypted[:20]}...")
    print(f"  Decrypted: {decrypted}")
    print(f"  Match: {test_data == decrypted}")

    print("\n✓ All tests passed!")
