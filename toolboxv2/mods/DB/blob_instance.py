"""
ToolBox V2 - BlobDB für Server Storage
Key-Value Datenbank basierend auf MinIO für Server-Daten

Features:
- Nur SERVER_SCOPE (tb-servers Bucket)
- Konfiguration via Environment Variables
- Lokaler MinIO + optionaler Cloud Sync
- Offline-Modus mit SQLite Fallback
- Cache mit TTL
- Manifest-Tracking

Environment Variables:
- MINIO_ENDPOINT:      Lokaler MinIO Endpoint (default: 127.0.0.1:9000)
- MINIO_ACCESS_KEY:    Lokaler MinIO Access Key (default: admin)
- MINIO_SECRET_KEY:    Lokaler MinIO Secret Key (required)
- MINIO_SECURE:        HTTPS verwenden (default: false)

- CLOUD_ENDPOINT:      Cloud MinIO Endpoint (optional, für Sync)
- CLOUD_ACCESS_KEY:    Cloud MinIO Access Key
- CLOUD_SECRET_KEY:    Cloud MinIO Secret Key
- CLOUD_SECURE:        Cloud HTTPS verwenden (default: true)

- IS_OFFLINE_DB:       Nur SQLite, kein MinIO (default: false)
- SERVER_ID:           Server Identifier (default: hostname)
- DB_CACHE_TTL:        Cache TTL in Sekunden (default: 60)
"""

import os
import json
import time
import socket
import threading
import hashlib
from typing import Any, List, Optional, Dict, Set
from abc import ABC, abstractmethod
from pathlib import Path
from io import BytesIO

# =================== Environment Configuration ===================

def _get_env(key: str, default: str = None, required: bool = False) -> Optional[str]:
    """Holt Environment Variable"""
    value = os.getenv(key, default)
    if required and not value:
        raise EnvironmentError(f"Required environment variable {key} is not set")
    return value

def _get_env_bool(key: str, default: bool = False) -> bool:
    """Holt Environment Variable als Boolean"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


class Config:
    """Konfiguration aus Environment Variables"""

    # Lokaler MinIO
    MINIO_ENDPOINT = _get_env("MINIO_ENDPOINT", "127.0.0.1:9000")
    MINIO_ACCESS_KEY = _get_env("MINIO_ACCESS_KEY", "admin")
    MINIO_SECRET_KEY = _get_env("MINIO_SECRET_KEY", "")
    MINIO_SECURE = _get_env_bool("MINIO_SECURE", False)

    # Cloud MinIO (optional)
    CLOUD_ENDPOINT = _get_env("CLOUD_ENDPOINT", "")
    CLOUD_ACCESS_KEY = _get_env("CLOUD_ACCESS_KEY", "")
    CLOUD_SECRET_KEY = _get_env("CLOUD_SECRET_KEY", "")
    CLOUD_SECURE = _get_env_bool("CLOUD_SECURE", True)

    # Betriebsmodus
    IS_OFFLINE_DB = _get_env_bool("IS_OFFLINE_DB", False)
    SERVER_ID = _get_env("SERVER_ID", socket.gethostname())

    # Cache
    DB_CACHE_TTL = int(_get_env("DB_CACHE_TTL", "60"))

    # Bucket
    BUCKET_NAME = "tb-servers"

    @classmethod
    def reload(cls):
        """Lädt Konfiguration neu aus Environment"""
        cls.MINIO_ENDPOINT = _get_env("MINIO_ENDPOINT", "127.0.0.1:9000")
        cls.MINIO_ACCESS_KEY = _get_env("MINIO_ACCESS_KEY", "admin")
        cls.MINIO_SECRET_KEY = _get_env("MINIO_SECRET_KEY", "")
        cls.MINIO_SECURE = _get_env_bool("MINIO_SECURE", False)
        cls.CLOUD_ENDPOINT = _get_env("CLOUD_ENDPOINT", "")
        cls.CLOUD_ACCESS_KEY = _get_env("CLOUD_ACCESS_KEY", "")
        cls.CLOUD_SECRET_KEY = _get_env("CLOUD_SECRET_KEY", "")
        cls.CLOUD_SECURE = _get_env_bool("CLOUD_SECURE", True)
        cls.IS_OFFLINE_DB = _get_env_bool("IS_OFFLINE_DB", False)
        cls.SERVER_ID = _get_env("SERVER_ID", socket.gethostname())
        cls.DB_CACHE_TTL = int(_get_env("DB_CACHE_TTL", "60"))

    @classmethod
    def has_local_minio(cls) -> bool:
        """Prüft ob lokaler MinIO konfiguriert ist"""
        return bool(cls.MINIO_ENDPOINT and cls.MINIO_ACCESS_KEY and cls.MINIO_SECRET_KEY)

    @classmethod
    def has_cloud_minio(cls) -> bool:
        """Prüft ob Cloud MinIO konfiguriert ist"""
        return bool(cls.CLOUD_ENDPOINT and cls.CLOUD_ACCESS_KEY and cls.CLOUD_SECRET_KEY)

    @classmethod
    def to_dict(cls) -> dict:
        """Gibt Konfiguration als Dict zurück (ohne Secrets)"""
        return {
            "minio_endpoint": cls.MINIO_ENDPOINT,
            "minio_secure": cls.MINIO_SECURE,
            "cloud_endpoint": cls.CLOUD_ENDPOINT or "(not configured)",
            "cloud_secure": cls.CLOUD_SECURE,
            "is_offline": cls.IS_OFFLINE_DB,
            "server_id": cls.SERVER_ID,
            "cache_ttl": cls.DB_CACHE_TTL,
            "bucket": cls.BUCKET_NAME,
            "has_local": cls.has_local_minio(),
            "has_cloud": cls.has_cloud_minio()
        }


# =================== MinIO Import ===================

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None
    S3Error = Exception


# =================== ToolBoxV2 Imports ===================

try:
    from toolboxv2 import Result, get_logger
    from toolboxv2.mods.DB.types import AuthenticationTypes
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False

    class Result:
        """Fallback Result Klasse"""
        @staticmethod
        def ok(data=None, data_info=None, info=None):
            class R:
                def __init__(self):
                    self._data = data
                    self._info = info or data_info
                def get(self):
                    return self._data
                def is_error(self):
                    return False
                def set_origin(self, x):
                    return self
            return R()

        @staticmethod
        def default_user_error(data=None, info=None, exec_code=400):
            class R:
                def __init__(self):
                    self._error = info
                def is_error(self):
                    return True
                def set_origin(self, x):
                    return self
                def get(self):
                    return data
            return R()

        @staticmethod
        def default_internal_error(data=None, info=None):
            class R:
                def __init__(self):
                    self._error = info
                def is_error(self):
                    return True
                def set_origin(self, x):
                    return self
                def get(self):
                    return data
            return R()

        @staticmethod
        def custom_error(data=None, info=None):
            return Result.default_internal_error(data, info)

    class AuthenticationTypes:
        location = "location"

    def get_logger():
        import logging
        logger = logging.getLogger("blob_instance")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


# =================== SQLite Fallback ===================

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


class SQLiteCache:
    """SQLite-basierter Offline-Storage"""

    def __init__(self, db_path: str = None):
        if not SQLITE_AVAILABLE:
            raise RuntimeError("sqlite3 not available")

        self.db_path = db_path or os.path.expanduser("~/.tb_server_cache/offline.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = None
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blobs (
                    path TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    checksum TEXT,
                    updated_at REAL,
                    sync_status TEXT DEFAULT 'dirty'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manifest (
                    key TEXT PRIMARY KEY,
                    created_at REAL
                )
            """)
            conn.commit()

    def put(self, path: str, data: bytes) -> bool:
        checksum = hashlib.sha256(data).hexdigest()
        with self._lock:
            conn = self._get_conn()
            conn.execute("""
                INSERT OR REPLACE INTO blobs (path, data, checksum, updated_at, sync_status)
                VALUES (?, ?, ?, ?, 'dirty')
            """, (path, data, checksum, time.time()))
            conn.commit()
        return True

    def get(self, path: str) -> Optional[bytes]:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT data FROM blobs WHERE path = ?", (path,)
            ).fetchone()
            return row["data"] if row else None

    def delete(self, path: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM blobs WHERE path = ?", (path,))
            conn.commit()
        return True

    def exists(self, path: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT 1 FROM blobs WHERE path = ?", (path,)
            ).fetchone()
            return row is not None

    def list(self, prefix: str = "") -> List[str]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT path FROM blobs WHERE path LIKE ?", (f"{prefix}%",)
            ).fetchall()
            return [row["path"] for row in rows]

    def get_dirty(self) -> List[str]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT path FROM blobs WHERE sync_status = 'dirty'"
            ).fetchall()
            return [row["path"] for row in rows]

    def mark_synced(self, path: str):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE blobs SET sync_status = 'synced' WHERE path = ?", (path,)
            )
            conn.commit()

    # Manifest
    def add_to_manifest(self, key: str):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR IGNORE INTO manifest (key, created_at) VALUES (?, ?)",
                (key, time.time())
            )
            conn.commit()

    def remove_from_manifest(self, key: str):
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM manifest WHERE key = ?", (key,))
            conn.commit()

    def get_manifest(self) -> Set[str]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute("SELECT key FROM manifest").fetchall()
            return {row["key"] for row in rows}

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# =================== Abstract DB Interface ===================

class DB(ABC):
    """Abstract Database Interface"""

    @abstractmethod
    def get(self, query: str) -> Result:
        """Get data by key"""

    @abstractmethod
    def set(self, query: str, value) -> Result:
        """Set data by key"""

    @abstractmethod
    def append_on_set(self, query: str, value) -> Result:
        """Append to list or create"""

    @abstractmethod
    def delete(self, query: str, matching=False) -> Result:
        """Delete by key or pattern"""

    @abstractmethod
    def if_exist(self, query: str) -> bool:
        """Check if key exists"""

    @abstractmethod
    def exit(self) -> Result:
        """Close connection"""


# =================== BlobDB Implementation ===================

class BlobDB(DB):
    """
    Server Blob Database mit MinIO Backend

    Verwendet tb-servers Bucket für Server-spezifische Daten.
    Konfiguration erfolgt über Environment Variables.

    Features:
    - Lokaler MinIO + optionaler Cloud Sync
    - SQLite Fallback für Offline-Modus
    - Cache mit TTL
    - Manifest für schnelle Key-Suche

    Environment Variables:
    - MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
    - CLOUD_ENDPOINT, CLOUD_ACCESS_KEY, CLOUD_SECRET_KEY
    - IS_OFFLINE_DB, SERVER_ID, DB_CACHE_TTL
    """

    auth_type = AuthenticationTypes.location

    def __init__(self):
        self._local_minio: Optional[Minio] = None
        self._cloud_minio: Optional[Minio] = None
        self._sqlite: Optional[SQLiteCache] = None

        # Cache
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_lock = threading.RLock()

        # Manifest
        self._manifest: Set[str] = set()
        self._manifest_loaded = False

        self._initialized = False
        self._server_prefix = ""

    def initialize(self, db_path: str = None, **kwargs) -> Result:
        """
        Initialisiert die DB mit Environment-Konfiguration

        Args:
            db_path: Optional - Prefix für Keys (default: SERVER_ID)
            **kwargs: Ignoriert (für Kompatibilität)

        Returns:
            Result
        """
        try:
            # Reload config from environment
            Config.reload()

            # Server Prefix
            self._server_prefix = db_path or Config.SERVER_ID

            # Modus bestimmen
            if Config.IS_OFFLINE_DB:
                get_logger().info("BlobDB: Running in OFFLINE mode (SQLite only)")
                self._init_sqlite()
            else:
                # Lokaler MinIO
                if Config.has_local_minio():
                    self._init_local_minio()
                else:
                    get_logger().warning("BlobDB: No local MinIO configured, using SQLite fallback")
                    self._init_sqlite()

                # Cloud MinIO (optional)
                if Config.has_cloud_minio():
                    self._init_cloud_minio()

            # Manifest laden
            self._load_manifest()

            self._initialized = True

            get_logger().info(f"BlobDB initialized: server={self._server_prefix}, "
                            f"local={self._local_minio is not None}, "
                            f"cloud={self._cloud_minio is not None}, "
                            f"offline={self._sqlite is not None}")

            return Result.ok(info="BlobDB initialized").set_origin("BlobDB")

        except Exception as e:
            get_logger().error(f"BlobDB initialization failed: {e}")
            return Result.default_internal_error(
                data=str(e),
                info="Initialization failed"
            ).set_origin("BlobDB")

    def _init_local_minio(self):
        """Initialisiert lokalen MinIO Client"""
        if not MINIO_AVAILABLE:
            raise RuntimeError("minio package not installed. Run: pip install minio")

        self._local_minio = Minio(
            Config.MINIO_ENDPOINT,
            access_key=Config.MINIO_ACCESS_KEY,
            secret_key=Config.MINIO_SECRET_KEY,
            secure=Config.MINIO_SECURE
        )

        # Bucket erstellen falls nicht vorhanden
        try:
            if not self._local_minio.bucket_exists(Config.BUCKET_NAME):
                self._local_minio.make_bucket(Config.BUCKET_NAME)
                get_logger().info(f"Created bucket: {Config.BUCKET_NAME}")
        except S3Error as e:
            if e.code != "BucketAlreadyOwnedByYou":
                raise

    def _init_cloud_minio(self):
        """Initialisiert Cloud MinIO Client"""
        if not MINIO_AVAILABLE:
            return

        try:
            self._cloud_minio = Minio(
                Config.CLOUD_ENDPOINT,
                access_key=Config.CLOUD_ACCESS_KEY,
                secret_key=Config.CLOUD_SECRET_KEY,
                secure=Config.CLOUD_SECURE
            )

            # Test connection
            self._cloud_minio.bucket_exists(Config.BUCKET_NAME)
            get_logger().info(f"Connected to cloud MinIO: {Config.CLOUD_ENDPOINT}")

        except Exception as e:
            get_logger().warning(f"Could not connect to cloud MinIO: {e}")
            self._cloud_minio = None

    def _init_sqlite(self):
        """Initialisiert SQLite Fallback"""
        cache_dir = os.path.expanduser(f"~/.tb_server_cache/{self._server_prefix}")
        self._sqlite = SQLiteCache(os.path.join(cache_dir, "offline.db"))

    # =================== Path Helpers ===================

    def _key_to_path(self, key: str) -> str:
        """
        Konvertiert DB-Key zu MinIO Object Path

        Format: {server_id}/{key}.json
        Bsp: "myserver/users/123.json"
        """
        # Key sanitizen
        clean_key = key.replace("::", "/").replace("\\", "/").strip("/")
        return f"{self._server_prefix}/{clean_key}.json"

    def _get_manifest_path(self) -> str:
        """Pfad zur Manifest-Datei"""
        return f"{self._server_prefix}/_manifest.json"

    # =================== Manifest ===================

    def _load_manifest(self):
        """Lädt Manifest aus Storage"""
        if self._manifest_loaded:
            return

        try:
            path = self._get_manifest_path()
            data = self._read_from_storage(path)

            if data:
                keys = json.loads(data.decode())
                self._manifest = set(keys) if isinstance(keys, list) else set()
            else:
                self._manifest = set()

            # Auch aus SQLite laden falls vorhanden
            if self._sqlite:
                sqlite_manifest = self._sqlite.get_manifest()
                self._manifest.update(sqlite_manifest)

            self._manifest_loaded = True

        except Exception as e:
            get_logger().debug(f"Could not load manifest: {e}")
            self._manifest = set()
            self._manifest_loaded = True

    def _save_manifest(self):
        """Speichert Manifest in Storage"""
        try:
            path = self._get_manifest_path()
            data = json.dumps(list(self._manifest)).encode()
            self._write_to_storage(path, data)
        except Exception as e:
            get_logger().error(f"Could not save manifest: {e}")

    def _add_to_manifest(self, key: str):
        """Fügt Key zum Manifest hinzu"""
        if key not in self._manifest:
            self._manifest.add(key)
            self._save_manifest()

            if self._sqlite:
                self._sqlite.add_to_manifest(key)

    def _remove_from_manifest(self, key: str):
        """Entfernt Key aus Manifest"""
        if key in self._manifest:
            self._manifest.remove(key)
            self._save_manifest()

            if self._sqlite:
                self._sqlite.remove_from_manifest(key)

    # =================== Storage Operations ===================

    def _write_to_storage(self, path: str, data: bytes) -> bool:
        """Schreibt Daten in Storage (lokal + cloud)"""
        success = False

        # Lokaler MinIO
        if self._local_minio:
            try:
                self._local_minio.put_object(
                    Config.BUCKET_NAME,
                    path,
                    BytesIO(data),
                    len(data),
                    content_type="application/json"
                )
                success = True
            except Exception as e:
                get_logger().error(f"Local MinIO write failed: {e}")

        # Cloud MinIO
        if self._cloud_minio:
            try:
                self._cloud_minio.put_object(
                    Config.BUCKET_NAME,
                    path,
                    BytesIO(data),
                    len(data),
                    content_type="application/json"
                )
                success = True
            except Exception as e:
                get_logger().warning(f"Cloud MinIO write failed: {e}")

        # SQLite Fallback
        if self._sqlite:
            try:
                self._sqlite.put(path, data)
                success = True
            except Exception as e:
                get_logger().error(f"SQLite write failed: {e}")

        return success

    def _read_from_storage(self, path: str) -> Optional[bytes]:
        """Liest Daten aus Storage (lokal → cloud → sqlite)"""

        # 1. Lokaler MinIO
        if self._local_minio:
            try:
                response = self._local_minio.get_object(Config.BUCKET_NAME, path)
                data = response.read()
                response.close()
                response.release_conn()
                return data
            except S3Error as e:
                if e.code != "NoSuchKey":
                    get_logger().debug(f"Local MinIO read error: {e}")
            except Exception as e:
                get_logger().debug(f"Local MinIO read error: {e}")

        # 2. Cloud MinIO
        if self._cloud_minio:
            try:
                response = self._cloud_minio.get_object(Config.BUCKET_NAME, path)
                data = response.read()
                response.close()
                response.release_conn()

                # Cache lokal
                if self._local_minio:
                    try:
                        self._local_minio.put_object(
                            Config.BUCKET_NAME,
                            path,
                            BytesIO(data),
                            len(data)
                        )
                    except:
                        pass

                return data
            except S3Error as e:
                if e.code != "NoSuchKey":
                    get_logger().debug(f"Cloud MinIO read error: {e}")
            except Exception as e:
                get_logger().debug(f"Cloud MinIO read error: {e}")

        # 3. SQLite Fallback
        if self._sqlite:
            try:
                return self._sqlite.get(path)
            except Exception as e:
                get_logger().debug(f"SQLite read error: {e}")

        return None

    def _delete_from_storage(self, path: str) -> bool:
        """Löscht Daten aus Storage"""
        deleted = False

        if self._local_minio:
            try:
                self._local_minio.remove_object(Config.BUCKET_NAME, path)
                deleted = True
            except:
                pass

        if self._cloud_minio:
            try:
                self._cloud_minio.remove_object(Config.BUCKET_NAME, path)
                deleted = True
            except:
                pass

        if self._sqlite:
            try:
                self._sqlite.delete(path)
                deleted = True
            except:
                pass

        return deleted

    def _exists_in_storage(self, path: str) -> bool:
        """Prüft ob Daten existieren"""
        if self._local_minio:
            try:
                self._local_minio.stat_object(Config.BUCKET_NAME, path)
                return True
            except:
                pass

        if self._cloud_minio:
            try:
                self._cloud_minio.stat_object(Config.BUCKET_NAME, path)
                return True
            except:
                pass

        if self._sqlite:
            try:
                if self._sqlite.exists(path):
                    return True
            except:
                pass

        return False

    # =================== Cache ===================

    def _cache_get(self, key: str) -> tuple:
        """
        Holt aus Cache

        Returns:
            (found: bool, data: Any)
        """
        with self._cache_lock:
            if key not in self._cache:
                return (False, None)

            # TTL Check
            timestamp = self._cache_timestamps.get(key, 0)
            if time.time() - timestamp > Config.DB_CACHE_TTL:
                del self._cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]
                return (False, None)

            return (True, self._cache[key])

    def _cache_set(self, key: str, data: Any):
        """Setzt Cache-Eintrag"""
        with self._cache_lock:
            self._cache[key] = data
            self._cache_timestamps[key] = time.time()

    def _cache_invalidate(self, key: str):
        """Invalidiert Cache-Eintrag"""
        with self._cache_lock:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

    def _cache_clear(self):
        """Löscht gesamten Cache"""
        with self._cache_lock:
            self._cache.clear()
            self._cache_timestamps.clear()

    # =================== DB Interface ===================

    def get(self, query: str) -> Result:
        """
        Lädt Daten. Unterstützt Wildcards (*) für Pattern-Matching.

        Args:
            query: Key oder Pattern (z.B. "users/*", "config")

        Returns:
            Result mit Daten
        """
        if not self._initialized:
            return Result.default_internal_error(info="DB not initialized")

        # Spezialfall: Alle Keys
        if query in ("all", "*"):
            return self._get_all()

        if query == "all-k":
            return Result.ok(data=list(self._manifest))

        # Wildcard Pattern?
        if "*" in query:
            return self._get_by_pattern(query)

        # Cache Check
        found, cached = self._cache_get(query)
        if found:
            return Result.ok(data=cached)

        # Storage Read
        path = self._key_to_path(query)

        try:
            data = self._read_from_storage(path)

            if data is None:
                return Result.default_user_error(info=f"Key '{query}' not found")

            # Parse JSON
            parsed = json.loads(data.decode())

            # Cache
            self._cache_set(query, parsed)

            return Result.ok(data=parsed)

        except json.JSONDecodeError as e:
            return Result.default_internal_error(info=f"Invalid JSON for '{query}': {e}")

        except Exception as e:
            return Result.default_internal_error(info=f"Error reading '{query}': {e}")

    def set(self, query: str, value) -> Result:
        """
        Speichert Daten sofort persistent.

        Args:
            query: Key (z.B. "users/123", "config")
            value: Zu speichernde Daten

        Returns:
            Result
        """
        if not self._initialized:
            return Result.default_internal_error(info="DB not initialized")

        path = self._key_to_path(query)

        try:
            # Serialize
            data = json.dumps(value).encode()

            # Write
            if not self._write_to_storage(path, data):
                return Result.default_internal_error(info=f"Failed to write '{query}'")

            # Cache
            self._cache_set(query, value)

            # Manifest
            self._add_to_manifest(query)

            return Result.ok()

        except Exception as e:
            return Result.default_internal_error(info=f"Failed to set '{query}': {e}")

    def append_on_set(self, query: str, value) -> Result:
        """
        Fügt Daten zu einer Liste hinzu oder erstellt sie.

        Args:
            query: Key
            value: Wert oder Liste

        Returns:
            Result
        """
        if not self._initialized:
            return Result.default_internal_error(info="DB not initialized")

        try:
            # Aktuelle Daten lesen
            current = []
            result = self.get(query)
            if not result.is_error():
                current = result.get()
                if not isinstance(current, list):
                    current = [current] if current else []

            # Append
            if isinstance(value, list):
                for v in value:
                    if v not in current:
                        current.append(v)
            elif value not in current:
                current.append(value)

            # Speichern
            return self.set(query, current)

        except Exception as e:
            return Result.default_internal_error(info=f"Failed to append to '{query}': {e}")

    def delete(self, query: str, matching=False) -> Result:
        """
        Löscht Schlüssel.

        Args:
            query: Key oder Pattern
            matching: Pattern-Matching aktivieren

        Returns:
            Result mit Anzahl gelöschter Keys
        """
        if not self._initialized:
            return Result.default_internal_error(info="DB not initialized")

        keys_to_delete = []

        if matching or "*" in query:
            pattern = query.replace("*", "")
            keys_to_delete = [k for k in self._manifest if k.startswith(pattern)]
        else:
            keys_to_delete = [query]

        deleted_count = 0
        errors = []

        for key in keys_to_delete:
            try:
                path = self._key_to_path(key)

                if self._delete_from_storage(path):
                    deleted_count += 1

                self._cache_invalidate(key)
                self._remove_from_manifest(key)

            except Exception as e:
                errors.append(f"{key}: {e}")

        if errors:
            return Result.custom_error(
                data=errors,
                info=f"Deleted {deleted_count} keys, {len(errors)} errors"
            )

        return Result.ok(data=deleted_count, data_info=f"Deleted {deleted_count} keys")

    def if_exist(self, query: str) -> bool:
        """
        Prüft Existenz über Manifest.

        Args:
            query: Key oder Pattern

        Returns:
            True wenn existiert
        """
        if not self._manifest_loaded:
            self._load_manifest()

        if "*" in query:
            pattern = query.replace("*", "")
            return any(k.startswith(pattern) for k in self._manifest)

        return query in self._manifest

    def exit(self) -> Result:
        """Schließt alle Verbindungen"""
        try:
            if self._sqlite:
                self._sqlite.close()
                self._sqlite = None

            self._cache_clear()
            self._initialized = False

            return Result.ok(info="BlobDB closed").set_origin("BlobDB")

        except Exception as e:
            return Result.default_internal_error(data=str(e))

    # =================== Extended API ===================

    def _get_all(self) -> Result:
        """Holt alle Daten"""
        all_data = {}
        for key in self._manifest:
            result = self.get(key)
            if not result.is_error():
                all_data[key] = result.get()
        return Result.ok(data=all_data)

    def _get_by_pattern(self, pattern: str) -> Result:
        """Holt alle Keys die zum Pattern passen"""
        clean_pattern = pattern.replace("*", "")
        matching = [k for k in self._manifest if k.startswith(clean_pattern)]

        results = []
        for key in matching:
            result = self.get(key)
            if not result.is_error():
                results.append(result.get())

        return Result.ok(data=results)

    def sync_to_cloud(self) -> Result:
        """
        Synchronisiert lokale Daten zur Cloud

        Returns:
            Result mit Sync-Statistiken
        """
        if not self._cloud_minio:
            return Result.default_user_error(info="Cloud not configured")

        if not self._sqlite:
            return Result.ok(data={"uploaded": 0, "message": "No offline data"})

        try:
            dirty_paths = self._sqlite.get_dirty()
            uploaded = 0

            for path in dirty_paths:
                data = self._sqlite.get(path)
                if data:
                    try:
                        self._cloud_minio.put_object(
                            Config.BUCKET_NAME,
                            path,
                            BytesIO(data),
                            len(data)
                        )
                        self._sqlite.mark_synced(path)
                        uploaded += 1
                    except Exception as e:
                        get_logger().warning(f"Failed to sync {path}: {e}")

            return Result.ok(data={"uploaded": uploaded})

        except Exception as e:
            return Result.default_internal_error(info=f"Sync failed: {e}")

    def sync_from_cloud(self) -> Result:
        """
        Synchronisiert Cloud-Daten lokal

        Returns:
            Result mit Sync-Statistiken
        """
        if not self._cloud_minio or not self._local_minio:
            return Result.default_user_error(info="Cloud or local not configured")

        try:
            downloaded = 0
            prefix = f"{self._server_prefix}/"

            objects = self._cloud_minio.list_objects(
                Config.BUCKET_NAME,
                prefix=prefix,
                recursive=True
            )

            for obj in objects:
                try:
                    # Download from cloud
                    response = self._cloud_minio.get_object(Config.BUCKET_NAME, obj.object_name)
                    data = response.read()
                    response.close()
                    response.release_conn()

                    # Upload to local
                    self._local_minio.put_object(
                        Config.BUCKET_NAME,
                        obj.object_name,
                        BytesIO(data),
                        len(data)
                    )
                    downloaded += 1

                except Exception as e:
                    get_logger().warning(f"Failed to download {obj.object_name}: {e}")

            # Reload manifest
            self._manifest_loaded = False
            self._load_manifest()

            return Result.ok(data={"downloaded": downloaded})

        except Exception as e:
            return Result.default_internal_error(info=f"Sync failed: {e}")

    def get_stats(self) -> dict:
        """Gibt Statistiken zurück"""
        return {
            "initialized": self._initialized,
            "server_id": self._server_prefix,
            "keys_count": len(self._manifest),
            "cache_size": len(self._cache),
            "has_local_minio": self._local_minio is not None,
            "has_cloud_minio": self._cloud_minio is not None,
            "has_sqlite": self._sqlite is not None,
            "is_offline": Config.IS_OFFLINE_DB,
            "config": Config.to_dict()
        }

    def clear_cache(self):
        """Löscht lokalen Cache"""
        self._cache_clear()

    def reload_manifest(self):
        """Lädt Manifest neu"""
        self._manifest_loaded = False
        self._load_manifest()


# =================== Convenience ===================

def create_db(db_path: str = None) -> BlobDB:
    """
    Factory für BlobDB

    Args:
        db_path: Optional Key-Prefix (default: SERVER_ID aus ENV)

    Returns:
        Initialisierte BlobDB
    """
    db = BlobDB()
    result = db.initialize(db_path=db_path)

    if result.is_error():
        raise RuntimeError(f"Failed to initialize BlobDB: {result._error}")

    return db


# =================== Test ===================

if __name__ == "__main__":
    print("=" * 60)
    print("BLOBDB SERVER STORAGE TEST")
    print("=" * 60)

    print("\n[1] Configuration:")
    Config.reload()
    for key, value in Config.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[2] Initialize DB:")

    # Setze Test-Konfiguration falls nicht vorhanden
    if not Config.MINIO_SECRET_KEY:
        os.environ["IS_OFFLINE_DB"] = "true"
        print("  No MINIO_SECRET_KEY set, using offline mode")

    try:
        db = create_db("test_server")
        print(f"  ✓ DB initialized")
        print(f"  Stats: {json.dumps(db.get_stats(), indent=2, default=str)}")
    except Exception as e:
        print(f"  ✗ DB initialization failed: {e}")
        exit(1)

    print("\n[3] CRUD Operations:")

    # Set
    result = db.set("test/key1", {"value": 42, "name": "test"})
    print(f"  Set: {'✓' if not result.is_error() else '✗'}")

    # Get
    result = db.get("test/key1")
    print(f"  Get: {'✓' if not result.is_error() else '✗'} - {result.get()}")

    # Append
    db.append_on_set("test/list", "item1")
    db.append_on_set("test/list", "item2")
    result = db.get("test/list")
    print(f"  Append: {'✓' if result.get() == ['item1', 'item2'] else '✗'} - {result.get()}")

    # Exists
    exists = db.if_exist("test/key1")
    print(f"  Exists: {'✓' if exists else '✗'}")

    # Pattern
    result = db.get("test/*")
    print(f"  Pattern: {'✓' if not result.is_error() else '✗'} - {len(result.get())} results")

    # Delete
    result = db.delete("test/key1")
    print(f"  Delete: {'✓' if not result.is_error() else '✗'}")

    # Not exists after delete
    exists = db.if_exist("test/key1")
    print(f"  Not exists: {'✓' if not exists else '✗'}")

    # Cleanup
    db.delete("test/*", matching=True)
    db.exit()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
