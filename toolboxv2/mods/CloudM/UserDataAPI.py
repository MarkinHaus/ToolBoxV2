"""
ToolBox V2 - Unified User Data API
Vereinheitlichte Schnittstelle für Mod-zu-Mod Datenzugriff mit Scoped Storage

SCOPES:
- PUBLIC_READ:   Alle lesen, nur Admin schreibt
- PUBLIC_RW:     Alle lesen/schreiben
- USER_PUBLIC:   Alle lesen, nur Owner schreibt unter eigenem Prefix
- USER_PRIVATE:  Nur Owner (lokal + verschlüsselter Cloud-Sync)
- SERVER_SCOPE:  Server-spezifische Daten
- MOD_DATA:      Modul-spezifische Daten

Features:
- Berechtigungsbasierter Zugriff auf Daten anderer Mods
- Audit-Log für Datenzugriffe
- Lokale Speicherung für USER_PRIVATE
- Caching für andere Scopes
- Integration mit Custom Auth
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

# ToolBoxV2 imports
try:
    from toolboxv2 import App, RequestData, Result, get_app, get_logger
    from toolboxv2.utils.security.cryp import Code
    from toolboxv2.utils.system.types import ApiResult, ToolBoxInterfaces
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    # Stubs für standalone Testing
    class App: pass
    class RequestData: pass
    class Result:
        @staticmethod
        def ok(data=None, data_info=None): return {"ok": True, "data": data}
        @staticmethod
        def default_user_error(info=None, exec_code=400): return {"ok": False, "error": info}
        @staticmethod
        def default_internal_error(info=None): return {"ok": False, "error": info}
    def get_app(name): return App()
    def get_logger():
        import logging
        return logging.getLogger()

# Local imports
from toolboxv2.utils.extras.db.scoped_storage import (
    Permission,
    Scope,
    ScopedBlobStorage,
    ScopePolicyEngine,
    UserContext,
    create_storage_from_clerk_session,
)

# Import MinIO Policy Helper for permission enforcement
try:
    from .auth.minio_policy import PermissionChecker, PolicyGenerator, CredentialBroker
    MINIO_POLICY_AVAILABLE = True
except ImportError:
    MINIO_POLICY_AVAILABLE = False
    CredentialBroker = None
    # Fallback implementations
    class PermissionChecker:
        LEVEL_GUEST = 0
        LEVEL_USER = 1
        LEVEL_MODERATOR = 5
        LEVEL_ADMIN = 10

        @classmethod
        def can_access_user_data(cls, requesting_user_id, requesting_user_level, target_user_id, scope):
            if requesting_user_id == target_user_id:
                return True
            if requesting_user_level >= cls.LEVEL_ADMIN:
                return True
            if scope == "USER_PUBLIC":
                return True
            return False

        @classmethod
        def can_modify_user_data(cls, requesting_user_id, requesting_user_level, target_user_id, scope):
            if requesting_user_id == target_user_id:
                return True
            if requesting_user_level >= cls.LEVEL_ADMIN:
                return True
            return False

Name = 'CloudM.UserDataAPI'
version = '2.0.0'

if TOOLBOX_AVAILABLE:
    export = get_app(f"{Name}.Export").tb
else:
    def export(**kwargs):
        def decorator(func):
            return func
        return decorator


# =================== Data Classes ===================

@dataclass
class ModPermission:
    """Berechtigung für Mod-Datenzugriff"""
    source_mod: str      # Mod die Zugriff anfragt
    target_mod: str      # Mod auf deren Daten zugegriffen wird
    permission_type: str # 'read', 'write', 'full'
    granted: bool = False
    granted_at: float = 0
    expires_at: float = 0  # 0 = never expires
    granted_keys: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class DataAccessLog:
    """Audit-Log Eintrag für Datenzugriff"""
    timestamp: float
    source_mod: str
    target_mod: str
    action: str  # 'read', 'write', 'delete'
    scope: str
    keys_accessed: List[str]
    success: bool
    user_id: str


# =================== Storage Provider ===================

class StorageProvider:
    """
    Zentrale Storage-Verwaltung pro User

    Verwaltet:
    - ScopedBlobStorage Instanz pro User
    - Mod-Permissions
    - Audit Logging
    """

    _instances: Dict[str, 'StorageProvider'] = {}

    def __init__(
        self,
        user_context: UserContext,
        minio_endpoint: str = None,
        minio_access_key: str = None,
        minio_secret_key: str = None,
        local_db_path: str = None
    ):
        self.user = user_context
        self.storage = ScopedBlobStorage(
            user_context=user_context,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            local_db_path=local_db_path
        )

        self._permissions: Dict[str, ModPermission] = {}
        self._access_log: List[DataAccessLog] = []
        self._load_permissions()

    @classmethod
    def get_instance(
        cls,
        user_context: UserContext,
        minio_endpoint: str = None,
        minio_access_key: str = None,
        minio_secret_key: str = None,
        local_db_path: str = None
    ) -> 'StorageProvider':
        """Singleton pro User"""
        user_id = user_context.user_id

        if user_id not in cls._instances:
            cls._instances[user_id] = cls(
                user_context,
                minio_endpoint,
                minio_access_key,
                minio_secret_key,
                local_db_path
            )

        return cls._instances[user_id]

    def _load_permissions(self):
        """Lädt Permissions aus Storage"""
        try:
            data = self.storage.read(
                "_system/permissions.json",
                scope=Scope.USER_PRIVATE
            )
            if data:
                perms = json.loads(data.decode())
                self._permissions = {
                    k: ModPermission(**v) for k, v in perms.items()
                }
        except:
            pass

    def _save_permissions(self):
        """Speichert Permissions in Storage"""
        data = {k: asdict(v) for k, v in self._permissions.items()}
        self.storage.write(
            "_system/permissions.json",
            json.dumps(data).encode(),
            scope=Scope.USER_PRIVATE
        )

    def _log_access(
        self,
        source_mod: str,
        target_mod: str,
        action: str,
        scope: Scope,
        keys: List[str],
        success: bool
    ):
        """Loggt Datenzugriff"""
        log = DataAccessLog(
            timestamp=time.time(),
            source_mod=source_mod,
            target_mod=target_mod,
            action=action,
            scope=scope.value,
            keys_accessed=keys,
            success=success,
            user_id=self.user.user_id
        )

        self._access_log.append(log)

        # Behalte nur letzte 100
        if len(self._access_log) > 100:
            self._access_log = self._access_log[-100:]

        # Speichere Log
        try:
            log_data = [asdict(l) for l in self._access_log]
            self.storage.write(
                "_system/access_log.json",
                json.dumps(log_data).encode(),
                scope=Scope.USER_PRIVATE
            )
        except:
            pass

    def check_mod_permission(
        self,
        source_mod: str,
        target_mod: str,
        permission_type: str,
        key: str = None
    ) -> bool:
        """
        Prüft Mod-zu-Mod Berechtigung

        Args:
            source_mod: Anfragende Mod
            target_mod: Ziel-Mod
            permission_type: 'read', 'write', 'delete'
            key: Optionaler spezifischer Key

        Returns:
            True wenn berechtigt
        """
        # Eigene Mod hat immer Zugriff
        if source_mod == target_mod:
            return True

        perm_key = f"{source_mod}::{target_mod}"

        if perm_key not in self._permissions:
            return False

        perm = self._permissions[perm_key]

        # Prüfe ob granted
        if not perm.granted:
            return False

        # Prüfe Ablauf
        if perm.expires_at > 0 and time.time() > perm.expires_at:
            return False

        # Prüfe Permission Type
        if perm.permission_type == 'full':
            pass
        elif perm.permission_type == 'read' and permission_type in ['write', 'delete'] or perm.permission_type == 'write' and permission_type == 'delete':
            return False

        # Prüfe Key-Restriction
        if perm.granted_keys and key and key not in perm.granted_keys:
            return False

        return True

    def grant_permission(
        self,
        source_mod: str,
        target_mod: str,
        permission_type: str = 'read',
        keys: List[str] = None,
        expires_hours: int = 0,
        reason: str = ""
    ):
        """Erteilt Mod-Permission"""
        perm_key = f"{source_mod}::{target_mod}"

        expires_at = 0
        if expires_hours > 0:
            expires_at = time.time() + (expires_hours * 3600)

        self._permissions[perm_key] = ModPermission(
            source_mod=source_mod,
            target_mod=target_mod,
            permission_type=permission_type,
            granted=True,
            granted_at=time.time(),
            expires_at=expires_at,
            granted_keys=keys or [],
            reason=reason
        )

        self._save_permissions()

    def revoke_permission(self, source_mod: str, target_mod: str):
        """Widerruft Mod-Permission"""
        perm_key = f"{source_mod}::{target_mod}"

        if perm_key in self._permissions:
            del self._permissions[perm_key]
            self._save_permissions()

    def list_permissions(self) -> List[dict]:
        """Listet alle Permissions"""
        return [asdict(p) for p in self._permissions.values()]

    def get_access_log(self, limit: int = 50) -> List[dict]:
        """Holt Access Log"""
        sorted_log = sorted(
            self._access_log,
            key=lambda x: x.timestamp,
            reverse=True
        )
        return [asdict(l) for l in sorted_log[:limit]]


# =================== Helper Functions ===================

_storage_providers: Dict[str, StorageProvider] = {}


async def _get_storage_provider(app: App, request: RequestData) -> StorageProvider | None:
    """Holt StorageProvider für aktuellen User"""
    user_data = await _get_current_user(app, request)
    if not user_data:
        return None

    user_id = user_data.uid or user_data.user_id
    if not user_id:
        return None

    username = user_data.username or (user_data.email.split('@')[0] if user_data.email else '')
    is_admin = getattr(user_data, 'level', 0) >= 10

    user_context = UserContext(
        user_id=user_id,
        username=username,
        is_admin=is_admin,
        is_authenticated=True
    )
    # Setze level für PermissionChecker
    user_context.level = getattr(user_data, 'level', 1)

    if user_id not in _storage_providers:
        import os
        _storage_providers[user_id] = StorageProvider(
            user_context=user_context,
            minio_endpoint=os.environ.get('MINIO_ENDPOINT'),
            minio_access_key=os.environ.get('MINIO_ACCESS_KEY'),
            minio_secret_key=os.environ.get('MINIO_SECRET_KEY'),
        )

    return _storage_providers[user_id]


async def _get_current_user(app: App, request: RequestData):
    """Aktuellen Benutzer aus Request via JWT (Auth.py) holen."""
    token = None
    if hasattr(request, 'session') and hasattr(request.session, 'get'):
        s = request.session if isinstance(request.session, dict) else (request.session.to_dict() if hasattr(request.session, 'to_dict') else {})
        token = s.get('session_token') or s.get('token')
    if not token and hasattr(request, 'request'):
        r = request.request if isinstance(request.request, dict) else (request.request.to_dict() if hasattr(request.request, 'to_dict') else {})
        headers = r.get('headers', {})
        auth = headers.get('authorization', '')
        if auth.startswith('Bearer '):
            token = auth[7:]
    if not token:
        return None
    try:
        result = await app.a_run_any(
            ("CloudM.Auth", "validate_session"), token=token, get_results=True
        )
        if hasattr(result, 'is_error') and result.is_error():
            return None
        data = result.get() if hasattr(result, 'get') else result
        if not data or not data.get('authenticated'):
            return None
        # Return als SimpleNamespace für Attribut-Zugriff
        from types import SimpleNamespace
        return SimpleNamespace(
            uid=data.get('user_id', ''),
            user_id=data.get('user_id', ''),
            username=data.get('user_name', data.get('username', '')),
            email=data.get('email', ''),
            level=data.get('level', 1),
        )
    except Exception:
        return None


async def _check_cross_user_permission(
    provider: StorageProvider,
    target_user_id: str,
    scope: Scope,
    action: str = "read"
) -> bool:
    """
    Prüft ob der aktuelle User berechtigt ist, auf Daten eines anderen Users zuzugreifen.

    Args:
        provider: StorageProvider des aktuellen Users
        target_user_id: User dessen Daten abgerufen werden sollen
        scope: Angefragter Scope
        action: 'read', 'write', oder 'delete'

    Returns:
        True wenn Zugriff erlaubt, False sonst
    """
    current_user_id = provider.user.user_id
    current_user_level = getattr(provider.user, 'level', PermissionChecker.LEVEL_USER)

    # Same user always has access
    if current_user_id == target_user_id:
        return True

    # Use PermissionChecker for cross-user access validation
    scope_str = scope.value if hasattr(scope, 'value') else str(scope)

    if action == "read":
        return PermissionChecker.can_access_user_data(
            current_user_id, current_user_level, target_user_id, scope_str
        )
    else:  # write or delete
        return PermissionChecker.can_modify_user_data(
            current_user_id, current_user_level, target_user_id, scope_str
        )


def _scope_from_string(scope_str: str) -> Scope:
    """Konvertiert String zu Scope"""
    scope_map = {
        'public_read': Scope.PUBLIC_READ,
        'public_rw': Scope.PUBLIC_RW,
        'user_public': Scope.USER_PUBLIC,
        'user_private': Scope.USER_PRIVATE,
        'server': Scope.SERVER_SCOPE,
        'mod_data': Scope.MOD_DATA,
        # Shortcuts
        'public': Scope.PUBLIC_RW,
        'private': Scope.USER_PRIVATE,
        'shared': Scope.USER_PUBLIC
    }
    return scope_map.get(scope_str.lower(), Scope.USER_PRIVATE)


# =================== Public API ===================

@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def get_data(
    app: App,
    request: RequestData,
    path: str,
    scope: str = "private",
    mod_name: str = None,
    owner_id: str = None
):
    """
    Universelle Daten-Abruf Funktion

    Args:
        path: Pfad zur Datei (relativ)
        scope: Storage Scope (public_read, public_rw, user_public, user_private, mod_data)
        mod_name: Modulname (nur für mod_data scope)
        owner_id: Owner-ID (für Zugriff auf fremde public Daten)

    Returns:
        Result mit Daten

    Examples:
        # Private Daten lesen
        result = await app.a_run_any('CloudM.UserDataAPI.get_data',
                                     path='settings.json', scope='private')

        # Public shared Daten eines anderen Users
        result = await app.a_run_any('CloudM.UserDataAPI.get_data',
                                     path='profile.json', scope='user_public',
                                     owner_id='other_user_id')

        # Mod-Daten
        result = await app.a_run_any('CloudM.UserDataAPI.get_data',
                                     path='config.json', scope='mod_data',
                                     mod_name='MyMod')
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    storage_scope = _scope_from_string(scope)

    # Cross-User Permission Check
    if owner_id and owner_id != provider.user.user_id:
        has_permission = await _check_cross_user_permission(
            provider, owner_id, storage_scope, "read"
        )
        if not has_permission:
            provider._log_access(
                source_mod=mod_name or "direct",
                target_mod=mod_name or "user",
                action="read",
                scope=storage_scope,
                keys=[path],
                success=False
            )
            return Result.default_user_error(
                info=f"Keine Berechtigung für Zugriff auf Daten von User '{owner_id}'",
                exec_code=403
            )

    try:
        data = provider.storage.read(
            path=path,
            scope=storage_scope,
            owner_id=owner_id,
            mod_name=mod_name
        )

        if data is None:
            return Result.default_user_error(info="Daten nicht gefunden", exec_code=404)

        # Log access
        provider._log_access(
            source_mod=mod_name or "direct",
            target_mod=mod_name or "user",
            action="read",
            scope=storage_scope,
            keys=[path],
            success=True
        )

        # Versuche als JSON zu parsen
        try:
            return Result.ok(data=json.loads(data.decode()))
        except:
            return Result.ok(data=data)

    except PermissionError as e:
        provider._log_access(
            source_mod=mod_name or "direct",
            target_mod=mod_name or "user",
            action="read",
            scope=storage_scope,
            keys=[path],
            success=False
        )
        return Result.default_user_error(info=str(e), exec_code=403)

    except Exception as e:
        return Result.default_internal_error(str(e))


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def set_data(
    app: App,
    request: RequestData,
    path: str,
    data: Any,
    scope: str = "private",
    mod_name: str = None,
    content_type: str = "application/json",
    owner_id: str = None
):
    """
    Universelle Daten-Speicher Funktion

    Args:
        path: Pfad zur Datei (relativ)
        data: Zu speichernde Daten (dict, list, str, bytes)
        scope: Storage Scope
        mod_name: Modulname (nur für mod_data scope)
        content_type: MIME Type
        owner_id: Owner-ID (für Schreibzugriff auf fremde Daten - nur Admin)

    Returns:
        Result mit Metadaten
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    storage_scope = _scope_from_string(scope)

    # Cross-User Write Permission Check (stricter than read)
    if owner_id and owner_id != provider.user.user_id:
        has_permission = await _check_cross_user_permission(
            provider, owner_id, storage_scope, "write"
        )
        if not has_permission:
            provider._log_access(
                source_mod=mod_name or "direct",
                target_mod=mod_name or "user",
                action="write",
                scope=storage_scope,
                keys=[path],
                success=False
            )
            return Result.default_user_error(
                info=f"Keine Schreibberechtigung für Daten von User '{owner_id}'",
                exec_code=403
            )

    # Konvertiere Daten zu bytes
    if isinstance(data, (dict, list)):
        store_data = json.dumps(data).encode()
        content_type = "application/json"
    elif isinstance(data, str):
        store_data = data.encode()
    elif isinstance(data, bytes):
        store_data = data
    else:
        store_data = str(data).encode()

    try:
        metadata = provider.storage.write(
            path=path,
            data=store_data,
            scope=storage_scope,
            mod_name=mod_name,
            content_type=content_type,
            owner_id=owner_id
        )

        provider._log_access(
            source_mod=mod_name or "direct",
            target_mod=mod_name or "user",
            action="write",
            scope=storage_scope,
            keys=[path],
            success=True
        )

        return Result.ok(data={
            "path": metadata.path,
            "size": metadata.size,
            "checksum": metadata.checksum,
            "encrypted": metadata.encrypted
        })

    except PermissionError as e:
        provider._log_access(
            source_mod=mod_name or "direct",
            target_mod=mod_name or "user",
            action="write",
            scope=storage_scope,
            keys=[path],
            success=False
        )
        return Result.default_user_error(info=str(e), exec_code=403)

    except Exception as e:
        return Result.default_internal_error(str(e))


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def delete_data(
    app: App,
    request: RequestData,
    path: str,
    scope: str = "private",
    mod_name: str = None
):
    """
    Löscht Daten

    Args:
        path: Pfad zur Datei
        scope: Storage Scope
        mod_name: Modulname

    Returns:
        Result
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    storage_scope = _scope_from_string(scope)

    try:
        deleted = provider.storage.delete(
            path=path,
            scope=storage_scope,
            mod_name=mod_name
        )

        provider._log_access(
            source_mod=mod_name or "direct",
            target_mod=mod_name or "user",
            action="delete",
            scope=storage_scope,
            keys=[path],
            success=deleted
        )

        if deleted:
            return Result.ok(data_info=f"Gelöscht: {path}")
        else:
            return Result.default_user_error(info="Nicht gefunden", exec_code=404)

    except PermissionError as e:
        return Result.default_user_error(info=str(e), exec_code=403)

    except Exception as e:
        return Result.default_internal_error(str(e))


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def list_data(
    app: App,
    request: RequestData,
    prefix: str = "",
    scope: str = "private",
    mod_name: str = None,
    owner_id: str = None
):
    """
    Listet Daten in einem Pfad

    Args:
        prefix: Pfad-Prefix
        scope: Storage Scope
        mod_name: Modulname
        owner_id: Owner-ID für fremde Daten

    Returns:
        Result mit Liste von Metadaten
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    storage_scope = _scope_from_string(scope)

    try:
        blobs = provider.storage.list(
            prefix=prefix,
            scope=storage_scope,
            owner_id=owner_id,
            mod_name=mod_name
        )

        return Result.ok(data=[
            {
                "path": b.path,
                "size": b.size,
                "updated_at": b.updated_at
            }
            for b in blobs
        ])

    except PermissionError as e:
        return Result.default_user_error(info=str(e), exec_code=403)

    except Exception as e:
        return Result.default_internal_error(str(e))


# =================== Mod-Specific API ===================

@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def get_mod_data(
    app: App,
    request: RequestData,
    source_mod: str,
    target_mod: str = None,
    key: str = None
):
    """
    Mod-Daten abrufen (Legacy API Kompatibilität)

    Args:
        source_mod: Name des anfragenden Moduls
        target_mod: Name des Ziel-Moduls (default: source_mod)
        key: Optionaler spezifischer Schlüssel

    Returns:
        Result mit Daten
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    if not target_mod:
        target_mod = source_mod

    # Prüfe Mod-Permission
    if source_mod != target_mod:
        if not provider.check_mod_permission(source_mod, target_mod, 'read', key):
            provider._log_access(
                source_mod=source_mod,
                target_mod=target_mod,
                action="read",
                scope=Scope.MOD_DATA,
                keys=[key] if key else [],
                success=False
            )
            return Result.default_user_error(
                info=f"Keine Berechtigung für '{source_mod}' auf Daten von '{target_mod}' zuzugreifen",
                exec_code=403
            )

    # Lese Mod-Daten
    path = f"{key}.json" if key else "data.json"

    try:
        data = provider.storage.read(
            path=path,
            scope=Scope.MOD_DATA,
            mod_name=target_mod
        )

        provider._log_access(
            source_mod=source_mod,
            target_mod=target_mod,
            action="read",
            scope=Scope.MOD_DATA,
            keys=[key] if key else ["*"],
            success=True
        )

        if data:
            return Result.ok(data=json.loads(data.decode()))
        return Result.ok(data={})

    except Exception as e:
        return Result.default_internal_error(str(e))


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def set_mod_data(
    app: App,
    request: RequestData,
    source_mod: str,
    data: Dict,
    target_mod: str = None,
    key: str = None,
    merge: bool = True
):
    """
    Mod-Daten speichern (Legacy API Kompatibilität)

    Args:
        source_mod: Name des anfragenden Moduls
        data: Zu speichernde Daten
        target_mod: Name des Ziel-Moduls (default: source_mod)
        key: Optionaler spezifischer Schlüssel
        merge: Daten mergen statt überschreiben

    Returns:
        Result
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    if not target_mod:
        target_mod = source_mod

    # Prüfe Mod-Permission für fremde Mods
    if source_mod != target_mod:
        if not provider.check_mod_permission(source_mod, target_mod, 'write', key):
            return Result.default_user_error(
                info=f"Keine Schreibberechtigung für '{source_mod}' auf Daten von '{target_mod}'",
                exec_code=403
            )

    path = f"{key}.json" if key else "data.json"

    try:
        if merge:
            # Lade existierende Daten
            existing = provider.storage.read(
                path=path,
                scope=Scope.MOD_DATA,
                mod_name=target_mod
            )

            if existing:
                existing_data = json.loads(existing.decode())
                existing_data.update(data)
                data = existing_data

        # Speichere
        provider.storage.write(
            path=path,
            data=json.dumps(data).encode(),
            scope=Scope.MOD_DATA,
            mod_name=target_mod
        )

        provider._log_access(
            source_mod=source_mod,
            target_mod=target_mod,
            action="write",
            scope=Scope.MOD_DATA,
            keys=list(data.keys()),
            success=True
        )

        return Result.ok(data_info="Daten gespeichert")

    except Exception as e:
        return Result.default_internal_error(str(e))


# =================== Permission Management ===================

@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def request_permission(
    app: App,
    request: RequestData,
    source_mod: str,
    target_mod: str,
    permission_type: str = 'read',
    reason: str = ""
):
    """
    Berechtigung für Zugriff auf andere Mod-Daten anfordern
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    # Erstelle pending Permission
    perm_key = f"{source_mod}::{target_mod}"
    provider._permissions[perm_key] = ModPermission(
        source_mod=source_mod,
        target_mod=target_mod,
        permission_type=permission_type,
        granted=False,
        reason=reason
    )
    provider._save_permissions()

    return Result.ok(
        data={'request_id': perm_key, 'status': 'pending'},
        data_info=f"Berechtigungsanfrage für '{target_mod}' erstellt"
    )


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def grant_permission(
    app: App,
    request: RequestData,
    source_mod: str,
    target_mod: str,
    permission_type: str = 'read',
    keys: List[str] = None,
    expires_hours: int = 0
):
    """
    Berechtigung erteilen (vom Benutzer aufgerufen)
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    provider.grant_permission(
        source_mod=source_mod,
        target_mod=target_mod,
        permission_type=permission_type,
        keys=keys,
        expires_hours=expires_hours
    )

    return Result.ok(data_info=f"Berechtigung für '{source_mod}' auf '{target_mod}' erteilt")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def revoke_permission(
    app: App,
    request: RequestData,
    source_mod: str,
    target_mod: str
):
    """
    Berechtigung widerrufen
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    provider.revoke_permission(source_mod, target_mod)

    return Result.ok(data_info=f"Berechtigung für '{source_mod}' auf '{target_mod}' widerrufen")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def list_permissions(app: App, request: RequestData):
    """
    Alle Berechtigungen auflisten
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    return Result.ok(data=provider.list_permissions())


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def get_access_log(app: App, request: RequestData, limit: int = 50):
    """
    Zugriffs-Log abrufen
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    return Result.ok(data=provider.get_access_log(limit))


# =================== Sync API ===================

@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def sync(app: App, request: RequestData):
    """
    Synchronisiert private Daten mit Cloud
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    stats = provider.storage.sync_private()

    return Result.ok(
        data=stats,
        data_info=f"Sync: {stats.get('uploaded', 0)} hochgeladen, {stats.get('downloaded', 0)} heruntergeladen"
    )


# =================== Convenience Client ===================

# =================== Credential Broker Endpoint (Szenario A) ===================

_credential_broker = None

def _get_credential_broker():
    """Lazy-init CredentialBroker from environment."""
    global _credential_broker
    if _credential_broker is None:
        import os
        endpoint = os.environ.get('MINIO_ENDPOINT')
        access_key = os.environ.get('MINIO_ACCESS_KEY')
        secret_key = os.environ.get('MINIO_SECRET_KEY')
        secure = os.environ.get('MINIO_SECURE', 'false').lower() == 'true'
        if not endpoint or not access_key:
            return None
        try:
            from .auth.minio_policy import CredentialBroker, MinIOPolicyConfig
            config = MinIOPolicyConfig(
                endpoint=endpoint, access_key=access_key, secret_key=secret_key,
                secure=secure,
            )
            _credential_broker = CredentialBroker(config)
        except Exception:
            return None
    return _credential_broker


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def get_secure_storage_credentials(app: App, request: RequestData):
    """
    Vend scoped MinIO credentials for the authenticated user.
    Client uses these to talk DIRECTLY to MinIO (encrypted, no server proxy).

    Returns:
        {endpoint, access_key, secret_key, secure, buckets, user_prefix}
    """
    provider = await _get_storage_provider(app, request)
    if not provider:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    broker = _get_credential_broker()
    if not broker:
        return Result.default_internal_error(
            info="Storage credentials not available (MINIO_ENDPOINT not configured)"
        )

    try:
        creds = broker.vend_user_credentials(provider.user.user_id)
        return Result.ok(
            data=creds,
            data_info="Scoped storage credentials",
        )
    except Exception as e:
        return Result.default_internal_error(info=f"Credential vending failed: {e}")


class ModDataClient:
    """
    Hilfsklasse für einfachen Zugriff auf Mod-Daten

    Usage:
        client = ModDataClient(app, request, 'MyModName')

        # Eigene Daten
        data = await client.get()
        await client.set({'key': 'value'})

        # Andere Scopes
        public = await client.get_public('announcement.json')
        await client.set_shared('profile.json', {'name': 'Test'})
    """

    def __init__(self, app: App, request: RequestData, mod_name: str):
        self.app = app
        self.request = request
        self.mod_name = mod_name

    async def get(self, key: str = None) -> dict:
        """Eigene Mod-Daten abrufen"""
        result = await get_mod_data(
            self.app, self.request,
            source_mod=self.mod_name,
            key=key
        )
        return result.get('data', {}) if isinstance(result, dict) else {}

    async def set(self, data: dict, merge: bool = True) -> bool:
        """Eigene Mod-Daten speichern"""
        result = await set_mod_data(
            self.app, self.request,
            source_mod=self.mod_name,
            data=data,
            merge=merge
        )
        return result.get('ok', False) if isinstance(result, dict) else False

    async def get_from(self, target_mod: str, key: str = None) -> dict:
        """Daten eines anderen Mods abrufen"""
        result = await get_mod_data(
            self.app, self.request,
            source_mod=self.mod_name,
            target_mod=target_mod,
            key=key
        )
        return result.get('data', {}) if isinstance(result, dict) else {}

    async def get_private(self, path: str) -> Any:
        """Private Daten abrufen"""
        result = await get_data(
            self.app, self.request,
            path=path,
            scope='private'
        )
        return result.get('data') if isinstance(result, dict) else None

    async def set_private(self, path: str, data: Any) -> bool:
        """Private Daten speichern"""
        result = await set_data(
            self.app, self.request,
            path=path,
            data=data,
            scope='private'
        )
        return result.get('ok', False) if isinstance(result, dict) else False

    async def get_public(self, path: str) -> Any:
        """Public read Daten abrufen"""
        result = await get_data(
            self.app, self.request,
            path=path,
            scope='public_read'
        )
        return result.get('data') if isinstance(result, dict) else None

    async def get_shared(self, path: str, owner_id: str = None) -> Any:
        """User public Daten abrufen"""
        result = await get_data(
            self.app, self.request,
            path=path,
            scope='user_public',
            owner_id=owner_id
        )
        return result.get('data') if isinstance(result, dict) else None

    async def set_shared(self, path: str, data: Any) -> bool:
        """Eigene shared Daten speichern"""
        result = await set_data(
            self.app, self.request,
            path=path,
            data=data,
            scope='user_public'
        )
        return result.get('ok', False) if isinstance(result, dict) else False




# =================== Test ===================

if __name__ == "__main__":
    print("=== User Data API Test ===\n")

    # Test Scope conversion
    print("Scope Tests:")
    for scope_str in ['private', 'public', 'shared', 'public_read', 'mod_data']:
        scope = _scope_from_string(scope_str)
        print(f"  '{scope_str}' -> {scope.name}")

    print("\n✓ Module loaded successfully!")
