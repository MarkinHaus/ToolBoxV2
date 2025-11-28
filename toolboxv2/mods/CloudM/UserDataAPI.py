# toolboxv2/mods/CloudM/UserDataAPI.py
"""
ToolBox V2 - User Data API
Programmatische Schnittstelle für Mod-zu-Mod Datenzugriff
Mit Berechtigungssystem (Permission-based access)

Features:
- Mods können eigene Daten speichern und abrufen
- Berechtigungsbasierter Zugriff auf Daten anderer Mods
- Benutzer kann Berechtigungen pro Mod verwalten
- Audit-Log für Datenzugriffe
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from toolboxv2 import App, RequestData, Result, get_app

Name = 'CloudM.UserDataAPI'
export = get_app(f"{Name}.Export").tb
version = '0.1.0'


# =================== Data Classes ===================

@dataclass
class ModPermission:
    """Berechtigung für Mod-Datenzugriff"""
    source_mod: str  # Mod die Zugriff anfragt
    target_mod: str  # Mod auf deren Daten zugegriffen wird
    permission_type: str  # 'read', 'write', 'full'
    granted: bool = False
    granted_at: float = 0
    expires_at: float = 0  # 0 = never expires
    granted_keys: List[str] = field(default_factory=list)  # Leere Liste = alle Keys


@dataclass
class DataAccessLog:
    """Audit-Log Eintrag für Datenzugriff"""
    timestamp: float
    source_mod: str
    target_mod: str
    action: str  # 'read', 'write', 'delete'
    keys_accessed: List[str]
    success: bool
    user_id: str


# =================== Helper Functions ===================

def _get_user_permissions(app: App, user_id: str) -> Dict[str, List[dict]]:
    """Berechtigungen eines Benutzers aus DB laden"""
    try:
        result = app.run_any('DB', 'get', query=f"ModPermissions::{user_id}", get_results=True)
        if result and not result.is_error() and result.get():
            data = result.get()
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            if isinstance(data, bytes):
                data = data.decode()
            if isinstance(data, str):
                return json.loads(data)
        return {}
    except Exception as e:
        app.logger.error(f"Error loading permissions: {e}")
        return {}


def _save_user_permissions(app: App, user_id: str, permissions: Dict):
    """Berechtigungen eines Benutzers in DB speichern"""
    try:
        app.run_any('DB', 'set',
                    query=f"ModPermissions::{user_id}",
                    data=json.dumps(permissions))
    except Exception as e:
        app.logger.error(f"Error saving permissions: {e}")


def _check_permission(permissions: Dict, source_mod: str, target_mod: str,
                      permission_type: str, key: Optional[str] = None) -> bool:
    """Überprüft ob eine Berechtigung existiert und gültig ist"""
    # Eigener Mod hat immer Zugriff auf eigene Daten
    if source_mod == target_mod:
        return True

    # Permission key erstellen
    perm_key = f"{source_mod}::{target_mod}"

    if perm_key not in permissions:
        return False

    perm = permissions[perm_key]

    # Prüfen ob Berechtigung erteilt wurde
    if not perm.get('granted', False):
        return False

    # Prüfen ob abgelaufen
    expires_at = perm.get('expires_at', 0)
    if expires_at > 0 and time.time() > expires_at:
        return False

    # Berechtigungstyp prüfen
    perm_type = perm.get('permission_type', '')
    if perm_type == 'full':
        pass  # Voller Zugriff
    elif perm_type == 'read' and permission_type in ['write', 'delete']:
        return False
    elif perm_type == 'write' and permission_type == 'delete':
        return False

    # Key-basierte Einschränkung prüfen
    granted_keys = perm.get('granted_keys', [])
    if granted_keys and key and key not in granted_keys:
        return False

    return True


def _log_access(app: App, user_id: str, source_mod: str, target_mod: str,
                action: str, keys: List[str], success: bool):
    """Audit-Log Eintrag erstellen"""
    try:
        log_entry = {
            'timestamp': time.time(),
            'source_mod': source_mod,
            'target_mod': target_mod,
            'action': action,
            'keys_accessed': keys,
            'success': success,
            'user_id': user_id
        }

        # Log in DB speichern (Ringpuffer - letzte 100 Einträge)
        logs_result = app.run_any('DB', 'get', query=f"ModAccessLog::{user_id}", get_results=True)
        logs = []
        if logs_result and not logs_result.is_error() and logs_result.get():
            data = logs_result.get()
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            if isinstance(data, bytes):
                data = data.decode()
            if isinstance(data, str):
                logs = json.loads(data)

        logs.append(log_entry)
        logs = logs[-100:]  # Nur letzte 100 behalten

        app.run_any('DB', 'set',
                    query=f"ModAccessLog::{user_id}",
                    data=json.dumps(logs))
    except Exception as e:
        app.logger.error(f"Error logging access: {e}")


async def _get_current_user(app: App, request: RequestData):
    """Aktuellen Benutzer aus Request holen"""
    from .UserAccountManager import get_current_user_from_request
    return await get_current_user_from_request(app, request)


# =================== Public API - Für andere Module ===================

@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def get_mod_data(app: App, request: RequestData,
                       source_mod: str, target_mod: str = None,
                       key: str = None):
    """
    Mod-Daten abrufen.

    Args:
        source_mod: Name des Moduls, das die Anfrage stellt
        target_mod: Name des Moduls, dessen Daten abgerufen werden (default: source_mod)
        key: Optionaler spezifischer Schlüssel (default: alle Daten)

    Returns:
        Result mit den angeforderten Daten

    Usage:
        # Eigene Daten abrufen
        result = await app.a_run_any('CloudM.UserDataAPI.get_mod_data',
                                     source_mod='MyMod', request=request)

        # Daten eines anderen Mods abrufen (erfordert Berechtigung)
        result = await app.a_run_any('CloudM.UserDataAPI.get_mod_data',
                                     source_mod='MyMod', target_mod='OtherMod',
                                     key='specific_setting', request=request)
    """
    user = await _get_current_user(app, request)
    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    user_id = getattr(user, 'uid', None) or getattr(user, 'clerk_user_id', None)
    if not user_id:
        return Result.default_user_error(info="Benutzer-ID nicht gefunden")

    # Default: Eigene Daten
    if not target_mod:
        target_mod = source_mod

    # Berechtigung prüfen
    permissions = _get_user_permissions(app, user_id)
    if not _check_permission(permissions, source_mod, target_mod, 'read', key):
        _log_access(app, user_id, source_mod, target_mod, 'read', [key] if key else [], False)
        return Result.default_user_error(
            info=f"Keine Berechtigung für '{source_mod}' auf Daten von '{target_mod}' zuzugreifen",
            exec_code=403
        )

    # Daten abrufen
    mod_data = {}
    if hasattr(user, 'mod_data') and user.mod_data:
        mod_data = user.mod_data.get(target_mod, {})
    elif hasattr(user, 'settings') and user.settings:
        mod_data = user.settings.get('mod_data', {}).get(target_mod, {})

    # Log erfolgreichen Zugriff
    accessed_keys = [key] if key else list(mod_data.keys())
    _log_access(app, user_id, source_mod, target_mod, 'read', accessed_keys, True)

    # Spezifischen Key oder alle Daten zurückgeben
    if key:
        return Result.ok(data={key: mod_data.get(key)})
    return Result.ok(data=mod_data)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def set_mod_data(app: App, request: RequestData,
                       source_mod: str, data: dict,
                       target_mod: str = None, merge: bool = True):
    """
    Mod-Daten speichern.

    Args:
        source_mod: Name des Moduls, das die Anfrage stellt
        data: Zu speichernde Daten (dict)
        target_mod: Name des Moduls, dessen Daten geändert werden (default: source_mod)
        merge: True = Daten mergen, False = Daten ersetzen

    Returns:
        Result mit aktualisierten Daten

    Usage:
        # Eigene Daten speichern
        result = await app.a_run_any('CloudM.UserDataAPI.set_mod_data',
                                     source_mod='MyMod',
                                     data={'score': 100, 'level': 5},
                                     request=request)
    """
    user = await _get_current_user(app, request)
    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    user_id = getattr(user, 'uid', None) or getattr(user, 'clerk_user_id', None)
    if not user_id:
        return Result.default_user_error(info="Benutzer-ID nicht gefunden")

    if not target_mod:
        target_mod = source_mod

    # Berechtigung prüfen
    permissions = _get_user_permissions(app, user_id)
    if not _check_permission(permissions, source_mod, target_mod, 'write'):
        _log_access(app, user_id, source_mod, target_mod, 'write', list(data.keys()), False)
        return Result.default_user_error(
            info=f"Keine Schreibberechtigung für '{source_mod}' auf Daten von '{target_mod}'",
            exec_code=403
        )

    # Daten aktualisieren
    try:
        if hasattr(user, 'mod_data'):
            if user.mod_data is None:
                user.mod_data = {}
            if target_mod not in user.mod_data:
                user.mod_data[target_mod] = {}

            if merge:
                user.mod_data[target_mod].update(data)
            else:
                user.mod_data[target_mod] = data

            updated_data = user.mod_data[target_mod]
        else:
            if user.settings is None:
                user.settings = {}
            if 'mod_data' not in user.settings:
                user.settings['mod_data'] = {}
            if target_mod not in user.settings['mod_data']:
                user.settings['mod_data'][target_mod] = {}

            if merge:
                user.settings['mod_data'][target_mod].update(data)
            else:
                user.settings['mod_data'][target_mod] = data

            updated_data = user.settings['mod_data'][target_mod]

        # Speichern
        from .UserAccountManager import _save_user_data
        save_result = _save_user_data(app, user) if hasattr(user, 'to_dict') else None

        if save_result is None:
            from .AuthManager import db_helper_save_user
            save_result = db_helper_save_user(app, asdict(user))

        if save_result and save_result.is_error():
            return save_result

        # Log
        _log_access(app, user_id, source_mod, target_mod, 'write', list(data.keys()), True)

        return Result.ok(data=updated_data, data_info="Daten gespeichert")

    except Exception as e:
        _log_access(app, user_id, source_mod, target_mod, 'write', list(data.keys()), False)
        return Result.default_internal_error(f"Fehler beim Speichern: {e}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def delete_mod_data(app: App, request: RequestData,
                          source_mod: str, keys: List[str] = None,
                          target_mod: str = None):
    """
    Mod-Daten löschen.

    Args:
        source_mod: Name des Moduls, das die Anfrage stellt
        keys: Liste der zu löschenden Schlüssel (None = alle Daten)
        target_mod: Name des Moduls, dessen Daten gelöscht werden

    Returns:
        Result mit Bestätigung
    """
    user = await _get_current_user(app, request)
    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    user_id = getattr(user, 'uid', None) or getattr(user, 'clerk_user_id', None)
    if not user_id:
        return Result.default_user_error(info="Benutzer-ID nicht gefunden")

    if not target_mod:
        target_mod = source_mod

    # Berechtigung prüfen (delete erfordert write oder full)
    permissions = _get_user_permissions(app, user_id)
    if not _check_permission(permissions, source_mod, target_mod, 'delete'):
        _log_access(app, user_id, source_mod, target_mod, 'delete', keys or ['*'], False)
        return Result.default_user_error(
            info=f"Keine Löschberechtigung für '{source_mod}' auf Daten von '{target_mod}'",
            exec_code=403
        )

    try:
        deleted_keys = []

        if hasattr(user, 'mod_data') and user.mod_data and target_mod in user.mod_data:
            if keys:
                for key in keys:
                    if key in user.mod_data[target_mod]:
                        del user.mod_data[target_mod][key]
                        deleted_keys.append(key)
            else:
                deleted_keys = list(user.mod_data[target_mod].keys())
                user.mod_data[target_mod] = {}
        elif hasattr(user, 'settings') and user.settings:
            mod_data = user.settings.get('mod_data', {}).get(target_mod, {})
            if keys:
                for key in keys:
                    if key in mod_data:
                        del mod_data[key]
                        deleted_keys.append(key)
            else:
                deleted_keys = list(mod_data.keys())
                if 'mod_data' in user.settings and target_mod in user.settings['mod_data']:
                    user.settings['mod_data'][target_mod] = {}

        # Speichern
        from .AuthManager import db_helper_save_user
        save_result = db_helper_save_user(app, asdict(user))

        if save_result and save_result.is_error():
            return save_result

        _log_access(app, user_id, source_mod, target_mod, 'delete', deleted_keys, True)

        return Result.ok(data={'deleted_keys': deleted_keys}, data_info="Daten gelöscht")

    except Exception as e:
        _log_access(app, user_id, source_mod, target_mod, 'delete', keys or ['*'], False)
        return Result.default_internal_error(f"Fehler beim Löschen: {e}")


# =================== Berechtigungs-Management ===================

@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def request_permission(app: App, request: RequestData,
                             source_mod: str, target_mod: str,
                             permission_type: str = 'read',
                             keys: List[str] = None,
                             reason: str = ""):
    """
    Berechtigung von einem anderen Mod anfordern.
    Der Benutzer muss dies noch genehmigen.

    Args:
        source_mod: Mod die Zugriff anfordert
        target_mod: Mod auf deren Daten zugegriffen werden soll
        permission_type: 'read', 'write', oder 'full'
        keys: Optionale Liste spezifischer Keys (None = alle)
        reason: Begründung für den Benutzer

    Returns:
        Result mit Anfrage-ID
    """
    user = await _get_current_user(app, request)
    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    user_id = getattr(user, 'uid', None) or getattr(user, 'clerk_user_id', None)
    if not user_id:
        return Result.default_user_error(info="Benutzer-ID nicht gefunden")

    if permission_type not in ['read', 'write', 'full']:
        return Result.default_user_error(info="Ungültiger Berechtigungstyp")

    # Bestehende Berechtigung prüfen
    permissions = _get_user_permissions(app, user_id)
    perm_key = f"{source_mod}::{target_mod}"

    if perm_key in permissions and permissions[perm_key].get('granted', False):
        return Result.ok(data={'status': 'already_granted'},
                         data_info="Berechtigung bereits erteilt")

    # Anfrage erstellen
    request_data = {
        'source_mod': source_mod,
        'target_mod': target_mod,
        'permission_type': permission_type,
        'granted_keys': keys or [],
        'reason': reason,
        'requested_at': time.time(),
        'granted': False
    }

    # Als pending speichern
    permissions[perm_key] = request_data
    _save_user_permissions(app, user_id, permissions)

    return Result.ok(
        data={'request_id': perm_key, 'status': 'pending'},
        data_info=f"Berechtigungsanfrage für '{target_mod}' erstellt. Benutzer muss genehmigen."
    )


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def grant_permission(app: App, request: RequestData,
                           source_mod: str, target_mod: str,
                           permission_type: str = 'read',
                           keys: List[str] = None,
                           expires_hours: int = 0):
    """
    Berechtigung erteilen (vom Benutzer aufgerufen).

    Args:
        source_mod: Mod die Zugriff erhält
        target_mod: Mod auf deren Daten zugegriffen werden darf
        permission_type: 'read', 'write', oder 'full'
        keys: Optionale Liste spezifischer Keys (None = alle)
        expires_hours: Ablaufzeit in Stunden (0 = nie)
    """
    user = await _get_current_user(app, request)
    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    user_id = getattr(user, 'uid', None) or getattr(user, 'clerk_user_id', None)

    permissions = _get_user_permissions(app, user_id)
    perm_key = f"{source_mod}::{target_mod}"

    expires_at = 0
    if expires_hours > 0:
        expires_at = time.time() + (expires_hours * 3600)

    permissions[perm_key] = {
        'source_mod': source_mod,
        'target_mod': target_mod,
        'permission_type': permission_type,
        'granted_keys': keys or [],
        'granted': True,
        'granted_at': time.time(),
        'expires_at': expires_at
    }

    _save_user_permissions(app, user_id, permissions)

    return Result.ok(data_info=f"Berechtigung für '{source_mod}' auf '{target_mod}' erteilt")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def revoke_permission(app: App, request: RequestData,
                            source_mod: str, target_mod: str):
    """
    Berechtigung widerrufen (vom Benutzer aufgerufen).
    """
    user = await _get_current_user(app, request)
    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    user_id = getattr(user, 'uid', None) or getattr(user, 'clerk_user_id', None)

    permissions = _get_user_permissions(app, user_id)
    perm_key = f"{source_mod}::{target_mod}"

    if perm_key in permissions:
        del permissions[perm_key]
        _save_user_permissions(app, user_id, permissions)
        return Result.ok(data_info=f"Berechtigung für '{source_mod}' auf '{target_mod}' widerrufen")

    return Result.ok(data_info="Keine Berechtigung gefunden")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def list_permissions(app: App, request: RequestData):
    """
    Alle Berechtigungen des aktuellen Benutzers auflisten.
    Für Dashboard-Anzeige.
    """
    user = await _get_current_user(app, request)
    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    user_id = getattr(user, 'uid', None) or getattr(user, 'clerk_user_id', None)

    permissions = _get_user_permissions(app, user_id)

    # In lesbare Liste umwandeln
    perm_list = []
    for perm_key, perm_data in permissions.items():
        perm_list.append({
            'source_mod': perm_data.get('source_mod'),
            'target_mod': perm_data.get('target_mod'),
            'permission_type': perm_data.get('permission_type'),
            'granted': perm_data.get('granted', False),
            'granted_at': perm_data.get('granted_at', 0),
            'expires_at': perm_data.get('expires_at', 0),
            'granted_keys': perm_data.get('granted_keys', []),
            'reason': perm_data.get('reason', '')
        })

    return Result.ok(data=perm_list)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)
async def get_access_log(app: App, request: RequestData, limit: int = 50):
    """
    Zugriffs-Log des aktuellen Benutzers abrufen.
    """
    user = await _get_current_user(app, request)
    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    user_id = getattr(user, 'uid', None) or getattr(user, 'clerk_user_id', None)

    try:
        logs_result = app.run_any('DB', 'get', query=f"ModAccessLog::{user_id}", get_results=True)
        logs = []
        if logs_result and not logs_result.is_error() and logs_result.get():
            data = logs_result.get()
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            if isinstance(data, bytes):
                data = data.decode()
            if isinstance(data, str):
                logs = json.loads(data)

        # Neueste zuerst, limitieren
        logs = sorted(logs, key=lambda x: x.get('timestamp', 0), reverse=True)[:limit]

        return Result.ok(data=logs)
    except Exception as e:
        return Result.default_internal_error(f"Fehler beim Laden der Logs: {e}")


# =================== Convenience Functions für Mods ===================

class ModDataClient:
    """
    Hilfsklasse für einfachen Zugriff auf Mod-Daten.

    Usage in einem Mod:
        from toolboxv2.mods.CloudM.UserDataAPI import ModDataClient

        async def my_function(app, request):
            client = ModDataClient(app, request, 'MyModName')

            # Eigene Daten lesen
            my_data = await client.get()

            # Eigene Daten schreiben
            await client.set({'score': 100})

            # Daten eines anderen Mods lesen (erfordert Berechtigung)
            other_data = await client.get_from('OtherMod')
    """

    def __init__(self, app: App, request: RequestData, mod_name: str):
        self.app = app
        self.request = request
        self.mod_name = mod_name

    async def get(self, key: str = None) -> dict:
        """Eigene Mod-Daten abrufen"""
        result = await self.app.a_run_any(
            f'{Name}.get_mod_data',
            source_mod=self.mod_name,
            key=key,
            request=self.request,
            get_results=True
        )
        if result.is_error():
            return {}
        return result.get() or {}

    async def set(self, data: dict, merge: bool = True) -> bool:
        """Eigene Mod-Daten speichern"""
        result = await self.app.a_run_any(
            f'{Name}.set_mod_data',
            source_mod=self.mod_name,
            data=data,
            merge=merge,
            request=self.request,
            get_results=True
        )
        return not result.is_error()

    async def delete(self, keys: List[str] = None) -> bool:
        """Eigene Mod-Daten löschen"""
        result = await self.app.a_run_any(
            f'{Name}.delete_mod_data',
            source_mod=self.mod_name,
            keys=keys,
            request=self.request,
            get_results=True
        )
        return not result.is_error()

    async def get_from(self, target_mod: str, key: str = None) -> dict:
        """Daten eines anderen Mods abrufen (erfordert Berechtigung)"""
        result = await self.app.a_run_any(
            f'{Name}.get_mod_data',
            source_mod=self.mod_name,
            target_mod=target_mod,
            key=key,
            request=self.request,
            get_results=True
        )
        if result.is_error():
            return {}
        return result.get() or {}

    async def request_access(self, target_mod: str,
                             permission_type: str = 'read',
                             reason: str = "") -> bool:
        """Zugriff auf Daten eines anderen Mods anfordern"""
        result = await self.app.a_run_any(
            f'{Name}.request_permission',
            source_mod=self.mod_name,
            target_mod=target_mod,
            permission_type=permission_type,
            reason=reason,
            request=self.request,
            get_results=True
        )
        return not result.is_error()
