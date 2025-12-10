# toolboxv2/mods/CloudM/UserAccountManager.py
"""
ToolBox V2 - User Account Manager
Benutzerkonten-Verwaltung mit Clerk-Integration
Stellt API-Endpunkte für Dashboard und programmatischen Zugriff bereit
"""

import time
from dataclasses import asdict
from typing import Optional, Dict, Any

from toolboxv2 import App, RequestData, Result, get_app

Name = 'CloudM.UserAccountManager'
export = get_app(f"{Name}.Export").tb
version = '0.1.1'


# =================== Core Helper Functions ===================

async def get_current_user_from_request(app: App, request: RequestData):
    """
    Holt den aktuellen Benutzer aus der Request-Session.
    Funktioniert mit Clerk und Legacy-Auth.

    Returns:
        User-Objekt (LocalUserData oder legacy User) oder None
    """
    if not request or not hasattr(request, 'session') or not request.session:
        app.logger.warning("UAM: Keine Session im Request gefunden")
        return None

    # Benutzer-Identifikator aus Session extrahieren
    clerk_user_id = None
    username = None

    # Clerk User ID prüfen
    if hasattr(request.session, 'clerk_user_id') and request.session.clerk_user_id:
        clerk_user_id = request.session.clerk_user_id
    elif hasattr(request.session, 'user_id') and request.session.user_id:
        clerk_user_id = request.session.user_id
    elif hasattr(request.session, 'user_name') and request.session.user_name:
        clerk_user_id = request.session.extra_data.get('clerk_user_id')
        username = request.session.user_name

    if not clerk_user_id:
        app.logger.debug("UAM: Kein gültiger Benutzer-Identifikator in Session")
        return None

    # Benutzer laden
    return await _load_user_data(app, clerk_user_id, username)


async def _load_user_data(app: App, clerk_user_id: str, username):
    """Lädt Benutzerdaten aus verschiedenen Quellen"""
    # Versuche zuerst Clerk/AuthClerk
    try:
        from .AuthClerk import load_local_user_data, _db_load_user_sync_data, LocalUserData

        local_data = load_local_user_data(clerk_user_id)
        if local_data:
            return local_data

        db_data = _db_load_user_sync_data(app, clerk_user_id)
        if db_data:
            return LocalUserData.from_dict(db_data)

    except ImportError:
        pass  # AuthClerk nicht verfügbar
    except Exception as e:
        app.logger.error(f"UAM: Fehler beim Laden via AuthClerk: {e}")

    return None


def _save_user_data(app: App, user_data) -> Result:
    """
    Speichert Benutzerdaten - unterstützt Clerk und Legacy.
    """
    try:
        # Clerk LocalUserData
        if hasattr(user_data, 'to_dict'):
            from .AuthClerk import save_local_user_data, _db_save_user_sync_data

            user_data.last_sync = time.time()
            save_local_user_data(user_data)
            _db_save_user_sync_data(app, user_data.clerk_user_id, user_data.to_dict())
            return Result.ok("Benutzerdaten gespeichert")

        # Legacy User Objekt
        from .AuthManager import db_helper_save_user
        return db_helper_save_user(app, asdict(user_data))

    except ImportError:
        # Nur Legacy verfügbar
        from .AuthManager import db_helper_save_user
        return db_helper_save_user(app, asdict(user_data))
    except Exception as e:
        return Result.default_internal_error(f"Fehler beim Speichern: {e}")


def _get_user_attribute(user, attr: str, default=None):
    """Sicheres Abrufen von Benutzerattributen"""
    if hasattr(user, attr):
        return getattr(user, attr)
    if hasattr(user, 'to_dict'):
        return user.to_dict().get(attr, default)
    return default


# =================== API Endpoints ===================

@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False)
async def get_current_user(app: App, request: RequestData):
    """
    API-Endpunkt: Aktuelle Benutzerdaten abrufen.
    Gibt öffentliche Benutzerdaten für Frontend zurück.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return Result.default_user_error(
            info="Benutzer nicht authentifiziert oder nicht gefunden",
            exec_code=401
        )

    # Öffentliche Daten zusammenstellen
    user_data = {
        "clerk_user_id": _get_user_attribute(user, 'clerk_user_id'),
        "username": _get_user_attribute(user, 'username') or _get_user_attribute(user, 'name'),
        "name": _get_user_attribute(user, 'name') or _get_user_attribute(user, 'username'),
        "email": _get_user_attribute(user, 'email'),
        "level": _get_user_attribute(user, 'level', 1),
        "settings": _get_user_attribute(user, 'settings', {}),
        "mod_data": _get_user_attribute(user, 'mod_data', {}),
        "is_persona": _get_user_attribute(user, 'is_persona', False),
        "uid": _get_user_attribute(user, 'uid')
    }

    return Result.ok(data=user_data)


# API Wrapper für Kompatibilität mit altem Namen
@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False,
        name="get_current_user_from_request_api_wrapper")
async def get_current_user_api_wrapper(app: App, request: RequestData):
    """Wrapper für Abwärtskompatibilität"""
    return await get_current_user(app, request=request)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True)
async def update_email(app: App, request: RequestData, new_email: str = None):
    """
    E-Mail-Adresse aktualisieren.
    Bei Clerk: Weiterleitung zu Clerk-Profil.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return """
            <div class="tb-alert tb-alert-error tb-p-4 tb-rounded">
                <p class="tb-font-semibold">Fehler</p>
                <p>Benutzer nicht authentifiziert.</p>
            </div>
        """

    current_email = _get_user_attribute(user, 'email', 'Nicht angegeben')
    is_clerk = hasattr(user, 'clerk_user_id') and user.clerk_user_id

    if is_clerk:
        return f"""
            <div class="tb-space-y-2">
                <p><strong>Aktuelle E-Mail:</strong> {current_email}</p>
                <p class="tb-text-sm tb-text-muted">
                    E-Mail-Änderungen werden aus Sicherheitsgründen über Clerk verwaltet.
                </p>
                <button onclick="window.TB?.user?.getClerkInstance()?.openUserProfile()"
                        class="tb-btn tb-btn-secondary tb-mt-2">
                    <span class="material-symbols-outlined tb-mr-1">settings</span>
                    Profil-Einstellungen öffnen
                </button>
            </div>
        """
    else:
        # Legacy: Direkte Aktualisierung
        if new_email and new_email != current_email:
            user.email = new_email
            save_result = _save_user_data(app, user)

            if save_result.is_error():
                return f"""
                    <div class="tb-alert tb-alert-error">
                        Fehler beim Speichern: {save_result.info}
                    </div>
                """

            return f"""
                <div class="tb-space-y-2">
                    <p><strong>E-Mail aktualisiert:</strong> {new_email}</p>
                    <p class="tb-text-success tb-text-sm">✓ Gespeichert</p>
                </div>
            """

        return f"""
            <div class="tb-space-y-2">
                <p><strong>Aktuelle E-Mail:</strong> {current_email}</p>
                <input type="email" name="new_email" value="{current_email if current_email != 'Nicht angegeben' else ''}"
                       class="tb-input tb-mt-2" placeholder="Neue E-Mail-Adresse">
                <button data-hx-post="/api/{Name}/update_email"
                        data-hx-include="[name='new_email']"
                        data-hx-target="closest div"
                        data-hx-swap="innerHTML"
                        class="tb-btn tb-btn-primary tb-mt-2">
                    <span class="material-symbols-outlined tb-mr-1">save</span>
                    E-Mail aktualisieren
                </button>
            </div>
        """


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True)
async def update_setting(app: App, request: RequestData, setting_key: str, setting_value: str):
    """
    Einzelne Benutzereinstellung aktualisieren.
    Gibt HTML für HTMX-Update zurück.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return "<div class='tb-alert tb-alert-error'>Fehler: Nicht authentifiziert.</div>"

    # Wert parsen
    if setting_value.lower() == 'true':
        actual_value = True
    elif setting_value.lower() == 'false':
        actual_value = False
    elif setting_value.isdigit():
        actual_value = int(setting_value)
    else:
        try:
            actual_value = float(setting_value)
        except ValueError:
            actual_value = setting_value

    # Einstellung aktualisieren
    if hasattr(user, 'settings'):
        if user.settings is None:
            user.settings = {}
        user.settings[setting_key] = actual_value
    else:
        setattr(user, 'settings', {setting_key: actual_value})

    # Speichern
    save_result = _save_user_data(app, user)

    if save_result.is_error():
        return f"""
            <div class="tb-alert tb-alert-error tb-text-sm">
                Fehler beim Speichern: {save_result.info if hasattr(save_result, 'info') else 'Unbekannt'}
            </div>
        """

    # Erfolgs-Response basierend auf Setting-Typ
    if setting_key == "experimental_features":
        is_checked = "checked" if actual_value else ""
        next_value = "false" if actual_value else "true"
        return f"""
            <label class="tb-label tb-flex tb-items-center tb-cursor-pointer">
                <input type="checkbox" {is_checked}
                       data-hx-post="/api/{Name}/update_setting"
                       data-hx-vals='{{"setting_key": "experimental_features", "setting_value": "{next_value}"}}'
                       data-hx-target="closest div"
                       data-hx-swap="innerHTML"
                       class="tb-checkbox tb-mr-2">
                <span class="tb-text-sm">Experimentelle Funktionen aktivieren</span>
            </label>
            <span class="tb-text-success tb-text-xs tb-ml-2">✓</span>
        """

    return f"""
        <div class="tb-text-success tb-text-sm">
            ✓ '{setting_key}' auf '{actual_value}' aktualisiert
        </div>
    """


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False)
async def update_settings_batch(app: App, request: RequestData, settings: dict):
    """
    Mehrere Einstellungen auf einmal aktualisieren.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    if not isinstance(settings, dict):
        return Result.default_user_error(info="Ungültiges Einstellungsformat")

    # Einstellungen aktualisieren
    if hasattr(user, 'settings'):
        if user.settings is None:
            user.settings = {}
        user.settings.update(settings)
    else:
        setattr(user, 'settings', settings)

    # Speichern
    save_result = _save_user_data(app, user)

    if save_result.is_error():
        return save_result

    return Result.ok(
        data=_get_user_attribute(user, 'settings', {}),
        data_info="Einstellungen gespeichert"
    )


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False)
async def update_mod_data(app: App, request: RequestData, mod_name: str, data: dict):
    """
    Mod-spezifische Daten für den aktuellen Benutzer aktualisieren.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    try:
        # Mod-Daten aktualisieren
        if hasattr(user, 'mod_data'):
            if user.mod_data is None:
                user.mod_data = {}
            if mod_name not in user.mod_data:
                user.mod_data[mod_name] = {}
            user.mod_data[mod_name].update(data)
            updated_data = user.mod_data[mod_name]
        else:
            # Fallback in settings speichern
            if not hasattr(user, 'settings') or user.settings is None:
                user.settings = {}
            if 'mod_data' not in user.settings:
                user.settings['mod_data'] = {}
            if mod_name not in user.settings['mod_data']:
                user.settings['mod_data'][mod_name] = {}
            user.settings['mod_data'][mod_name].update(data)
            updated_data = user.settings['mod_data'][mod_name]

        # Speichern
        save_result = _save_user_data(app, user)

        if save_result.is_error():
            return save_result

        return Result.ok(data=updated_data, data_info=f"Mod-Daten für '{mod_name}' aktualisiert")

    except Exception as e:
        return Result.default_internal_error(f"Fehler: {e}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False)
async def get_mod_data(app: App, request: RequestData, mod_name: str):
    """
    Mod-spezifische Daten für den aktuellen Benutzer abrufen.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    mod_data = {}
    if hasattr(user, 'mod_data') and user.mod_data:
        mod_data = user.mod_data.get(mod_name, {})
    elif hasattr(user, 'settings') and user.settings:
        mod_data = user.settings.get('mod_data', {}).get(mod_name, {})

    return Result.ok(data=mod_data)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False)
async def delete_mod_data(app: App, request: RequestData, mod_name: str, keys: list = None):
    """
    Mod-Daten löschen (bestimmte Keys oder alle).
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    deleted_keys = []

    try:
        if hasattr(user, 'mod_data') and user.mod_data and mod_name in user.mod_data:
            if keys:
                for key in keys:
                    if key in user.mod_data[mod_name]:
                        del user.mod_data[mod_name][key]
                        deleted_keys.append(key)
            else:
                deleted_keys = list(user.mod_data[mod_name].keys())
                user.mod_data[mod_name] = {}
        elif hasattr(user, 'settings') and user.settings:
            mod_data = user.settings.get('mod_data', {}).get(mod_name, {})
            if keys:
                for key in keys:
                    if key in mod_data:
                        del mod_data[key]
                        deleted_keys.append(key)
            else:
                deleted_keys = list(mod_data.keys())
                if 'mod_data' in user.settings and mod_name in user.settings['mod_data']:
                    user.settings['mod_data'][mod_name] = {}

        # Speichern
        save_result = _save_user_data(app, user)

        if save_result.is_error():
            return save_result

        return Result.ok(
            data={'deleted_keys': deleted_keys},
            data_info=f"{len(deleted_keys)} Schlüssel gelöscht"
        )

    except Exception as e:
        return Result.default_internal_error(f"Fehler: {e}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True)
async def get_account_section_html(app: App, request: RequestData):
    """
    HTML für den Account-Bereich im Dashboard generieren.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return """
            <div class="tb-card tb-p-4">
                <h3 class="tb-text-lg tb-font-semibold tb-mb-4">Kontoeinstellungen</h3>
                <p class="tb-text-warning">Bitte melden Sie sich an.</p>
                <button onclick="window.TB?.user?.signIn()" class="tb-btn tb-btn-primary tb-mt-4">
                    <span class="material-symbols-outlined tb-mr-1">login</span>
                    Anmelden
                </button>
            </div>
        """

    username = _get_user_attribute(user, 'username') or _get_user_attribute(user, 'name', 'Unbekannt')
    email = _get_user_attribute(user, 'email', 'Nicht angegeben')
    level = _get_user_attribute(user, 'level', 1)
    settings = _get_user_attribute(user, 'settings', {})
    is_clerk = hasattr(user, 'clerk_user_id') and user.clerk_user_id

    exp_features = settings.get('experimental_features', False)
    exp_checked = 'checked' if exp_features else ''
    exp_next = 'false' if exp_features else 'true'

    return f"""
        <div class="tb-card tb-p-4">
            <h3 class="tb-text-lg tb-font-semibold tb-mb-4">Kontoeinstellungen</h3>

            <div class="tb-space-y-4">
                <!-- Benutzerinfo -->
                <div class="tb-border-b tb-pb-4">
                    <p><strong>Benutzername:</strong> {username}</p>
                    <p><strong>E-Mail:</strong> {email}</p>
                    <p><strong>Level:</strong> {level}</p>
                </div>

                <!-- Profil-Button für Clerk -->
                {'<div><button onclick="window.TB?.user?.getClerkInstance()?.openUserProfile()" class="tb-btn tb-btn-secondary">Profil-Einstellungen öffnen</button></div>' if is_clerk else ''}

                <!-- App-Einstellungen -->
                <div class="tb-border-t tb-pt-4">
                    <h4 class="tb-font-semibold tb-mb-2">Anwendungseinstellungen</h4>

                    <div id="setting-experimental" class="tb-mb-2">
                        <label class="tb-label tb-flex tb-items-center tb-cursor-pointer">
                            <input type="checkbox" {exp_checked}
                                   data-hx-post="/api/{Name}/update_setting"
                                   data-hx-vals='{{"setting_key": "experimental_features", "setting_value": "{exp_next}"}}'
                                   data-hx-target="closest div"
                                   data-hx-swap="innerHTML"
                                   class="tb-checkbox tb-mr-2">
                            Experimentelle Funktionen aktivieren
                        </label>
                    </div>
                </div>

                <!-- Abmelden -->
                <div class="tb-border-t tb-pt-4">
                    <button onclick="window.TB?.user?.signOut()" class="tb-btn tb-btn-danger">
                        <span class="material-symbols-outlined tb-mr-1">logout</span>
                        Abmelden
                    </button>
                </div>
            </div>
        </div>
    """


# =================== Hilfsfunktionen für andere Module ===================

async def get_user_settings(app: App, request: RequestData) -> dict:
    """
    Convenience-Funktion: Nur Benutzereinstellungen abrufen.
    """
    user = await get_current_user_from_request(app, request)
    if not user:
        return {}
    return _get_user_attribute(user, 'settings', {})


async def get_user_mod_data(app: App, request: RequestData, mod_name: str) -> dict:
    """
    Convenience-Funktion: Mod-Daten für einen bestimmten Mod abrufen.
    """
    user = await get_current_user_from_request(app, request)
    if not user:
        return {}

    if hasattr(user, 'mod_data') and user.mod_data:
        return user.mod_data.get(mod_name, {})
    elif hasattr(user, 'settings') and user.settings:
        return user.settings.get('mod_data', {}).get(mod_name, {})
    return {}


async def set_user_mod_data(app: App, request: RequestData, mod_name: str, data: dict) -> bool:
    """
    Convenience-Funktion: Mod-Daten speichern.
    Gibt True bei Erfolg zurück.
    """
    result = await update_mod_data(app, request=request, mod_name=mod_name, data=data)
    return not result.is_error()
