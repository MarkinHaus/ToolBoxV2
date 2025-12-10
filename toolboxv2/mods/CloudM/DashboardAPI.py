"""
ToolBox V2 - Dashboard API Endpoints mit Minu Integration
=========================================================
Backend-Endpunkte für die Minu-basierten Dashboards mit:
- Zuverlässige Logout-Logik
- Session-Management
- Event-Handling für Minu Views
"""

import json
from dataclasses import asdict
from typing import Dict, Any, Optional

from toolboxv2 import App, RequestData, Result, get_app, TBEF
from toolboxv2.mods.Minu import register_view, get_or_create_session

# Imports für User/Auth Management
from toolboxv2.mods.CloudM.AuthManager import db_helper_save_user
from toolboxv2.mods.CloudM.UserAccountManager import get_current_user_from_request
from toolboxv2.mods.CloudM.types import User

Name = "CloudM.DashboardAPI"
export = get_app(Name + ".Export").tb
version = "0.1.0"


# ============================================================================
# LOGOUT-HANDLER - Kritische Funktion
# ============================================================================


@export(
    mod_name=Name,
    api=True,
    version=version,
    request_as_kwarg=True,
    api_methods=["POST", "GET"],
)
async def logout(app: App, request: RequestData):
    """
    Zuverlässiger Logout-Endpunkt.

    Führt folgende Schritte aus:
    1. Invalidiert Server-Session
    2. Löscht Session-Cookies
    3. Benachrichtigt Clerk (falls verwendet)
    4. Räumt Minu-Sessions auf
    5. Leitet zur Login-Seite weiter

    Kann sowohl via POST (AJAX) als auch GET (Link) aufgerufen werden.
    """
    try:
        # 1. Aktuellen User holen (falls vorhanden)
        current_user = await get_current_user_from_request(app, request)
        user_id = None

        if current_user:
            user_id = getattr(current_user, "uid", None) or getattr(
                current_user, "clerk_user_id", None
            )
            app.logger.info(
                f"[Logout] Logging out user: {getattr(current_user, 'name', 'unknown')}"
            )

        # 2. Minu-Session aufräumen (falls vorhanden)
        if user_id:
            from toolboxv2.mods.Minu import cleanup_session

            try:
                cleanup_session(user_id)
                app.logger.debug(f"[Logout] Minu session cleaned up for {user_id}")
            except Exception as e:
                app.logger.warning(f"[Logout] Could not cleanup Minu session: {e}")

        # 3. User Instance schließen (falls vorhanden)
        if user_id:
            try:
                from toolboxv2.mods.CloudM.UserInstances import close_user_instance

                close_user_instance(user_id)
                app.logger.debug(f"[Logout] User instance closed for {user_id}")
            except Exception as e:
                app.logger.warning(f"[Logout] Could not close user instance: {e}")

        # 4. Server-seitiges Session-Token invalidieren
        try:
            # Session aus Request holen und invalidieren
            session_data = request.session if hasattr(request, "session") else {}
            session_id = session_data.get("session_id")

            if session_id:
                # Session in DB als ungültig markieren
                await app.a_run_any(
                    TBEF.DB.DELETE, query=f"Session::{session_id}", get_results=True
                )
                app.logger.debug(f"[Logout] Server session invalidated: {session_id}")
        except Exception as e:
            app.logger.warning(f"[Logout] Could not invalidate server session: {e}")

        # 5. Response mit Cookie-Löschung erstellen
        # Headers zum Löschen aller relevanten Cookies
        clear_cookie_headers = {
            "Set-Cookie": [
                "session=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT; HttpOnly; SameSite=Strict",
                "token=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT; HttpOnly; SameSite=Strict",
                "__session=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT; HttpOnly; SameSite=Strict",
                "__clerk_db_jwt=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT; HttpOnly; SameSite=Strict",
            ]
        }

        # 6. Prüfen ob AJAX oder Browser-Request
        accept_header = (
            request.request.headers.get("accept", "")
            if hasattr(request, "request")
            else ""
        )
        is_ajax = "application/json" in accept_header

        if is_ajax:
            # AJAX: JSON Response mit Anweisungen
            return Result.json(
                data={
                    "success": True,
                    "message": "Erfolgreich abgemeldet",
                    "redirect": "/web/assets/login.html",
                    "clear_local_storage": True,
                    "actions": [
                        {
                            "type": "clear_storage",
                            "keys": ["tbjs_user_session", "tbjs_app_state_user"],
                        },
                        {"type": "redirect", "url": "/web/assets/login.html"},
                    ],
                },
                data_info="Logout successful",
            )
        else:
            # Browser: Redirect zur Login-Seite
            return Result.redirect("/web/assets/login.html")

    except Exception as e:
        app.logger.error(f"[Logout] Error during logout: {e}", exc_info=True)
        # Auch bei Fehler zur Login-Seite weiterleiten
        return Result.redirect("/web/assets/login.html")


# ============================================================================
# MINU VIEW RENDER ENDPOINTS
# ============================================================================


@export(
    mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=["GET"]
)
async def render_user_dashboard(app: App, request: RequestData):
    """
    Rendert das User Dashboard als Minu View.

    GET /api/CloudM.DashboardAPI/render_user_dashboard
    """
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.redirect("/web/assets/login.html")

    try:
        # User-Daten vorbereiten
        user_data = {
            "username": getattr(current_user, "username", None)
            or getattr(current_user, "name", "Benutzer"),
            "email": getattr(current_user, "email", ""),
            "level": getattr(current_user, "level", 1),
            "uid": getattr(current_user, "uid", None)
            or getattr(current_user, "clerk_user_id", ""),
            "settings": getattr(current_user, "settings", {}) or {},
            "mod_data": getattr(current_user, "mod_data", {}) or {},
        }

        # Instance-Daten laden
        instance_data = {}
        uid = user_data.get("uid")
        if uid:
            try:
                from toolboxv2.mods.CloudM.UserInstances import (
                    get_user_instance_with_cli_sessions,
                )

                instance_raw = get_user_instance_with_cli_sessions(uid, hydrate=True)
                if instance_raw:
                    live_modules = []
                    if instance_raw.get("live"):
                        for mod_name in instance_raw.get("live", {}).keys():
                            live_modules.append({"name": mod_name})

                    instance_data = {
                        "live_modules": live_modules,
                        "saved_modules": instance_raw.get("save", {}).get("mods", []),
                        "active_cli_sessions": len(instance_raw.get("cli_sessions", [])),
                    }
            except Exception as e:
                app.logger.warning(f"Could not load user instance: {e}")

        # Minu View rendern
        from toolboxv2.mods.Minu import render_view

        return Result.html((await render_view(
            app,
            request,
            view="user_dashboard",
            props={
                "user_data": user_data,
                "instance_data": instance_data,
                "loading": False,
            },
            ssr="true",
            format="html",
        )).get())

    except Exception as e:
        app.logger.error(f"Error rendering user dashboard: {e}", exc_info=True)
        return Result.default_internal_error(info=str(e))


@export(
    mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=["GET"]
)
async def render_admin_dashboard(app: App, request: RequestData):
    """
    Rendert das Admin Dashboard als Minu View.

    GET /api/CloudM.DashboardAPI/render_admin_dashboard
    """
    # Admin-Check
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.redirect("/web/assets/login.html")

    # Nur Admins (Level 0) oder spezielle User erlauben
    user_level = getattr(current_user, "level", 1)
    username = getattr(current_user, "username", "") or getattr(current_user, "name", "")

    if user_level != 0 and username not in ["root", "loot"]:
        return Result.html(
            "<h1>Zugriff verweigert</h1><p>Sie haben keine Berechtigung für diese Seite.</p>",
            status=403,
        )

    try:
        # Admin-Daten vorbereiten
        admin_data = {
            "name": username,
            "email": getattr(current_user, "email", ""),
            "level": user_level,
            "uid": getattr(current_user, "uid", None)
            or getattr(current_user, "clerk_user_id", ""),
            "settings": getattr(current_user, "settings", {}) or {},
        }

        # System-Status laden
        from toolboxv2.mods.CloudM import mini

        status_str = mini.get_service_status("./.info")
        system_status = _parse_service_status(status_str)

        # Benutzer laden
        users = await _load_all_users(app)

        # Warteliste laden
        waiting_list = await _load_waiting_list(app)

        # Module laden
        modules = list(app.get_all_mods())

        # SPPs laden
        spps = _load_spps(app)

        # Minu View rendern
        from toolboxv2.mods.Minu import render_view

        return Result.html((await render_view(
            app,
            request,
            view="admin_dashboard",
            props={
                "admin_user": admin_data,
                "system_status": system_status,
                "users": users,
                "waiting_list": waiting_list,
                "modules": modules,
                "spps": spps,
                "loading": False,
            },
            ssr="true",
            format="html",
        )).get())

    except Exception as e:
        app.logger.error(f"Error rendering admin dashboard: {e}", exc_info=True)
        return Result.default_internal_error(info=str(e))


# ============================================================================
# EVENT HANDLERS FÜR MINU VIEWS
# ============================================================================


@export(
    mod_name=Name, api=True, version=version, request_as_kwarg=True, api_methods=["POST"]
)
async def handle_dashboard_event(app: App, request: RequestData, data: dict):
    """
    Verarbeitet Events von Minu Dashboard Views.

    POST /api/CloudM.DashboardAPI/handle_dashboard_event
    {
        "action": "logout",
        "payload": {...}
    }
    """
    current_user = await get_current_user_from_request(app, request)
    if not current_user:
        return Result.default_user_error(info="Nicht authentifiziert", exec_code=401)

    action = data.get("action", "")
    payload = data.get("payload", {})

    # Action Router
    handlers = {
        # Allgemeine Actions
        "logout": lambda: _handle_logout(app, request, current_user),
        "navigate": lambda: Result.ok(data={"navigate": payload.get("section")}),
        # User Dashboard Actions
        "load_module": lambda: _handle_load_module(app, current_user, payload),
        "unload_module": lambda: _handle_unload_module(app, current_user, payload),
        "save_module": lambda: _handle_save_module(app, current_user, payload),
        "remove_saved_module": lambda: _handle_remove_saved_module(
            app, current_user, payload
        ),
        "update_setting": lambda: _handle_update_setting(app, current_user, payload),
        "set_theme": lambda: Result.ok(data={"theme": payload.get("theme")}),
        "request_magic_link": lambda: _handle_request_magic_link(app, current_user),
        "edit_profile": lambda: Result.ok(data={"action": "open_clerk_profile"}),
        "register_persona": lambda: Result.ok(
            data={"action": "start_webauthn_registration"}
        ),
        # Admin Dashboard Actions
        "refresh_system_status": lambda: _handle_refresh_status(app),
        "restart_service": lambda: _handle_restart_service(app, payload),
        "edit_user": lambda: Result.ok(
            data={"show_modal": "edit_user", "user": payload.get("user")}
        ),
        "delete_user": lambda: _handle_delete_user(app, payload),
        "send_invite": lambda: _handle_send_invite(app, payload),
        "remove_from_waiting": lambda: _handle_remove_from_waiting(app, payload),
        "reload_module": lambda: _handle_reload_module(app, payload),
        "open_spp": lambda: Result.ok(data={"open_url": payload.get("path")}),
    }

    handler = handlers.get(action)
    if handler:
        try:
            result = handler()
            if hasattr(result, "__await__"):
                result = await result
            return result
        except Exception as e:
            app.logger.error(f"Error handling action '{action}': {e}", exc_info=True)
            return Result.default_internal_error(info=str(e))

    return Result.default_user_error(info=f"Unbekannte Aktion: {action}")


# ============================================================================
# HANDLER IMPLEMENTIERUNGEN
# ============================================================================


async def _handle_logout(app: App, request: RequestData, user):
    """Logout-Handler Implementierung"""
    uid = getattr(user, "uid", None) or getattr(user, "clerk_user_id", None)

    # Cleanup
    if uid:
        try:
            from toolboxv2.mods.Minu import cleanup_session

            cleanup_session(uid)
        except:
            pass

        try:
            from toolboxv2.mods.CloudM.UserInstances import close_user_instance

            close_user_instance(uid)
        except:
            pass

    return Result.ok(
        data={
            "success": True,
            "redirect": "/web/assets/login.html",
            "clear_storage": True,
        }
    )


async def _handle_load_module(app: App, user, payload: dict):
    """Modul laden"""
    module_name = payload.get("module")
    if not module_name:
        return Result.default_user_error(info="Modulname fehlt")

    uid = getattr(user, "uid", None) or getattr(user, "clerk_user_id", None)

    try:
        from toolboxv2.mods.CloudM.UserInstances import get_user_instance as get_instance

        instance = get_instance(uid, hydrate=False)

        if not instance:
            return Result.default_internal_error(info="Instanz nicht gefunden")

        if module_name not in app.get_all_mods():
            return Result.default_user_error(
                info=f"Modul '{module_name}' nicht verfügbar"
            )

        spec = app.save_load(module_name)
        if spec:
            if "live" not in instance:
                instance["live"] = {}
            instance["live"][module_name] = spec

            from toolboxv2.mods.CloudM.UserInstances import save_user_instances

            save_user_instances(instance)

            return Result.ok(info=f"Modul '{module_name}' geladen")

        return Result.default_internal_error(
            info=f"Fehler beim Laden von '{module_name}'"
        )

    except Exception as e:
        return Result.default_internal_error(info=str(e))


async def _handle_unload_module(app: App, user, payload: dict):
    """Modul entladen"""
    module_name = payload.get("module")
    if not module_name:
        return Result.default_user_error(info="Modulname fehlt")

    uid = getattr(user, "uid", None) or getattr(user, "clerk_user_id", None)

    try:
        from toolboxv2.mods.CloudM.UserInstances import (
            get_user_instance as get_instance,
            save_user_instances,
        )

        instance = get_instance(uid, hydrate=False)

        if instance and "live" in instance and module_name in instance["live"]:
            spec = instance["live"][module_name]
            app.remove_mod(mod_name=module_name, spec=spec, delete=False)
            del instance["live"][module_name]
            save_user_instances(instance)
            return Result.ok(info=f"Modul '{module_name}' entladen")

        return Result.default_user_error(info=f"Modul '{module_name}' nicht geladen")

    except Exception as e:
        return Result.default_internal_error(info=str(e))


async def _handle_save_module(app: App, user, payload: dict):
    """Modul zu gespeicherten hinzufügen"""
    module_name = payload.get("module")
    if not module_name:
        return Result.default_user_error(info="Modulname fehlt")

    uid = getattr(user, "uid", None) or getattr(user, "clerk_user_id", None)

    try:
        from toolboxv2.mods.CloudM.UserInstances import (
            get_user_instance as get_instance,
            save_user_instances,
        )

        instance = get_instance(uid, hydrate=False)

        if not instance:
            return Result.default_internal_error(info="Instanz nicht gefunden")

        if "save" not in instance:
            instance["save"] = {"mods": [], "uid": uid}
        if "mods" not in instance["save"]:
            instance["save"]["mods"] = []

        if module_name not in instance["save"]["mods"]:
            instance["save"]["mods"].append(module_name)
            save_user_instances(instance)

            # In DB speichern
            app.run_any(
                "DB",
                "set",
                query=f"User::Instance::{uid}",
                data=json.dumps({"saves": instance["save"]}),
            )

        return Result.ok(info=f"Modul '{module_name}' gespeichert")

    except Exception as e:
        return Result.default_internal_error(info=str(e))


async def _handle_remove_saved_module(app: App, user, payload: dict):
    """Modul aus gespeicherten entfernen"""
    module_name = payload.get("module")
    if not module_name:
        return Result.default_user_error(info="Modulname fehlt")

    uid = getattr(user, "uid", None) or getattr(user, "clerk_user_id", None)

    try:
        from toolboxv2.mods.CloudM.UserInstances import (
            get_user_instance as get_instance,
            save_user_instances,
        )

        instance = get_instance(uid, hydrate=False)

        if instance and "save" in instance and "mods" in instance["save"]:
            if module_name in instance["save"]["mods"]:
                instance["save"]["mods"].remove(module_name)
                save_user_instances(instance)

                app.run_any(
                    "DB",
                    "set",
                    query=f"User::Instance::{uid}",
                    data=json.dumps({"saves": instance["save"]}),
                )

                return Result.ok(info=f"Modul '{module_name}' entfernt")

        return Result.default_user_error(info=f"Modul '{module_name}' nicht gefunden")

    except Exception as e:
        return Result.default_internal_error(info=str(e))


async def _handle_update_setting(app: App, user, payload: dict):
    """Einstellung aktualisieren"""
    key = payload.get("key")
    value = payload.get("value")

    if not key:
        return Result.default_user_error(info="Einstellungsschlüssel fehlt")

    try:
        if user.settings is None:
            user.settings = {}

        # Wert konvertieren
        if isinstance(value, str):
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False

        user.settings[key] = value

        save_result = db_helper_save_user(app, asdict(user))
        if save_result.is_error():
            return save_result

        return Result.ok(
            info=f"Einstellung '{key}' aktualisiert", data={"key": key, "value": value}
        )

    except Exception as e:
        return Result.default_internal_error(info=str(e))


async def _handle_request_magic_link(app: App, user):
    """Magic Link anfordern"""
    username = getattr(user, "username", None) or getattr(user, "name", None)

    if not username:
        return Result.default_user_error(info="Benutzername nicht gefunden")

    try:
        from toolboxv2.mods.CloudM.AuthManager import get_magic_link_email

        result = await get_magic_link_email(app, username=username)

        if not result.as_result().is_error():
            return Result.ok(info="Magic Link wurde an Ihre E-Mail gesendet")

        return Result.default_internal_error(info="Fehler beim Senden des Magic Links")

    except Exception as e:
        return Result.default_internal_error(info=str(e))


async def _handle_refresh_status(app: App):
    """System-Status aktualisieren"""
    try:
        from toolboxv2.mods.CloudM import mini

        status_str = mini.get_service_status("./.info")
        system_status = _parse_service_status(status_str)
        return Result.ok(data=system_status)
    except Exception as e:
        return Result.default_internal_error(info=str(e))


async def _handle_restart_service(app: App, payload: dict):
    """Service neustarten (Placeholder)"""
    service = payload.get("service")
    # Tatsächlicher Neustart würde hier implementiert
    return Result.ok(info=f"Neustart für '{service}' initiiert (Placeholder)")


async def _handle_delete_user(app: App, payload: dict):
    """Benutzer löschen"""
    user = payload.get("user", {})
    uid = user.get("uid")
    username = user.get("name")

    if not uid or not username:
        return Result.default_user_error(info="Benutzer-ID und Name erforderlich")

    try:
        from toolboxv2.mods.CloudM.AuthManager import db_helper_delete_user

        result = db_helper_delete_user(app, username, uid, matching=True)

        if result.is_error():
            return result

        return Result.ok(info=f"Benutzer '{username}' gelöscht")

    except Exception as e:
        return Result.default_internal_error(info=str(e))


async def _handle_send_invite(app: App, payload: dict):
    """Einladung senden"""
    email = payload.get("email")
    username = payload.get("username", email.split("@")[0] if email else None)

    if not email:
        return Result.default_user_error(info="E-Mail erforderlich")

    try:
        from toolboxv2.mods.CloudM.email_services import send_signup_invitation_email

        result = send_signup_invitation_email(
            app,
            invited_user_email=email,
            invited_username=username,
            inviter_username="Admin",
        )

        if result.is_error():
            return result

        return Result.ok(info=f"Einladung an '{email}' gesendet")

    except Exception as e:
        return Result.default_internal_error(info=str(e))


async def _handle_remove_from_waiting(app: App, payload: dict):
    """Von Warteliste entfernen"""
    email = payload.get("email")

    if not email:
        return Result.default_user_error(info="E-Mail erforderlich")

    try:
        waiting_result = await app.a_run_any(
            TBEF.DB.GET, query="email_waiting_list", get_results=True
        )
        current_list = []

        if not waiting_result.is_error() and waiting_result.get():
            raw = waiting_result.get()
            if isinstance(raw, list) and len(raw) > 0:
                raw = raw[0]
            if isinstance(raw, bytes):
                raw = raw.decode()
            if isinstance(raw, str):
                data = json.loads(raw.replace("'", '"'))
                current_list = data.get("set", []) if isinstance(data, dict) else data

        updated_list = [e for e in current_list if e != email]

        await app.a_run_any(
            TBEF.DB.SET,
            query="email_waiting_list",
            data=json.dumps({"set": updated_list}),
            get_results=True,
        )

        return Result.ok(info=f"'{email}' von Warteliste entfernt")

    except Exception as e:
        return Result.default_internal_error(info=str(e))


async def _handle_reload_module(app: App, payload: dict):
    """Modul neu laden"""
    module_name = payload.get("module")

    if not module_name:
        return Result.default_user_error(info="Modulname erforderlich")

    try:
        if module_name in app.get_all_mods():
            if hasattr(app, "reload_mod"):
                app.reload_mod(module_name)
            else:
                app.remove_mod(module_name)
                app.save_load(module_name)

            return Result.ok(info=f"Modul '{module_name}' neu geladen")

        return Result.default_user_error(info=f"Modul '{module_name}' nicht gefunden")

    except Exception as e:
        return Result.default_internal_error(info=str(e))


# ============================================================================
# HELPER FUNKTIONEN
# ============================================================================


def _parse_service_status(status_str: str) -> Dict[str, Any]:
    """Parst Service-Status String zu Dictionary"""
    services = {}

    if not status_str or status_str == "No services found":
        return services

    lines = status_str.split("\n")
    for line in lines:
        if not line.strip() or line.startswith("Service(s):"):
            continue

        parts = line.split("(PID:")
        if len(parts) >= 2:
            name_part = parts[0].strip()
            pid = parts[1].replace(")", "").strip()

            status_indicator = name_part[0] if name_part else "🟡"
            service_name = name_part[2:].strip() if len(name_part) > 2 else "Unknown"

            services[service_name] = {"status_indicator": status_indicator, "pid": pid}

    return services


async def _load_all_users(app: App) -> list:
    """Lädt alle Benutzer aus der Datenbank"""
    users = []

    try:
        result = await app.a_run_any(TBEF.DB.GET, query="USER::*", get_results=True)

        if result.is_error() or not result.get():
            return users

        raw_data = result.get()
        if not isinstance(raw_data, list):
            raw_data = [raw_data]

        for item in raw_data:
            try:
                if isinstance(item, bytes):
                    item = item.decode()

                user_dict = json.loads(item) if isinstance(item, str) else item

                users.append(
                    {
                        "uid": user_dict.get("uid", "N/A"),
                        "name": user_dict.get("name", "N/A"),
                        "email": user_dict.get("email"),
                        "level": user_dict.get("level", -1),
                        "settings": user_dict.get("settings", {}),
                    }
                )
            except Exception as e:
                app.logger.warning(f"Could not parse user data: {e}")

    except Exception as e:
        app.logger.error(f"Error loading users: {e}")

    return users


async def _load_waiting_list(app: App) -> list:
    """Lädt die Warteliste"""
    waiting = []

    try:
        result = await app.a_run_any(
            TBEF.DB.GET, query="email_waiting_list", get_results=True
        )

        if result.is_error() or not result.get():
            return waiting

        raw = result.get()
        if isinstance(raw, list) and len(raw) > 0:
            raw = raw[0]
        if isinstance(raw, bytes):
            raw = raw.decode()
        if isinstance(raw, str):
            data = json.loads(raw.replace("'", '"'))
            waiting = data.get("set", []) if isinstance(data, dict) else data

    except Exception as e:
        app.logger.error(f"Error loading waiting list: {e}")

    return waiting


def _load_spps(app: App) -> list:
    """Lädt registrierte SPPs"""
    spps = []

    try:
        ui_config = app.config_fh.get_file_handler("CloudM::UI", "{}")
        ui_data = json.loads(ui_config)

        for name, details in ui_data.items():
            spps.append(
                {
                    "name": name,
                    "title": details.get("title", name),
                    "path": details.get("path", ""),
                    "description": details.get("description", ""),
                    "auth": details.get("auth", False),
                }
            )

    except Exception as e:
        pass

    return spps
