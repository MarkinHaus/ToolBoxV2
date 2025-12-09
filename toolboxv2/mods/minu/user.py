# toolboxv2/mods/Minu/user.py
"""
Minu User System
================
Einheitlicher Zugriff auf Nutzerdaten in allen MinuViews.

Features:
- Automatische User-Property in jeder MinuView
- Angemeldete Nutzer: Echtes User-Objekt + ModDataClient
- Anonyme Nutzer: Pseudo-User mit Session-basierter Datenspeicherung
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2 import App, RequestData


# ============================================================================
# USER DATA CLASSES
# ============================================================================


@dataclass
class AnonymousUser:
    """
    Pseudo-User für nicht angemeldete Nutzer.
    Speichert Daten in der Session statt in der DB.
    """

    name: str = "anonymous"
    level: int = -1
    session_id: str = ""
    created_at: float = field(default_factory=time.time)

    # Session-basierte Datenspeicherung
    _session_data: Dict[str, Any] = field(default_factory=dict)
    _request: Optional[RequestData] = field(default=None, repr=False)

    @property
    def uid(self) -> str:
        """Eindeutige ID basierend auf Session"""
        return f"anon_{self.session_id}"

    @property
    def is_authenticated(self) -> bool:
        return False

    @property
    def is_anonymous(self) -> bool:
        return True

    def get_mod_data(self, mod_name: str, key: str = None) -> Dict[str, Any]:
        """
        Mod-Daten aus Session lesen.
        Synchrone Version für anonyme Nutzer.
        """
        mod_data = self._session_data.get(f"mod_data:{mod_name}", {})
        if key:
            return {key: mod_data.get(key)}
        return mod_data

    def set_mod_data(
        self, mod_name: str, data: Dict[str, Any], merge: bool = True
    ) -> bool:
        """
        Mod-Daten in Session speichern.
        Synchrone Version für anonyme Nutzer.
        """
        key = f"mod_data:{mod_name}"
        if merge and key in self._session_data:
            self._session_data[key] = {**self._session_data[key], **data}
        else:
            self._session_data[key] = data

        # Session persistieren wenn request vorhanden
        if self._request and hasattr(self._request, "session"):
            self._request.session["anon_mod_data"] = self._session_data

        return True

    def delete_mod_data(self, mod_name: str, keys: List[str] = None) -> bool:
        """Mod-Daten aus Session löschen."""
        session_key = f"mod_data:{mod_name}"
        if session_key not in self._session_data:
            return True

        if keys:
            for key in keys:
                self._session_data[session_key].pop(key, None)
        else:
            del self._session_data[session_key]

        if self._request and hasattr(self._request, "session"):
            self._request.session["anon_mod_data"] = self._session_data

        return True


@dataclass
class AuthenticatedUserWrapper:
    """
    Wrapper für authentifizierte Nutzer.
    Bietet einheitliches Interface und ModDataClient-Integration.
    """

    _user: Any  # Das echte User-Objekt aus CloudM
    _app: Optional[App] = field(default=None, repr=False)
    _request: Optional[RequestData] = field(default=None, repr=False)
    _mod_client_cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return (
            getattr(self._user, "username", None)
            or getattr(self._user, "name", None)
            or getattr(self._user, "email", "User")
        )

    @property
    def level(self) -> int:
        return getattr(self._user, "level", 0)

    @property
    def uid(self) -> str:
        return (
            getattr(self._user, "uid", None)
            or getattr(self._user, "clerk_user_id", None)
            or str(id(self._user))
        )

    @property
    def email(self) -> Optional[str]:
        return getattr(self._user, "email", None)

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def is_anonymous(self) -> bool:
        return False

    @property
    def raw(self) -> Any:
        """Zugriff auf das originale User-Objekt"""
        return self._user

    def __getattr__(self, name: str) -> Any:
        """Proxy für alle anderen Attribute zum originalen User"""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return getattr(self._user, name)

    def get_mod_client(self, mod_name: str):
        """
        ModDataClient für ein Modul erstellen.
        Cached pro mod_name.
        """
        if mod_name not in self._mod_client_cache:
            from toolboxv2.mods.CloudM.UserDataAPI import ModDataClient

            self._mod_client_cache[mod_name] = ModDataClient(
                self._app, self._request, mod_name
            )
        return self._mod_client_cache[mod_name]

    async def get_mod_data(self, mod_name: str, key: str = None) -> Dict[str, Any]:
        """Mod-Daten über ModDataClient abrufen"""
        client = self.get_mod_client(mod_name)
        return await client.get(key)

    async def set_mod_data(
        self, mod_name: str, data: Dict[str, Any], merge: bool = True
    ) -> bool:
        """Mod-Daten über ModDataClient speichern"""
        client = self.get_mod_client(mod_name)
        return await client.set(data, merge)

    async def delete_mod_data(self, mod_name: str, keys: List[str] = None) -> bool:
        """Mod-Daten über ModDataClient löschen"""
        client = self.get_mod_client(mod_name)
        return await client.delete(keys)


# ============================================================================
# USER FACTORY
# ============================================================================


class MinuUser:
    """
    Factory und Utility-Klasse für User-Erstellung.
    Wird von MinuView verwendet um die `user` Property bereitzustellen.
    """

    @staticmethod
    async def from_request(
        app: App, request: RequestData
    ) -> AuthenticatedUserWrapper | AnonymousUser:
        """
        User aus Request erstellen.
        Gibt AuthenticatedUserWrapper oder AnonymousUser zurück.
        """
        # Versuche authentifizierten User zu laden
        try:
            from toolboxv2.mods.CloudM.UserAccountManager import (
                get_current_user_from_request,
            )

            user = await get_current_user_from_request(app, request)

            if user:
                return AuthenticatedUserWrapper(_user=user, _app=app, _request=request)
        except ImportError:
            pass
        except Exception as e:
            if app:
                app.logger.warning(f"[MinuUser] Error loading user: {e}")

        # Fallback: Anonymer User
        return MinuUser.create_anonymous(request)

    @staticmethod
    def from_request_sync(
        app: App, request: RequestData
    ) -> AuthenticatedUserWrapper | AnonymousUser:
        """
        Synchrone Version - versucht gecachten User zu nutzen.
        Für Fälle wo async nicht möglich ist.
        """
        # Prüfe ob User bereits im Request gecacht ist
        if hasattr(request, "_cached_minu_user"):
            return request._cached_minu_user

        # Prüfe Session auf User-Info
        session = getattr(request, "session", {}) or {}

        # Wenn User-ID in Session, ist der Nutzer vermutlich eingeloggt
        # Aber wir können async nicht aufrufen, also anonymen User zurückgeben
        # Der wird dann durch async from_request ersetzt sobald möglich
        return MinuUser.create_anonymous(request)

    @staticmethod
    def create_anonymous(request: RequestData) -> AnonymousUser:
        """Anonymen User aus Request erstellen"""
        session = getattr(request, "session", {}) or {}
        session_id = session.get("session_id", f"anon-{uuid.uuid4().hex[:12]}")

        # Lade existierende Session-Daten
        session_data = session.get("anon_mod_data", {})

        return AnonymousUser(
            session_id=session_id, _session_data=session_data, _request=request
        )


# ============================================================================
# MIXIN FOR MINUVIEW
# ============================================================================


class UserMixin:
    """
    Mixin für MinuView um User-Property bereitzustellen.

    Usage:
        class MyView(MinuView, UserMixin):
            def render(self):
                if self.user.is_authenticated:
                    return Text(f"Willkommen, {self.user.name}!")
                return Text("Bitte anmelden")
    """

    _user_cache: AuthenticatedUserWrapper | AnonymousUser | None = None
    _app: Optional[App] = None
    request_data: Optional[RequestData] = None

    @property
    def user(self) -> AuthenticatedUserWrapper | AnonymousUser:
        """
        Aktueller User (angemeldet oder anonym).

        Für angemeldete Nutzer:
            - user.name, user.uid, user.email, etc.
            - user.get_mod_client('ModName') für ModDataClient
            - await user.get_mod_data('ModName')
            - await user.set_mod_data('ModName', {...})

        Für anonyme Nutzer:
            - user.name == "anonymous"
            - user.level == -1
            - user.uid == "anon_<session_id>"
            - user.get_mod_data('ModName') (synchron, Session-basiert)
            - user.set_mod_data('ModName', {...}) (synchron, Session-basiert)
        """
        if self._user_cache is not None:
            return self._user_cache

        # Sync fallback wenn async nicht möglich
        if self.request_data:
            self._user_cache = MinuUser.from_request_sync(self._app, self.request_data)
            return self._user_cache

        # Default: Anonymous ohne Session
        return AnonymousUser(session_id=f"no-session-{uuid.uuid4().hex[:8]}")

    async def ensure_user(self) -> AuthenticatedUserWrapper | AnonymousUser:
        """
        Async User-Laden.
        Sollte zu Beginn eines Event-Handlers aufgerufen werden.

        Usage:
            async def on_submit(self, event):
                user = await self.ensure_user()
                if user.is_authenticated:
                    await user.set_mod_data('MyMod', {'score': 100})
        """
        if self._user_cache is not None and self._user_cache.is_authenticated:
            return self._user_cache

        if self.request_data and self._app:
            self._user_cache = await MinuUser.from_request(self._app, self.request_data)
            # Cache im Request für spätere Zugriffe
            if self.request_data:
                self.request_data._cached_minu_user = self._user_cache

        return self._user_cache or AnonymousUser()

    def set_app(self, app: App):
        """App-Referenz setzen (wird von Session-Handler aufgerufen)"""
        self._app = app


# ============================================================================
# CONVENIENCE EXPORTS
# ============================================================================


__all__ = [
    "MinuUser",
    "AnonymousUser",
    "AuthenticatedUserWrapper",
    "UserMixin",
]
