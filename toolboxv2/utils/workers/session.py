#!/usr/bin/env python3
"""
session.py - Stateless Session Management with Signed Cookies

Implements signed cookies for horizontal scaling without shared storage.
Session data is encoded in the cookie itself, signed with HMAC-SHA256.

Features:
- Stateless: No server-side session storage needed
- Secure: HMAC-SHA256 signature prevents tampering
- Expiry: Built-in TTL support
- Clerk integration: Verify sessions via CloudM.AuthClerk
- Multi-worker support: All session state in signed cookie
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote, unquote

logger = logging.getLogger(__name__)


# ============================================================================
# Access Level Constants
# ============================================================================


class AccessLevel:
    """User access levels for authorization."""
    ADMIN = -1           # Full access to everything
    NOT_LOGGED_IN = 0    # Anonymous user, only public endpoints
    LOGGED_IN = 1        # Authenticated user
    TRUSTED = 2          # Trusted/verified user


# ============================================================================
# Session Data Structure
# ============================================================================


@dataclass
class SessionData:
    """Session payload stored in signed cookie."""

    # Core identification
    user_id: str = ""
    session_id: str = ""
    user_name: str = "anonymous"

    # Authorization
    level: int = AccessLevel.NOT_LOGGED_IN  # Permission level
    spec: str = ""  # User specification/role

    # Expiration
    exp: float = 0.0  # Expiration timestamp

    # Clerk integration
    clerk_user_id: str = ""

    # Session state
    validated: bool = False  # Whether session was validated with Clerk
    anonymous: bool = True   # Anonymous session flag

    # Additional custom data
    extra: Dict[str, Any] = field(default_factory=dict)
    live_data: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    _dirty: bool = field(default=False, repr=False, compare=False)

    @property
    def is_authenticated(self) -> bool:
        """Check if session represents an authenticated user."""
        return (
            self.validated and
            not self.anonymous and
            self.level >= AccessLevel.LOGGED_IN and
            self.user_id != "" and
            not self.is_expired
        )

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.exp <= 0:
            return False
        return time.time() > self.exp

    def mark_dirty(self):
        """Mark session as modified (needs to be saved)."""
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        """Check if session has unsaved changes."""
        return self._dirty

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "user_name": self.user_name,
            "level": self.level,
            "spec": self.spec,
            "exp": self.exp,
            "clerk_user_id": self.clerk_user_id,
            "validated": self.validated,
            "anonymous": self.anonymous,
            "extra": self.extra,
            "live_data": self.live_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        return cls(
            user_id=data.get("user_id", ""),
            session_id=data.get("session_id", ""),
            user_name=data.get("user_name", "anonymous"),
            level=data.get("level", AccessLevel.NOT_LOGGED_IN),
            spec=data.get("spec", ""),
            exp=data.get("exp", 0.0),
            clerk_user_id=data.get("clerk_user_id", ""),
            validated=data.get("validated", False),
            anonymous=data.get("anonymous", True),
            extra=data.get("extra", {}),
            live_data=data.get("live_data", {}),
        )

    @classmethod
    def anonymous_session(cls, session_id: str = None) -> "SessionData":
        """Create anonymous session."""
        return cls(
            user_id="",
            session_id=session_id or f"anon_{uuid.uuid4().hex[:16]}",
            user_name="anonymous",
            level=AccessLevel.NOT_LOGGED_IN,
            validated=False,
            anonymous=True,
        )

    @classmethod
    def authenticated_session(
        cls,
        user_id: str,
        user_name: str,
        level: int = AccessLevel.LOGGED_IN,
        clerk_user_id: str = "",
        spec: str = "",
        max_age: int = 604800,
        **extra
    ) -> "SessionData":
        """Create authenticated session."""
        return cls(
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            user_name=user_name,
            level=level,
            spec=spec,
            exp=time.time() + max_age,
            clerk_user_id=clerk_user_id,
            validated=True,
            anonymous=False,
            extra=extra,
            live_data={
                "clerk_user_id": clerk_user_id,
                "level": str(level),
            },
        )

    def invalidate(self):
        """Invalidate this session."""
        self.validated = False
        self.anonymous = True
        self.level = AccessLevel.NOT_LOGGED_IN
        self.user_id = ""
        self.clerk_user_id = ""
        self._dirty = True

    # Backwards compatibility
    @classmethod
    def anonymous(cls) -> "SessionData":
        """Alias for anonymous_session."""
        return cls.anonymous_session()


# ============================================================================
# Signed Cookie Implementation
# ============================================================================


class SignedCookieSession:
    """
    Stateless session manager using signed cookies.

    Cookie format: base64(json_payload).signature
    Signature: HMAC-SHA256(secret, payload)
    """

    SEPARATOR = "."

    def __init__(
        self,
        secret: str,
        cookie_name: str = "tb_session",
        max_age: int = 604800,  # 7 days
        secure: bool = True,
        httponly: bool = True,
        samesite: str = "Lax",
        path: str = "/",
        domain: Optional[str] = None,
    ):
        if not secret or len(secret) < 32:
            raise ValueError("Cookie secret must be at least 32 characters")

        self._secret = secret.encode()
        self.cookie_name = cookie_name
        self.max_age = max_age
        self.secure = secure
        self.httponly = httponly
        self.samesite = samesite
        self.path = path
        self.domain = domain

    def _sign(self, payload: bytes) -> str:
        """Create HMAC-SHA256 signature."""
        signature = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(signature).decode().rstrip("=")

    def _verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify HMAC-SHA256 signature."""
        # Restore padding
        padding = 4 - len(signature) % 4
        if padding != 4:
            signature += "=" * padding

        try:
            expected = base64.urlsafe_b64decode(signature)
        except Exception:
            return False

        actual = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return hmac.compare_digest(expected, actual)

    def encode(self, session: SessionData) -> str:
        """Encode session data to signed cookie value."""
        payload = json.dumps(session.to_dict(), separators=(",", ":")).encode()
        encoded_payload = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        signature = self._sign(payload)
        return f"{encoded_payload}{self.SEPARATOR}{signature}"

    def decode(self, cookie_value: str) -> Optional[SessionData]:
        """Decode and verify signed cookie value."""
        if not cookie_value or self.SEPARATOR not in cookie_value:
            return None

        try:
            encoded_payload, signature = cookie_value.rsplit(self.SEPARATOR, 1)

            # Restore padding
            padding = 4 - len(encoded_payload) % 4
            if padding != 4:
                encoded_payload += "=" * padding

            payload = base64.urlsafe_b64decode(encoded_payload)

            # Verify signature
            if not self._verify_signature(payload, signature):
                logger.warning("Invalid cookie signature")
                return None

            data = json.loads(payload.decode())
            session = SessionData.from_dict(data)

            # Check expiration
            if session.is_expired:
                logger.debug("Session expired")
                return None

            return session

        except Exception as e:
            logger.warning(f"Cookie decode error: {e}")
            return None

    def create_cookie_header(
        self,
        session: SessionData,
        max_age: Optional[int] = None,
    ) -> str:
        """Create Set-Cookie header value."""
        value = self.encode(session)

        parts = [f"{self.cookie_name}={quote(value)}"]

        if max_age is None:
            max_age = self.max_age

        parts.append(f"Max-Age={max_age}")
        parts.append(f"Path={self.path}")

        if self.domain:
            parts.append(f"Domain={self.domain}")

        if self.secure:
            parts.append("Secure")

        if self.httponly:
            parts.append("HttpOnly")

        if self.samesite:
            parts.append(f"SameSite={self.samesite}")

        return "; ".join(parts)

    def create_logout_cookie_header(self) -> str:
        """Create Set-Cookie header that clears the session."""
        parts = [
            f"{self.cookie_name}=",
            "Max-Age=0",
            f"Path={self.path}",
        ]

        if self.domain:
            parts.append(f"Domain={self.domain}")

        return "; ".join(parts)

    def get_from_cookie_header(self, cookie_header: str) -> Optional[SessionData]:
        """Extract session from Cookie header."""
        if not cookie_header:
            return None

        cookies = SimpleCookie()
        try:
            cookies.load(cookie_header)
        except Exception:
            return None

        if self.cookie_name not in cookies:
            return None

        value = unquote(cookies[self.cookie_name].value)
        return self.decode(value)

    def get_from_environ(self, environ: Dict) -> Optional[SessionData]:
        """Extract session from WSGI environ."""
        cookie_header = environ.get("HTTP_COOKIE", "")
        return self.get_from_cookie_header(cookie_header)


# ============================================================================
# Clerk Integration
# ============================================================================


class ClerkSessionVerifier:
    """
    Verify sessions using CloudM.AuthClerk from ToolBoxV2.

    Falls back to signed cookie if Clerk is not available.
    """

    def __init__(
        self,
        app,  # ToolBoxV2 App instance
        auth_module: str = "CloudM.AuthClerk",
        verify_func: str = "verify_session",
    ):
        self.app = app
        self.auth_module = auth_module
        self.verify_func = verify_func
        self._clerk_available = None

    def _check_clerk_available(self) -> bool:
        """Check if Clerk module is available."""
        if self._clerk_available is not None:
            return self._clerk_available

        try:
            if hasattr(self.app, "get_mod"):
                mod = self.app.get_mod(self.auth_module.split(".")[0])
                self._clerk_available = mod is not None
            else:
                self._clerk_available = False
        except Exception:
            self._clerk_available = False

        return self._clerk_available

    async def verify_session_async(
        self,
        session_token: str,
    ) -> Tuple[bool, Optional[SessionData]]:
        """
        Verify session token via Clerk.

        Returns:
            Tuple of (is_valid, session_data)
        """
        if not self._check_clerk_available():
            return False, None

        try:
            result = await self.app.a_run_any(
                (self.auth_module, self.verify_func),
                session_token=session_token,
                get_results=True,
            )

            if result.is_error():
                return False, None

            data = result.get()

            if not data or not data.get("valid", False):
                return False, None

            # Convert Clerk response to SessionData
            session = SessionData(
                user_id=data.get("user_id", ""),
                session_id=data.get("session_id", str(uuid.uuid4())),
                user_name=data.get("user_name", data.get("username", "anonymous")),
                level=data.get("level", AccessLevel.LOGGED_IN),
                spec=data.get("spec", ""),
                exp=data.get("exp", 0),
                clerk_user_id=data.get("clerk_user_id", ""),
                validated=True,
                anonymous=False,
                extra={
                    "email": data.get("email"),
                },
                live_data={
                    "clerk_user_id": data.get("clerk_user_id", ""),
                    "level": str(data.get("level", AccessLevel.LOGGED_IN)),
                },
            )

            return True, session

        except Exception as e:
            logger.error(f"Clerk verification error: {e}")
            return False, None

    def verify_session_sync(
        self,
        session_token: str,
    ) -> Tuple[bool, Optional[SessionData]]:
        """Synchronous version of verify_session."""
        if not self._check_clerk_available():
            return False, None

        try:
            result = self.app.run_any(
                (self.auth_module, self.verify_func),
                session_token=session_token,
                get_results=True,
            )

            if result.is_error():
                return False, None

            data = result.get()

            if not data or not data.get("valid", False):
                return False, None

            session = SessionData(
                user_id=data.get("user_id", ""),
                session_id=data.get("session_id", str(uuid.uuid4())),
                user_name=data.get("user_name", data.get("username", "anonymous")),
                level=data.get("level", AccessLevel.LOGGED_IN),
                spec=data.get("spec", ""),
                exp=data.get("exp", 0),
                clerk_user_id=data.get("clerk_user_id", ""),
                validated=True,
                anonymous=False,
                extra={
                    "email": data.get("email"),
                },
                live_data={
                    "clerk_user_id": data.get("clerk_user_id", ""),
                    "level": str(data.get("level", AccessLevel.LOGGED_IN)),
                },
            )

            return True, session

        except Exception as e:
            logger.error(f"Clerk verification error: {e}")
            return False, None


# ============================================================================
# Combined Session Manager (Multi-Worker Ready)
# ============================================================================


class SessionManager:
    """
    Combined session manager supporting:
    - Signed cookies (stateless, multi-worker safe)
    - Clerk verification
    - Bearer token auth
    - API key auth

    For multi-worker setup, all session state is in the signed cookie.
    No server-side storage needed.
    """

    def __init__(
        self,
        cookie_secret: str,
        cookie_name: str = "tb_session",
        cookie_max_age: int = 604800,
        cookie_secure: bool = True,
        cookie_httponly: bool = True,
        cookie_samesite: str = "Lax",
        cookie_path: str = "/",
        cookie_domain: Optional[str] = None,
        app=None,
        clerk_enabled: bool = True,
        api_key_header: str = "X-API-Key",
        bearer_header: str = "Authorization",
    ):
        self.cookie_session = SignedCookieSession(
            secret=cookie_secret,
            cookie_name=cookie_name,
            max_age=cookie_max_age,
            secure=cookie_secure,
            httponly=cookie_httponly,
            samesite=cookie_samesite,
            path=cookie_path,
            domain=cookie_domain,
        )

        self.clerk_verifier = None
        if app and clerk_enabled:
            self.clerk_verifier = ClerkSessionVerifier(app)

        self.api_key_header = api_key_header
        self.bearer_header = bearer_header
        self.cookie_max_age = cookie_max_age

        # API key storage (consider using Redis for multi-worker)
        self._api_keys: Dict[str, SessionData] = {}

        # Track sessions that need cookie updates
        # Key: session_id, Value: SessionData
        self._pending_updates: Dict[str, SessionData] = {}

    # =========================================================================
    # Session Creation
    # =========================================================================

    def create_session(
        self,
        user_id: str = "",
        user_name: str = "anonymous",
        level: int = AccessLevel.NOT_LOGGED_IN,
        spec: str = "",
        clerk_user_id: str = "",
        client_ip: str = "",
        token: str = "",
        max_age: Optional[int] = None,
        **extra
    ) -> str:
        """
        Create a new session and return the session ID.

        The session data is stored in a signed cookie, not server-side.

        Returns:
            session_id: The unique session identifier
        """
        if max_age is None:
            max_age = self.cookie_max_age

        session_id = str(uuid.uuid4())

        # Determine if this is an anonymous or authenticated session
        is_anonymous = not user_id or level <= AccessLevel.NOT_LOGGED_IN

        session = SessionData(
            user_id=user_id,
            session_id=session_id,
            user_name=user_name,
            level=level,
            spec=spec,
            exp=time.time() + max_age,
            clerk_user_id=clerk_user_id,
            validated=not is_anonymous,
            anonymous=is_anonymous,
            extra={
                "client_ip": client_ip,
                "created_at": time.time(),
                **extra,
            },
            live_data={
                "clerk_user_id": clerk_user_id,
                "level": str(level),
            },
        )

        # Mark for cookie update
        session._dirty = True
        self._pending_updates[session_id] = session

        logger.debug(f"Created session {session_id} for user {user_id or 'anonymous'}")

        return session_id

    def create_authenticated_session(
        self,
        user_id: str,
        user_name: str,
        level: int = AccessLevel.LOGGED_IN,
        clerk_user_id: str = "",
        spec: str = "",
        max_age: Optional[int] = None,
        **extra
    ) -> Tuple[SessionData, str]:
        """
        Create an authenticated session and return both session and cookie header.

        Returns:
            Tuple of (session_data, set_cookie_header)
        """
        if max_age is None:
            max_age = self.cookie_max_age

        session = SessionData.authenticated_session(
            user_id=user_id,
            user_name=user_name,
            level=level,
            clerk_user_id=clerk_user_id,
            spec=spec,
            max_age=max_age,
            **extra
        )

        cookie_header = self.cookie_session.create_cookie_header(session, max_age)

        return session, cookie_header

    # =========================================================================
    # Session Retrieval
    # =========================================================================

    def get_session(self, session_id: str) -> SessionData:
        """
        Get session by ID.

        In stateless mode, this returns from pending updates or creates anonymous.
        The actual session data comes from the cookie, not server storage.
        """
        # Check pending updates first
        if session_id in self._pending_updates:
            return self._pending_updates[session_id]

        # In stateless mode, we don't have server-side storage
        # Return anonymous session as fallback
        return SessionData.anonymous_session(session_id)

    async def get_session_from_request(
        self,
        environ: Dict,
        headers: Optional[Dict[str, str]] = None,
    ) -> SessionData:
        """
        Extract and verify session from request.

        Checks in order:
        1. API Key header
        2. Bearer token (Clerk)
        3. Signed cookie
        4. Returns anonymous session
        """
        if headers is None:
            headers = {}
            for key, value in environ.items():
                if key.startswith("HTTP_"):
                    header_name = key[5:].replace("_", "-").title()
                    headers[header_name] = value

        # 1. Check API key
        api_key = headers.get(self.api_key_header) or headers.get(
            self.api_key_header.lower()
        )
        if api_key and api_key in self._api_keys:
            session = self._api_keys[api_key]
            if not session.is_expired:
                return session

        # 2. Check Bearer token (Clerk)
        auth_header = headers.get(self.bearer_header) or headers.get(
            self.bearer_header.lower()
        )
        if auth_header and auth_header.startswith("Bearer ") and self.clerk_verifier:
            token = auth_header[7:]
            is_valid, session = await self.clerk_verifier.verify_session_async(token)
            if is_valid and session:
                return session

        # 3. Check signed cookie
        cookie_session = self.cookie_session.get_from_environ(environ)
        if cookie_session:
            # Check if there's a pending update for this session
            if cookie_session.session_id in self._pending_updates:
                return self._pending_updates[cookie_session.session_id]
            if cookie_session.is_authenticated or not cookie_session.anonymous:
                return cookie_session

        # 4. Return anonymous
        return SessionData.anonymous()

    def get_session_from_request_sync(
        self,
        environ: Dict,
        headers: Optional[Dict[str, str]] = None,
    ) -> SessionData:
        """Synchronous version of get_session_from_request."""
        if headers is None:
            headers = {}
            for key, value in environ.items():
                if key.startswith("HTTP_"):
                    header_name = key[5:].replace("_", "-").title()
                    headers[header_name] = value

        # 1. Check API key
        api_key = headers.get(self.api_key_header) or headers.get(
            self.api_key_header.lower()
        )
        if api_key and api_key in self._api_keys:
            session = self._api_keys[api_key]
            if not session.is_expired:
                return session

        # 2. Check Bearer token
        auth_header = headers.get(self.bearer_header) or headers.get(
            self.bearer_header.lower()
        )
        if auth_header and auth_header.startswith("Bearer ") and self.clerk_verifier:
            token = auth_header[7:]
            is_valid, session = self.clerk_verifier.verify_session_sync(token)
            if is_valid and session:
                return session

        # 3. Check signed cookie
        cookie_session = self.cookie_session.get_from_environ(environ)
        if cookie_session:
            # Check if there's a pending update for this session
            if cookie_session.session_id in self._pending_updates:
                return self._pending_updates[cookie_session.session_id]
            if cookie_session.is_authenticated or not cookie_session.anonymous:
                return cookie_session

        # 4. Return anonymous
        return SessionData.anonymous()

    # =========================================================================
    # Session Update
    # =========================================================================

    def update_session(self, session: SessionData):
        """
        Mark session for update.

        In stateless mode, this queues the session for cookie update.
        """
        session._dirty = True
        self._pending_updates[session.session_id] = session
        logger.debug(f"Session {session.session_id} marked for update")

    def set_session_data(
        self,
        session: SessionData,
        user_id: str = None,
        user_name: str = None,
        level: int = None,
        clerk_user_id: str = None,
        validated: bool = None,
        anonymous: bool = None,
        **extra
    ) -> SessionData:
        """
        Update session fields and mark as dirty.

        Returns the updated session.
        """
        if user_id is not None:
            session.user_id = user_id
        if user_name is not None:
            session.user_name = user_name
        if level is not None:
            session.level = level
            session.live_data["level"] = str(level)
        if clerk_user_id is not None:
            session.clerk_user_id = clerk_user_id
            session.live_data["clerk_user_id"] = clerk_user_id
        if validated is not None:
            session.validated = validated
        if anonymous is not None:
            session.anonymous = anonymous

        for key, value in extra.items():
            session.extra[key] = value

        session._dirty = True
        self._pending_updates[session.session_id] = session

        return session

    # =========================================================================
    # Session Deletion
    # =========================================================================

    def delete_session(self, session_id: str):
        """
        Delete/invalidate a session.

        In stateless mode, this marks the session for cookie clearing.
        """
        # Remove from pending updates
        self._pending_updates.pop(session_id, None)

        logger.debug(f"Session {session_id} deleted")

    def invalidate_session(self, session: SessionData = None) -> str:
        """
        Invalidate session and return Set-Cookie header that clears cookie.

        Returns:
            Set-Cookie header value
        """
        if session:
            session.invalidate()
            self._pending_updates.pop(session.session_id, None)

        return self.cookie_session.create_logout_cookie_header()

    # =========================================================================
    # Cookie Header Generation
    # =========================================================================

    def get_set_cookie_header(self, session: SessionData) -> Optional[str]:
        """
        Get Set-Cookie header for a session if it needs updating.

        Returns:
            Set-Cookie header string, or None if no update needed
        """
        if not session:
            return None

        # Check if session needs update
        if session._dirty or session.session_id in self._pending_updates:
            # Get the most recent version
            if session.session_id in self._pending_updates:
                session = self._pending_updates[session.session_id]

            # Clear from pending
            self._pending_updates.pop(session.session_id, None)
            session._dirty = False

            # Generate cookie header
            return self.cookie_session.create_cookie_header(session)

        return None

    def create_cookie_header_for_session(
        self,
        session: SessionData,
        max_age: Optional[int] = None
    ) -> str:
        """
        Create Set-Cookie header for a specific session.

        Always generates header regardless of dirty state.
        """
        if max_age is None:
            max_age = self.cookie_max_age
        return self.cookie_session.create_cookie_header(session, max_age)

    def get_logout_cookie_header(self) -> str:
        """Get Set-Cookie header that clears the session cookie."""
        return self.cookie_session.create_logout_cookie_header()

    # =========================================================================
    # API Key Management
    # =========================================================================

    def register_api_key(self, api_key: str, session: SessionData):
        """Register an API key with associated session data."""
        self._api_keys[api_key] = session

    def revoke_api_key(self, api_key: str):
        """Revoke an API key."""
        self._api_keys.pop(api_key, None)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def verify_session_token(self, token: str) -> Tuple[bool, Optional[SessionData]]:
        """
        Verify a session token (sync).

        Returns:
            Tuple of (is_valid, session_data)
        """
        if self.clerk_verifier:
            return self.clerk_verifier.verify_session_sync(token)
        return False, None

    async def verify_session_token_async(self, token: str) -> Tuple[bool, Optional[SessionData]]:
        """
        Verify a session token (async).

        Returns:
            Tuple of (is_valid, session_data)
        """
        if self.clerk_verifier:
            return await self.clerk_verifier.verify_session_async(token)
        return False, None

    def clear_pending_updates(self):
        """Clear all pending session updates."""
        self._pending_updates.clear()


# ============================================================================
# WSGI Middleware
# ============================================================================


class SessionMiddleware:
    """WSGI middleware that adds session to environ and handles cookie updates."""

    def __init__(
        self,
        app,
        session_manager: SessionManager,
        environ_key: str = "tb.session",
    ):
        self.app = app
        self.session_manager = session_manager
        self.environ_key = environ_key

    def __call__(self, environ, start_response):
        """Process request and add session to environ."""
        session = self.session_manager.get_session_from_request_sync(environ)
        environ[self.environ_key] = session

        def custom_start_response(status, headers, exc_info=None):
            # Add Set-Cookie header if session was modified
            cookie_header = self.session_manager.get_set_cookie_header(session)
            if cookie_header:
                headers.append(("Set-Cookie", cookie_header))
            return start_response(status, headers, exc_info)

        return self.app(environ, custom_start_response)


# ============================================================================
# Utility Functions
# ============================================================================


def generate_secret(length: int = 64) -> str:
    """Generate a secure random secret."""
    return base64.urlsafe_b64encode(os.urandom(length)).decode()


def require_auth(min_level: int = AccessLevel.LOGGED_IN):
    """Decorator to require authentication for handlers."""

    def decorator(func):
        async def wrapper(environ, session: SessionData, *args, **kwargs):
            if not session.is_authenticated:
                return (
                    401,
                    {"Content-Type": "application/json"},
                    b'{"error": "Unauthorized"}',
                )
            if session.level < min_level and session.level != AccessLevel.ADMIN:
                return (
                    403,
                    {"Content-Type": "application/json"},
                    b'{"error": "Forbidden"}',
                )
            return await func(environ, session, *args, **kwargs)

        return wrapper

    return decorator


def require_level(level: int):
    """Decorator to require specific access level."""
    return require_auth(level)


# ============================================================================
# CLI
# ============================================================================


def main():
    """CLI for session management tools."""
    import argparse

    parser = argparse.ArgumentParser(description="Session Management Tools", prog="tb session")
    subparsers = parser.add_subparsers(dest="command")

    # Generate secret
    gen_parser = subparsers.add_parser("generate-secret", help="Generate cookie secret")
    gen_parser.add_argument("-l", "--length", type=int, default=64)

    # Test encode/decode
    test_parser = subparsers.add_parser("test", help="Test session encoding")
    test_parser.add_argument("-s", "--secret", required=True)

    args = parser.parse_args()

    if args.command == "generate-secret":
        secret = generate_secret(args.length)
        print(f"Generated secret ({args.length} bytes):")
        print(secret)

    elif args.command == "test":
        session_mgr = SignedCookieSession(secret=args.secret)

        # Create test session
        session = SessionData.authenticated_session(
            user_id="test_123",
            user_name="testuser",
            level=AccessLevel.LOGGED_IN,
            clerk_user_id="clerk_abc",
        )

        # Encode
        encoded = session_mgr.encode(session)
        print(f"Encoded cookie value ({len(encoded)} chars):")
        print(encoded)

        # Decode
        decoded = session_mgr.decode(encoded)
        print(f"\nDecoded session:")
        print(json.dumps(decoded.to_dict(), indent=2))

        # Verify
        print(f"\nAuthenticated: {decoded.is_authenticated}")
        print(f"Expired: {decoded.is_expired}")
        print(f"Level: {decoded.level}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
