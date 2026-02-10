"""
Custom Session Management for ToolBoxV2
Replaces Clerk-based session system with provider-agnostic custom auth.

Compatible with:
- Discord OAuth
- Google OAuth
- Passkeys/WebAuthn
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import threading
import json


@dataclass
class SessionData:
    """
    Session data structure for custom authentication.

    Replaces Clerk-specific session data with provider-agnostic format.
    """
    # User identification
    user_id: str = ""  # Internal ToolBoxV2 user ID
    username: str = ""
    email: str = ""
    level: int = 0

    # Provider information
    provider: str = ""  # "discord", "google", "passkey"
    provider_user_id: str = ""  # External provider user ID

    # Token data
    token: str = ""  # JWT access token
    refresh_token: str = ""

    # Metadata
    expires_at: float = 0.0  # Unix timestamp
    created_at: float = 0.0
    last_validated: float = 0.0

    # Additional provider-specific data
    provider_data: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if session is still valid based on expiration."""
        import time
        return time.time() < self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert session data to dictionary for storage/transmission."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "level": self.level,
            "provider": self.provider,
            "provider_user_id": self.provider_user_id,
            "token": self.token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
            "last_validated": self.last_validated,
            "provider_data": self.provider_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionData":
        """Create SessionData from dictionary."""
        return cls(
            user_id=data.get("user_id", ""),
            username=data.get("username", ""),
            email=data.get("email", ""),
            level=data.get("level", 0),
            provider=data.get("provider", ""),
            provider_user_id=data.get("provider_user_id", ""),
            token=data.get("token", ""),
            refresh_token=data.get("refresh_token", ""),
            expires_at=data.get("expires_at", 0.0),
            created_at=data.get("created_at", 0.0),
            last_validated=data.get("last_validated", 0.0),
            provider_data=data.get("provider_data", {})
        )

    def to_frontend_format(self) -> Dict[str, Any]:
        """
        Convert to frontend-compatible format.
        Maintains compatibility with existing frontend state structure.
        """
        return {
            "isAuthenticated": self.is_valid(),
            "userId": self.user_id,
            "username": self.username,
            "email": self.email,
            "userLevel": self.level,
            "token": self.token,
            "refresh_token": self.refresh_token,
            "provider": self.provider,
            "userData": {
                "firstName": self.username.split()[0] if self.username else "",
                "lastName": " ".join(self.username.split()[1:]) if self.username and len(self.username.split()) > 1 else "",
                "imageUrl": self.provider_data.get("avatar", ""),
                "providerId": self.provider_user_id
            }
        }


class CustomSessionVerifier:
    """
    Custom session verifier replacing ClerkSessionVerifier.

    Uses the CloudM.Auth module for token validation and user data retrieval.
    Provider-agnostic - works with Discord, Google, and Passkeys.
    """

    _instance: Optional["CustomSessionVerifier"] = None
    _lock = threading.Lock()
    _test_mode = False

    def __new__(cls, app=None, auth_module: str = "CloudM.Auth"):
        """Singleton pattern with test-mode reset support."""
        if cls._test_mode:
            # In test mode, create new instance each time
            return super().__new__(cls)

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing with IsolatedTestCase)."""
        with cls._lock:
            cls._instance = None
            cls._test_mode = True

    def __init__(self, app=None, auth_module: str = "CloudM.Auth"):
        """
        Initialize the custom session verifier.

        Args:
            app: ToolBoxV2 application instance
            auth_module: Module name for auth functions (default: CloudM.Auth)
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.app = app
        self.auth_module = auth_module
        self._cache: Dict[str, SessionData] = {}
        self._cache_lock = threading.Lock()
        self._initialized = True

    async def verify_session(self, token: str) -> SessionData:
        """
        Verify a JWT token and retrieve session data.

        Args:
            token: JWT access token

        Returns:
            SessionData object with user information
        """
        if not token or not token.strip():
            return SessionData()

        # Check cache first
        with self._cache_lock:
            if token in self._cache and self._cache[token].is_valid():
                return self._cache[token]

        try:
            # Call CloudM.Auth.validate_session
            result = await self.app.a_run_any(
                (self.auth_module, "validate_session"),
                token=token
            )

            if not result or not hasattr(result, "ok") or not result.ok():
                return SessionData()

            data = result.unwrap_or({})

            # Create session data from response
            session_data = SessionData(
                user_id=data.get("user_id", ""),
                username=data.get("user_name", ""),
                email=data.get("email", ""),
                level=data.get("level", 0),
                provider=data.get("provider", ""),
                provider_user_id=data.get("provider_user_id", ""),
                token=token,
                refresh_token=data.get("refresh_token", ""),
                expires_at=data.get("exp", 0.0),
                created_at=data.get("created_at", 0.0),
                last_validated=datetime.now().timestamp(),
                provider_data=data.get("provider_data", {})
            )

            # Cache the session
            with self._cache_lock:
                self._cache[token] = session_data

            return session_data

        except Exception as e:
            if self.app:
                self.app.logger.error(f"Session verification failed: {e}")
            return SessionData()

    async def refresh_session(self, refresh_token: str) -> SessionData:
        """
        Refresh an expired session using refresh token.

        Args:
            refresh_token: JWT refresh token

        Returns:
            New SessionData with updated tokens
        """
        if not refresh_token:
            return SessionData()

        try:
            result = await self.app.a_run_any(
                (self.auth_module, "refresh_token"),
                refresh_token=refresh_token
            )

            if not result or not result.ok():
                return SessionData()

            data = result.unwrap_or({})

            new_token = data.get("token", "")
            new_refresh = data.get("refresh_token", refresh_token)

            # Get user data for new token
            session_data = SessionData(
                user_id=data.get("user_id", ""),
                username=data.get("user_name", ""),
                email=data.get("email", ""),
                level=data.get("level", 0),
                provider=data.get("provider", ""),
                provider_user_id=data.get("provider_user_id", ""),
                token=new_token,
                refresh_token=new_refresh,
                expires_at=data.get("exp", 0.0),
                created_at=data.get("created_at", 0.0),
                last_validated=datetime.now().timestamp(),
                provider_data=data.get("provider_data", {})
            )

            return session_data

        except Exception as e:
            if self.app:
                self.app.logger.error(f"Session refresh failed: {e}")
            return SessionData()

    def invalidate_session(self, token: str) -> bool:
        """
        Invalidate a cached session.

        Args:
            token: Token to invalidate

        Returns:
            True if session was invalidated
        """
        with self._cache_lock:
            if token in self._cache:
                del self._cache[token]
                return True
        return False

    def clear_cache(self):
        """Clear all cached sessions (for testing)."""
        with self._cache_lock:
            self._cache.clear()


# Legacy compatibility alias
def get_session_verifier(app=None, auth_module: str = "CloudM.Auth") -> CustomSessionVerifier:
    """
    Get or create the session verifier instance.

    Args:
        app: ToolBoxV2 application instance
        auth_module: Auth module name

    Returns:
        CustomSessionVerifier instance
    """
    return CustomSessionVerifier(app, auth_module)
