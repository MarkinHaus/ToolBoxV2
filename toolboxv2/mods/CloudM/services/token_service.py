"""
Token Service - Standard JWT Token Management
Version: 2.0.0

Verwaltet JWT Tokens mit Server Secret (HS256).
Ersetzt die alte User-Key-basierte Token-Generierung.
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from ..models import SessionToken, TokenType, User


class TokenService:
    """
    Service für JWT Token Management.

    Verwendet HS256 (HMAC) mit Server Secret statt User-Keys.
    """

    # Token Lifetimes
    ACCESS_TOKEN_LIFETIME = timedelta(minutes=15)      # 15 Minuten
    REFRESH_TOKEN_LIFETIME = timedelta(days=7)         # 7 Tage
    DEVICE_INVITE_LIFETIME = timedelta(minutes=15)     # 15 Minuten
    CLI_SESSION_LIFETIME = timedelta(hours=1)          # 1 Stunde

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize TokenService.

        Args:
            secret_key: JWT secret key. If None, reads from environment variable TOKEN_SECRET
        """
        self.secret_key = secret_key or os.getenv('TOKEN_SECRET')

        if not self.secret_key:
            raise ValueError(
                "TOKEN_SECRET not configured. Set environment variable TOKEN_SECRET or pass secret_key parameter."
            )

        self.algorithm = 'HS256'

    def _get_lifetime(self, token_type: str) -> timedelta:
        """Get lifetime for token type"""
        lifetimes = {
            TokenType.ACCESS: self.ACCESS_TOKEN_LIFETIME,
            TokenType.REFRESH: self.REFRESH_TOKEN_LIFETIME,
            TokenType.DEVICE_INVITE: self.DEVICE_INVITE_LIFETIME,
            TokenType.CLI_SESSION: self.CLI_SESSION_LIFETIME,
        }
        return lifetimes.get(token_type, self.ACCESS_TOKEN_LIFETIME)

    def create_token(
        self,
        user: User,
        token_type: str,
        device_label: Optional[str] = None,
        custom_lifetime: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT token.

        Args:
            user: User object
            token_type: Token type (access/refresh/device_invite/cli_session)
            device_label: Optional device label for tracking
            custom_lifetime: Optional custom token lifetime

        Returns:
            Encoded JWT token string
        """
        now = datetime.utcnow()
        lifetime = custom_lifetime or self._get_lifetime(token_type)

        token_data = SessionToken(
            sub=user.username,
            uid=user.uid,
            type=token_type,
            exp=int((now + lifetime).timestamp()),
            iat=int(now.timestamp()),
            device_label=device_label
        )

        payload = token_data.model_dump()

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_token_pair(
        self,
        user: User,
        device_label: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create access + refresh token pair.

        Args:
            user: User object
            device_label: Optional device label

        Returns:
            Dict with 'access_token' and 'refresh_token'
        """
        access_token = self.create_token(user, TokenType.ACCESS, device_label)
        refresh_token = self.create_token(user, TokenType.REFRESH, device_label)

        return {
            'access_token': access_token,
            'refresh_token': refresh_token
        }

    def validate_token(self, token: str, expected_type: Optional[str] = None) -> Optional[SessionToken]:
        """
        Validate and decode a JWT token.

        Args:
            token: JWT token string
            expected_type: Expected token type (optional validation)

        Returns:
            SessionToken if valid, None if invalid/expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            token_data = SessionToken(**payload)

            # Validate token type if specified
            if expected_type and token_data.type != expected_type:
                return None

            return token_data

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None

    def refresh_access_token(self, refresh_token: str, user: User) -> Optional[str]:
        """
        Create new access token from refresh token.

        Args:
            refresh_token: Valid refresh token
            user: User object (for validation)

        Returns:
            New access token if refresh token is valid, None otherwise
        """
        token_data = self.validate_token(refresh_token, expected_type=TokenType.REFRESH)

        if not token_data:
            return None

        # Validate user matches
        if token_data.sub != user.username or token_data.uid != user.uid:
            return None

        # Create new access token
        return self.create_token(user, TokenType.ACCESS, token_data.device_label)

    def decode_token_unsafe(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode token without validation (for debugging).

        Args:
            token: JWT token string

        Returns:
            Decoded payload dict or None
        """
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except Exception:
            return None


# Global singleton instance
_token_service_instance: Optional[TokenService] = None


def get_token_service(secret_key: Optional[str] = None) -> TokenService:
    """Get or create the global TokenService instance"""
    global _token_service_instance

    if _token_service_instance is None:
        _token_service_instance = TokenService(secret_key)

    return _token_service_instance

