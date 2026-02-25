"""
CloudM.Auth Client for Token Validation

This client validates JWT tokens issued by CloudM.Auth module.

Architecture:
- tb-registry is a separate FastAPI service
- CloudM.Auth runs in ToolBoxV2 main app
- Validation can be done via:
  1. Local JWT validation (preferred - shared secret)
  2. Remote API call to ToolBoxV2 /auth/validateSession

Classes:
    CloudMAuthClient: Main client class

Functions:
    validate_token(token: str) -> Optional[dict]
        Validate JWT token, return payload if valid

Configuration:
    CLOUDM_JWT_SECRET: Shared secret for JWT validation
    CLOUDM_AUTH_URL: URL to ToolBoxV2 instance (fallback)
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass

import jwt

logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    """CloudM.Auth JWT token payload."""
    user_id: str
    username: str
    email: str
    level: int
    provider: str
    exp: int  # Expiry timestamp


class CloudMAuthClient:
    """Client for validating CloudM.Auth JWT tokens."""

    def __init__(self, jwt_secret: str, auth_url: Optional[str] = None):
        """Initialize client.

        Args:
            jwt_secret: Shared secret for JWT validation (HS256)
            auth_url: Optional URL for remote validation fallback
        """
        self.jwt_secret = jwt_secret
        self.auth_url = auth_url

    def validate_token(self, token: str) -> Optional[TokenPayload]:
        """Validate JWT token locally.

        Uses PyJWT library to decode and verify HS256 signed tokens.

        Args:
            token: JWT token string

        Returns:
            TokenPayload if valid, None if invalid

        Token Structure (from CloudM.Auth):
        {
            "user_id": str,
            "username": str,
            "email": str,
            "level": int,
            "provider": str,  # "discord", "google", "magic_link", etc.
            "exp": int,      # Unix timestamp
            "iat": int,      # Issued at
            "jti": str       # JWT ID for blacklist
        }
        """
        try:
            # Decode and verify signature
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"],
                options={
                    "verify_signature": True,
                    "require": ["user_id", "username", "email", "exp"],
                }
            )

            # Check expiry (already done by jwt.decode, but explicit check)
            if payload.get("exp", 0) < int(time.time()):
                logger.warning("Token expired")
                return None

            # Return structured payload
            return TokenPayload(
                user_id=payload["user_id"],
                username=payload["username"],
                email=payload["email"],
                level=payload.get("level", 1),
                provider=payload.get("provider", "unknown"),
                exp=payload["exp"],
            )

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
