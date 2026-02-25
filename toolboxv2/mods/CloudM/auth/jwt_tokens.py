"""
JWT token generation and validation.
"""

import time
import uuid

import jwt as pyjwt

from toolboxv2 import App

from .config import JWT_ALGORITHM, ACCESS_TOKEN_EXPIRY, REFRESH_TOKEN_EXPIRY, get_jwt_secret
from .models import UserData
from .state import _is_blacklisted


def _generate_access_token(user_id: str, username: str, level: int, provider: str = "", email: str = "") -> str:
    payload = {
        "sub": user_id,
        "user_id": user_id,  # alias for cloudm_client compatibility
        "username": username,
        "email": email,
        "level": level,
        "provider": provider,
        "type": "access",
        "iat": time.time(),
        "exp": time.time() + ACCESS_TOKEN_EXPIRY,
        "jti": str(uuid.uuid4()),
    }
    return pyjwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def _generate_refresh_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "type": "refresh",
        "iat": time.time(),
        "exp": time.time() + REFRESH_TOKEN_EXPIRY,
        "jti": str(uuid.uuid4()),
    }
    return pyjwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def _generate_tokens(user: UserData, provider: str = "") -> dict:
    """Generate access + refresh token pair."""
    return {
        "access_token": _generate_access_token(
            user.user_id, user.username, user.level, provider,
            email=getattr(user, "email", ""),
        ),
        "refresh_token": _generate_refresh_token(user.user_id),
        "expires_in": ACCESS_TOKEN_EXPIRY,
        "token_type": "Bearer",
    }


async def _validate_jwt(app: App, token_str: str, token_type: str = "access") -> tuple[bool, dict]:
    """Validate JWT: signature, type, expiry, blacklist."""
    if not token_str:
        return False, {"error": "No token provided"}
    try:
        payload = pyjwt.decode(token_str, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
    except pyjwt.ExpiredSignatureError:
        return False, {"error": "Token expired"}
    except pyjwt.InvalidTokenError as e:
        return False, {"error": f"Invalid token: {e}"}

    if payload.get("type") != token_type:
        return False, {"error": f"Expected {token_type} token"}

    jti = payload.get("jti", "")
    if jti and await _is_blacklisted(app, jti):
        return False, {"error": "Token has been revoked"}

    return True, payload
