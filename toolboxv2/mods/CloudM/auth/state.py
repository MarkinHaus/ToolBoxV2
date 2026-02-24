"""
Ephemeral state management — OAuth state, WebAuthn challenges, token blacklist.
All stored in TBEF.DB with timestamp-based expiry.
"""

import time
import secrets
from typing import Optional

import jwt as pyjwt

from toolboxv2 import App

from .config import STATE_EXPIRY, CHALLENGE_EXPIRY
from .db_helpers import _db_set, _db_get, _db_delete, _db_exists


# =================== OAuth State ===================

async def _store_oauth_state(app: App, provider: str, extra: dict = None) -> str:
    """Create and store OAuth CSRF state in DB."""
    state = secrets.token_urlsafe(32)
    data = {"provider": provider, "created_at": time.time(), "extra": extra or {}}
    key = f"AUTH_STATE::{state}"
    await _db_set(app, key, data)
    from toolboxv2 import get_logger
    get_logger().debug(f"[CloudM.Auth] Stored OAuth state: {key[:20]}... for {provider}, redirect_after: {extra}")
    return state


async def _validate_and_consume_state(app: App, state: str, provider: str) -> tuple[bool, dict]:
    """Validate OAuth state: check provider, expiry, delete after use."""
    key = f"AUTH_STATE::{state}"
    from toolboxv2 import get_logger
    get_logger().debug(f"[CloudM.Auth] Looking for OAuth state: {key[:20]}... for {provider}")
    data = await _db_get(app, key)
    if not data:
        get_logger().warning(f"[CloudM.Auth] OAuth state not found: {key[:20]}... for {provider}")
        return False, {}
    await _db_delete(app, key)
    if data.get("provider") != provider:
        return False, {}
    if time.time() - data.get("created_at", 0) > STATE_EXPIRY:
        return False, {}
    return True, data.get("extra", {})


# =================== WebAuthn Challenge ===================

async def _store_challenge(app: App, challenge: str, data: dict):
    """Store WebAuthn challenge in DB."""
    data["created_at"] = time.time()
    await _db_set(app, f"AUTH_CHALLENGE::{challenge}", data)


async def _validate_and_consume_challenge(app: App, challenge: str, expected_type: str) -> Optional[dict]:
    """Validate challenge: check type, expiry, delete after use."""
    data = await _db_get(app, f"AUTH_CHALLENGE::{challenge}")
    if not data:
        return None
    await _db_delete(app, f"AUTH_CHALLENGE::{challenge}")
    if data.get("type") != expected_type:
        return None
    if time.time() - data.get("created_at", 0) > CHALLENGE_EXPIRY:
        return None
    return data


# =================== Token Blacklist ===================

async def _blacklist_token(app: App, token_str: str):
    """Add token JTI to blacklist."""
    try:
        payload = pyjwt.decode(token_str, options={"verify_signature": False})
        jti = payload.get("jti")
        if jti:
            await _db_set(app, f"AUTH_BLACKLIST::{jti}", {
                "blacklisted_at": time.time(),
                "expires": payload.get("exp", 0),
            })
    except Exception:
        pass


async def _is_blacklisted(app: App, jti: str) -> bool:
    """Check if token JTI is blacklisted."""
    return await _db_exists(app, f"AUTH_BLACKLIST::{jti}")
