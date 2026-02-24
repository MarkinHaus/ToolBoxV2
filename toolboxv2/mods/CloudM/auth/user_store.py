"""
User CRUD operations — load, save, find by provider/email.
"""

import time
import secrets
from typing import Optional

from toolboxv2 import App

from .models import UserData, OAuthProvider
from .db_helpers import _db_set, _db_get, _db_get_raw, _parse_db_result


async def _save_user(app: App, user: UserData):
    """Save user profile + all indexes."""
    await _db_set(app, f"AUTH_USER::{user.user_id}", user.to_dict())
    if user.email:
        await _db_set(app, f"AUTH_USER_EMAIL::{user.email}", user.user_id)
    for provider, pdata in user.oauth_providers.items():
        pid = pdata.get("provider_id", "")
        if pid:
            await _db_set(app, f"AUTH_USER_PROVIDER::{provider}::{pid}", user.user_id)
    for pk in user.passkeys:
        cid = pk.get("credential_id", "")
        if cid:
            await _db_set(app, f"AUTH_USER_PROVIDER::passkey::{cid}", user.user_id)


async def _load_user(app: App, user_id: str) -> Optional[UserData]:
    """Load user by user_id."""
    data = await _db_get(app, f"AUTH_USER::{user_id}")
    return UserData.from_dict(data) if data else None


async def _find_user_by_provider(app: App, provider: str, provider_id: str) -> Optional[UserData]:
    """Find user via provider index."""
    raw = await _db_get_raw(app, f"AUTH_USER_PROVIDER::{provider}::{provider_id}")
    if not raw:
        return None
    user_id = raw.strip('"')
    return await _load_user(app, user_id)


async def _find_user_by_email(app: App, email: str) -> Optional[UserData]:
    """Find user via email index."""
    raw = await _db_get_raw(app, f"AUTH_USER_EMAIL::{email}")
    if not raw:
        return None
    user_id = raw.strip('"')
    return await _load_user(app, user_id)


async def _create_or_update_user(
    app: App, provider: str, provider_data: dict, oauth_tokens: dict
) -> tuple[UserData, bool]:
    """Find existing user by provider or create new one. Returns (user, is_new)."""
    provider_id = provider_data["provider_id"]

    user = await _find_user_by_provider(app, provider, provider_id)
    is_new = False

    if not user and provider_data.get("email"):
        user = await _find_user_by_email(app, provider_data["email"])

    if not user:
        is_new = True
        user = UserData(
            user_id=f"usr_{secrets.token_hex(12)}",
            username=provider_data["username"],
            email=provider_data.get("email", ""),
            created_at=time.time(),
        )

    # Update provider data
    oauth_entry = OAuthProvider(
        provider_id=provider_id,
        provider=provider,
        access_token=oauth_tokens.get("access_token", ""),
        refresh_token=oauth_tokens.get("refresh_token", ""),
        token_expires=time.time() + oauth_tokens.get("expires_in", 3600),
        username=provider_data["username"],
        email=provider_data.get("email", ""),
        avatar=provider_data.get("avatar", ""),
    )
    user.oauth_providers[provider] = oauth_entry.to_dict()
    user.last_login = time.time()

    if not user.email and provider_data.get("email"):
        user.email = provider_data["email"]

    await _save_user(app, user)
    return user, is_new
