"""WP1 K1: Device account - the local marker that ties this install to one
SimpleCore account. device_owner.json holds identity only (never tokens);
tokens live in the existing session token store (F17a).

Migration: an existing local admin (LOCAL_ADMIN_EMAIL) hands its settings
(without anonymous/local_admin flags) and mod_data to the account user;
the account wins on conflict. The old record and its email index are
deleted, migrated_from records the origin.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

from toolboxv2 import App

from .config import LOCAL_ADMIN_EMAIL
from .models import UserData
from .user_store import _find_user_by_email, _load_user, _save_user
from .db_helpers import _db_delete

DEFAULT_BASE_URL = "https://simplecore.app"
_ADMIN_ONLY_SETTINGS = ("anonymous", "local_admin")


def device_owner_file(app: App) -> Path:
    return Path(getattr(app, "data_dir", ".")) / "device_owner.json"


def read_device_owner(app: App) -> Optional[dict]:
    f = device_owner_file(app)
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) and data.get("user_id") else None


def local_trust_provider(app, is_production: bool) -> Optional[dict]:
    """Provider for SessionManager.enable_local_trust (WP1 K2).
    Only profile local/developer, never production, only with a linked
    device account. No local-admin fallback (account is mandatory)."""
    if is_production:
        return None
    profile = getattr(getattr(getattr(app, "manifest", None), "app", None), "profile", None)
    value = getattr(profile, "value", profile)
    if value not in ("local", "developer"):
        return None
    owner = read_device_owner(app)
    if not owner or not owner.get("user_id"):
        return None
    return {"user_id": owner["user_id"], "username": owner.get("username", ""), "level": -1}


async def link_account(app: App, payload: dict) -> UserData:
    """Link this device to a SimpleCore account (idempotent per account).

    payload: dict with access_token, refresh_token, user_id, username
    (as delivered by cli_remote_login, F17). Raises RuntimeError when
    already linked to another account or the remote login fails.
    """
    user_id = payload.get("user_id")
    if not user_id:
        raise RuntimeError("payload missing user_id")

    owner = read_device_owner(app)
    if owner is not None:
        if owner.get("user_id") != user_id:
            raise RuntimeError("device already linked")
        user = await _load_user(app, user_id)
        if user is None:  # marker survived, record lost - restore shell
            user = UserData(user_id=user_id,
                            username=owner.get("username") or payload.get("username", ""),
                            level=-1)
            await _save_user(app, user)
        # WP1 Teil 10: initial account sync right after linking (best-effort).
    try:
        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import run_sync_in_background
        run_sync_in_background(app)
    except Exception:
        pass
    return user

    # Tokens via the existing session path (F17a); base points at the remote.
    s = app.session
    s.base = os.environ.get("TOOLBOXV2_REMOTE_BASE") or DEFAULT_BASE_URL
    s._save_session_token(payload.get("access_token", ""),
                          payload.get("refresh_token", ""), user_id)
    if not await s.login():
        raise RuntimeError("remote login failed - device not linked")

    user = await _load_user(app, user_id)
    if user is None:
        user = UserData(user_id=user_id, email="",
                        username=payload.get("username", ""), level=-1)

    migrated_from = None
    admin = await _find_user_by_email(app, LOCAL_ADMIN_EMAIL)
    if admin is not None and admin.user_id != user_id:
        for k, v in admin.settings.items():
            if k not in _ADMIN_ONLY_SETTINGS and k not in user.settings:
                user.settings[k] = v
        for k, v in admin.mod_data.items():
            if k not in user.mod_data:
                user.mod_data[k] = v
        migrated_from = admin.user_id
        await _db_delete(app, f"AUTH_USER::{admin.user_id}")
        await _db_delete(app, f"AUTH_USER_EMAIL::{LOCAL_ADMIN_EMAIL}")

    await _save_user(app, user)

    f = device_owner_file(app)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({
        "user_id": user_id,
        "username": payload.get("username", ""),
        "base_url": s.base,
        "linked_at": time.time(),
        "migrated_from": migrated_from,
    }, indent=2), encoding="utf-8")

    return user


async def unlink_account(app: App) -> None:
    """Remove the device marker. Does not touch remote or user records."""
    device_owner_file(app).unlink(missing_ok=True)

