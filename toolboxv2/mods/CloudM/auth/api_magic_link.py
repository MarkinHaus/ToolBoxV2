"""
API: Magic link email auth and device invite codes (CLI auth).
"""

import time
import secrets

from toolboxv2 import App, Result, get_app, get_logger
from toolboxv2.utils.system.types import ApiResult

from .config import get_base_url, MAGIC_LINK_EXPIRY, DEVICE_INVITE_EXPIRY
from .models import UserData
from .db_helpers import _db_set, _db_get, _db_delete
from .user_store import _load_user, _save_user, _find_user_by_email
from .jwt_tokens import _generate_tokens

Name = "CloudM.Auth"
version = "2.0.0"
export = get_app(f"{Name}.Export").tb


@export(mod_name=Name, version=version, api=True)
async def request_magic_link(app: App = None, email: str = None, data: dict = None) -> ApiResult:
    """Generate magic link, store in DB, send email."""
    if app is None:
        app = get_app(f"{Name}.request_magic_link")
    if data:
        email = email or data.get("email")
    if not email or "@" not in email:
        return Result.default_user_error("Valid email required")

    token = secrets.token_urlsafe(32)
    await _db_set(app, f"AUTH_MAGIC_LINK::{token}", {
        "email": email,
        "created_at": time.time(),
        "verified": False,
    })

    base_url = get_base_url()
    link = f"{base_url}/auth/magic/{token}"

    try:
        from ..email_services import send_magic_link_email
        send_magic_link_email(app, email, link)
    except Exception as e:
        get_logger().error(f"[{Name}] Failed to send magic link email: {e}")
        import traceback
        traceback.print_exc()
        return Result.default_internal_error(f"Failed to send email: {e}")

    return Result.ok({"message": "Magic link sent", "token_hint": token[:8]})


@export(mod_name=Name, version=version, api=True)
async def verify_magic_link(app: App = None, token: str = None, data: dict = None) -> ApiResult:
    """Verify magic link token, create/find user, return JWT."""
    if app is None:
        app = get_app(f"{Name}.verify_magic_link")
    if data:
        token = token or data.get("token")
    if not token:
        return Result.default_user_error("Token required")

    ml_data = await _db_get(app, f"AUTH_MAGIC_LINK::{token}")
    if not ml_data:
        try:
            app.audit_logger.log_action(
                user_id="unknown", action="auth.login.magic_link",
                resource="/auth/magic/verify", status="FAILURE",
                details={"reason": "invalid_token"}
            )
        except Exception: pass
        return Result.default_user_error("Invalid or expired link")

    await _db_delete(app, f"AUTH_MAGIC_LINK::{token}")

    if time.time() - ml_data.get("created_at", 0) > MAGIC_LINK_EXPIRY:
        try:
            app.audit_logger.log_action(
                user_id="unknown", action="auth.login.magic_link",
                resource="/auth/magic/verify", status="FAILURE",
                details={"reason": "link_expired", "email": ml_data.get("email")}
            )
        except Exception: pass
        return Result.default_user_error("Link expired")

    email = ml_data["email"]
    user = await _find_user_by_email(app, email)
    if not user:
        username = email.split("@")[0]
        user = UserData(
            user_id=f"usr_{secrets.token_hex(12)}",
            username=username,
            email=email,
            created_at=time.time(),
        )
        await _save_user(app, user)

    user.last_login = time.time()
    await _save_user(app, user)

    jwt_tokens = _generate_tokens(user, "magic_link")
    get_logger().info(f"[{Name}] Magic link login: {user.user_id}")

    try:
        app.audit_logger.log_action(
            user_id=user.user_id, action="auth.login.magic_link",
            resource="/auth/magic/verify", status="SUCCESS",
            details={"email": email, "auto_created": user is None}
        )
    except Exception: pass

    return Result.ok({
        "authenticated": True,
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "level": user.level,
        "provider": "magic_link",
        **jwt_tokens,
    })


@export(mod_name=Name, version=version, api=True)
async def check_magic_link_status(app: App = None, token_hint: str = None, data: dict = None) -> ApiResult:
    """CLI polls this to check if magic link was clicked. (For future polling flow.)"""
    return Result.default_user_error("Use verify_magic_link directly")


@export(mod_name=Name, version=version, api=True)
async def create_device_invite(app: App = None, user_id: str = None, data: dict = None) -> ApiResult:
    """Generate a 6-digit device invite code for CLI pairing."""
    if app is None:
        app = get_app(f"{Name}.create_device_invite")
    if data:
        user_id = user_id or data.get("user_id")
    if not user_id:
        return Result.default_user_error("User ID required (must be logged in)")

    user = await _load_user(app, user_id)
    if not user:
        return Result.default_user_error("User not found")

    code = f"{secrets.randbelow(1000000):06d}"
    await _db_set(app, f"AUTH_DEVICE_INVITE::{code}", {
        "user_id": user_id,
        "created_at": time.time(),
    })

    return Result.ok({"code": code, "expires_in": DEVICE_INVITE_EXPIRY})


@export(mod_name=Name, version=version, api=True)
async def verify_device_invite(app: App = None, code: str = None, data: dict = None) -> ApiResult:
    """Verify device invite code and generate JWT for new device."""
    if app is None:
        app = get_app(f"{Name}.verify_device_invite")
    if data:
        code = code or data.get("code")
    if not code:
        return Result.default_user_error("Invite code required")

    invite_data = await _db_get(app, f"AUTH_DEVICE_INVITE::{code}")
    if not invite_data:
        return Result.default_user_error("Invalid or expired invite code")

    await _db_delete(app, f"AUTH_DEVICE_INVITE::{code}")

    if time.time() - invite_data.get("created_at", 0) > DEVICE_INVITE_EXPIRY:
        return Result.default_user_error("Invite code expired")

    user = await _load_user(app, invite_data["user_id"])
    if not user:
        return Result.default_user_error("User not found")

    jwt_tokens = _generate_tokens(user, "device_invite")
    get_logger().info(f"[{Name}] Device invite login: {user.user_id}")

    try:
        app.audit_logger.log_action(
            user_id=user.user_id, action="auth.login.device_invite",
            resource="/auth/device/verify", status="SUCCESS",
            details={"code": code}
        )
    except Exception: pass

    return Result.ok({
        "authenticated": True,
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "level": user.level,
        "provider": "device_invite",
        **jwt_tokens,
    })
