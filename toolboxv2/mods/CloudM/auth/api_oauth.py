"""
API: OAuth URL generation and callback handlers (Discord, Google).
"""

from urllib.parse import urlencode

from toolboxv2 import App, Result, get_app, get_logger
from toolboxv2.utils.system.types import ApiResult

from .config import get_discord_config, get_google_config
from .state import _store_oauth_state, _validate_and_consume_state
from .oauth import _exchange_oauth_code, _get_discord_user, _get_google_user
from .user_store import _create_or_update_user
from .jwt_tokens import _generate_tokens

Name = "CloudM.Auth"
version = "2.0.0"
export = get_app(f"{Name}.Export").tb


@export(mod_name=Name, version=version, api=True)
async def get_discord_auth_url(app: App = None, redirect_after: str = None) -> ApiResult:
    """Generate Discord OAuth authorization URL."""
    if app is None:
        app = get_app(f"{Name}.get_discord_auth_url")
    config = get_discord_config()
    if not config["client_id"]:
        return Result.default_internal_error("Discord OAuth not configured")
    state = await _store_oauth_state(app, "discord", {"redirect_after": redirect_after})
    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "response_type": "code",
        "scope": " ".join(config["scopes"]),
        "state": state,
    }
    return Result.ok({"auth_url": f"{config['authorize_url']}?{urlencode(params)}", "state": state})


@export(mod_name=Name, version=version, api=True)
async def get_google_auth_url(app: App = None, redirect_after: str = None) -> ApiResult:
    """Generate Google OAuth authorization URL."""
    if app is None:
        app = get_app(f"{Name}.get_google_auth_url")
    config = get_google_config()
    if not config["client_id"]:
        return Result.default_internal_error("Google OAuth not configured")
    state = await _store_oauth_state(app, "google", {"redirect_after": redirect_after})
    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "response_type": "code",
        "scope": " ".join(config["scopes"]),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    return Result.ok({"auth_url": f"{config['authorize_url']}?{urlencode(params)}", "state": state})


@export(mod_name=Name, version=version, api=True)
async def login_discord(app: App = None, code: str = None, state: str = None, data: dict = None) -> ApiResult:
    """Discord OAuth callback handler."""
    if app is None:
        app = get_app(f"{Name}.login_discord")
    log = get_logger()
    if data:
        code = code or data.get("code")
        state = state or data.get("state")
    if not code:
        return Result.default_user_error("Authorization code required")

    redirect_after = None
    if state:
        valid, _extra = await _validate_and_consume_state(app, state, "discord")
        if not valid:
            log.warning(f"[{Name}] Invalid OAuth state for Discord: {state[:8]}...")
            try:
                app.audit_logger.log_action(
                    user_id="unknown", action="auth.login.discord",
                    resource="/auth/login/discord", status="FAILURE",
                    details={"reason": "invalid_state"}
                )
            except Exception: pass
            return Result.default_user_error(f"Invalid or expired OAuth state (state: {state[:8]}...). Please try again.")
        redirect_after = (_extra or {}).get("redirect_after")

    config = get_discord_config()
    ok, tokens = await _exchange_oauth_code(config, code)
    if not ok:
        log.error(f"[{Name}] Discord token exchange failed: {tokens}")
        try:
            app.audit_logger.log_action(
                user_id="unknown", action="auth.login.discord",
                resource="/auth/login/discord", status="FAILURE",
                details={"reason": "token_exchange_failed"}
            )
        except Exception: pass
        return Result.default_internal_error(tokens.get("error", "Token exchange failed"))

    ok, user_info = await _get_discord_user(tokens["access_token"])
    if not ok:
        log.error(f"[{Name}] Discord user fetch failed: {user_info}")
        try:
            app.audit_logger.log_action(
                user_id="unknown", action="auth.login.discord",
                resource="/auth/login/discord", status="FAILURE",
                details={"reason": "user_fetch_failed"}
            )
        except Exception: pass
        return Result.default_internal_error(user_info.get("error", "Failed to get user"))

    user, is_new = await _create_or_update_user(app, "discord", user_info, tokens)
    jwt_tokens = _generate_tokens(user, "discord")
    log.info(f"[{Name}] Discord login: {user.user_id} (new={is_new})")

    try:
        app.audit_logger.log_action(
            user_id=user.user_id, action="auth.login.discord",
            resource="/auth/login/discord", status="SUCCESS",
            details={"provider_user_id": user_info.get("id"), "is_new_user": is_new}
        )
    except Exception: pass

    result_data = {
        "authenticated": True,
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "level": user.level,
        "provider": "discord",
        "is_new_user": is_new,
        **jwt_tokens,
    }
    if redirect_after:
        result_data["redirect_after"] = redirect_after
    return Result.ok(result_data)


@export(mod_name=Name, version=version, api=True)
async def login_google(app: App = None, code: str = None, state: str = None, data: dict = None) -> ApiResult:
    """Google OAuth callback handler."""
    if app is None:
        app = get_app(f"{Name}.login_google")
    log = get_logger()
    if data:
        code = code or data.get("code")
        state = state or data.get("state")
    if not code:
        return Result.default_user_error("Authorization code required")

    redirect_after = None
    if state:
        valid, _extra = await _validate_and_consume_state(app, state, "google")
        if not valid:
            log.warning(f"[{Name}] Invalid OAuth state for Google: {state[:8]}...")
            try:
                app.audit_logger.log_action(
                    user_id="unknown", action="auth.login.google",
                    resource="/auth/login/google", status="FAILURE",
                    details={"reason": "invalid_state"}
                )
            except Exception: pass
            return Result.default_user_error(f"Invalid or expired OAuth state (state: {state[:8]}...). Please try again.")
        redirect_after = (_extra or {}).get("redirect_after")

    config = get_google_config()
    ok, tokens = await _exchange_oauth_code(config, code)
    if not ok:
        log.error(f"[{Name}] Google token exchange failed: {tokens}")
        try:
            app.audit_logger.log_action(
                user_id="unknown", action="auth.login.google",
                resource="/auth/login/google", status="FAILURE",
                details={"reason": "token_exchange_failed"}
            )
        except Exception: pass
        return Result.default_internal_error(tokens.get("error", "Token exchange failed"))

    ok, user_info = await _get_google_user(tokens["access_token"])
    if not ok:
        log.error(f"[{Name}] Google user fetch failed: {user_info}")
        try:
            app.audit_logger.log_action(
                user_id="unknown", action="auth.login.google",
                resource="/auth/login/google", status="FAILURE",
                details={"reason": "user_fetch_failed"}
            )
        except Exception: pass
        return Result.default_internal_error(user_info.get("error", "Failed to get user"))

    user, is_new = await _create_or_update_user(app, "google", user_info, tokens)
    jwt_tokens = _generate_tokens(user, "google")
    log.info(f"[{Name}] Google login: {user.user_id} (new={is_new})")

    try:
        app.audit_logger.log_action(
            user_id=user.user_id, action="auth.login.google",
            resource="/auth/login/google", status="SUCCESS",
            details={"provider_user_id": user_info.get("id"), "is_new_user": is_new}
        )
    except Exception: pass

    result_data = {
        "authenticated": True,
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "level": user.level,
        "provider": "google",
        "is_new_user": is_new,
        **jwt_tokens,
    }
    if redirect_after:
        result_data["redirect_after"] = redirect_after
    return Result.ok(result_data)
