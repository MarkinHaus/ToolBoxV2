"""
API: Session validation, token refresh, logout, user data.
"""

from toolboxv2 import App, Result, get_app, get_logger
from toolboxv2.utils.system.types import ApiResult, ToolBoxInterfaces

from .jwt_tokens import _validate_jwt, _generate_tokens
from .state import _blacklist_token
from .user_store import _load_user, _save_user
from .db_helpers import _parse_db_result

from toolboxv2 import TBEF

Name = "CloudM.Auth"
version = "2.0.0"
export = get_app(f"{Name}.Export").tb


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def validate_session(
    app: App = None, request=None, token: str = None, session_token: str = None, data: dict = None
) -> ApiResult:
    """Validate JWT access token. Called by middleware and frontend."""
    if app is None:
        app = get_app(f"{Name}.validate_session")

    # Extract token from multiple sources
    jwt_token = token or session_token
    if not jwt_token and data:
        jwt_token = data.get("token") or data.get("session_token") or data.get("Jwt_claim")
    if not jwt_token and request:
        auth_header = ""
        if hasattr(request, "request") and hasattr(request.request, "headers"):
            auth_header = request.request.headers.get("Authorization", "")
        elif hasattr(request, "headers"):
            if isinstance(request.headers, dict):
                auth_header = request.headers.get("Authorization", "") or request.headers.get("authorization", "")
            else:
                auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            jwt_token = auth_header[7:]

    if not jwt_token:
        return Result.default_user_error("No token provided", data={"authenticated": False})

    valid, payload = await _validate_jwt(app, jwt_token, "access")
    if not valid:
        return Result.default_user_error(payload.get("error", "Invalid token"), data={"authenticated": False})

    user = await _load_user(app, payload["sub"])
    if not user:
        return Result.default_user_error("User not found", data={"authenticated": False})

    return Result.ok({
        "authenticated": True,
        "user_id": user.user_id,
        "user_name": user.username,
        "email": user.email,
        "level": user.level,
        "provider": payload.get("provider", ""),
        "settings": user.settings,
    })


# Compatibility alias
verify_session_token = validate_session


@export(mod_name=Name, version=version, api=True)
async def refresh_token(app: App = None, refresh_token: str = None, data: dict = None) -> ApiResult:
    """Refresh access token using refresh token."""
    if app is None:
        app = get_app(f"{Name}.refresh_token")
    if data:
        refresh_token = refresh_token or data.get("refresh_token")
    if not refresh_token:
        return Result.default_user_error("Refresh token required")

    valid, payload = await _validate_jwt(app, refresh_token, "refresh")
    if not valid:
        try:
            app.audit_logger.log_action(
                user_id="unknown", action="auth.token.refresh",
                resource="/auth/refresh", status="FAILURE",
                details={"reason": "invalid_refresh_token"}
            )
        except Exception: pass
        return Result.default_user_error(payload.get("error", "Invalid refresh token"))

    user = await _load_user(app, payload["sub"])
    if not user:
        return Result.default_user_error("User not found")

    tokens = _generate_tokens(user)
    try:
        app.audit_logger.log_action(
            user_id=user.user_id, action="auth.token.refresh",
            resource="/auth/refresh", status="SUCCESS"
        )
    except Exception: pass
    get_logger().info(f"[{Name}] Token refreshed for {user.user_id}")
    return Result.ok(tokens)


@export(mod_name=Name, version=version, api=True)
async def logout(app: App = None, token: str = None, data: dict = None) -> ApiResult:
    """Logout: blacklist current access token."""
    if app is None:
        app = get_app(f"{Name}.logout")
    if data:
        token = token or data.get("token")
    if token:
        await _blacklist_token(app, token)
        try:
            app.audit_logger.log_action(
                user_id="unknown", action="auth.logout",
                resource="/auth/logout", status="SUCCESS",
                details={"token_blacklisted": True}
            )
        except Exception: pass
    get_logger().info(f"[{Name}] Logout completed")
    return Result.ok({"logged_out": True})


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def get_user_data(app: App = None, request=None, user_id: str = None, data: dict = None) -> ApiResult:
    """Get user data by user_id."""
    if app is None:
        app = get_app(f"{Name}.get_user_data")
    if data:
        user_id = user_id or data.get("user_id")
    if not user_id:
        return Result.default_user_error("User ID required")
    user = await _load_user(app, user_id)
    if not user:
        return Result.default_user_error("User not found")

    try:
        app.audit_logger.log_action(
            user_id=user_id, action="user.read",
            resource=f"/users/{user_id}", status="SUCCESS",
            details={"accessed_by": "api"}
        )
    except Exception: pass
    return Result.ok(user.to_dict())


@export(mod_name=Name, version=version, api=True)
async def update_user_data(
    app: App = None, user_id: str = None, settings: dict = None,
    level: int = None, mod_data: dict = None, data: dict = None,
) -> ApiResult:
    """Update user profile fields."""
    if app is None:
        app = get_app(f"{Name}.update_user_data")
    if data:
        user_id = user_id or data.get("user_id")
        settings = settings or data.get("settings")
        level = level if level is not None else data.get("level")
        mod_data = mod_data or data.get("mod_data")
    if not user_id:
        return Result.default_user_error("User ID required")
    user = await _load_user(app, user_id)
    if not user:
        return Result.default_user_error("User not found")
    if settings is not None:
        user.settings.update(settings)
    if level is not None:
        user.level = level
    if mod_data is not None:
        user.mod_data.update(mod_data)
    await _save_user(app, user)

    try:
        app.audit_logger.log_action(
            user_id=user_id, action="user.update",
            resource=f"/users/{user_id}", status="SUCCESS",
            details={"updated_fields": list(filter(None, [settings, level, mod_data]))}
        )
    except Exception: pass
    return Result.ok(user.to_dict())


@export(mod_name=Name, version=version, api=False, interface=ToolBoxInterfaces.native)
def list_users(app: App = None) -> Result:
    """List all users from DB (admin/internal use)."""
    if app is None:
        app = get_app(f"{Name}.list_users")
    try:
        result = app.run_any(TBEF.DB.GET, query="AUTH_USER::*", get_results=True)
        if result.is_error():
            return Result.ok(data=[])
        raw = result.get()
        users = []
        if isinstance(raw, list):
            for entry in raw:
                parsed = _parse_db_result(entry)
                if parsed:
                    users.append({
                        "user_id": parsed.get("user_id"),
                        "username": parsed.get("username"),
                        "email": parsed.get("email"),
                        "level": parsed.get("level"),
                        "created_at": parsed.get("created_at"),
                    })
        elif isinstance(raw, dict):
            for _key, val in raw.items():
                parsed = _parse_db_result(val)
                if parsed:
                    users.append({
                        "user_id": parsed.get("user_id"),
                        "username": parsed.get("username"),
                        "email": parsed.get("email"),
                        "level": parsed.get("level"),
                        "created_at": parsed.get("created_at"),
                    })
        return Result.ok(data=users)
    except Exception as e:
        get_logger().error(f"[{Name}] Error listing users: {e}")
        return Result.default_internal_error(str(e))
