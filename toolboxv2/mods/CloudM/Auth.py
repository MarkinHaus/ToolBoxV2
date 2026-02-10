"""
ToolBox V2 - Custom Authentication System
Replaces Clerk with self-hosted OAuth2 + Passkey + Magic Link system.

Features:
- Discord OAuth2
- Google OAuth2
- WebAuthn Passkeys (py_webauthn)
- JWT Token Management (self-hosted, HS256)
- Magic Link Email (CLI Auth)
- Device Invite Codes (CLI Auth)
- Token Blacklisting

ARCHITECTURE:
- TBEF.DB = Single Source of Truth (NO BlobFile, NO global dicts)
- Provider-agnostic: system is "blind" to auth provider
- All state externalized via TBEF.DB namespaces
- Multi-worker safe (no in-memory state)

DB NAMESPACES:
- AUTH_USER::{user_id}                          → User profile (permanent)
- AUTH_USER_PROVIDER::{provider}::{provider_id} → Provider→User index (permanent)
- AUTH_USER_EMAIL::{email}                      → Email→User index (permanent)
- AUTH_STATE::{state}                           → OAuth CSRF state (10min, timestamp)
- AUTH_CHALLENGE::{challenge}                   → WebAuthn challenge (5min, timestamp)
- AUTH_BLACKLIST::{jti}                         → Token blacklist (until token expiry)
- AUTH_MAGIC_LINK::{token}                      → Magic link (10min, timestamp)
- AUTH_DEVICE_INVITE::{code}                    → Device invite (5min, timestamp)
"""

import os
import json
import time
import uuid
import secrets
import base64
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
from urllib.parse import urlencode

import jwt
import httpx

from toolboxv2 import App, Result, get_app, get_logger, TBEF
from toolboxv2.utils.system.types import ApiResult, ToolBoxInterfaces

Name = "CloudM.Auth"
version = "2.0.0"
export = get_app(f"{Name}.Export").tb

# =================== Constants ===================

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRY = 15 * 60       # 15 minutes
REFRESH_TOKEN_EXPIRY = 7 * 24 * 3600  # 7 days
STATE_EXPIRY = 10 * 60              # 10 minutes for OAuth state
CHALLENGE_EXPIRY = 5 * 60           # 5 minutes for WebAuthn challenges
MAGIC_LINK_EXPIRY = 10 * 60         # 10 minutes for magic links
DEVICE_INVITE_EXPIRY = 5 * 60       # 5 minutes for device invites


# =================== Data Types ===================

@dataclass
class OAuthProvider:
    """OAuth Provider data for a user"""
    provider_id: str
    provider: str       # "discord", "google"
    access_token: str = ""
    refresh_token: str = ""
    token_expires: float = 0.0
    username: str = ""
    email: str = ""
    avatar: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OAuthProvider":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Passkey:
    """WebAuthn Passkey credential"""
    credential_id: str
    public_key: str           # base64-encoded
    sign_count: int = 0
    name: str = "Passkey"
    transports: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Passkey":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UserData:
    """User profile stored in AUTH_USER::{user_id}"""
    user_id: str
    username: str
    email: str
    level: int = 1
    created_at: float = field(default_factory=time.time)
    last_login: float = 0.0
    settings: dict = field(default_factory=dict)
    mod_data: dict = field(default_factory=dict)
    oauth_providers: Dict[str, dict] = field(default_factory=dict)
    passkeys: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserData":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =================== Environment & Config ===================

def is_production() -> bool:
    return os.getenv("TB_ENV", "development").lower() == "production"


def get_jwt_secret() -> str:
    secret = os.getenv("TB_JWT_SECRET") or os.getenv("TB_COOKIE_SECRET")
    if not secret:
        raise ValueError("TB_JWT_SECRET or TB_COOKIE_SECRET not set")
    return secret


def get_base_url() -> str:
    if is_production():
        return os.getenv("APP_BASE_URL_PROD", "https://simplecore.app")
    return os.getenv("APP_BASE_URL", "http://localhost:8000")


def get_discord_config() -> dict:
    return {
        "client_id": os.getenv("DISCORD_CLIENT_ID", ""),
        "client_secret": os.getenv("DISCORD_CLIENT_SECRET", ""),
        "redirect_uri": os.getenv("DISCORD_REDIRECT_URI", f"{get_base_url()}/auth/discord/callback"),
        "scopes": ["identify", "email"],
        "authorize_url": "https://discord.com/api/oauth2/authorize",
        "token_url": "https://discord.com/api/oauth2/token",
        "user_url": "https://discord.com/api/users/@me",
    }


def get_google_config() -> dict:
    return {
        "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
        "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI", f"{get_base_url()}/auth/google/callback"),
        "scopes": ["openid", "email", "profile"],
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "user_url": "https://www.googleapis.com/oauth2/v2/userinfo",
    }


def get_passkey_config() -> dict:
    rp_id = os.getenv("PASSKEY_RP_ID")
    if not rp_id:
        base = get_base_url()
        rp_id = base.replace("https://", "").replace("http://", "").split(":")[0]
    return {
        "rp_id": rp_id,
        "rp_name": os.getenv("PASSKEY_RP_NAME", "ToolBoxV2"),
        "origin": os.getenv("PASSKEY_ORIGIN", get_base_url()),
    }


# =================== DB Helpers (TBEF.DB = Single Source of Truth) ===================

def _parse_db_result(raw) -> Optional[dict]:
    """Safely parse a DB result into a dict."""
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    return raw if isinstance(raw, dict) else None


async def _db_set(app: App, key: str, data) -> Result:
    """Store data in TBEF.DB."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    return await app.a_run_any(TBEF.DB.SET, query=key, data=payload, get_results=True)


async def _db_get(app: App, key: str) -> Optional[dict]:
    """Load data from TBEF.DB, returns parsed dict or None."""
    result = await app.a_run_any(TBEF.DB.GET, query=key, get_results=True)
    if result.is_error():
        return None
    return _parse_db_result(result.get())


async def _db_get_raw(app: App, key: str) -> Optional[str]:
    """Load raw string from TBEF.DB."""
    result = await app.a_run_any(TBEF.DB.GET, query=key, get_results=True)
    if result.is_error():
        return None
    raw = result.get()
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, bytes):
        raw = raw.decode()
    return raw


async def _db_delete(app: App, key: str) -> Result:
    """Delete data from TBEF.DB."""
    return await app.a_run_any(TBEF.DB.DELETE, query=key, get_results=True)


async def _db_exists(app: App, key: str) -> bool:
    """Check if key exists in TBEF.DB."""
    result = await app.a_run_any(TBEF.DB.IF_EXIST, query=key, get_results=True)
    if result.is_error():
        return False
    count = result.get()
    return isinstance(count, int) and count > 0


# =================== User Storage ===================

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


# =================== OAuth State (DB-backed, no global dict) ===================

async def _store_oauth_state(app: App, provider: str, extra: dict = None) -> str:
    """Create and store OAuth CSRF state in DB."""
    state = secrets.token_urlsafe(32)
    data = {"provider": provider, "created_at": time.time(), "extra": extra or {}}
    await _db_set(app, f"AUTH_STATE::{state}", data)
    return state


async def _validate_and_consume_state(app: App, state: str, provider: str) -> tuple[bool, dict]:
    """Validate OAuth state: check provider, expiry, delete after use."""
    data = await _db_get(app, f"AUTH_STATE::{state}")
    if not data:
        return False, {}
    await _db_delete(app, f"AUTH_STATE::{state}")
    if data.get("provider") != provider:
        return False, {}
    if time.time() - data.get("created_at", 0) > STATE_EXPIRY:
        return False, {}
    return True, data.get("extra", {})


# =================== WebAuthn Challenge (DB-backed) ===================

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


# =================== Token Blacklist (DB-backed) ===================

async def _blacklist_token(app: App, token_str: str):
    """Add token JTI to blacklist."""
    try:
        payload = jwt.decode(token_str, options={"verify_signature": False})
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


# =================== JWT Token Management ===================

def _generate_access_token(user_id: str, username: str, level: int, provider: str = "") -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "level": level,
        "provider": provider,
        "type": "access",
        "iat": time.time(),
        "exp": time.time() + ACCESS_TOKEN_EXPIRY,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def _generate_refresh_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "type": "refresh",
        "iat": time.time(),
        "exp": time.time() + REFRESH_TOKEN_EXPIRY,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def _generate_tokens(user: UserData, provider: str = "") -> dict:
    """Generate access + refresh token pair."""
    return {
        "access_token": _generate_access_token(user.user_id, user.username, user.level, provider),
        "refresh_token": _generate_refresh_token(user.user_id),
        "expires_in": ACCESS_TOKEN_EXPIRY,
        "token_type": "Bearer",
    }


async def _validate_jwt(app: App, token_str: str, token_type: str = "access") -> tuple[bool, dict]:
    """Validate JWT: signature, type, expiry, blacklist."""
    if not token_str:
        return False, {"error": "No token provided"}
    try:
        payload = jwt.decode(token_str, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return False, {"error": "Token expired"}
    except jwt.InvalidTokenError as e:
        return False, {"error": f"Invalid token: {e}"}

    if payload.get("type") != token_type:
        return False, {"error": f"Expected {token_type} token"}

    jti = payload.get("jti", "")
    if jti and await _is_blacklisted(app, jti):
        return False, {"error": "Token has been revoked"}

    return True, payload


# =================== OAuth HTTP Helpers ===================

async def _exchange_oauth_code(config: dict, code: str) -> tuple[bool, dict]:
    """Exchange OAuth authorization code for tokens."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(config["token_url"], data={
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": config["redirect_uri"],
            }, headers={"Accept": "application/json"})
            if resp.status_code != 200:
                return False, {"error": f"Token exchange failed ({resp.status_code}): {resp.text}"}
            return True, resp.json()
    except Exception as e:
        return False, {"error": str(e)}


async def _get_discord_user(access_token: str) -> tuple[bool, dict]:
    """Fetch Discord user profile."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://discord.com/api/users/@me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code != 200:
                return False, {"error": f"Discord user fetch failed: {resp.text}"}
            d = resp.json()
            avatar = ""
            if d.get("avatar"):
                avatar = f"https://cdn.discordapp.com/avatars/{d['id']}/{d['avatar']}.png"
            return True, {
                "provider_id": d["id"],
                "username": d["username"],
                "email": d.get("email", ""),
                "avatar": avatar,
            }
    except Exception as e:
        return False, {"error": str(e)}


async def _get_google_user(access_token: str) -> tuple[bool, dict]:
    """Fetch Google user profile."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code != 200:
                return False, {"error": f"Google user fetch failed: {resp.text}"}
            d = resp.json()
            return True, {
                "provider_id": d["id"],
                "username": d.get("name", d.get("email", "").split("@")[0]),
                "email": d.get("email", ""),
                "avatar": d.get("picture", ""),
            }
    except Exception as e:
        return False, {"error": str(e)}


# =================== User Create/Update ===================

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


# =================== API: Auth Config ===================

@export(mod_name=Name, version=version, api=True)
async def get_auth_config(app: App = None) -> ApiResult:
    """Return auth provider config for frontend."""
    discord = get_discord_config()
    google = get_google_config()
    return Result.ok({
        "providers": {
            "discord": {"enabled": bool(discord["client_id"]), "client_id": discord["client_id"]},
            "google": {"enabled": bool(google["client_id"]), "client_id": google["client_id"]},
            "passkeys": {"enabled": True, "rp_id": get_passkey_config()["rp_id"]},
        },
        "sign_in_url": "/web/scripts/login.html",
        "after_sign_in_url": "/web/mainContent.html",
    })


# =================== API: OAuth URLs ===================

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


# =================== API: OAuth Callbacks ===================

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

    if state:
        valid, _extra = await _validate_and_consume_state(app, state, "discord")
        if not valid:
            log.warning(f"[{Name}] Invalid OAuth state for Discord")
            return Result.default_user_error("Invalid or expired state")

    config = get_discord_config()
    ok, tokens = await _exchange_oauth_code(config, code)
    if not ok:
        log.error(f"[{Name}] Discord token exchange failed: {tokens}")
        return Result.default_internal_error(tokens.get("error", "Token exchange failed"))

    ok, user_info = await _get_discord_user(tokens["access_token"])
    if not ok:
        log.error(f"[{Name}] Discord user fetch failed: {user_info}")
        return Result.default_internal_error(user_info.get("error", "Failed to get user"))

    user, is_new = await _create_or_update_user(app, "discord", user_info, tokens)
    jwt_tokens = _generate_tokens(user, "discord")
    log.info(f"[{Name}] Discord login: {user.user_id} (new={is_new})")

    return Result.ok({
        "authenticated": True,
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "level": user.level,
        "provider": "discord",
        "is_new_user": is_new,
        **jwt_tokens,
    })


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

    if state:
        valid, _extra = await _validate_and_consume_state(app, state, "google")
        if not valid:
            log.warning(f"[{Name}] Invalid OAuth state for Google")
            return Result.default_user_error("Invalid or expired state")

    config = get_google_config()
    ok, tokens = await _exchange_oauth_code(config, code)
    if not ok:
        log.error(f"[{Name}] Google token exchange failed: {tokens}")
        return Result.default_internal_error(tokens.get("error", "Token exchange failed"))

    ok, user_info = await _get_google_user(tokens["access_token"])
    if not ok:
        log.error(f"[{Name}] Google user fetch failed: {user_info}")
        return Result.default_internal_error(user_info.get("error", "Failed to get user"))

    user, is_new = await _create_or_update_user(app, "google", user_info, tokens)
    jwt_tokens = _generate_tokens(user, "google")
    log.info(f"[{Name}] Google login: {user.user_id} (new={is_new})")

    return Result.ok({
        "authenticated": True,
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "level": user.level,
        "provider": "google",
        "is_new_user": is_new,
        **jwt_tokens,
    })


# =================== API: Session Validation ===================

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


# =================== API: Token Refresh ===================

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
        return Result.default_user_error(payload.get("error", "Invalid refresh token"))

    user = await _load_user(app, payload["sub"])
    if not user:
        return Result.default_user_error("User not found")

    tokens = _generate_tokens(user)
    get_logger().info(f"[{Name}] Token refreshed for {user.user_id}")
    return Result.ok(tokens)


# =================== API: Logout (with blacklist) ===================

@export(mod_name=Name, version=version, api=True)
async def logout(app: App = None, token: str = None, data: dict = None) -> ApiResult:
    """Logout: blacklist current access token."""
    if app is None:
        app = get_app(f"{Name}.logout")
    if data:
        token = token or data.get("token")
    if token:
        await _blacklist_token(app, token)
    get_logger().info(f"[{Name}] Logout completed")
    return Result.ok({"logged_out": True})


# =================== API: User Data ===================

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
    return Result.ok(user.to_dict())


# =================== API: WebAuthn Passkeys ===================

@export(mod_name=Name, version=version, api=True)
async def passkey_register_start(
    app: App = None, user_id: str = None, username: str = None, data: dict = None
) -> ApiResult:
    """Start WebAuthn registration - generate challenge and options."""
    if app is None:
        app = get_app(f"{Name}.passkey_register_start")
    if data:
        user_id = user_id or data.get("user_id")
        username = username or data.get("username")
    if not user_id or not username:
        return Result.default_user_error("User ID and username required")

    try:
        from webauthn import generate_registration_options, options_to_json
        from webauthn.helpers.structs import (
            PublicKeyCredentialDescriptor,
            AuthenticatorSelectionCriteria,
            UserVerificationRequirement,
            ResidentKeyRequirement,
        )
    except ImportError:
        return Result.default_internal_error("py_webauthn not installed")

    config = get_passkey_config()

    # Get existing credentials to exclude
    user = await _load_user(app, user_id)
    exclude_credentials = []
    if user:
        for pk in user.passkeys:
            exclude_credentials.append(PublicKeyCredentialDescriptor(
                id=base64.urlsafe_b64decode(pk["credential_id"] + "=="),
            ))

    options = generate_registration_options(
        rp_id=config["rp_id"],
        rp_name=config["rp_name"],
        user_id=user_id.encode(),
        user_name=username,
        user_display_name=username,
        exclude_credentials=exclude_credentials,
        authenticator_selection=AuthenticatorSelectionCriteria(
            user_verification=UserVerificationRequirement.PREFERRED,
            resident_key=ResidentKeyRequirement.PREFERRED,
        ),
    )

    # Store challenge in DB
    challenge_b64 = base64.urlsafe_b64encode(options.challenge).decode().rstrip("=")
    await _store_challenge(app, challenge_b64, {
        "user_id": user_id,
        "username": username,
        "type": "registration",
        "challenge_bytes": base64.b64encode(options.challenge).decode(),
    })

    return Result.ok(json.loads(options_to_json(options)))


@export(mod_name=Name, version=version, api=True)
async def passkey_register_finish(
    app: App = None, challenge: str = None, credential: dict = None, data: dict = None
) -> ApiResult:
    """Complete WebAuthn registration - verify attestation and store credential."""
    if app is None:
        app = get_app(f"{Name}.passkey_register_finish")
    if data:
        challenge = challenge or data.get("challenge")
        credential = credential or data.get("credential")
    if not challenge or not credential:
        return Result.default_user_error("Challenge and credential required")

    challenge_data = await _validate_and_consume_challenge(app, challenge, "registration")
    if not challenge_data:
        return Result.default_user_error("Invalid or expired challenge")

    user_id = challenge_data["user_id"]
    config = get_passkey_config()

    try:
        from webauthn import verify_registration_response
        from webauthn.helpers.structs import RegistrationCredential

        credential_obj = RegistrationCredential.model_validate(credential)
        verification = verify_registration_response(
            credential=credential_obj,
            expected_challenge=base64.b64decode(challenge_data["challenge_bytes"]),
            expected_rp_id=config["rp_id"],
            expected_origin=config["origin"],
        )
    except ImportError:
        return Result.default_internal_error("py_webauthn not installed")
    except Exception as e:
        get_logger().error(f"[{Name}] WebAuthn registration verification failed: {e}")
        return Result.default_user_error(f"Registration verification failed: {e}")

    user = await _load_user(app, user_id)
    if not user:
        return Result.default_user_error("User not found")

    cred_id_b64 = base64.urlsafe_b64encode(verification.credential_id).decode().rstrip("=")
    pub_key_b64 = base64.b64encode(verification.credential_public_key).decode()

    passkey = Passkey(
        credential_id=cred_id_b64,
        public_key=pub_key_b64,
        sign_count=verification.sign_count,
        name=credential.get("name", "Passkey"),
    )
    user.passkeys.append(passkey.to_dict())
    await _save_user(app, user)

    return Result.ok({"success": True, "credential_id": cred_id_b64})


@export(mod_name=Name, version=version, api=True)
async def passkey_login_start(app: App = None, data: dict = None) -> ApiResult:
    """Start WebAuthn authentication - generate challenge."""
    if app is None:
        app = get_app(f"{Name}.passkey_login_start")

    try:
        from webauthn import generate_authentication_options, options_to_json
    except ImportError:
        return Result.default_internal_error("py_webauthn not installed")

    config = get_passkey_config()
    options = generate_authentication_options(rp_id=config["rp_id"])

    challenge_b64 = base64.urlsafe_b64encode(options.challenge).decode().rstrip("=")
    await _store_challenge(app, challenge_b64, {
        "type": "authentication",
        "challenge_bytes": base64.b64encode(options.challenge).decode(),
    })

    return Result.ok(json.loads(options_to_json(options)))


@export(mod_name=Name, version=version, api=True)
async def passkey_login_finish(
    app: App = None, challenge: str = None, credential: dict = None, data: dict = None
) -> ApiResult:
    """Complete WebAuthn authentication - verify assertion."""
    if app is None:
        app = get_app(f"{Name}.passkey_login_finish")
    log = get_logger()
    if data:
        challenge = challenge or data.get("challenge")
        credential = credential or data.get("credential")
    if not challenge or not credential:
        return Result.default_user_error("Challenge and credential required")

    challenge_data = await _validate_and_consume_challenge(app, challenge, "authentication")
    if not challenge_data:
        return Result.default_user_error("Invalid or expired challenge")

    # Find user by credential ID
    cred_id = credential.get("id", "")
    user = await _find_user_by_provider(app, "passkey", cred_id)
    if not user:
        return Result.default_user_error("Passkey not registered")

    # Find the matching stored passkey
    stored_pk = None
    for pk in user.passkeys:
        if pk.get("credential_id") == cred_id:
            stored_pk = pk
            break
    if not stored_pk:
        return Result.default_user_error("Passkey credential not found")

    config = get_passkey_config()

    try:
        from webauthn import verify_authentication_response
        from webauthn.helpers.structs import AuthenticationCredential

        credential_obj = AuthenticationCredential.model_validate(credential)
        verification = verify_authentication_response(
            credential=credential_obj,
            expected_challenge=base64.b64decode(challenge_data["challenge_bytes"]),
            expected_rp_id=config["rp_id"],
            expected_origin=config["origin"],
            credential_public_key=base64.b64decode(stored_pk["public_key"]),
            credential_current_sign_count=stored_pk.get("sign_count", 0),
        )
    except ImportError:
        return Result.default_internal_error("py_webauthn not installed")
    except Exception as e:
        log.error(f"[{Name}] WebAuthn auth verification failed: {e}")
        return Result.default_user_error(f"Authentication verification failed: {e}")

    # Update sign count
    stored_pk["sign_count"] = verification.new_sign_count
    user.last_login = time.time()
    await _save_user(app, user)

    jwt_tokens = _generate_tokens(user, "passkey")
    log.info(f"[{Name}] Passkey login: {user.user_id}")

    return Result.ok({
        "authenticated": True,
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "level": user.level,
        "provider": "passkey",
        **jwt_tokens,
    })


# =================== API: Magic Link (CLI Auth) ===================

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
        from .email_services import send_magic_link_email
        send_magic_link_email(app, email, link)
    except Exception as e:
        get_logger().error(f"[{Name}] Failed to send magic link email: {e}")
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
        return Result.default_user_error("Invalid or expired link")

    await _db_delete(app, f"AUTH_MAGIC_LINK::{token}")

    if time.time() - ml_data.get("created_at", 0) > MAGIC_LINK_EXPIRY:
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
    # Placeholder: In a full implementation, the magic link verify endpoint
    # would mark the link as verified, and this endpoint would check that status.
    return Result.default_user_error("Use verify_magic_link directly")


# =================== API: Device Invite (CLI Auth) ===================

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

    return Result.ok({
        "authenticated": True,
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "level": user.level,
        "provider": "device_invite",
        **jwt_tokens,
    })


# =================== Admin Functions ===================

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
