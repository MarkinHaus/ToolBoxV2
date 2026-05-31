"""
Environment and configuration for the auth system.
"""

import os

# =================== Constants ===================

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRY = 15 * 60          # 15 minutes
REFRESH_TOKEN_EXPIRY = 7 * 24 * 3600   # 7 days
STATE_EXPIRY = 10 * 60                 # 10 minutes for OAuth state
CHALLENGE_EXPIRY = 5 * 60              # 5 minutes for WebAuthn challenges
MAGIC_LINK_EXPIRY = 10 * 60            # 10 minutes for magic links
DEVICE_INVITE_EXPIRY = 5 * 60          # 5 minutes for device invites


# =================== Environment Helpers ===================

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


def get_discord_config(redirect_after=None) -> dict:
    return {
        "client_id": os.getenv("DISCORD_CLIENT_ID", ""),
        "client_secret": os.getenv("DISCORD_CLIENT_SECRET", ""),
        "redirect_uri": redirect_after or os.getenv("DISCORD_REDIRECT_URI", f"{get_base_url()}/auth/discord/callback"),
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


def get_redirect_whitelist() -> list[str]:
    """Allowed origins for OAuth redirect_after (token bridge target).

    Always includes the deployment base URL. Loopback (127.0.0.1/localhost on any
    port) is allowed separately in is_allowed_redirect to support the local CLI
    loopback receiver. Extra origins via TB_OAUTH_REDIRECT_WHITELIST (comma-sep).
    """
    raw = os.getenv("TB_OAUTH_REDIRECT_WHITELIST", "")
    extra = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return [get_base_url().rstrip("/")] + extra


def is_allowed_redirect(redirect_after: str) -> bool:
    """Validate an OAuth redirect_after target against the trusted list.

    Empty/None is allowed (means: no cross-origin redirect, stay on this origin).
    Loopback hosts are always allowed (CLI receiver). Otherwise the scheme+host
    [+port] must match an entry in get_redirect_whitelist().
    """
    if redirect_after == "/":
        return True
    if not redirect_after:
        return True
    from urllib.parse import urlparse
    try:
        u = urlparse(redirect_after)
    except Exception:
        return False
    if not u.scheme or not u.hostname:
        return False
    if u.hostname in ("127.0.0.1", "localhost"):
        return True
    origin = f"{u.scheme}://{u.netloc}".rstrip("/")
    return origin in get_redirect_whitelist()
