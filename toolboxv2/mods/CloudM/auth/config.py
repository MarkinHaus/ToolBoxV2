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
