"""
API: Auth configuration endpoint.
"""

from toolboxv2 import Result, get_app
from toolboxv2.utils.system.types import ApiResult

from .config import get_discord_config, get_google_config, get_passkey_config

Name = "CloudM.Auth"
version = "2.0.0"
export = get_app(f"{Name}.Export").tb


@export(mod_name=Name, version=version, api=True)
async def get_auth_config(app=None) -> ApiResult:
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
