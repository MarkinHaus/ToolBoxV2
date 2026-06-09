"""
Local anonymous root admin — created automatically on first local auth touch.
No token is shown to or required from the user locally; tokens are remote-only.
Level semantics: root is always -1 (admin); anonymity is a settings flag,
removed when the user picks their own username (see update_user_data).
"""
import secrets
import time

from toolboxv2 import App, Result, get_app
from toolboxv2.utils.system.types import ToolBoxInterfaces

from .config import LOCAL_ADMIN_EMAIL, LOCAL_ADMIN_USERNAME
from .models import UserData
from .user_store import _find_user_by_email, _save_user

Name = "CloudM.Auth"
version = "2.0.0"
export = get_app(f"{Name}.Export").tb


async def ensure_local_admin(app: App) -> UserData:
    """Idempotent: find or create the local anonymous root user (level -1)."""
    user = await _find_user_by_email(app, LOCAL_ADMIN_EMAIL)
    if user:
        return user
    user = UserData(
        user_id=f"usr_{secrets.token_hex(12)}",
        username=LOCAL_ADMIN_USERNAME,
        email=LOCAL_ADMIN_EMAIL,
        level=-1,
        created_at=time.time(),
        settings={"anonymous": True, "local_admin": True},
    )
    await _save_user(app, user)
    return user


@export(mod_name=Name, version=version, api=False, interface=ToolBoxInterfaces.native)
async def get_local_admin(app: App = None) -> Result:
    """Native-only (never HTTP): returns the local admin user, creating it if missing."""
    if app is None:
        app = get_app(f"{Name}.get_local_admin")
    user = await ensure_local_admin(app)
    return Result.ok(user.to_dict())
