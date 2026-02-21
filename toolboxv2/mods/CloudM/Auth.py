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
- Split into sub-modules under auth/ for maintainability

DB NAMESPACES:
- AUTH_USER::{user_id}                          -> User profile (permanent)
- AUTH_USER_PROVIDER::{provider}::{provider_id} -> Provider->User index (permanent)
- AUTH_USER_EMAIL::{email}                      -> Email->User index (permanent)
- AUTH_STATE::{state}                           -> OAuth CSRF state (10min, timestamp)
- AUTH_CHALLENGE::{challenge}                   -> WebAuthn challenge (5min, timestamp)
- AUTH_BLACKLIST::{jti}                         -> Token blacklist (until token expiry)
- AUTH_MAGIC_LINK::{token}                      -> Magic link (10min, timestamp)
- AUTH_DEVICE_INVITE::{code}                    -> Device invite (5min, timestamp)

SUB-MODULES:
- auth/models.py       - Data classes (OAuthProvider, Passkey, UserData)
- auth/config.py       - Constants & environment helpers
- auth/db_helpers.py   - Low-level TBEF.DB operations
- auth/user_store.py   - User CRUD (save, load, find)
- auth/state.py        - OAuth state, WebAuthn challenges, token blacklist
- auth/jwt_tokens.py   - JWT generation & validation
- auth/oauth.py        - OAuth HTTP helpers (token exchange, user fetch)
- auth/api_config.py   - API: get_auth_config
- auth/api_oauth.py    - API: OAuth URLs + callbacks
- auth/api_session.py  - API: validate_session, refresh, logout, user data, admin
- auth/api_passkey.py  - API: WebAuthn passkey registration & login
- auth/api_magic_link.py - API: Magic links & device invites
"""

from toolboxv2 import get_app

Name = "CloudM.Auth"
version = "2.0.0"
export = get_app(f"{Name}.Export").tb

# =================== Re-exports for backwards compatibility ===================
# All internal consumers import from here (e.g. `from toolboxv2.mods.CloudM.Auth import _load_user`)

# Models
from .auth.models import OAuthProvider, Passkey, UserData  # noqa: F401, E402

# Config
from .auth.config import (  # noqa: F401, E402
    JWT_ALGORITHM, ACCESS_TOKEN_EXPIRY, REFRESH_TOKEN_EXPIRY,
    STATE_EXPIRY, CHALLENGE_EXPIRY, MAGIC_LINK_EXPIRY, DEVICE_INVITE_EXPIRY,
    is_production, get_jwt_secret, get_base_url,
    get_discord_config, get_google_config, get_passkey_config,
)

# DB helpers
from .auth.db_helpers import (  # noqa: F401, E402
    _parse_db_result, _db_set, _db_get, _db_get_raw, _db_delete, _db_exists,
)

# User storage
from .auth.user_store import (  # noqa: F401, E402
    _save_user, _load_user, _find_user_by_provider, _find_user_by_email,
    _create_or_update_user,
)

# State management
from .auth.state import (  # noqa: F401, E402
    _store_oauth_state, _validate_and_consume_state,
    _store_challenge, _validate_and_consume_challenge,
    _blacklist_token, _is_blacklisted,
)

# JWT tokens
from .auth.jwt_tokens import (  # noqa: F401, E402
    _generate_access_token, _generate_refresh_token, _generate_tokens,
    _validate_jwt,
)

# OAuth HTTP helpers
from .auth.oauth import (  # noqa: F401, E402
    _exchange_oauth_code, _get_discord_user, _get_google_user,
)

# =================== API endpoints (decorated with @export) ===================
# These must be imported so the @export decorators register the functions.

from .auth.api_config import get_auth_config  # noqa: F401, E402
from .auth.api_oauth import (  # noqa: F401, E402
    get_discord_auth_url, get_google_auth_url,
    login_discord, login_google,
)
from .auth.api_session import (  # noqa: F401, E402
    validate_session, verify_session_token,
    refresh_token, logout,
    get_user_data, update_user_data,
    list_users,
)
from .auth.api_passkey import (  # noqa: F401, E402
    passkey_register_start, passkey_register_finish,
    passkey_login_start, passkey_login_finish,
)
from .auth.api_magic_link import (  # noqa: F401, E402
    request_magic_link, verify_magic_link, check_magic_link_status,
    create_device_invite, verify_device_invite,
)
