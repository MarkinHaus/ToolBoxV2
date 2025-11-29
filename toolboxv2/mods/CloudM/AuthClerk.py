"""
ToolBox V2 - Clerk Authentication Integration
Replaces AuthManager.py with Clerk-based authentication

WICHTIG:
- NO Passkeys (Premium Feature in Clerk Free Tier)
- Email + Code verification (keine Magic Links mit URLs)
- Nur eine Session pro User erlaubt
- Lokale Speicherung in BlobFile für Offline/Sync
"""

import os
import json
import time
import asyncio
from typing import Optional, Any
from dataclasses import dataclass, asdict, field

from clerk_backend_api import Clerk, GetUserListRequest, CreateSessionRequestBody
from clerk_backend_api.models import User as ClerkUser

from toolboxv2 import App, Result, get_app, get_logger, TBEF
from toolboxv2.utils.system.types import ApiResult, ToolBoxInterfaces
from toolboxv2.utils.extras.blobs import BlobFile
from toolboxv2.utils.security.cryp import Code

Name = "CloudM.AuthClerk"
version = "1.0.0"
export = get_app(f"{Name}.Export").tb


# =================== Datentypen ===================

@dataclass
class LocalUserData:
    """Lokale User-Daten die in BlobFile gespeichert werden (dezentral)"""
    clerk_user_id: str
    username: str
    email: str
    level: int = 1
    settings: dict = field(default_factory=dict)
    mod_data: dict = field(default_factory=dict)  # Mod-spezifische Daten
    last_sync: float = 0.0
    session_token: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LocalUserData":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =================== Clerk Client ===================

_clerk_client: Optional[Clerk] = None

def get_clerk_client() -> Clerk:
    """Get or create Clerk client instance"""
    global _clerk_client
    if _clerk_client is None:
        secret_key = os.getenv('CLERK_SECRET_KEY')
        if not secret_key:
            raise ValueError("CLERK_SECRET_KEY not set in environment. Please add it to your .env file.")
        _clerk_client = Clerk(bearer_auth=secret_key)
    return _clerk_client


def get_publishable_key() -> str:
    """Get Clerk publishable key for frontend"""
    key = os.getenv('CLERK_PUBLISHABLE_KEY')
    if not key:
        raise ValueError("CLERK_PUBLISHABLE_KEY not set in environment")
    return key


# =================== Local Storage (BlobFile) ===================

def _get_user_blob_path(identifier: str) -> str:
    """Get BlobFile path for user data"""
    safe_id = Code.one_way_hash(identifier, "user-blob")[:16]
    return f"clerk/users/{safe_id}/data.json"


def _get_session_blob_path(identifier: str) -> str:
    """Get BlobFile path for session data"""
    safe_id = Code.one_way_hash(identifier, "session-blob")[:16]
    return f"clerk/sessions/{safe_id}/session.json"


def save_local_user_data(user_data: LocalUserData) -> bool:
    """Save user data to local BlobFile (dezentral)"""
    try:
        blob_path = _get_user_blob_path(user_data.clerk_user_id)
        with BlobFile(blob_path, key=Code.DK()(), mode="w") as blob:
            blob.clear()
            blob.write(json.dumps(user_data.to_dict()).encode())
        return True
    except Exception as e:
        get_logger().error(f"[{Name}] Failed to save local user data: {e}")
        return False


def load_local_user_data(clerk_user_id: str) -> Optional[LocalUserData]:
    """Load user data from local BlobFile"""
    try:
        blob_path = _get_user_blob_path(clerk_user_id)
        with BlobFile(blob_path, key=Code.DK()(), mode="r") as blob:
            data = blob.read()
            if data and data != b'Error decoding':
                return LocalUserData.from_dict(json.loads(data.decode()))
    except Exception as e:
        get_logger().debug(f"[{Name}] No local user data found: {e}")
    return None


def save_session_token(identifier: str, token: str, username: str) -> bool:
    """Save session token to BlobFile"""
    try:
        blob_path = _get_session_blob_path(identifier)
        session_data = {
            "token": token,
            "username": username,
            "created_at": time.time()
        }
        with BlobFile(blob_path, key=Code.DK()(), mode="w") as blob:
            blob.clear()
            blob.write(json.dumps(session_data).encode())
        return True
    except Exception as e:
        get_logger().error(f"[{Name}] Failed to save session token: {e}")
        return False


def load_session_token(identifier: str) -> Optional[dict]:
    """Load session token from BlobFile"""
    try:
        blob_path = _get_session_blob_path(identifier)
        with BlobFile(blob_path, key=Code.DK()(), mode="r") as blob:
            data = blob.read()
            if data and data != b'Error decoding':
                return json.loads(data.decode())
    except Exception as e:
        get_logger().debug(f"[{Name}] No session token found: {e}")
    return None


def clear_session_token(identifier: str) -> bool:
    """Clear session token from BlobFile"""
    try:
        blob_path = _get_session_blob_path(identifier)
        with BlobFile(blob_path, key=Code.DK()(), mode="w") as blob:
            blob.clear()
        return True
    except Exception as e:
        get_logger().error(f"[{Name}] Failed to clear session token: {e}")
        return False


# =================== Database Storage (für Sync) ===================

def _db_save_user_sync_data(app: App, clerk_user_id: str, data: dict) -> Result:
    """Save user sync data to database (für Vendor Lock-in Prevention)"""
    return app.run_any(
        TBEF.DB.SET,
        query=f"CLERK_USER::{clerk_user_id}",
        data=data,
        get_results=True
    )


def _db_load_user_sync_data(app: App, clerk_user_id: str) -> Optional[dict]:
    """Load user sync data from database"""
    result = app.run_any(
        TBEF.DB.GET,
        query=f"CLERK_USER::{clerk_user_id}",
        get_results=True
    )
    if result.is_error():
        return None
    data = result.get()
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    if isinstance(data, bytes):
        data = data.decode()
    if isinstance(data, str):
        try:
            return eval(data)
        except:
            return json.loads(data)
    return data if isinstance(data, dict) else None


# =================== Clerk API Endpoints ===================

@export(mod_name=Name, version=version, api=True)
async def get_clerk_config(app: App = None) -> ApiResult:
    """
    Get Clerk configuration for frontend
    Returns publishable key and settings
    """
    try:
        return Result.ok({
            "publishable_key": get_publishable_key(),
            "sign_in_url": "/web/assets/login.html",
            "sign_up_url": "/web/assets/signup.html",
            "after_sign_in_url": "/web/mainContent.html",
            "after_sign_up_url": "/web/mainContent.html",
        })
    except ValueError as e:
        return Result.default_internal_error(str(e))


# =================== Clerk Session Token Verification ===================
# Füge dies am Anfang von AuthClerk.py hinzu (nach den bestehenden imports)

import httpx
from typing import Optional, Tuple
from dataclasses import dataclass

# Versuche die offiziellen SDK-Helper zu importieren
try:
    from clerk_backend_api.security import authenticate_request
    from clerk_backend_api.security.types import AuthenticateRequestOptions

    CLERK_SDK_AUTH_AVAILABLE = True
except ImportError:
    CLERK_SDK_AUTH_AVAILABLE = False


@dataclass
class TokenVerificationResult:
    """Result of token verification"""
    is_valid: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    claims: Optional[dict] = None
    error: Optional[str] = None


def verify_session_token(token: str, authorized_parties: list = None) -> TokenVerificationResult:
    """
    Verify Clerk session token using the official SDK.

    Args:
        token: The session token (JWT) from the frontend
        authorized_parties: List of allowed origins (e.g., ['https://example.com'])

    Returns:
        TokenVerificationResult with verification status and user info
    """
    logger = get_logger()

    if not token:
        return TokenVerificationResult(is_valid=False, error="No token provided")

    try:
        clerk = get_clerk_client()

        # Erstelle einen httpx.Request mit dem Token im Authorization Header
        # Das ist das Format, das authenticate_request erwartet
        fake_request = httpx.Request(
            method="GET",
            url="http://localhost/verify",
            headers={"Authorization": f"Bearer {token}"}
        )

        # Konfiguriere authorized_parties (CSRF-Schutz)
        if authorized_parties is None:
            # Fallback: Erlaube localhost für Entwicklung
            authorized_parties = [
                "http://localhost:8080",
                "http://localhost:3000",
                "http://127.0.0.1:8080",
            ]
            # Füge Produktions-Domain hinzu falls konfiguriert
            prod_domain = os.getenv('APP_BASE_URL')
            if prod_domain:
                authorized_parties.append(prod_domain)

        if CLERK_SDK_AUTH_AVAILABLE:
            # Nutze die offizielle SDK-Methode
            request_state = clerk.authenticate_request(
                fake_request,
                AuthenticateRequestOptions(
                    authorized_parties=authorized_parties
                )
            )

            if request_state.is_signed_in:
                # Token ist gültig - extrahiere Claims
                payload = request_state.payload or {}
                return TokenVerificationResult(
                    is_valid=True,
                    user_id=payload.get("sub"),  # subject = user_id
                    session_id=payload.get("sid"),  # session_id
                    claims=payload
                )
            else:
                return TokenVerificationResult(
                    is_valid=False,
                    error=request_state.reason or "Token verification failed"
                )
        else:
            # Fallback: Nutze sessions.get_session mit Session-ID aus Token
            # Dekodiere Token ohne Verifikation um Session-ID zu bekommen
            import jwt
            unverified = jwt.decode(token, options={"verify_signature": False})
            session_id = unverified.get("sid")
            user_id = unverified.get("sub")

            if not session_id:
                return TokenVerificationResult(is_valid=False, error="No session ID in token")

            # Verifiziere Session über Clerk API
            try:
                session = clerk.sessions.get(session_id=session_id)
                if session and session.status == "active":
                    return TokenVerificationResult(
                        is_valid=True,
                        user_id=user_id or session.user_id,
                        session_id=session_id,
                        claims=unverified
                    )
                else:
                    return TokenVerificationResult(
                        is_valid=False,
                        error=f"Session not active: {session.status if session else 'not found'}"
                    )
            except Exception as e:
                logger.warning(f"[{Name}] Session lookup failed: {e}")
                return TokenVerificationResult(is_valid=False, error=str(e))

    except Exception as e:
        logger.error(f"[{Name}] Token verification error: {e}")
        return TokenVerificationResult(is_valid=False, error=str(e))


# =================== Updated verify_session ===================

@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def verify_session(app: App = None, request=None, session_token: str = None,
                         clerk_user_id: str = None) -> ApiResult:
    """
    Verify Clerk session token.
    Called by middleware/frontend to validate authentication.
    """
    if app is None:
        app = get_app(f"{Name}.verify_session")

    logger = get_logger()

    try:
        # Get token from multiple sources
        token = session_token
        if not token and request:
            # Try Authorization header
            auth_header = ""
            if hasattr(request, 'request') and hasattr(request.request, 'headers'):
                auth_header = request.request.headers.get("Authorization", "")
            elif hasattr(request, 'headers'):
                auth_header = request.headers.get("Authorization", "")

            if auth_header.startswith("Bearer "):
                token = auth_header[7:]

            # Try request body
            if not token and hasattr(request, 'data'):
                data = request.data
                if isinstance(data, dict):
                    token = data.get("session_token") or data.get("Jwt_claim")

        if not token:
            logger.warning(f"[{Name}] No session token provided")
            return Result.default_user_error("No session token provided", data={"authenticated": False})

        logger.info(f"[{Name}] Verifying session token (length: {len(token)})")

        # Verify token
        result = verify_session_token(token)

        if not result.is_valid:
            logger.warning(f"[{Name}] Token verification failed: {result.error}")
            return Result.default_user_error(
                "Invalid or expired session",
                data={"authenticated": False}
            )

        user_id = result.user_id or clerk_user_id

        if not user_id:
            logger.warning(f"[{Name}] No user ID in verified token")
            return Result.default_user_error("Invalid token", data={"authenticated": False})

        logger.info(f"[{Name}] Token verified for user: {user_id}")

        # Get user info from Clerk
        try:
            clerk = get_clerk_client()
            user = clerk.users.get(user_id=user_id)
        except Exception as e:
            logger.error(f"[{Name}] Failed to get user: {e}")
            return Result.default_user_error("User not found", data={"authenticated": False})

        # Extract user info
        email = ""
        if user.email_addresses and len(user.email_addresses) > 0:
            email = user.email_addresses[0].email_address

        username = user.username or (email.split("@")[0] if email else f"user_{user_id[:8]}")

        # Load or create local user data
        local_data = load_local_user_data(user_id)

        if not local_data:
            local_data = LocalUserData(
                clerk_user_id=user_id,
                username=username,
                email=email,
                level=1,
                settings={},
                mod_data={},
                session_token=token,
                last_sync=time.time()
            )
            save_local_user_data(local_data)
            _db_save_user_sync_data(app, user_id, local_data.to_dict())
            logger.info(f"[{Name}] Created local user data for {user_id}")
        else:
            local_data.session_token = token
            local_data.last_sync = time.time()
            if user.username:
                local_data.username = user.username
            if email:
                local_data.email = email
            save_local_user_data(local_data)

        return Result.ok({
            "authenticated": True,
            "user_id": user_id,
            "username": local_data.username,
            "email": local_data.email,
            "level": local_data.level,
            "settings": local_data.settings
        })

    except ValueError as ve:
        logger.error(f"[{Name}] Configuration error: {ve}")
        return Result.default_internal_error("Authentication service not configured")

    except Exception as e:
        logger.error(f"[{Name}] Error in verify_session: {e}")
        return Result.default_internal_error("Authentication error")

@export(mod_name=Name, version=version, api=True)
async def get_user_data(app: App = None, clerk_user_id: str = None, data=None) -> ApiResult:
    """
    Get user data (local + synced)
    Combines Clerk data with local BlobFile data
    """
    if app is None:
        app = get_app(f"{Name}.get_user_data")

    clerk_user_id = clerk_user_id or ( data.get("clerk_user_id") if data else None )
    if not clerk_user_id:
        return Result.default_user_error("User ID required")

    try:
        # Load local data
        local_data = load_local_user_data(clerk_user_id)

        # Load synced data from DB
        db_data = _db_load_user_sync_data(app, clerk_user_id)

        if local_data:
            # Merge with DB data if newer
            if db_data and db_data.get("last_sync", 0) > local_data.last_sync:
                local_data.settings = db_data.get("settings", local_data.settings)
                local_data.level = db_data.get("level", local_data.level)
                local_data.mod_data = db_data.get("mod_data", local_data.mod_data)
                local_data.last_sync = db_data.get("last_sync", local_data.last_sync)
                save_local_user_data(local_data)

            return Result.ok(local_data.to_dict())
        elif db_data:
            # Create local from DB
            local_data = LocalUserData.from_dict(db_data)
            save_local_user_data(local_data)
            return Result.ok(local_data.to_dict())
        else:
            return Result.default_user_error("User data not found")

    except Exception as e:
        get_logger().error(f"[{Name}] Error getting user data: {e}")
        return Result.default_internal_error(str(e))


@export(mod_name=Name, version=version, api=True)
async def update_user_data(
    app: App = None,
    clerk_user_id: str = None,
    settings: dict = None,
    level: int = None,
    mod_data: dict = None
) -> ApiResult:
    """
    Update user data (both local and synced)
    """
    if app is None:
        app = get_app(f"{Name}.update_user_data")

    if not clerk_user_id:
        return Result.default_user_error("User ID required")

    try:
        # Load current data
        local_data = load_local_user_data(clerk_user_id)
        if not local_data:
            return Result.default_user_error("User not found")

        # Update fields
        if settings is not None:
            local_data.settings.update(settings)
        if level is not None:
            local_data.level = level
        if mod_data is not None:
            local_data.mod_data.update(mod_data)

        local_data.last_sync = time.time()

        # Save locally
        save_local_user_data(local_data)

        # Sync to database (für Vendor Lock-in Prevention)
        sync_data = {
            "clerk_user_id": local_data.clerk_user_id,
            "username": local_data.username,
            "email": local_data.email,
            "level": local_data.level,
            "settings": local_data.settings,
            "mod_data": local_data.mod_data,
            "last_sync": local_data.last_sync
        }
        _db_save_user_sync_data(app, clerk_user_id, sync_data)

        return Result.ok(local_data.to_dict())

    except Exception as e:
        get_logger().error(f"[{Name}] Error updating user data: {e}")
        return Result.default_internal_error(str(e))


# =================== CLI Authentication (Email + Code) ===================

# Temporärer Speicher für Email-Verifizierungscodes (in Production: Redis/DB)
_verification_codes: dict[str, dict] = {}


@export(mod_name=Name, version=version, api=True)
async def cli_request_code(app: App = None, email: str = None) -> ApiResult:
    """
    CLI: Request verification code via email
    Clerk sends the code, we track it for CLI polling
    """
    if app is None:
        app = get_app(f"{Name}.cli_request_code")

    if not email:
        return Result.default_user_error("Email required")

    try:
        clerk = get_clerk_client()

        # Check if user exists
        users = clerk.users.list(request=GetUserListRequest(email_address=[email]))
        user_list = list(users)

        if not user_list:
            return Result.default_user_error(f"No user found with email: {email}")

        user = user_list[0]

        # Generate CLI session ID for tracking
        cli_session_id = Code.generate_symmetric_key()[:32]

        # Store pending verification
        _verification_codes[cli_session_id] = {
            "email": email,
            "user_id": user.id,
            "username": user.username or email.split("@")[0],
            "created_at": time.time(),
            "verified": False,
            "session_token": None
        }

        # Clerk will send email with code via Sign-In flow
        # For CLI, we create a sign-in attempt
        sign_in = clerk.sign_ins.create(
            identifier=email,
            strategy="email_code"
        )

        # Store sign-in ID for verification
        _verification_codes[cli_session_id]["sign_in_id"] = sign_in.id

        return Result.ok({
            "cli_session_id": cli_session_id,
            "message": f"Verification code sent to {email}",
            "user_id": user.id
        })

    except Exception as e:
        get_logger().error(f"[{Name}] Error requesting CLI code: {e}")
        return Result.default_internal_error(str(e))


@export(mod_name=Name, version=version, api=True)
async def cli_verify_code(
    app: App = None,
    cli_session_id: str = None,
    code: str = None
) -> ApiResult:
    """
    CLI: Verify the code entered by user
    """
    if app is None:
        app = get_app(f"{Name}.cli_verify_code")

    if not cli_session_id or not code:
        return Result.default_user_error("Session ID and code required")

    if cli_session_id not in _verification_codes:
        return Result.default_user_error("Invalid or expired session")

    session_data = _verification_codes[cli_session_id]

    # Check expiry (10 minutes)
    if time.time() - session_data["created_at"] > 600:
        del _verification_codes[cli_session_id]
        return Result.default_user_error("Verification code expired")

    try:
        clerk = get_clerk_client()

        # Verify the code with Clerk
        sign_in = clerk.sign_ins.attempt_first_factor(
            sign_in_id=session_data["sign_in_id"],
            strategy="email_code",
            code=code
        )

        if sign_in.status == "complete":
            # Create session
            session = clerk.sessions.create(request=CreateSessionRequestBody(user_id=session_data["user_id"]))

            # Get session token
            session_token = session.id  # In real implementation, get JWT

            # Update verification data
            session_data["verified"] = True
            session_data["session_token"] = session_token

            # Save to BlobFile
            save_session_token(
                session_data["user_id"],
                session_token,
                session_data["username"]
            )

            # Create/update local user data
            local_data = load_local_user_data(session_data["user_id"])
            if not local_data:
                local_data = LocalUserData(
                    clerk_user_id=session_data["user_id"],
                    username=session_data["username"],
                    email=session_data["email"],
                    session_token=session_token
                )
            else:
                local_data.session_token = session_token
            save_local_user_data(local_data)

            # Sync to DB
            if app:
                _db_save_user_sync_data(app, session_data["user_id"], local_data.to_dict())

            return Result.ok({
                "authenticated": True,
                "user_id": session_data["user_id"],
                "username": session_data["username"],
                "session_token": session_token
            })
        else:
            return Result.default_user_error("Invalid verification code")

    except Exception as e:
        get_logger().error(f"[{Name}] Error verifying CLI code: {e}")
        return Result.default_internal_error(str(e))


@export(mod_name=Name, version=version, api=True)
async def cli_check_auth(app: App = None, cli_session_id: str = None) -> ApiResult:
    """
    CLI: Check if authentication is complete (polling endpoint)
    """
    if not cli_session_id:
        return Result.ok({"authenticated": False})

    if cli_session_id not in _verification_codes:
        return Result.ok({"authenticated": False, "expired": True})

    session_data = _verification_codes[cli_session_id]

    if session_data.get("verified"):
        # Clean up
        result = {
            "authenticated": True,
            "user_id": session_data["user_id"],
            "username": session_data["username"],
            "session_token": session_data["session_token"]
        }
        del _verification_codes[cli_session_id]
        return Result.ok(result)

    return Result.ok({"authenticated": False})


# =================== Web Authentication Callbacks ===================

@export(mod_name=Name, version=version, api=True)
async def on_sign_in(app: App = None, user_data: dict = None) -> ApiResult:
    """
    Webhook/Callback when user signs in via Clerk UI
    Creates local user data and syncs to DB
    """
    if app is None:
        app = get_app(f"{Name}.on_sign_in")

    if not user_data:
        return Result.default_user_error("User data required")

    try:
        clerk_user_id = user_data.get("id")
        email = user_data.get("email_addresses", [{}])[0].get("email_address", "")
        username = user_data.get("username") or email.split("@")[0]

        # Load or create local data
        local_data = load_local_user_data(clerk_user_id)

        if not local_data:
            # New user - create local data
            local_data = LocalUserData(
                clerk_user_id=clerk_user_id,
                username=username,
                email=email,
                level=1,
                settings={},
                mod_data={}
            )

        # Update session token if provided
        session_token = user_data.get("session_token", "")
        if session_token:
            local_data.session_token = session_token

        local_data.last_sync = time.time()

        # Save locally
        save_local_user_data(local_data)

        # Sync to DB
        _db_save_user_sync_data(app, clerk_user_id, local_data.to_dict())

        return Result.ok({
            "success": True,
            "user_id": clerk_user_id,
            "username": username
        })

    except Exception as e:
        get_logger().error(f"[{Name}] Error in on_sign_in: {e}")
        return Result.default_internal_error(str(e))


@export(mod_name=Name, version=version, api=True)
async def on_sign_out(app: App = None, clerk_user_id: str = None) -> ApiResult:
    """
    Callback when user signs out
    Clears session but preserves local data
    """
    if app is None:
        app = get_app(f"{Name}.on_sign_out")

    if clerk_user_id:
        clear_session_token(clerk_user_id)

        # Update local data
        local_data = load_local_user_data(clerk_user_id)
        if local_data:
            local_data.session_token = ""
            save_local_user_data(local_data)

    return Result.ok({"success": True})


# =================== Admin Functions ===================

@export(mod_name=Name, version=version, api=False, interface=ToolBoxInterfaces.native)
def list_users(app: App = None) -> Result:
    """List all users from Clerk"""
    if app is None:
        app = get_app(f"{Name}.list_users")

    try:
        clerk = get_clerk_client()
        users = clerk.users.list()

        user_list = []
        for user in users:
            user_list.append({
                "id": user.id,
                "username": user.username,
                "email": user.email_addresses[0].email_address if user.email_addresses else None,
                "created_at": user.created_at
            })

        return Result.ok(data=user_list)

    except Exception as e:
        get_logger().error(f"[{Name}] Error listing users: {e}")
        return Result.default_internal_error(str(e))


@export(mod_name=Name, version=version, api=False, interface=ToolBoxInterfaces.native, test=False)
def delete_user(app: App = None, clerk_user_id: str = None) -> Result:
    """Delete a user from Clerk and local storage"""
    if app is None:
        app = get_app(f"{Name}.delete_user")

    if not clerk_user_id:
        return Result.default_user_error("User ID required")

    try:
        clerk = get_clerk_client()

        # Delete from Clerk
        clerk.users.delete(user_id=clerk_user_id)

        # Clear local data
        clear_session_token(clerk_user_id)

        # Delete from DB
        app.run_any(
            TBEF.DB.DELETE,
            query=f"CLERK_USER::{clerk_user_id}",
            get_results=True
        )

        return Result.ok(f"User {clerk_user_id} deleted")

    except Exception as e:
        get_logger().error(f"[{Name}] Error deleting user: {e}")
        return Result.default_internal_error(str(e))
