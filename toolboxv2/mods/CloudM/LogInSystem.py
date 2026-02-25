"""
ToolBox V2 - CLI & Web Login System (Custom Auth)
Handles authentication via Magic Link Email + Device Invite Code.

NO Clerk dependency. NO BlobFile.
Token storage via TBEF.DB for multi-worker safety.

CLI Auth Methods:
1. Magic Link Email: User enters email -> receives link -> clicks -> JWT issued
2. Device Invite Code: Logged-in user generates 6-digit code -> CLI enters code -> JWT issued

Web Login:
Redirects to /web/scripts/login.html (Discord/Google/Passkey UI)
"""

import json
import pickle
import time
from typing import Optional

from toolboxv2 import App, get_app, Result, TBEF

# CLI Printing Utilities
from toolboxv2.utils.clis.cli_printing import (
    print_box_header,
    print_box_content,
    print_box_footer,
    print_status,
    print_separator
)

Name = 'CloudM'
version = '0.0.6'
export = get_app(f"{Name}.EXPORT").tb

AUTH_MODULE = "CloudM.Auth"


# =================== CLI Token Storage (via BlobFile - Encrypted) ===================

def _get_cli_session_storage_dir():
    """Get unified storage directory for CLI sessions."""
    from pathlib import Path
    from toolboxv2 import tb_root_dir

    # Use toolbox data directory for consistent storage
    storage_dir = tb_root_dir / ".data" / "cli_sessions"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return str(storage_dir)


def _get_blob_storage():
    """Get unified BlobStorage instance for CLI sessions."""
    from toolboxv2.utils.extras.blobs import BlobStorage, StorageMode

    storage = BlobStorage(
        mode=StorageMode.OFFLINE,
        storage_directory=_get_cli_session_storage_dir()
    )
    return storage


async def _save_cli_token(app: App, username: str, token_data: dict) -> bool:
    """Save CLI session token via BlobFile (encrypted with device key)."""
    try:
        from toolboxv2.utils.extras.blobs import BlobFile
        from toolboxv2.utils.security.cryp import Code

        key = Code.DK()()  # Device Key für Verschlüsselung

        # Speichere verschlüsselt via BlobFile
        with BlobFile(f"cli_sessions/{username}/session.json", mode="w", key=key, storage=_get_blob_storage()) as blob:
            blob.write_json(token_data)

        return True
    except Exception as e:
        print_status(f"Failed to save session: {e}", "error")
        return False


async def _load_cli_token(app: App, username: str) -> Optional[dict]:
    """Load CLI session token via BlobFile (encrypted with device key)."""
    try:
        from toolboxv2.utils.extras.blobs import BlobFile
        from toolboxv2.utils.security.cryp import Code

        key = Code.DK()()

        with BlobFile(f"cli_sessions/{username}/session.json", mode="r", key=key, storage=_get_blob_storage()) as blob:
            return blob.read_json()
    except Exception:
        return None


async def _clear_cli_token(app: App, username: str) -> bool:
    """Clear CLI session token via BlobFile."""
    try:
        storage = _get_blob_storage()
        storage.delete_blob(f"cli_sessions/{username}")
        return True
    except Exception:
        return False


# =================== Main CLI Login ===================

@export(mod_name=Name, version=version)
async def cli_login(app: App = None, email: str = None, method: str = None):
    """
    CLI Login - Two methods:
    1. Magic Link Email (default)
    2. Device Invite Code

    Args:
        app: ToolBoxV2 app instance
        email: Email for magic link (optional, will prompt)
        method: "magic" or "invite" (optional, will prompt)
    """
    if app is None:
        app = get_app("CloudM.cli_login")

    # Check existing session
    existing = await _check_existing_session(app)
    if existing:
        print_box_header("Already Authenticated", "+")
        print_box_content(f"Logged in as: {existing.get('username', 'Unknown')}", "success")
        print_box_footer()

        choice = input("\033[96m> Continue with existing session? (y/n): \033[0m").strip().lower()
        if choice == 'y':
            return Result.ok(info="Already authenticated", data=existing)
        else:
            await cli_logout(app)

    # Choose login method
    if not method:
        print_box_header("ToolBoxV2 Authentication", "#")
        print()
        print("  [1] Magic Link (Email)")
        print("  [2] Device Invite Code")
        print()
        choice = input("\033[96m> Choose method (1/2): \033[0m").strip()
        method = "invite" if choice == "2" else "magic"
        print()

    if method == "invite":
        return await _cli_login_invite(app)
    else:
        return await _cli_login_magic(app, email)


# =================== Magic Link Login ===================

async def _cli_login_magic(app: App, email: str = None) -> Result:
    """CLI login via magic link email."""
    if not email:
        print_box_header("Magic Link Login", "@")
        print()
        email = input("\033[96m> Enter your email: \033[0m").strip()
        print()

    if not email or "@" not in email:
        print_status("Invalid email address", "error")
        return Result.default_user_error("Invalid email address")

    print_status(f"Sending magic link to {email}...", "progress")

    # Request magic link via CloudM.Auth
    result = await app.a_run_any(
        (AUTH_MODULE, "request_magic_link"),
        email=email,
        get_results=True,
    )

    if result.is_error():
        error_msg = "Failed to send magic link"
        if hasattr(result, 'info') and hasattr(result.info, 'help_text'):
            error_msg = result.info.help_text
        print_status(error_msg, "error")
        return result

    print_status("Magic link sent! Check your email.", "success")
    print()
    print_separator("-")
    print()
    print("  Open the link in your email to complete authentication.")
    print("  Then paste the token from the URL below.")
    print()

    token = input("\033[96m> Token (from URL after /auth/magic/): \033[0m").strip()

    if not token:
        print_status("No token entered", "error")
        return Result.default_user_error("No token provided")

    # Verify the magic link token
    verify_result = await app.a_run_any(
        (AUTH_MODULE, "verify_magic_link"),
        token=token,
        get_results=True,
    )

    if verify_result.is_error():
        error_msg = "Invalid or expired magic link"
        if hasattr(verify_result, 'info') and hasattr(verify_result.info, 'help_text'):
            error_msg = verify_result.info.help_text
        print_status(error_msg, "error")
        return verify_result

    return await _complete_login(app, verify_result, email)


# =================== Device Invite Login ===================

async def _cli_login_invite(app: App) -> Result:
    """CLI login via device invite code."""
    print_box_header("Device Invite Login", "*")
    print()
    print("  Ask a logged-in user to generate an invite code")
    print("  (from Web Dashboard or another CLI session)")
    print()

    max_attempts = 3
    for attempt in range(max_attempts):
        code = input(f"\033[96m> Enter 6-digit invite code ({attempt + 1}/{max_attempts}): \033[0m").strip()
        print()

        if not code:
            print_status("No code entered", "warning")
            continue

        # Clean up code
        code = code.replace(" ", "").replace("-", "")

        if len(code) != 6 or not code.isdigit():
            print_status("Code must be 6 digits", "warning")
            continue

        print_status("Verifying invite code...", "progress")

        result = await app.a_run_any(
            (AUTH_MODULE, "verify_device_invite"),
            code=code,
            get_results=True,
        )

        if result.is_error():
            error_msg = "Invalid or expired code"
            if hasattr(result, 'info') and hasattr(result.info, 'help_text'):
                error_msg = result.info.help_text
            print_status(error_msg, "error")
            if attempt < max_attempts - 1:
                print_status("Please try again", "info")
            continue

        return await _complete_login(app, result)

    print()
    print_box_header("Authentication Failed", "X")
    print_box_content("Maximum attempts reached", "error")
    print_box_footer()
    return Result.default_user_error("Maximum verification attempts reached")


# =================== Login Completion ===================

async def _complete_login(app: App, result: Result, email: str = None) -> Result:
    """Complete login: save token, update app session, print success."""
    data = result.get()
    if not data or not isinstance(data, dict):
        print_status("Unexpected response from auth", "error")
        return Result.default_internal_error("Bad auth response")

    username = data.get("username", "")
    session_data = {
        "username": username,
        "email": email or data.get("email", ""),
        "user_id": data.get("user_id", ""),
        "level": data.get("level", 1),
        "access_token": data.get("access_token", ""),
        "refresh_token": data.get("refresh_token", ""),
        "provider": data.get("provider", ""),
        "authenticated_at": time.time(),
    }
    await _save_cli_token(app, username, session_data)

    # FIX: Korrekte SessionData-Attribute setzen
    _apply_session_to_app(app, session_data)

    print()
    print_box_header("Login Successful", "+")
    print_box_content(f"Welcome, {username}!", "success")
    print_box_content(f"Provider: {data.get('provider', '')}", "info")
    print_box_content("Your CLI session has been established", "info")
    print_box_footer()

    return Result.ok(info="Login successful", data=session_data)



def _apply_session_to_app(app: App, session_data: dict):
    """Setzt app.session korrekt (user_name/validated, nicht username/valid)."""
    s = getattr(app, 'session', None)
    if s is None:
        return

    username = session_data.get("username", "")
    user_id = session_data.get("user_id", "")

    # SessionData-Attribute (session.py korrekt)
    for attr, val in [
        ("user_name", username), ("validated", True), ("anonymous", False),
        ("user_id", user_id), ("provider_user_id", user_id),
    ]:
        if hasattr(s, attr):
            setattr(s, attr, val)

    if hasattr(s, 'level'):
        try:
            from toolboxv2.utils.workers.session import AccessLevel
            s.level = AccessLevel.LOGGED_IN
        except ImportError:
            s.level = 1

    # Legacy-Attribute
    if hasattr(s, 'username'):
        s.username = username
    if hasattr(s, 'valid'):
        s.valid = True

    if hasattr(s, 'mark_dirty'):
        s.mark_dirty()


def _invalidate_app_session(app: App):
    """Invalidiert app.session korrekt."""
    s = getattr(app, 'session', None)
    if s is None:
        return
    for attr, val in [
        ("validated", False), ("anonymous", True), ("user_name", "anonymous"),
        ("user_id", ""), ("valid", False), ("username", None),
    ]:
        if hasattr(s, attr):
            setattr(s, attr, val)
    if hasattr(s, 'level'):
        s.level = 0



async def auto_login_from_blob(app) -> bool:
    """
    Auto-Login aus BlobFile mit Token-Refresh. Ersetzt log_in() in main().

    Usage:
        from toolboxv2.mods.CloudM.LogInSystem import auto_login_from_blob
        tb_app.run_bg_task_advanced(auto_login_from_blob, tb_app)
    """
    try:
        # Use _check_existing_session for auto-refresh support
        session_data = await _check_existing_session(app)
        if session_data:
            _apply_session_to_app(app, session_data)
            app.print(f"🔓 Logged in as: {session_data.get('username', 'Unknown')}")
            app.set_username(session_data.get('username', ''))
            return True
        return False
    except Exception as e:
        if hasattr(app, 'logger'):
            app.logger.debug(f"Auto-login failed: {e}")
        return False

# =================== Session Check ===================

def _find_cli_session(app: App = None, username: str = None) -> Optional[dict]:
    """
    Findet gespeicherte CLI-Session aus dem BlobFile-Speicher.
    Liest den Blob "cli_sessions" direkt und iteriert seine Keys.

    Returns:
        dict {"username": str, "session_data": dict} oder None
    """
    from toolboxv2.utils.extras.blobs import BlobFile
    from toolboxv2.utils.security.cryp import Code

    storage = _get_blob_storage()

    # Blob "cli_sessions" direkt lesen (raw pickle dict)
    raw = storage.read_blob("cli_sessions", decrypt=False)
    if raw is None:
        return None

    # Entschlüsseln auf Blob-Ebene (Device-Key, ohne custom key)
    try:
        raw = storage.crypto.decrypt(raw)
    except Exception:
        pass  # Vielleicht schon entschlüsselt

    try:
        blob_content = pickle.loads(raw)
    except Exception:
        return None

    if not isinstance(blob_content, dict) or not blob_content:
        return None

    # Username-Discovery: entweder spezifisch oder erster gefundener
    if username:
        usernames = [username] if username in blob_content else []
    else:
        usernames = list(blob_content.keys())

    for uname in usernames:
        user_folder = blob_content.get(uname, {})
        if not isinstance(user_folder, dict):
            continue

        file_data = user_folder.get("session.json")
        if file_data is None:
            continue

        # Entschlüsseln mit Device-Key (BlobFile custom key)
        try:
            key = Code.DK()()
            decrypted = storage.crypto.decrypt(file_data, key)
            import json
            session_data = json.loads(decrypted.decode())
            if session_data and session_data.get("access_token"):
                return {"username": uname, "session_data": session_data}
        except Exception:
            continue

    return None


async def _check_existing_session(app: App, username: str = None) -> Optional[dict]:
    """Check for existing valid CLI session. Auto-refreshes if access token expired/expiring."""
    try:
        result = _find_cli_session(app, username)
        if not result:
            return None

        session_data = result["session_data"]
        uname = result["username"]

        # 1) Try validating access token
        validate = await app.a_run_any(
            (AUTH_MODULE, "validate_session"),
            token=session_data["access_token"],
            get_results=True,
        )
        if not validate.is_error():
            # Token valid — check if expiring soon (< 2 min) → preemptive refresh
            try:
                import jwt as pyjwt
                payload = pyjwt.decode(
                    session_data["access_token"],
                    options={"verify_signature": False, "verify_exp": False},
                )
                remaining = payload.get("exp", 0) - time.time()
                if remaining > 120:  # mehr als 2 min → alles gut
                    return session_data
                # < 2 min → weiter zu refresh
            except Exception:
                return session_data  # decode failed but token valid → use as-is

        # 2) Access token expired/expiring → try refresh
        refresh_token = session_data.get("refresh_token")
        if not refresh_token:
            print_status("Session expired — no refresh token, please login again (tb login)", "warning")
            return None

        # Validate refresh token JWT locally
        from toolboxv2.mods.CloudM.auth.jwt_tokens import (
            _validate_jwt as _validate_jwt_local,
            _generate_access_token,
            _generate_refresh_token,
        )

        valid, ref_payload = await _validate_jwt_local(app, refresh_token, token_type="refresh")
        if not valid:
            print_status("Session expired — please login again (tb login)", "warning")
            await _clear_cli_token(app, uname)
            return None

        # 3) Generate new token pair from stored session info
        #    Extract level from old access token (even if expired)
        level = session_data.get("level", 1)
        if level <= 0:
            try:
                import jwt as pyjwt
                old = pyjwt.decode(
                    session_data["access_token"],
                    options={"verify_signature": False, "verify_exp": False},
                )
                level = old.get("level", 1)
            except Exception:
                level = 1

        new_access = _generate_access_token(
            user_id=session_data["user_id"],
            username=uname,
            level=level,
            provider=session_data.get("provider", ""),
            email=session_data.get("email", ""),
        )
        new_refresh = _generate_refresh_token(session_data["user_id"])

        # 4) Update & persist session
        session_data["access_token"] = new_access
        session_data["refresh_token"] = new_refresh
        session_data["level"] = level
        session_data["refreshed_at"] = time.time()

        await _save_cli_token(app, uname, session_data)

        if hasattr(app, 'logger'):
            app.logger.info(f"CLI session auto-refreshed for {uname}")

        return session_data

    except Exception as e:
        if hasattr(app, 'logger'):
            app.logger.debug(f"Session check failed: {e}")
        return None

# =================== Logout ===================

@export(mod_name=Name, version=version)
async def cli_logout(app: App = None):
    """Logout — BlobFile-korrekt."""
    if app is None:
        app = get_app("CloudM.cli_logout")

    print_box_header("Logout", "~")

    result = _find_cli_session(app)

    if result:
        username = result["username"]
        sd = result["session_data"]
        print_status(f"Logging out {username}...", "progress")

        # Token blacklisten
        if sd.get("access_token"):
            try:
                await app.a_run_any(
                    (AUTH_MODULE, "logout"),
                    token=sd["access_token"],
                    get_results=False,
                )
            except Exception:
                pass

        # Session aus Blob entfernen (leeres write überschreibt)
        await _clear_cli_token(app, username)

    # App-Session invalidieren
    _invalidate_app_session(app)

    print_status("Logged out successfully", "success")
    print_box_footer()
    return Result.ok("Logout successful")



# =================== Session Status ===================

@export(mod_name=Name, version=version)
async def cli_status(app: App = None):
    """Show current CLI session status — BlobFile-korrekt."""
    if app is None:
        app = get_app("CloudM.cli_status")

    print_box_header("Session Status", "i")

    try:
        result = _find_cli_session(app)

        if result:
            sd = result["session_data"]
            username = result["username"]
            provider = sd.get("provider", "unknown")
            auth_time = sd.get("authenticated_at", 0)

            print_box_content(f"+ Authenticated as: {username}", "success")

            if auth_time:
                elapsed = time.time() - auth_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                print_box_content(f"  Provider: {provider}", "info")
                print_box_content(f"  Session age: {hours}h {minutes}m", "info")

            print_box_content("Session is valid", "info")
        else:
            print_box_content("X Not authenticated", "warning")
            print_box_content("Run 'tb login' to authenticate", "info")
    except Exception as e:
        print_box_content("X Not authenticated", "warning")
        print_box_content(f"Error: {e}", "info")

    print_box_footer()
    return Result.ok()


# =================== Generate Device Invite (for logged-in users) ===================

@export(mod_name=Name, version=version)
async def cli_generate_invite(app: App = None):
    """Generate a device invite code for CLI pairing via BlobFile."""
    if app is None:
        app = get_app("CloudM.cli_generate_invite")

    # Check if logged in via BlobFile
    from toolboxv2.utils.extras.blobs import BlobFile
    from toolboxv2.utils.security.cryp import Code

    storage = _get_blob_storage()
    blobs = storage.list_blobs(prefix="cli_sessions/")

    if not blobs:
        print_status("You must be logged in to generate an invite code", "error")
        return Result.default_user_error("Not authenticated")

    username = blobs[0]['blob_id'].replace('cli_sessions/', '')

    key = Code.DK()()
    with BlobFile(f"cli_sessions/{username}/session.json", mode="r", key=key, storage=_get_blob_storage()) as blob:
        token_data = blob.read_json()

    if not token_data or not token_data.get("user_id"):
        print_status("No valid session found", "error")
        return Result.default_user_error("No valid session")

    print_status("Generating invite code...", "progress")

    result = await app.a_run_any(
        (AUTH_MODULE, "create_device_invite"),
        user_id=token_data["user_id"],
        get_results=True,
    )

    if result.is_error():
        error_msg = "Failed to generate invite code"
        if hasattr(result, 'info') and hasattr(result.info, 'help_text'):
            error_msg = result.info.help_text
        print_status(error_msg, "error")
        return result

    data = result.get()
    code = data.get("code", "")
    expires_in = data.get("expires_in", 300)

    print()
    print_box_header("Device Invite Code", "*")
    print()
    print(f"    Code: \033[1;93m{code}\033[0m")
    print()
    print_box_content(f"Expires in {expires_in // 60} minutes", "info")
    print_box_content("Enter this code on the other device", "info")
    print_box_footer()

    return Result.ok(data={"code": code, "expires_in": expires_in})


# =================== Web Login Page (API endpoint) ===================

@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def open_web_login_web(app: App = None, request=None, **kwargs):
    """
    Web login page - redirects to custom login.html.
    No Clerk SDK dependency.
    """
    return Result.redirect("/web/scripts/login.html")
