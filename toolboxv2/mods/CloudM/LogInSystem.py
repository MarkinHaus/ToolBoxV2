"""
ToolBox V2 - CLI Login System with Clerk
Handles CLI authentication via Email + Code (NO browser opening)

WICHTIG:
- Kein Webbrowser mehr öffnen
- Direkter Code-Eingabe in CLI
- BlobFile für Token-Speicherung
"""

import asyncio
import json
import os
import sys
import time
from typing import Optional

from toolboxv2 import App, get_app, Result, Code
from toolboxv2.utils.extras.blobs import BlobFile

# CLI Printing Utilities
from toolboxv2.utils.clis.cli_printing import (
    print_box_header,
    print_box_content,
    print_box_footer,
    print_status,
    print_separator
)

Name = 'CloudM'
version = '0.0.5'
export = get_app(f"{Name}.EXPORT").tb


# =================== Helper Functions ===================

def _get_session_path(username: str) -> str:
    """Get BlobFile path for CLI session"""
    safe_name = Code.one_way_hash(username, "cli-session")[:16]
    return f"clerk/cli/{safe_name}/session.json"


def _save_cli_session(username: str, session_data: dict) -> bool:
    """Save CLI session to BlobFile"""
    try:
        path = _get_session_path(username)
        with BlobFile(path, key=Code.DK()(), mode="w") as blob:
            blob.clear()
            blob.write(json.dumps(session_data).encode())
        return True
    except Exception as e:
        print_status(f"Failed to save session: {e}", "error")
        return False


def _load_cli_session(username: str) -> Optional[dict]:
    """Load CLI session from BlobFile"""
    try:
        path = _get_session_path(username)
        with BlobFile(path, key=Code.DK()(), mode="r") as blob:
            data = blob.read()
            if data and data != b'Error decoding':
                return json.loads(data.decode())
    except:
        pass
    return None


def _clear_cli_session(username: str) -> bool:
    """Clear CLI session from BlobFile"""
    try:
        path = _get_session_path(username)
        with BlobFile(path, key=Code.DK()(), mode="w") as blob:
            blob.clear()
        return True
    except:
        return False


# =================== Main CLI Login ===================

async def cli_login(app: App = None, email: str = None):
    """
    CLI Login with Clerk Email + Code verification
    NO browser opening - direct code input
    """
    if app is None:
        app = get_app("CloudM.cli_login")

    # Check if already logged in
    existing_session = _check_existing_session(app)
    if existing_session:
        print_box_header("Already Authenticated", "✓")
        print_box_content(f"Logged in as: {existing_session.get('username', 'Unknown')}", "success")
        print_box_footer()

        choice = input("\033[96m❯ Continue with existing session? (y/n): \033[0m").strip().lower()
        if choice == 'y':
            return Result.ok("Already authenticated", data=existing_session)
        else:
            await cli_logout(app)

    # Get email if not provided
    if not email:
        print_box_header("Clerk Authentication", "🔐")
        print()
        email = input("\033[96m❯ Enter your email: \033[0m").strip()
        print()

    if not email or "@" not in email:
        print_status("Invalid email address", "error")
        return Result.default_user_error("Invalid email address")

    print_status(f"Requesting verification code for {email}...", "progress")

    # Request verification code
    try:
        result = await _request_verification_code(app, email)

        if result.is_error():
            print_status(result.info.help_text or "Failed to request code", "error")
            return result

        cli_session_id = result.get().get("cli_session_id")

        print_status("Verification code sent to your email!", "success")
        print()
        print_separator("─")
        print()

        # Wait for code input
        return await _wait_for_code_input(app, cli_session_id, email)

    except Exception as e:
        print_status(f"Error: {e}", "error")
        return Result.default_internal_error(str(e))


async def _request_verification_code(app: App, email: str) -> Result:
    """Request verification code from Clerk via API"""
    try:
        # Call AuthClerk API
        result = await app.a_run_any(
            "CloudM.AuthClerk.cli_request_code",
            email=email,
            get_results=True
        )
        return result
    except Exception as e:
        # Fallback: Direct API call
        response = await app.session.fetch(
            "/api/CloudM.AuthClerk/cli_request_code",
            method="POST",
            data={"email": email}
        )
        if hasattr(response, 'json'):
            response = await response.json()
        return Result.result_from_dict(**response)


async def _wait_for_code_input(app: App, cli_session_id: str, email: str) -> Result:
    """Wait for user to input verification code"""
    max_attempts = 3

    for attempt in range(max_attempts):
        print_box_header(f"Enter Verification Code (Attempt {attempt + 1}/{max_attempts})", "📧")
        print()

        code = input("\033[96m❯ Enter 6-digit code: \033[0m").strip()
        print()

        if not code:
            print_status("No code entered", "warning")
            continue

        # Clean up code (remove spaces, dashes)
        code = code.replace(" ", "").replace("-", "")

        if len(code) != 6 or not code.isdigit():
            print_status("Code must be 6 digits", "warning")
            continue

        print_status("Verifying code...", "progress")

        # Verify code
        result = await _verify_code(app, cli_session_id, code)

        if result.is_error():
            print_status(result.info.help_text or "Invalid code", "error")
            if attempt < max_attempts - 1:
                print_status("Please try again", "info")
            continue

        # Success!
        data = result.get()
        username = data.get("username", email.split("@")[0])
        session_token = data.get("session_token", "")
        user_id = data.get("user_id", "")

        # Save session
        session_data = {
            "username": username,
            "email": email,
            "user_id": user_id,
            "session_token": session_token,
            "authenticated_at": time.time()
        }
        _save_cli_session(username, session_data)

        # Also save in app session
        if app.session:
            app.session.username = username
            app.session.valid = True

        print()
        print_box_header("Login Successful", "✓")
        print_box_content(f"Welcome, {username}!", "success")
        print_box_content("Your CLI session has been established", "info")
        print_box_footer()

        return Result.ok("Login successful", data=session_data)

    # Max attempts reached
    print()
    print_box_header("Authentication Failed", "✗")
    print_box_content("Maximum verification attempts reached", "error")
    print_box_content("Please try again later", "info")
    print_box_footer()

    return Result.default_user_error("Maximum verification attempts reached")


async def _verify_code(app: App, cli_session_id: str, code: str) -> Result:
    """Verify the entered code with Clerk"""
    try:
        result = await app.a_run_any(
            "CloudM.AuthClerk.cli_verify_code",
            cli_session_id=cli_session_id,
            code=code,
            get_results=True
        )
        return result
    except Exception as e:
        # Fallback: Direct API call
        response = await app.session.fetch(
            "/api/CloudM.AuthClerk/cli_verify_code",
            method="POST",
            data={"cli_session_id": cli_session_id, "code": code}
        )
        if hasattr(response, 'json'):
            response = await response.json()
        return Result.result_from_dict(**response)


def _check_existing_session(app: App) -> Optional[dict]:
    """Check for existing valid session"""
    # Check app session
    if app.session and app.session.valid:
        return {"username": app.session.username}

    # Check all saved sessions
    # This is a simplified check - in production, iterate through possible users
    return None


# =================== Logout ===================

async def cli_logout(app: App = None):
    """Logout from CLI session"""
    if app is None:
        app = get_app("CloudM.cli_logout")

    print_box_header("Logout", "🔓")

    username = app.get_username() if hasattr(app, 'get_username') else None

    if username:
        print_status(f"Logging out {username}...", "progress")
        _clear_cli_session(username)

    # Clear app session
    if app.session:
        app.session.valid = False
        app.session.username = None

    # Notify server
    try:
        await app.a_run_any(
            "CloudM.AuthClerk.on_sign_out",
            clerk_user_id=username,
            get_results=True
        )
    except:
        pass

    print_status("Logged out successfully", "success")
    print_box_footer()

    return Result.ok("Logout successful")


# =================== Session Status ===================

async def cli_status(app: App = None):
    """Show current CLI session status"""
    if app is None:
        app = get_app("CloudM.cli_status")

    print_box_header("Session Status", "ℹ")

    if app.session and app.session.valid:
        print_box_content(f"✓ Authenticated as: {app.session.username}", "success")
        print_box_content("Session is valid", "info")
    else:
        print_box_content("✗ Not authenticated", "warning")
        print_box_content("Run 'tb login' to authenticate", "info")

    print_box_footer()

    return Result.ok()


# =================== Web Login Page (für API) ===================

@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def open_web_login_web(app: App, request=None, session_id=None, return_to=None):
    """
    Web login page using Clerk UI components
    Returns HTML that loads Clerk's sign-in component
    """
    if request is None:
        return Result.default_internal_error("No request specified")

    # Get Clerk publishable key
    publishable_key = os.getenv('CLERK_PUBLISHABLE_KEY', '')

    template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ToolBox V2 - Login</title>
    <script src="https://cdn.jsdelivr.net/npm/@clerk/clerk-js@latest/dist/clerk.browser.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }}
        #clerk-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}
        .loading {{
            color: #666;
            text-align: center;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div id="clerk-container">
        <div class="loading">Loading authentication...</div>
    </div>

    <script>
        const clerkPubKey = '{publishable_key}';
        const sessionId = '{session_id or ""}';
        const returnTo = '{return_to or "/web/mainContent.html"}';

        async function initClerk() {{
            const clerk = new Clerk(clerkPubKey);
            await clerk.load();

            const container = document.getElementById('clerk-container');

            if (clerk.user) {{
                // Already signed in
                container.innerHTML = '<p>Already signed in! Redirecting...</p>';

                // Notify CLI if this is a CLI auth flow
                if (sessionId) {{
                    await notifyCliAuth(clerk);
                }}

                setTimeout(() => window.location.href = returnTo, 1000);
            }} else {{
                // Show sign-in component
                clerk.mountSignIn(container, {{
                    afterSignInUrl: returnTo,
                    signUpUrl: '/web/assets/signup.html'
                }});

                // Listen for sign-in completion
                clerk.addListener((event) => {{
                    if (event.user && sessionId) {{
                        notifyCliAuth(clerk);
                    }}
                }});
            }}
        }}

        async function notifyCliAuth(clerk) {{
            if (!sessionId) return;

            try {{
                const response = await fetch('/api/CloudM/open_complete_cli_auth', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        session_id: sessionId,
                        user_id: clerk.user.id,
                        username: clerk.user.username || clerk.user.emailAddresses[0]?.emailAddress?.split('@')[0],
                        session_token: await clerk.session.getToken()
                    }})
                }});
                console.log('CLI auth notified:', await response.json());
            }} catch (e) {{
                console.error('Failed to notify CLI:', e);
            }}
        }}

        initClerk().catch(console.error);
    </script>
</body>
</html>"""

    return Result.html(template)


@export(mod_name=Name, version=version, api=True)
async def open_check_cli_auth(session_id: str, app: App = None):
    """Check if CLI authentication is complete (polling endpoint)"""
    if app is None:
        app = get_app("CloudM.open_check_cli_auth")

    # Delegate to AuthClerk
    result = await app.a_run_any(
        "CloudM.AuthClerk.cli_check_auth",
        cli_session_id=session_id,
        get_results=True
    )

    return result


@export(mod_name=Name, version=version, api=True)
async def open_complete_cli_auth(
    session_id: str,
    user_id: str = None,
    username: str = None,
    session_token: str = None,
    app: App = None
):
    """Complete CLI authentication (called from web after Clerk sign-in)"""
    if app is None:
        app = get_app("CloudM.open_complete_cli_auth")

    # This is called from the web page after successful Clerk sign-in
    # to notify the CLI polling that auth is complete

    from .AuthClerk import _verification_codes

    if session_id in _verification_codes:
        _verification_codes[session_id].update({
            "verified": True,
            "user_id": user_id,
            "username": username,
            "session_token": session_token
        })
        return Result.ok({"success": True})

    return Result.default_user_error("Invalid session ID")
