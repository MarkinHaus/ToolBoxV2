import asyncio
import json
import os
import sys
import uuid
import webbrowser
import time
from typing import Any
from urllib.parse import quote, parse_qs, urlparse
from toolboxv2 import App, get_app, Result, Code, Style, Spinner, RequestData
from toolboxv2.utils.extras.blobs import BlobFile
from toolboxv2.utils.system.session import Session

# --- CLI Printing Utilities ---
from toolboxv2.utils.clis.cli_printing import (
    print_box_header,
    print_box_content,
    print_box_footer,
    print_status,
    print_separator
)

Name = 'CloudM'
version = '0.0.4'
export = get_app(f"{Name}.EXPORT").tb


def print_menu_option(number: int, text: str, selected: bool = False):
    """Print a menu option"""
    if selected:
        print(f"  ▶ {number}. {text}")
    else:
        print(f"    {number}. {text}")


# =================== Main Login Functions ===================

async def cli_web_login(app: App = None, force_remote: bool = False, force_local: bool = False):
    """
    Enhanced CLI web login with remote/local options and modern visual feedback
    """
    if app is None:
        app = get_app("CloudM.cli_web_login")

    # Check if already logged in
    if app.session and app.session.valid:
        print_box_header("Already Authenticated", "✓")
        print_box_content("You are already logged in!", "success")
        print_box_content(f"Username: {app.get_username()}", "info")
        print_box_footer()
        return Result.ok("Already authenticated")

    # Determine login method
    login_method = await _determine_login_method(app, force_remote, force_local)

    if login_method == "remote":
        return await _remote_web_login(app)
    elif login_method == "local":
        return await _local_web_login(app)
    else:
        print_status("Login cancelled by user", "warning")
        return Result.default_user_error("Login cancelled by user")


async def _determine_login_method(app: App, force_remote: bool, force_local: bool):
    """Determine whether to use remote or local login"""
    if force_remote:
        print_status("Using remote login (forced)", "info")
        return "remote"

    if force_local:
        print_status("Using local login (forced)", "info")
        return "local"

    print_box_header("Login Method Selection", "🔐")
    print()
    print_menu_option(1, "🌐 Remote Login (SimpleCore Hub)")
    print_menu_option(2, "🏠 Local Login (Local Server)")
    print_menu_option(3, "❌ Cancel")
    print()
    print_box_footer()

    while True:
        choice = input("\033[96m❯ Enter choice (1-3): \033[0m").strip()

        if choice == "1":
            print_status("Remote login selected", "success")
            return "remote"
        elif choice == "2":
            print_status("Local login selected", "success")
            return "local"
        elif choice == "3":
            # print_status("Login cancelled", "warning")
            return "cancel"
        else:
            print_status("Invalid choice, please enter 1, 2, or 3", "error")


async def _remote_web_login(app: App):
    """Handle remote web login flow with enhanced visuals"""
    remote_base = os.getenv('TOOLBOXV2_REMOTE_BASE') or os.getenv('APP_BASE_URL', 'https://simplecore.app')

    print_box_header("Remote Login", "🌐")
    print_box_content(f"Server: {remote_base}", "info")
    print_box_footer()

    # Generate session token for CLI
    cli_session_id = uuid.uuid4().hex

    # Create login URL with CLI session tracking
    login_url = f"{remote_base}/api/CloudM/open_web_login_web?session_id={cli_session_id}&return_to=cli"

    print_status("Generating secure session...", "progress")
    print_status("Session ready!", "success")

    print()
    print_separator("═")
    print("\033[1m  📱 Browser Authentication Required\033[0m")
    print_separator("═")
    print()

    print_status("Opening browser for authentication...", "info")
    print_status("Please complete the login process in your browser", "waiting")

    try:
        webbrowser.open(login_url, new=0, autoraise=True)
        print_status("Browser opened successfully", "success")
    except Exception as e:
        print_status(f"Could not auto-open browser: {e}", "warning")
        print()
        print("  📋 Please manually open this URL:")
        print(f"  \033[94m{login_url}\033[0m")
        print()

    # Poll for authentication completion
    return await _poll_for_auth_completion(app, remote_base, cli_session_id)


async def _local_web_login(app: App):
    """Handle local web login flow with enhanced visuals"""
    local_base = os.getenv('APP_BASE_URL', 'http://localhost:8080')

    print_box_header("Local Login", "🏠")
    print_box_content(f"Server: {local_base}", "info")
    print_box_footer()

    # Check if local server is running
    print_status("Checking local server status...", "progress")
    with Spinner("Checking local server status..."):
        server_running = await _check_local_server(app, local_base)

    if not server_running:
        print_status("Local server not detected", "warning")
        setup_result = await _setup_local_server(app)
        if setup_result.is_error():
            return setup_result
    else:
        print_status("Local server is running", "success")

    # Generate session token for CLI
    cli_session_id = uuid.uuid4().hex

    # Create login URL
    login_url = f"{local_base}/api/CloudM/open_web_login_web?session_id={cli_session_id}&return_to=cli"

    print()
    print_separator("═")
    print("\033[1m  📱 Browser Authentication Required\033[0m")
    print_separator("═")
    print()

    print_status("Opening browser for authentication...", "info")

    try:
        webbrowser.open(login_url, new=0, autoraise=True)
        print_status("Browser opened successfully", "success")
    except Exception as e:
        print_status(f"Could not auto-open browser: {e}", "warning")
        print()
        print("  📋 Please manually open this URL:")
        print(f"  \033[94m{login_url}\033[0m")
        print()

    return await _poll_for_auth_completion(app, local_base, cli_session_id)


async def _check_local_server(app: App, base_url: str) -> bool:
    """Check if local server is running"""
    try:
        base_sto = app.session.base
        app.session.base = ''
        response = await app.session.fetch(f"{base_url}/api/CloudM/openVersion", timeout=5)
        app.session.base = base_sto
        return response.status == 200
    except Exception as e:
        return False


async def _setup_local_server(app: App):
    """Setup local server if not running with enhanced UI"""
    print_box_header("Local Server Setup", "⚙_")
    print()
    print_menu_option(1, "🚀 Start local server with API")
    print_menu_option(2, "🔧 Setup background service")
    print_menu_option(3, "❌ Cancel")
    print()
    print_box_footer()

    while True:
        choice = input("\033[96m❯ Enter choice (1-3): \033[0m").strip()

        if choice == "1":
            print_status("Starting local server...", "progress")
            with Spinner("Starting server..."):
                from toolboxv2.utils.clis.api import manage_server
                if not manage_server("start"):
                    return Result.default_internal_error("Failed to start server")
                await asyncio.sleep(5)
            # Start server logic here
            print_status("Server started successfully", "success")
            return Result.ok("Server started")

        elif choice == "2":
            print_status("Setting up background service...", "progress")
            from toolboxv2.__main__ import setup_service_linux, setup_service_windows
            from toolboxv2 import get_app
            tb_app = get_app()
            if tb_app.system_flag == "Linux":
                setup_service_linux()
            if tb_app.system_flag == "Windows":
                await setup_service_windows()
            # Setup service logic here
            print_status("Service configured successfully", "success")
            return Result.ok("Service setup")

        elif choice == "3":
            print_status("Setup cancelled", "warning")
            return Result.default_user_error("Setup cancelled")

        else:
            print_status("Invalid choice, please enter 1, 2, or 3", "error")

@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def open_web_login_web(app: App, request: RequestData, session_id=None, return_to=None, log_in_for="CLI"):
    """Handle web login flow for CLI and Browser"""
    if request is None:
        return Result.default_internal_error("No request specified")

    template = """<div">
        <div class="login-card">
            <h1>🔐 XXX Authentication</h1>
            <p>Please authenticate to continue with XXX access</p>

            <div id="loginForm">
                <input type="text" id="username" placeholder="Username" required>
                <button type="button" onclick="startLogin()">Login</button>
            </div>

            <div id="statusMessage" style="display: none; padding: 10px; margin: 10px 0; border-radius: 5px;"></div>
            <div id="successMessage" style="display: none;">
                <h3>✅ Authentication Successful!</h3>
                <p>You can now return to your XXX. This window will close automatically.</p>
            </div>
        </div>

    <script unsave="true">

        const urlParams = new URLSearchParams(window.location.search);
        const sessionId = urlParams.get('session_id');
        const returnTo = urlParams.get('return_to');

        console.log('[XXX Login] Session ID:', sessionId);
        console.log('[XXX Login] Return to:', returnTo);

        async function startLogin() {
            const username = document.getElementById('username').value;
            if (!username) {
                showStatus('Please enter a username', 'error');
                return;
            }

            if (!sessionId) {
                showStatus('Error: No session ID provided. Please restart the XXX login process.', 'error');
                return;
            }

            showStatus('Authenticating...', 'info');

            try {
                // Use TB.js login system
                console.log('[XXX Login] Attempting login for:', username);
                const result = await window.TB.user.loginWithDeviceKey(username);
                console.log('[XXX Login] Login result:', result);

                if (result.success) {
                    showStatus('Login successful! Notifying XXX...', 'info');

                    // Get the JWT token from TB.user state
                    const jwtToken = window.TB.user.getToken();
                    console.log('[XXX Login] JWT Token:', jwtToken ? 'Found' : 'Not found');

                    if (!jwtToken) {
                        showStatus('Error: No authentication token found. Please try again.', 'error');
                        return;
                    }

                    // Complete XXX authentication by notifying the backend
                    console.log('[XXX Login] Calling complete_cli_auth with session_id:', sessionId);
                    const completeResponse = await window.TB.api.request(
                        'CloudM',
                        'open_complete_cli_auth',
                        {
                            session_id: sessionId,
                            jwt_token: jwtToken,
                            username: username
                        },
                        'POST'
                    );

                    console.log('[XXX Login] complete_cli_auth response:', completeResponse);

                    if (completeResponse.error === window.TB.ToolBoxError.none) {
                        showSuccess();
                        console.log('[XXX Login] XXX authentication completed successfully');
                        setTimeout(() => {
                            console.log('[XXX Login] Closing window...');
                            window.close();
                        }, 3000);
                    } else {
                        showStatus('Error notifying XXX: ' + (completeResponse.info?.help_text || 'Unknown error'), 'error');
                    }
                } else {
                    showStatus(result.message || 'Login failed', 'error');
                }
            } catch (error) {
                console.error('[XXX Login] Error:', error);
                showStatus('Authentication error: ' + error.message, 'error');
            }
        }

        function showStatus(message, type) {
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.style.display = 'block';

            // Style based on type
            if (type === 'error') {
                statusEl.style.backgroundColor = '#fee';
                statusEl.style.color = '#c00';
                statusEl.style.border = '1px solid #c00';
            } else if (type === 'info') {
                statusEl.style.backgroundColor = '#eef';
                statusEl.style.color = '#006';
                statusEl.style.border = '1px solid #006';
            } else if (type === 'success') {
                statusEl.style.backgroundColor = '#efe';
                statusEl.style.color = '#060';
                statusEl.style.border = '1px solid #060';
            }
        }

        function showSuccess() {
            document.getElementById('loginForm').style.display = 'none';
            document.getElementById('statusMessage').style.display = 'none';
            document.getElementById('successMessage').style.display = 'block';
        }

        // Auto-focus username field
        if (document.getElementById('username')) {
            document.getElementById('username').focus();

            // Allow Enter key to submit
            document.getElementById('username').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                e.preventDefault();
                    startLogin();
                }
            });
        }

        // Check if TB is available
        if (!window.TB) {
            console.error('[XXX Login] TB framework not loaded!');
            showStatus('Error: TB framework not loaded. Please refresh the page.', 'error');
        } else {
            console.log('[XXX Login] TB framework loaded successfully');
        }


    </script>
</div>
""".replace("XXX", log_in_for)

    return Result.html(template)

async def _poll_for_auth_completion(app: App, base_url: str, session_id: str, timeout: int = 300):
    """Poll for authentication completion with progress indicator"""
    import sys

    print()
    print_separator("═")
    print("\033[1m  ⏳ Waiting for Authentication\033[0m")
    print_separator("═")
    print()

    print_status(f"Timeout in {timeout} seconds", "info")
    print_status("Complete the login in your browser...", "waiting")
    print()

    start_time = time.time()
    dots = 0
    check_count = 0

    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    while time.time() - start_time < timeout:
        try:
            # Visual progress indicator
            elapsed = int(time.time() - start_time)
            remaining = timeout - elapsed
            spinner_char = spinner[check_count % len(spinner)]

            # Progress bar
            progress = min(elapsed / timeout, 1.0)
            bar_length = 50
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)

            sys.stdout.write(f'\r\033[96m{spinner_char} Checking... [{bar}] {remaining}s remaining\033[0m')
            sys.stdout.flush()

            # Check if authentication is complete
            base_sto = app.session.base
            app.session.base = ''
            response = await app.session.fetch(
                f"{base_url}/api/CloudM/open_check_cli_auth",
                method="POST",
                data={'session_id': session_id}
            )
            app.session.base = base_sto

            # Convert response to JSON if it's a ClientResponse object
            if hasattr(response, 'json'):
                response = await response.json()
            elif response is False:
                # Connection failed, retry
                await asyncio.sleep(2)
                check_count += 1
                continue

            result = Result.result_from_dict(**response)

            if result.is_error():
                print_status(result.info.help_text, "error")
                break

            if result.get('authenticated'):
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                sys.stdout.flush()

                jwt_token = result.get('jwt_token')
                username = result.get('username')

                if jwt_token:
                    print_status("Authentication detected!", "success")
                    with Spinner("Finalizing login..."):

                        # Store JWT token
                        await _store_jwt_token(app, jwt_token)

                    print()
                    print_box_header("Login Successful", "✓")
                    print_box_content(f"Welcome back, {username}!", "success")
                    print_box_content("Your session has been established", "info")
                    print_box_footer()

                    return Result.ok("Login successful")

            await asyncio.sleep(2)  # Poll every 2 seconds
            check_count += 1

        except Exception as e:
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            sys.stdout.flush()
            print_status(f"Connection issue: {e}", "warning")
            break
            # print_status("Retrying...", "progress")
            # await asyncio.sleep(5)
            # check_count += 1

    # Timeout reached
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()

    print()
    print_box_header("Authentication Timeout", "⏱")
    print_box_content("Authentication window has expired", "error")
    print_box_content("Please try again", "info")
    print_box_footer()

    return Result.default_user_error("Authentication timeout")


async def _store_jwt_token(app: App, jwt_token: str):
    """Store JWT token and register CLI session"""
    from toolboxv2.utils.extras.blobs import BlobFile
    from .UserInstances import register_cli_session

    username = app.get_username()

    # Store JWT token
    with BlobFile(f"claim/{username}/jwt.c", key=Code.DK()(), mode="w") as blob:
        blob.clear()
        blob.write(jwt_token.encode())

    # Register CLI session
    session_info = {
        'login_method': 'web_cli',
        'user_agent': 'CLI',
        'ip_address': 'localhost'
    }

    register_cli_session(username, jwt_token, session_info)

async def _set_db_helper(app: App, data: dict[str, Any]):
    with BlobFile(f"CLOUDM/Login/{data['query']}/user.session", key=Code.DK()(), mode="w") as blob:
        blob.write(data['value'].encode(encoding=sys.stdout.encoding or 'utf-8'))

async def _get_db_helper(app: App, query: str):
    with BlobFile(f"CLOUDM/Login/{query}/user.session", key=Code.DK()(), mode="r") as blob:
        return blob.read().decode(encoding=sys.stdout.encoding or 'utf-8')

async def _remove_db_helper(app: App, query: str):
    with BlobFile(f"CLOUDM/Login/{query}/user.session", key=Code.DK()(), mode="w") as blob:
        blob.clear()

# =================== Server-side Endpoints (unchanged) ===================

@export(mod_name=Name, version=version, api=True)
async def open_check_cli_auth(session_id: str, app: App = None):
    """Check if CLI authentication is complete"""
    if app is None:
        app = get_app("CloudM.open_check_cli_auth")

    # Check session storage for completed authentication
    auth_data_str = await _get_db_helper(app, f"cli_auth_{session_id}")

    if auth_data_str:
        try:
            auth_data = json.loads(auth_data_str)
            return {
                'authenticated': True,
                'jwt_token': auth_data.get('jwt_token'),
                'username': auth_data.get('username')
            }
        except (json.JSONDecodeError, TypeError) as e:
            app.logger.error(f"Error parsing CLI auth data for session {session_id}: {e}")
            return {'authenticated': False}

    return {'authenticated': False}


@export(mod_name=Name, version=version, api=True)
async def open_complete_cli_auth(session_id: str, jwt_token: str, username: str, app: App = None):
    """Complete CLI authentication process - No auth required as this IS the auth process"""
    if app is None:
        app = get_app("CloudM.open_complete_cli_auth")

    app.logger.info(f"CLI auth completion requested for session {session_id}, user {username}")

    # Store authentication data temporarily
    auth_data = {
        'jwt_token': jwt_token,
        'username': username,
        'timestamp': time.time()
    }

    try:
        await _set_db_helper(app, {"query": f"cli_auth_{session_id}", "value": json.dumps(auth_data)})
        app.logger.info(f"CLI auth data stored for session {session_id}")

        # Clean up after 10 minutes
        asyncio.create_task(_cleanup_auth_session(app, session_id, 600))

        return Result.ok("CLI authentication completed")
    except Exception as e:
        app.logger.error(f"Error storing CLI auth data for session {session_id}: {e}")
        return Result.default_internal_error(f"Failed to complete CLI authentication: {str(e)}")


async def _cleanup_auth_session(app: App, session_id: str, delay: int):
    """Clean up authentication session after delay"""
    await asyncio.sleep(delay)
    await _remove_db_helper(app, f"cli_auth_{session_id}")


# =================== Logout Function (enhanced) ===================

async def cli_logout(app: App = None):
    """Enhanced logout with modern visual feedback"""
    if app is None:
        app = get_app("CloudM.cli_logout")

    # Clear screen
    print('\033[2J\033[H')

    print_box_header("Logout Process", "🔓")
    print_box_content("Terminating session...", "info")
    print_box_footer()

    username = app.get_username()

    if not username:
        print_status("No active session found", "warning")
        return Result.ok("No session to logout")

    print_status(f"Logging out user: {username}", "progress")
    with Spinner("Finalizing logout..."):
        # Close CLI session
        from .UserInstances import UserInstances, close_cli_session

        try:
            cli_session_id = UserInstances.get_cli_session_id(username).get()
            close_cli_session(cli_session_id)
            print_status("CLI session closed", "success")
        except Exception as e:
            print_status(f"Session cleanup warning: {e}", "warning")

        # Clear JWT token
        from toolboxv2.utils.extras.blobs import BlobFile
        try:
            with BlobFile(f"claim/{username}/jwt.c", key=Code.DK()(), mode="w") as blob:
                blob.clear()
            print_status("Credentials cleared", "success")
        except Exception as e:
            print_status(f"Credential cleanup warning: {e}", "warning")

        # Clear session
        if app.session:
            app.session.valid = False
            app.session.username = None

    print()
    print_box_header("Logout Complete", "✓")
    print_box_content(f"User '{username}' logged out successfully", "success")
    print_box_content("All credentials have been cleared", "info")
    print_box_footer()

    return Result.ok("Logout successful")
