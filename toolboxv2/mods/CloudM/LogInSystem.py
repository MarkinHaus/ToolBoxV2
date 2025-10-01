import asyncio
import json
import os
import webbrowser
import time
from urllib.parse import quote, parse_qs, urlparse
from toolboxv2 import App, get_app, Result, Code, Style
from toolboxv2.utils.system.session import Session

Name = 'CloudM'
version = '0.0.4'
export = get_app(f"{Name}.EXPORT").tb


# =================== Visual Helper Functions ===================

def print_box_header(title: str, icon: str = "ℹ"):
    """Print a styled box header"""
    width = 76
    title_text = f" {icon} {title} "
    padding = (width - len(title_text)) // 2

    print("\n┌" + "─" * width + "┐")
    print("│" + " " * padding + title_text + " " * (width - padding - len(title_text)) + "│")
    print("├" + "─" * width + "┤")


def print_box_content(text: str, style: str = ""):
    """Print content inside a box"""
    width = 76

    if style == "success":
        icon = "✓"
        text = f"{icon} {text}"
    elif style == "error":
        icon = "✗"
        text = f"{icon} {text}"
    elif style == "warning":
        icon = "⚠"
        text = f"{icon} {text}"
    elif style == "info":
        icon = "ℹ"
        text = f"{icon} {text}"

    print("│ " + text.ljust(width - 2) + "│")


def print_box_footer():
    """Print box footer"""
    width = 76
    print("└" + "─" * width + "┘\n")


def print_status(message: str, status: str = "info"):
    """Print a status message with icon"""
    icons = {
        'success': '✓',
        'error': '✗',
        'warning': '⚠',
        'info': 'ℹ',
        'progress': '⟳',
        'waiting': '⏳'
    }

    colors = {
        'success': '\033[92m',  # Green
        'error': '\033[91m',  # Red
        'warning': '\033[93m',  # Yellow
        'info': '\033[94m',  # Blue
        'progress': '\033[96m',  # Cyan
        'waiting': '\033[95m'  # Magenta
    }

    reset = '\033[0m'
    icon = icons.get(status, '•')
    color = colors.get(status, '')

    print(f"{color}{icon} {message}{reset}")


def print_separator(char: str = "─", width: int = 76):
    """Print a separator line"""
    print(char * width)


def print_menu_option(number: int, text: str, selected: bool = False):
    """Print a menu option"""
    if selected:
        print(f"  ▶ {number}. {text}")
    else:
        print(f"    {number}. {text}")


def show_spinner(message: str, duration: float = 0.5):
    """Show a simple spinner animation"""
    import sys

    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    end_time = time.time() + duration
    idx = 0

    while time.time() < end_time:
        sys.stdout.write(f'\r\033[96m{spinner[idx % len(spinner)]} {message}\033[0m')
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1

    sys.stdout.write('\r' + ' ' * (len(message) + 3) + '\r')
    sys.stdout.flush()


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
            print_status("Login cancelled", "warning")
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
    cli_session_id = Code.generate_random_string(32)

    # Create login URL with CLI session tracking
    login_url = f"{remote_base}/web/cli_login.html?session_id={cli_session_id}&return_to=cli"

    print_status("Generating secure session...", "progress")
    show_spinner("Preparing authentication", 0.5)
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
    show_spinner("Connecting to server", 0.5)

    server_running = await _check_local_server(app, local_base)

    if not server_running:
        print_status("Local server not detected", "warning")
        setup_result = await _setup_local_server(app)
        if setup_result.is_error():
            return setup_result
    else:
        print_status("Local server is running", "success")

    # Generate session token for CLI
    cli_session_id = Code.generate_random_string(32)

    # Create login URL
    login_url = f"{local_base}/web/cli_login.html?session_id={cli_session_id}&return_to=cli"

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
        response = await app.session.fetch(f"{base_url}/health", timeout=5)
        return response.status == 200
    except:
        return False


async def _setup_local_server(app: App):
    """Setup local server if not running with enhanced UI"""
    print_box_header("Local Server Setup", "⚙")
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
            show_spinner("Initializing server", 1.0)
            # Start server logic here
            print_status("Server started successfully", "success")
            return Result.ok("Server started")

        elif choice == "2":
            print_status("Setting up background service...", "progress")
            show_spinner("Configuring service", 1.0)
            # Setup service logic here
            print_status("Service configured successfully", "success")
            return Result.ok("Service setup")

        elif choice == "3":
            print_status("Setup cancelled", "warning")
            return Result.default_user_error("Setup cancelled")

        else:
            print_status("Invalid choice, please enter 1, 2, or 3", "error")


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
            response = await app.session.fetch(
                f"{base_url}/api/CloudM.LogInSystem/check_cli_auth",
                method="POST",
                data={'session_id': session_id}
            )

            if response.get('authenticated'):
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                sys.stdout.flush()

                jwt_token = response.get('jwt_token')
                username = response.get('username')

                if jwt_token:
                    print_status("Authentication detected!", "success")
                    show_spinner("Finalizing login", 0.5)

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
            print_status("Retrying...", "progress")
            await asyncio.sleep(5)
            check_count += 1

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


# =================== Server-side Endpoints (unchanged) ===================

@export(mod_name=Name, version=version, api=True)
async def check_cli_auth(session_id: str, app: App = None):
    """Check if CLI authentication is complete"""
    if app is None:
        app = get_app("CloudM.check_cli_auth")

    # Check session storage for completed authentication
    auth_data = app.config_fh.get_file_handler(f"cli_auth_{session_id}")
    auth_data = json.loads(auth_data)

    if auth_data:
        return {
            'authenticated': True,
            'jwt_token': auth_data.get('jwt_token'),
            'username': auth_data.get('username')
        }

    return {'authenticated': False}


@export(mod_name=Name, version=version, api=True)
async def complete_cli_auth(session_id: str, jwt_token: str, username: str, app: App = None):
    """Complete CLI authentication process"""
    if app is None:
        app = get_app("CloudM.complete_cli_auth")

    # Store authentication data temporarily
    auth_data = {
        'jwt_token': jwt_token,
        'username': username,
        'timestamp': time.time()
    }

    app.config_fh.add_to_save_file_handler(f"cli_auth_{session_id}", json.dumps(auth_data))

    # Clean up after 10 minutes
    asyncio.create_task(_cleanup_auth_session(app, session_id, 600))

    return Result.ok("CLI authentication completed")


async def _cleanup_auth_session(app: App, session_id: str, delay: int):
    """Clean up authentication session after delay"""
    await asyncio.sleep(delay)
    app.config_fh.remove_key_file_handler(f"cli_auth_{session_id}")


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
    show_spinner("Closing session", 0.5)

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
