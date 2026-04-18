# toolboxv2/mods/CloudM/extras.py
"""
CloudM Extra Functions with Clerk Integration
Provides utility functions, UI registration, and initialization
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote
from typing import Optional

from toolboxv2 import TBEF, App, Code, Result, Style, get_app

Name = 'CloudM'
version = '0.1.0'
export = get_app(f"{Name}.EXPORT").tb
no_test = export(mod_name=Name, test=False, version=version)
test_only = export(mod_name=Name, test=True, version=version, test_only=True)
to_api = export(mod_name=Name, api=True, version=version)


# =================== UI Management ===================

@no_test
def add_ui(app: App, name: str, title: str, path: str, description: str, auth: bool = False, icon: str = "apps", bg_img_url: Optional[str] = None):
    """
    Register a UI component in the CloudM UI registry.

    Args:
        app: Application instance
        name: Unique name for the UI
        title: Display title
        path: API path to load the UI
        description: Description of the UI
        auth: Whether authentication is required
    """
    if app is None:
        app = get_app("add_ui")

    uis = json.loads(app.config_fh.get_file_handler("CloudM::UI", "{}"))
    print(f"ADDING UI: {name}")
    uis[name] = {
        "auth": auth,
        "path": path,
        "title": title,
        "description": description,
        "icon": icon,
        "bg_img_url": bg_img_url
    }
    app.config_fh.add_to_save_file_handler("CloudM::UI", json.dumps(uis))


@export(mod_name=Name, api=True, version=version)
def openui(app: App=None):
    """Get all registered UIs"""
    if app is None:
        app = get_app("openui")

    x = app.config_fh.get_file_handler("CloudM::UI", "{}")
    uis = json.loads(x)
    return [uis[name] for name in uis]


@export(mod_name=Name, api=True, version=version)
def openVersion(self):
    """Return module version"""
    return self.version


# =================== Module Management ===================

@no_test
def new_module(self, mod_name: str, *options):
    """
    Create a new module from boilerplate.

    Args:
        mod_name: Name of the new module
        *options: Additional options (-fh for FileHandler, -func for functional style)
    """
    self.logger.info(f"Creating new module: {mod_name}")

    boilerplate = '''import logging
from toolboxv2 import MainTool, FileHandler, App, Style


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.1"
        self.name = "NAME"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "WHITE"
        # ~ self.keys = {}
        self.tools = {
            "all": [["Version", "Shows current Version"]],
            "name": "NAME",
            "Version": self.show_version,
        }
        # ~ FileHandler.__init__(self, "File name", app.id if app else __name__, keys=self.keys, defaults={})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                        name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)

    def on_start(self):
        self.logger.info(f"Starting NAME")
        # ~ self.load_file_handler()

    def on_exit(self):
        self.logger.info(f"Closing NAME")
        # ~ self.save_file_handler()

'''

    helper_functions_class = '''
    def show_version(self):
        self.print("Version: ", self.version)
        return self.version
'''

    helper_functions_func = '''
def get_tool(app: App):
    return app.AC_MOD


def show_version(_, app: App):
    welcome_f: Tools = get_tool(app)
    welcome_f.print(f"Version: {welcome_f.version}")
    return welcome_f.version

'''

    self.logger.info("Creating boilerplate")

    if '-fh' in options:
        boilerplate = boilerplate.replace('pass', '').replace('# ~ ', '')
        self.logger.info("Adding FileHandler")

    if '-func' in options:
        boilerplate += helper_functions_func
        self.logger.info("Adding functional based")
    else:
        boilerplate += helper_functions_class
        self.logger.info("Adding class based")

    if os.path.exists(f"mods/{mod_name}.py") or os.path.exists(f"mods_dev/{mod_name}.py"):
        self.print(Style.Bold(Style.RED("MODULE exists, please use another name")))
        return False

    fle = Path(f"mods_dev/{mod_name}.py")
    fle.touch(exist_ok=True)

    with open(f"mods_dev/{mod_name}.py", "wb") as mod_file:
        mod_file.write(bytes(boilerplate.replace('NAME', mod_name), 'ISO-8859-1'))

    self.print("Successfully created new module")
    return True


# =================== Account Functions ===================

@no_test
def create_account(self):
    """Open signup page in browser"""
    version_command = self.app.config_fh.get_file_handler("provider::")
    url = "https://simplecore.app/web/assets/signup.html"

    if version_command is not None:
        url = version_command + "/web/assets/signup.html"

    try:
        import webbrowser
        webbrowser.open(url, new=0, autoraise=True)
    except Exception as e:
        os.system(f"start {url}")
        self.logger.error(Style.YELLOW(str(e)))
        return False
    return True


# =================== Git & Update Functions ===================

@no_test
def init_git(_):
    """Initialize git repository"""
    os.system("git init")
    os.system("git remote add origin https://github.com/MarkinHaus/ToolBoxV2.git")
    print("Stashing changes...")
    os.system("git stash")
    print("Pulling latest changes...")
    os.system("git pull origin master")
    print("Applying stashed changes...")
    os.system("git stash pop")


import os
import subprocess
import sys
from pathlib import Path


@no_test
def update_core(self, backup=False, name=""):
    """Update ToolBox core"""
    from toolboxv2 import tb_root_dir

    # Sicherstellen, dass tb_home_dir ein Path-Objekt ist
    tb_home_dir = Path(tb_root_dir).parent

    def is_git_installed():
        try:
            # cwd hier eigentlich egal, aber zur Sicherheit drin
            subprocess.run(['git', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def is_git_repository():
        return (tb_home_dir / '.git').is_dir()

    if is_git_installed() and is_git_repository():
        self.update_core_git(backup, name, tb_home_dir)
    else:
        self.update_core_pip(tb_home_dir)


def update_core_pip(self, tb_home_dir):
    """Update via pip"""
    self.print("Updating via pip...")
    try:
        # sys.executable stellt sicher, dass das gleiche Python-Env genutzt wird
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "ToolBoxV2"],
            check=True,
            cwd=str(tb_home_dir)
        )
        self.app.print_ok()
    except subprocess.CalledProcessError as e:
        print(f"Error updating via pip: {e}")


def update_core_git(self, backup=False, name="base", tb_home_dir=None):
    """Update via git"""
    self.print("Updating via git...")

    def run_git(args):
        """Helper to run git commands in the correct directory"""
        return subprocess.run(
            ['git'] + args,
            cwd=str(tb_home_dir),
            capture_output=True,
            text=True
        )

    try:
        if backup:
            # Fetch all updates
            run_git(["fetch", "--all"])

            # Create backup branch
            branch_name = f"backup-master-{self.app.id}-{self.version}-{name}"
            run_git(["branch", branch_name])

            # Reset to origin/master
            run_git(["reset", "--hard", "origin/master"])

        # Der eigentliche Pull
        result = run_git(["pull"])

        if result.returncode == 0:
            self.app.remove_all_modules()
            self.app.print_ok()
            if result.stdout:
                print(result.stdout.strip())
        else:
            print(f"Error updating: {result.stderr}")

    except Exception as e:
        print(f"An unexpected error occurred during git update: {e}")


# =================== User Registration (Clerk) ===================

@no_test
async def register_initial_loot_user(app: App, email: str = None, user_name: str = "loot"):
    """
    Register initial admin user.
    With Clerk, this guides user to web registration.

    Args:
        app: Application instance
        email: User email (optional, prompts if not provided)
        user_name: Username for the admin account

    Returns:
        Result with registration URL or instructions
    """
    # Check if custom auth is configured
    auth_configured = bool(os.getenv('TB_JWT_SECRET') or os.getenv('DISCORD_CLIENT_ID'))

    if auth_configured:
        # Custom auth is configured - direct to web registration
        base_url = os.getenv('APP_BASE_URL', 'http://localhost:8080')
        signup_url = f"{base_url}/web/assets/signup.html"

        print("\n" + "=" * 60)
        print("  Custom Authentication Configured")
        print("=" * 60)
        print(f"\nPlease register your admin account via the web interface:")
        print(f"\n  {signup_url}")
        print("\nAfter registration, use 'tb login' for CLI access.")
        print("=" * 60 + "\n")

        # Try to show QR code
        try:
            from ...utils.extras.qr import print_qrcode_to_console
            print_qrcode_to_console(signup_url)
        except:
            pass

        return Result.ok(signup_url)

    return Result.default_user_error("Auth not configured")


@no_test
def create_magic_log_in(app: App, username: str):
    """
    Create magic login URL for a user.
    Note: With custom auth, this is replaced by email code verification.
    """
    # Check if custom auth is configured
    if os.getenv('TB_JWT_SECRET') or os.getenv('DISCORD_CLIENT_ID'):
        print("\nWith custom auth, magic links are replaced by email code verification.")
        print("Use 'tb login' and enter your email to receive a verification code.\n")
        return Result.ok("Use 'tb login' for email verification")

    # Legacy flow
    user = app.run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=username)

    if not hasattr(user, 'user_pass_sync'):
        return Result.default_internal_error("Invalid user or db connection")

    key = "01#" + Code.one_way_hash(user.user_pass_sync, "CM", "get_magic_link_email")
    url = f"{os.getenv('APP_BASE_URL', 'http://localhost:8080')}/web/assets/m_log_in.html?key={quote(key)}&name={user.name}"

    try:
        from ...utils.extras.qr import print_qrcode_to_console
        print_qrcode_to_console(url)
    except:
        pass

    return url


# =================== Database Functions ===================

@no_test
def clear_db(self, do_root=False):
    """Clear the database (use with caution!)"""
    db = self.app.get_mod('DB', spec=self.spec)

    if db.data_base is None or not db:
        self.print("No database instance available")
        return "Please connect to a database first"

    if not do_root:
        if 'y' not in input(Style.RED("Are you sure? The DB will be cleared. Type 'y' to confirm: ")):
            return "Cancelled"

    db.delete('*', matching=True)

    i = 0
    for _ in db.get('all').get(default=[]):
        print(_)
        i += 1

    if i != 0:
        self.print("Database not fully cleared")
        return f"{i} entries remaining"

    return True


# =================== Version & Status ===================


@to_api
def show_version(self):
    """Show module version"""
    self.print(f"Version: {self.version} {self.api_version}")
    return self.version


LEN_FUNCTIONS = [0, None]
@to_api
def docs(app=None):
    """Show APP api documentation"""
    if app is None:
        app = get_app()
    if len(app.functions) != LEN_FUNCTIONS[0]:
        LEN_FUNCTIONS[0] = len(app.functions)
        LEN_FUNCTIONS[1] = app.generate_openapi_html()
    return LEN_FUNCTIONS[1]

@export(mod_name=Name, version=version, state=False, request_as_kwarg=True)
async def get_eco(app=None, request=None):
    """Debug endpoint - returns request info"""
    return str(request)

# =================== Initialization ===================

@export(mod_name=Name, version=version, initial=True)
def initialize_admin_panel(app: App):
    """
    Initialize the CloudM admin panel.
    Registers UI components and sets up initial configuration.
    """
    if app is None:
        app = get_app()

    app.logger.info(f"Admin Panel ({Name} v{version}) initialized.")

    # Register main dashboard UI
    app.run_any(
        ("CloudM", "add_ui"),
        name="UserDashboard",
        title=Name,
        path="/api/CloudM.UI.widget/get_widget",
        description="main",
        auth=True
    )

    # Check auth configuration
    auth_configured = bool(os.getenv('TB_JWT_SECRET') or os.getenv('DISCORD_CLIENT_ID'))

    return Result.ok(
        info="Admin Panel Online",
        data={
            "auth_enabled": auth_configured,
            "version": version
        }
    ).set_origin("CloudM.initialize_admin_panel")

def cleanup_dashboard_api(app: App):
    """Entfernt UIs beim Entladen des Moduls."""
    app.run_any(("CloudM", "remove_ui"), name="UserDashboard")
