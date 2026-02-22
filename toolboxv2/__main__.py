"""Console script for toolboxv2."""

import argparse
import asyncio
import pprint

# Import default Pages
import textwrap
import time
from functools import wraps
from platform import node, system
from typing import List, Tuple

from dotenv import load_dotenv

from toolboxv2 import tb_root_dir
from toolboxv2.utils.system.feature_manager import FeatureManager

# from sqlalchemy.testing.suite.test_reflection.metadata
from toolboxv2.flows import flows_dict as flows_dict_func
from toolboxv2.setup_helper import run_command
from toolboxv2.tests.a_util import async_test
from toolboxv2.utils import get_app
from toolboxv2.utils.clis.cli_worker_manager import main as cli_worker_manager
from toolboxv2.utils.clis.db_cli_manager import cli_db_runner
from toolboxv2.utils.clis.minio_user_manager import main as minio_user_manager_main
from toolboxv2.utils.clis.tb_lang_cli import cli_tbx_main
from toolboxv2.utils.clis.tcm_p2p_cli import cli_tcm_runner

# TEMPORARY FIX: Disabled user_dashboard import due to pywintypes dependency issue
from toolboxv2.utils.clis.user_dashboard import interactive_user_dashboard
from toolboxv2.utils.daemon import DaemonApp
from toolboxv2.utils.extras.Style import Spinner, Style
from toolboxv2.utils.proxy import ProxyApp
from toolboxv2.utils.system import CallingObject, get_state_from_app
from toolboxv2.utils.system.getting_and_closing_app import a_get_proxy_app
from toolboxv2.utils.system.main_tool import MainTool, get_version_from_pyproject
from toolboxv2.utils.workers import cli_event, cli_http_worker, cli_session, cli_ws_worker

from .utils.toolbox import App as TbApp

try:
    from .mcp_server.__main__ import main as cli_mcp_server
except ImportError:
    cli_mcp_server = lambda: None

load_dotenv()

import os
import sys

# Set UTF-8 encoding for Windows console (place at top of your script)
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")  # Change console to UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

DEFAULT_MODI = "cli"

_hook = [None]

try:
    import hmr

    HOT_RELOADER = True
except ImportError:
    HOT_RELOADER = False

try:
    import cProfile
    import io
    import pstats

    def profile_execute_all_functions(app=None, m_query="", f_query=""):
        # Erstellen Sie eine Instanz Ihrer Klasse
        instance = app if app is not None else get_app(from_="Profiler")

        # Erstellen eines Profilers
        profiler = cProfile.Profile()

        def timeit(func_):
            @wraps(func_)
            def timeit_wrapper(*args, **kwargs):
                profiler.enable()
                start_time = time.perf_counter()
                result = func_(*args, **kwargs)
                end_time = time.perf_counter()
                profiler.disable()
                total_time_ = end_time - start_time
                print(
                    f"Function {func_.__name__}{args} {kwargs} Took {total_time_:.4f} seconds"
                )
                return result

            return timeit_wrapper

        items = list(instance.functions.items()).copy()
        for module_name, functions in items:
            if not module_name.startswith(m_query):
                continue
            f_items = list(functions.items()).copy()
            for function_name, function_data in f_items:
                if not isinstance(function_data, dict):
                    continue
                if not function_name.startswith(f_query):
                    continue
                test: list = function_data.get("do_test")
                print(test, module_name, function_name, function_data)
                if test is False:
                    continue
                instance.functions[module_name][function_name]["func"] = timeit(
                    function_data.get("func")
                )

                # Starten des Profilers und Ausführen der Funktion
        instance.execute_all_functions(m_query=m_query, f_query=f_query)

        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        print("\n================================" * 12)
        s = io.StringIO()
        sortby = (
            "time"  # Sortierung nach der Gesamtzeit, die in jeder Funktion verbracht wird
        )

        # Erstellen eines pstats-Objekts und Ausgabe der Top-Funktionen
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()

        # Ausgabe der Ergebnisse
        print(s.getvalue())

        # Erstellen eines Streams für die Profilergebnisse

except ImportError:

    def profile_execute_all_functions(*args):
        return print(args)

    raise ValueError("Failed to import function for profiling")

try:
    from toolboxv2.utils.system.tb_logger import (
        edit_log_files,
        loggerNameOfToolboxv2,
        unstyle_log_files,
    )
except ModuleNotFoundError:
    from toolboxv2.utils.system.tb_logger import (
        edit_log_files,
        loggerNameOfToolboxv2,
        unstyle_log_files,
    )

import os
import subprocess


def start(pidname, args, filename):
    caller = args[0]
    args = args[1:]
    args = ["-bgr" if arg == "-bg" else arg for arg in args]

    if caller.endswith("tb"):
        args = ["tb"] + args
    else:
        args = [sys.executable, "-m", "tb"] + args
    if system() == "Windows":
        DETACHED_PROCESS = 0x00000008
        p = subprocess.Popen(args, creationflags=DETACHED_PROCESS)
    else:  # sys.executable, "-m",
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pid = p.pid
    with open(filename, "w", encoding="utf8") as f:
        f.write(str(pid))
    get_app().sprint(f"Service {pidname} started")


def stop(pidfile, pidname):
    try:
        with open(pidfile, encoding="utf8") as f:
            procID = f.readline().strip()
    except OSError:
        get_app().logger.error("Process file does not exist")
        return

    if procID:
        if system() == "Windows":
            subprocess.Popen(["taskkill", "/PID", procID, "/F"])
        else:
            subprocess.Popen(["kill", "-SIGTERM", procID])

        get_app().logger.info(f"Service {pidname} {procID} stopped")
        os.remove(pidfile)


def create_service_file(user, group, working_dir, runner):
    service_content = f"""[Unit]
Description=ToolBoxService
After=network.target

[Service]
Type=oneshot
User={user}
Group={group}
WorkingDirectory={working_dir}
ExecStart=tb --sm

[Install]
WantedBy=multi-user.target
"""
    with open("tb.service", "w", encoding="utf8") as f:
        f.write(service_content)


def init_service():
    user = input("Enter the user name: ")
    group = input("Enter the group name: ")
    runner = "bg"
    if runner_ := input("enter a runner default bg: ").strip():
        runner = runner_
    working_dir = get_app().start_dir

    create_service_file(user, group, working_dir, runner)

    subprocess.run(["sudo", "mv", "tb.service", "/etc/systemd/system/"])
    subprocess.run(["sudo", "systemctl", "daemon-reload"])


def manage_service(action):
    subprocess.run(["sudo", "systemctl", action, "tb.service"])


def show_service_status():
    subprocess.run(["sudo", "systemctl", "status", "tb.service"])


def uninstall_service():
    subprocess.run(["sudo", "systemctl", "disable", "tb.service"])
    subprocess.run(["sudo", "systemctl", "stop", "tb.service"])
    subprocess.run(["sudo", "rm", "/etc/systemd/system/tb.service"])
    subprocess.run(["sudo", "systemctl", "daemon-reload"])


async def setup_service_windows():
    path = "C:/ProgramData/Microsoft/Windows/Start Menu/Programs/Startup"
    print("Select mode:")
    print("1. Init (first-time setup)")
    print("2. Uninstall")
    print("3. Show window")
    print("4. hide window")
    print("0. Exit")

    mode = input("Enter the mode number: ").strip()

    if not os.path.exists(path):
        print("pleas press win + r and enter")
        print("1. for system -> shell:common startup")
        print("2. for user -> shell:startup")
        path = input("Enter the path that opened: ")

    if mode == "1":
        if os.path.exists(path + "/tb_start.bat"):
            os.remove(path + "/tb_start.bat")
        try:
            with open(path + "/tb_start.bat", "a", encoding="utf8") as f:
                command = f"-m tb --sm"
                f.write(f"""{sys.executable} {command}""")
            print(f"Init Service in {path}")
        except PermissionError:
            print("Pleas run as Admin")
    elif mode == "3":
        await get_app().show_console()
    elif mode == "4":
        await get_app().hide_console()
    elif mode == "0":
        pass
    elif mode == "2":
        os.remove(path + "/tb_start.bat")
        print(f"Removed Service from {path}")
    else:
        await setup_service_windows()


def setup_service_linux():
    print("Select mode:")
    print("1. Init (first-time setup)")
    print("2. Start / Stop / Restart")
    print("3. Status")
    print("4. Uninstall")

    print("5. Show window")
    print("6. hide window")

    mode = int(input("Enter the mode number: "))

    if mode == 1:
        init_service()
    elif mode == 2:
        action = input("Enter 'start', 'stop', or 'restart': ")
        manage_service(action)
    elif mode == 3:
        show_service_status()
    elif mode == 4:
        uninstall_service()
    elif mode == 5:
        get_app().show_console()
    elif mode == 6:
        get_app().hide_console()
    else:
        print("Invalid mode")


# =================== Constants ===================

RUNNER_KEYS = [
    "venv",
    "ipy",
    "db",
    "gui",
    "p2p",
    "default",
    "status",
    "browser",
    "mcp",
    "login",
    "logout",
    "run",
    "mods",
    "flow",
    "user",
    "workers",
    "session",
    "event",
    "broker",
    "http_worker",
    "ws_worker",
    "services",
    "registry",
    "manifest",
    "llm-gateway",
    "docksh",
    "docker-image"
]

DEFAULT_MODI = "cli"


# =================== Helper Functions ===================


def split_args_by_runner(
    args: List[str], runner_keys: List[str]
) -> Tuple[List[str], str, List[str]]:
    """Split arguments into main args, runner name, and runner args."""
    for i, arg in enumerate(args):
        if arg in runner_keys:
            return args[:i], arg, args[i + 1 :]
    return args, None, []


def parse_kwargs(kwargs_list: List[str]) -> dict:
    """Parse key=value pairs into dictionary."""
    kwargs = {}
    for item in kwargs_list:
        if "=" in item:
            key, value = item.split("=", 1)
            kwargs[key.strip()] = value.strip()
        elif ":" in item:
            key, value = item.split(":", 1)
            kwargs[key.strip()] = value.strip()
    return kwargs


# =================== Modern Help Formatter ===================


class ModernHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Modern help formatter with better spacing and alignment."""

    def __init__(self, prog, indent_increment=2, max_help_position=40, width=None):
        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_action(self, action):
        # Get the default formatting
        result = super()._format_action(action)

        # Add spacing for subcommands
        if action.nargs == 0 and action.option_strings:
            return result

        return result


# =================== Guide System ===================


def show_interactive_guide():
    """Show interactive guide with examples and tips."""
    guide = textwrap.dedent("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                     🧰 ToolBoxV2 - Interactive Guide 🧰                    ║
    ╚════════════════════════════════════════════════════════════════════════════╝

    ┌─ QUICK START ──────────────────────────────────────────────────────────────┐
    │                                                                            │
    │  First Time Setup:                                                         │
    │    $ tb -init main                # Initialize ToolBoxV2                   │
    │    $ tb -init config              # Initialize ToolBoxV2 manifest          │
    │    $ tb -c helper init_system     # Setup system configuration             │
    │    $ tb -u main                   # Update to the latest version           │
    │                                                                            │
    │  Start ToolBoxV2:                                                          │
    │    $ tb                           # Start in CLI mode                      │
    │    $ tb --sm                      # Start all Services from config         │
    │    $ tb gui                       # Start with GUI                         │
    │    $ tb workers start             # Start API server                       │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌─ EXTENSION COMMANDS ───────────────────────────────────────────────────────┐
    │                                                                            │
    │  🔐 Authentication:                                                        │
    │    $ tb login                     # Login to ToolBoxV2                     │
    │    $ tb logout                    # Logout from ToolBoxV2                  │
    │    $ tb status                    # Check system status                    │
    │                                                                            │
    │  📦 Module Management:                                                     │
    │    $ tb mods                      # Open module manager (interactive)      │
    │    $ tb -i [MODULE]               # Install module                         │
    │    $ tb -u [MODULE]               # Update module                          │
    │    $ tb -r [MODULE]               # Remove module                          │
    │                                                                            │
    │  🌐 Services & Workers:                                                    │
    │    $ tb workers [start|stop|status]   # Manage worker system               │
    │    $ tb session [generate-secret|test]# Session management                 │
    │    $ tb event                         # Event broker management            │
    │    $ tb broker                        # ZMQ event broker                   │
    │    $ tb http_worker                   # HTTP worker                        │
    │    $ tb ws_worker                     # WebSocket worker                   │
    │    $ tb services                      # Service manager                    │
    │    $ tb manifest                      # Manifest configuration             │
    │                                                                            │
    │  🖥️  Interfaces:                                                           │
    │    $ tb gui                           # Launch GUI interface               │
    │    $ tb mcp                           # Start MCP server (for agents)      │
    │    $ tb p2p [start|stop]              # Manage P2P client                  │
    │                                                                            │
    │  🗄️  Database:                                                             │
    │    $ tb db [command]              # Manage DB                              │
    │                                                                            │
    │  🌍 Browser Extension:                                                     │
    │    $ tb browser build             # Build browser extension                │
    │    $ tb browser install           # Install extension                      │
    │                                                                            │
    │  📦 Virtual Environment:                                                   │
    │    $ tb venv [command]           # Run venv commands                       │
    │                                                                            │
    │  ▶️  Flow Execution:                                                       │
    │    $ tb run              # ToolBox TBX Lang                                │
    │    $ tb -m [flow name]   # Run Flow from default flow folder               │
    │    $ tb flow --flow [file] # Execute flows from file or --remote + .gist   │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌─ COMMAND EXECUTION ────────────────────────────────────────────────────────┐
    │                                                                            │
    │  Basic Syntax:                                                             │
    │    $ tb -c [MODULE] [FUNCTION] [ARGS...]                                   │
    │                                                                            │
    │  Show Module Info:                                                         │
    │    $ tb -c                        # List all functions and idele           │
    │    $ tb -c helper                 # List all helper functions              │
    │    $ tb -c CloudM                 # List all CloudM functions              │
    │                                                                            │
    │  Execute Functions:                                                        │
    │    $ tb -c CloudM Version         # Get CloudM version                     │
    │    $ tb -c CloudM get_mod_snapshot CloudM                                  │
    │    $ tb -c helper create-user john john@example.com                        │
    │                                                                            │
    │  With Kwargs:                                                              │
    │    $ tb -c CloudM get_mod_snapshot --kwargs mod_name=CloudM                │
    │    $ tb -c MyMod my_func --kwargs key1:value1 key2:value2                  │
    │                                                                            │
    │  Multiple Commands:                                                        │
    │    $ tb -c CloudM Version -c CloudM get_mod_snapshot CloudM                │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌─ ACCOUNT MANAGEMENT ───────────────────────────────────────────────────────┐
    │                                                                            │
    │  $ tb -c helper init_system                # Initialize system             │
    │  $ tb -c helper create-user USER EMAIL     # Create new user               │
    │  $ tb -c helper delete-user USER           # Delete user                   │
    │  $ tb -c helper list-users                 # List all users                │
    │  $ tb -c helper create-invitation USER     # Create invitation             │
    │  $ tb -c helper send-magic-link USER       # Send magic login link         │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌─ IPYTHON INTEGRATION ──────────────────────────────────────────────────────┐
    │                                                                            │
    │  Start IPython Shell:                                                      │
    │    $ tb --ipy                     # Enter IPython toolbox shell            │
    │                                                                            │
    │  IPython Magic Commands:                                                   │
    │    In [1]: tb save NAME           # Save current session                   │
    │    In [2]: tb inject NAME         # Inject session into file               │
    │    In [3]: tb loadx NAME          # Load & run session                     │
    │    In [4]: tb loade NAME          # Reload session                         │
    │    In [5]: tb open NAME           # Open in Jupyter                        │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌─ ADVANCED USAGE ───────────────────────────────────────────────────────────┐
    │                                                                            │
    │  Docker Mode:                                                              │
    │    $ tb --docker -m [test|live|dev] -p 8000                                │
    │    $ tb --build                   # Build Docker image                     │
    │                                                                            │
    │  Remote Mode:                                                              │
    │    $ tb --remote -w 0.0.0.0 -p 5000                                        │
    │                                                                            │
    │  Debug Mode:                                                               │
    │    $ tb --debug --sysPrint        # Enable verbose output                  │
    │                                                                            │
    │  Instance Management:                                                      │
    │    $ tb -n myinstance             # Start with custom name                 │
    │    $ tb --kill                    # Kill running instance                  │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌─ SERVICE MANAGEMENT ───────────────────────────────────────────────────────┐
    │                                                                            │
    │  Service Manager (Windows):                                                │
    │    $ tb --sm                      # Manage auto-start/restart              │
    │                                                                            │
    │  Log Manager:                                                              │
    │    $ tb --lm                      # Manage log files                       │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌─ DATA OPERATIONS (⚠️  DANGER ZONE) ────────────────────────────────────────┐
    │                                                                            │
    │  ⚠️  WARNING: These operations cause DATA LOSS!                            │
    │                                                                            │
    │    $ tb --delete-config NAME      # Delete specific config                 │
    │    $ tb --delete-data NAME        # Delete specific data                   │
    │    $ tb --delete-config-all       # Delete ALL configs (!)                 │
    │    $ tb --delete-data-all         # Delete ALL data (!)                    │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌─ DEVELOPMENT & TESTING ────────────────────────────────────────────────────┐
    │                                                                            │
    │  $ tb --test                      # Run test suite                         │
    │  $ tb --profiler                  # Profile functions                      │
    │  $ tb -l                          # Load all modules                       │
    │  $ tb -sfe -l                     # Generate function enums                │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌─ PRACTICAL EXAMPLES ───────────────────────────────────────────────────────┐
    │                                                                            │
    │  Install and run module:                                                   │
    │    $ tb -i MyModule                                                        │
    │    $ tb -c MyModule my_function                                            │
    │                                                                            │
    │  Run in Docker with GUI:                                                   │
    │    $ tb --docker -m dev gui -p 8000 -w 0.0.0.0                             │
    │                                                                            │
    │  Check system status:                                                      │
    │    $ tb status                    # Shows DB, API, P2P status              │
    │                                                                            │
    │  Interactive module management:                                            │
    │    $ tb mods                      # Opens interactive manager              │
    │                                                                            │
    │  Create user and send magic link:                                          │
    │    $ tb -c helper create-user alice alice@mail.com                         │
    │    $ tb -c helper send-magic-link alice                                    │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌─ TIPS & TRICKS ────────────────────────────────────────────────────────────┐
    │                                                                            │
    │  • Use `tb [command] -h` for detailed help on any command                  │
    │  • Most commands support tab completion in modern shells                   │
    │  • Use `--sysPrint` for verbose output when debugging                      │
    │  • Runner commands can be combined: `tb workers start -bg`                 │
    │  • Use `-n` to run multiple instances with different names                 │
    │  • Module functions are auto-discovered when using `-l`                    │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ┌─ GETTING HELP ─────────────────────────────────────────────────────────────┐
    │                                                                            │
    │  General Help:                                                             │
    │    $ tb -h                        # Show main help                         │
    │    $ tb --guide                   # Show this guide                        │
    │                                                                            │
    │  Command-Specific Help:                                                    │
    │    $ tb workers -h                # API command help                       │
    │    $ tb venv -h                   # Conda command help                     │
    │    $ tb db -h                     # Database command help                  │
    │                                                                            │
    │  Module Information:                                                       │
    │    $ tb -c [MODULE]               # List module functions                  │
    │    $ tb -v -l                     # Show all module versions               │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    ╔════════════════════════════════════════════════════════════════════════════╗
    ║  For more information, visit: https://markinhaus.github.io/ToolBoxV2/      ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)

    print(guide)


# =================== Modern Parser ===================


def parse_args():
    """Create modern argument parser with improved help text."""

    # Handle runner commands early
    runner_keys = RUNNER_KEYS
    main_args, runner_name, runner_args = split_args_by_runner(sys.argv[1:], runner_keys)

    # Temporarily adjust sys.argv for main parsing
    if runner_name:
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]] + main_args

    # Create parser with modern help
    parser = argparse.ArgumentParser(
        prog="tb",
        description=textwrap.dedent("""
        ╔════════════════════════════════════════════════════════════════════════╗
        ║                   🧰 ToolBoxV2 - CLI Interface 🧰                        ║
        ╚════════════════════════════════════════════════════════════════════════╝

        A powerful, modular Python framework for building and managing tools.

        Quick Start:
          $ tb                    # Start CLI interface
          $ tb gui                # Launch GUI
          $ tb --guide            # Show guide
          $ tb -c [MOD] [FUNC]    # Execute module function

        """),
        epilog=textwrap.dedent("""
        ┌─ EXAMPLES ─────────────────────────────────────────────────────────────┐
        │                                                                        │
        │  Basic Usage:                                                          │
        │    $ tb gui                              # Start GUI                   │
        │    $ tb workkers start                   # Start API server            │
        │    $ tb status                           # Check status                │
        │    $ tb --ipy                            # IPython shell               │
        │                                                                        │
        │  Module Commands:                                                      │
        │    $ tb -c helper                        # List helper functions       │
        │    $ tb -c CloudM Version                # Get version                 │
        │    $ tb -c CloudM get_mod_snapshot CloudM                              │
        │                                                                        │
        │  Advanced:                                                             │
        │    $ tb --docker -m dev -p 8000          # Docker mode                 │
        │    $ tb -c helper create-user bob bob@mail.com                         │
        │    $ tb -c MyMod func --kwargs key=val   # With kwargs                 │
        │                                                                        │
        └────────────────────────────────────────────────────────────────────────┘

        For detailed guide: tb --guide
        For command help: tb [command] -h
        """),
        formatter_class=ModernHelpFormatter,
        add_help=False,  # We'll add custom help
    )

    # =================== EXTENSION COMMANDS ===================
    extensions = parser.add_argument_group("🚀 Extension Commands")

    extensions.add_argument(
        "gui",
        help="Launch graphical user interface",
        nargs="?",
        const=True,
        default=False,
    )

    extensions.add_argument(
        "mods",
        help="Open interactive module manager",
        nargs="?",
        const=True,
        default=False,
    )

    # extensions.add_argument("p2p",
    #                         help="Manage P2P client (use: tb p2p -h)",
    #                         nargs='?',
    #                         const=True,
    #                         default=False)

    extensions.add_argument(
        "db",
        help="Database commands (use: tb db -h)",
        nargs="?",
        const=True,
        default=False,
    )

    extensions.add_argument(
        "venv",
        help="Conda environment commands (use: tb venv -h)",
        nargs="?",
        const=True,
        default=False,
    )

    extensions.add_argument(
        "mcp", help="Start MCP server for agents", nargs="?", const=True, default=False
    )

    extensions.add_argument(
        "browser",
        help="Browser extension installer",
        nargs="?",
        const=True,
        default=False,
    )

    extensions.add_argument(
        "flow",
        help="Execute flows/mod from file or directory",
        nargs="?",
        const=True,
        default=False,
    )

    extensions.add_argument(
        "run", help="Execute .tbx file and setup", nargs="?", const=True, default=False
    )

    extensions.add_argument(
        "login", help="Login to ToolBoxV2", nargs="?", const=True, default=False
    )

    extensions.add_argument(
        "logout", help="Logout from ToolBoxV2", nargs="?", const=True, default=False
    )

    extensions.add_argument(
        "status",
        help="Display system status (DB, API, P2P, Workers)",
        nargs="?",
        const=True,
        default=False,
    )
    extensions.add_argument(
        "user", help="User management", nargs="?", const=True, default=False
    )
    extensions.add_argument(
        "workers",
        help="Worker management (use: tb workers -h)",
        nargs="?",
        const=True,
        default=False,
    )
    extensions.add_argument(
        "session",
        help="Session management for workers (use: tb session -h)",
        nargs="?",
        const=True,
        default=False,
    )
    extensions.add_argument(
        "broker",
        help="ZMQ event broker (use: tb broker -h)",
        nargs="?",
        const=True,
        default=False,
    )
    extensions.add_argument(
        "http_worker", help="HTTP worker", nargs="?", const=True, default=False
    )
    extensions.add_argument(
        "ws_worker", help="WebSocket worker", nargs="?", const=True, default=False
    )
    extensions.add_argument(
        "docksh", help="Start docker cli", nargs="?", const=True, default=False
    )

    # =================== CORE OPTIONS ===================
    core = parser.add_argument_group("⚙️  Core Options")

    core.add_argument("-h", "--help", action="store_true", help="Show this help message")

    core.add_argument("--guide", action="store_true", help="Show interactive usage guide")

    core.add_argument(
        "-v",
        "--get-version",
        action="store_true",
        help="Display ToolBox version and modules (use with -l for all)",
    )

    core.add_argument(
        "-init",
        type=str,
        metavar="TYPE",
        help="Initialize ToolBoxV2 [main|config|manifest]",
    )

    core.add_argument(
        "-l",
        "--load-all-mod-in-files",
        action="store_true",
        help="Load all modules from mod directory",
    )

    core.add_argument(
        "-c",
        "--command",
        nargs="*",
        action="append",
        metavar=("MODULE", "FUNCTION"),
        help="Execute module command: -c MODULE FUNCTION [ARGS...]",
    )

    core.add_argument(
        "--ipy",
        action="store_true",
        help="Enter IPython toolbox shell with magic commands",
    )

    # =================== MODULE MANAGEMENT ===================
    modules = parser.add_argument_group("📦 Module Management")

    modules.add_argument(
        "-i",
        "--install",
        type=str,
        metavar="MODULE",
        help="Install module or interface by name",
    )

    modules.add_argument(
        "-u",
        "--update",
        type=str,
        metavar="MODULE",
        help="Update module or interface by name",
    )

    modules.add_argument(
        "-r",
        "--remove",
        type=str,
        metavar="MODULE",
        help="Uninstall module or interface by name",
    )

    modules.add_argument(
        "-m",
        "--modi",
        type=str,
        metavar="MODE",
        default=DEFAULT_MODI,
        help=f"Interface mode (default: {DEFAULT_MODI})",
    )

    # =================== RUNTIME CONTROL ===================
    runtime = parser.add_argument_group("🎮 Runtime Control")

    runtime.add_argument(
        "--kill", action="store_true", help="Terminate running ToolBox instance"
    )

    runtime.add_argument(
        "-bg",
        "--background-application",
        action="store_true",
        help="Run interface in background mode",
    )

    runtime.add_argument(
        "-bgr",
        "--background-application-runner",
        action="store_true",
        help="Background runner flag for current process",
    )

    runtime.add_argument(
        "-fg",
        "--live-application",
        action="store_true",
        help="Run proxy interface in foreground",
    )

    runtime.add_argument(
        "--remote", action="store_true", help="Enable remote access mode"
    )

    runtime.add_argument(
        "--debug", action="store_true", help="Enable debug mode with hot-reload"
    )

    # =================== DOCKER ===================
    docker = parser.add_argument_group("🐳 Docker Options")

    docker.add_argument(
        "--docker", action="store_true", help="Run in Docker container [test|live|dev]"
    )

    docker.add_argument(
        "--build", action="store_true", help="Build Docker image from local source"
    )

    # =================== NETWORKING ===================
    network = parser.add_argument_group("🌐 Network Configuration")

    network.add_argument(
        "-n",
        "--name",
        type=str,
        metavar="ID",
        default="main",
        help="Instance identifier (default: main)",
    )

    network.add_argument(
        "-p",
        "--port",
        type=int,
        metavar="PORT",
        default=5000,
        help="Interface port number (default: 5000)",
    )

    network.add_argument(
        "-w",
        "--host",
        type=str,
        metavar="HOST",
        default="0.0.0.0",
        help="Interface host address (default: 0.0.0.0)",
    )

    # =================== SERVICE MANAGEMENT ===================
    services = parser.add_argument_group("🔧 Service Management")

    services.add_argument(
        "--init-sm",
        action="store_true",
        help=f"Service Manager for '{system()}' (auto-start/restart)",
    )

    services.add_argument(
        "--sm",
        action="store_true",
        help=f"Service Manager for '{system()}' (auto-start/restart)",
    )

    services.add_argument(
        "--lm", action="store_true", help="Log Manager (view/remove/edit logs)"
    )

    # =================== DATA OPERATIONS ===================
    data_ops = parser.add_argument_group("💾 Data Operations (⚠️  Use with caution!)")

    data_ops.add_argument(
        "--delete-config", action="store_true", help="⚠️  Delete named config folder"
    )

    data_ops.add_argument(
        "--delete-data", action="store_true", help="⚠️  Delete named data folder"
    )

    data_ops.add_argument(
        "--delete-config-all",
        action="store_true",
        help="🚨 DANGER: Delete ALL config files (DATA LOSS!)",
    )

    data_ops.add_argument(
        "--delete-data-all",
        action="store_true",
        help="🚨 DANGER: Delete ALL data folders (DATA LOSS!)",
    )

    # =================== DEVELOPMENT ===================
    dev = parser.add_argument_group("🔬 Development & Testing")

    dev.add_argument("--test", action="store_true", help="Run complete test suite")

    dev.add_argument(
        "--profiler", action="store_true", help="Profile all registered functions"
    )

    dev.add_argument(
        "-sfe",
        "--save-function-enums-in-file",
        action="store_true",
        help="Generate all_function_enums.py (requires -l)",
    )

    dev.add_argument(
        "--sysPrint", action="store_true", help="Enable verbose system output"
    )

    # =================== ADVANCED ===================
    advanced = parser.add_argument_group("🎯 Advanced Options")

    advanced.add_argument(
        "--kwargs",
        nargs="*",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Pass key-value pairs: --kwargs key1=value1 key2=value2",
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle custom help
    if args.help:
        parser.print_help()
        sys.exit(0)

    # Handle guide
    if args.guide:
        show_interactive_guide()
        sys.exit(0)

    # Add runner information
    if runner_name:
        args.runner_name = runner_name
        args.runner_args = runner_args
        sys.argv = original_argv
    else:
        args.runner_name = None
        args.runner_args = []

    # Parse kwargs
    if args.kwargs:
        kwargs_list = args.kwargs.copy()
        args.kwargs = []
        for k in kwargs_list:
            args.kwargs.append(parse_kwargs(k))

    if not args.kwargs or len(args.kwargs) == 0:
        args.kwargs = [{}]

    return args


def edit_logs():
    name = input(f"Name of logger \ndefault {loggerNameOfToolboxv2}\n:")
    name = name if name else loggerNameOfToolboxv2

    def date_in_format(_date):
        ymd = _date.split("-")
        if len(ymd) != 3:
            print("Not enough segments")
            return False
        if len(ymd[1]) != 2:
            print("incorrect format")
            return False
        if len(ymd[2]) != 2:
            print("incorrect format")
            return False

        return True

    def level_in_format(_level):
        if _level in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]:
            _level = [50, 40, 30, 20, 10, 0][
                ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"].index(_level)
            ]
            return True, _level
        try:
            _level = int(_level)
        except ValueError:
            print("incorrect format pleas enter integer 50, 40, 30, 20, 10, 0")
            return False, -1
        return _level in [50, 40, 30, 20, 10, 0], _level

    date = input(
        "Date of log format : YYYY-MM-DD replace M||D with xx for multiple editing\n:"
    )

    while not date_in_format(date):
        date = input("Date of log format : YYYY-MM-DD :")

    level = input(
        f"Level : {list(zip(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'], [50, 40, 30, 20, 10, 0], strict=False))}"
        f" : enter number\n:"
    )

    while not level_in_format(level)[0]:
        level = input("Level : ")

    level = level_in_format(level)[1]

    do = input("Do function : default remove (r) or uncoler (uc)")
    if do == "uc":
        edit_log_files(name=name, date=date, level=level, n=0, do=unstyle_log_files)
    else:
        edit_log_files(name=name, date=date, level=level, n=0)


def run_tests(test_path):
    # Konstruiere den Befehl für den Unittest-Testaufruf
    command = [sys.executable, "-m", "unittest", "discover", "-s", test_path]

    # Führe den Befehl mit subprocess aus
    try:
        result = subprocess.run(command, check=True, encoding="cp850")
        # Überprüfe den Rückgabewert des Prozesses und gib entsprechend True oder False zurück
        if result.returncode != 0:
            return False
    except subprocess.CalledProcessError as e:
        get_app().logger.error(f"Fehler beim Ausführen der Unittests: {e}")
        return False
    except Exception as e:
        get_app().logger.error(f"Fehler beim Ausführen der Unittests:{e}")
        return False

    return True

    # try:
    #     from . import tb_root_dir
    #     command = ["npm", "test", "--prefix", tb_root_dir.as_posix()]
    #     result = subprocess.run(command, check=True, encoding='cp850', cwd=tb_root_dir)
    #     return result.returncode == 0
    # except subprocess.CalledProcessError as e:
    #     print(f"Fehler beim Ausführen der npm-Tests: {e}")
    #     return False
    # except Exception as e:
    #     print(f"Fehler beim Ausführen der npm-Tests:{e}")
    #     return False


async def setup_app(ov_name=None, App=TbApp):
    args = parse_args()
    if ov_name:
        args.name = ov_name

    abspath = os.path.dirname(os.path.abspath(__file__))

    identification = args.name + "-" + node() + "\\"

    data_folder = abspath + "\\.data\\"
    config_folder = abspath + "\\.config\\"
    info_folder = abspath + "\\.info\\"

    os.makedirs(info_folder, exist_ok=True)

    app_config_file = config_folder + identification
    app_data_folder = data_folder + identification

    if args.delete_config_all:
        os.remove(config_folder)
    if args.delete_data_all:
        os.remove(data_folder)
    if args.delete_config:
        os.remove(app_config_file)
    if args.delete_data:
        os.remove(app_data_folder)

    if args.test:
        test_path = os.path.dirname(os.path.abspath(__file__))
        if system() == "Windows":
            test_path = test_path + "\\tests"
        else:
            test_path = test_path + "/tests"
        get_app().logger.info(f"Testing in {test_path}")
        if not run_tests(test_path):
            get_app().logger.error("Error in tests")
            exit(1)
        exit(0)

    abspath = os.path.dirname(os.path.abspath(__file__))
    info_folder = abspath + "\\.info\\pids\\"
    os.makedirs(info_folder, exist_ok=True)
    if not args.sysPrint and not (
        args.debug or args.background_application_runner or args.install or args.kill
    ):
        TbApp.sprint = lambda text, *_args, **kwargs: False
    tb_app = get_app(from_="InitialStartUp", name=args.name, args=args, app_con=TbApp)

    # Initialize FeatureManager with app reference
    features_dir = tb_root_dir / "toolboxv2" / "features"
    feature_manager = FeatureManager(app=tb_app, features_dir=str(features_dir))
    tb_app.feature_manager = feature_manager

    tb_app.loop = asyncio.get_running_loop()

    if args.load_all_mod_in_files:
        _min_info = await tb_app.load_all_mods_in_file()
        with Spinner("Crating State"):
            import threading

            st = threading.Thread(
                target=get_state_from_app,
                args=(
                    tb_app,
                    os.environ.get("TOOLBOXV2_REMOTE_BASE", "https://simplecore.app"),
                    "https://github.com/MarkinHaus/ToolBoxV2/tree/master/toolboxv2/",
                ),
                daemon=True,
            )
        st.start()
        # tb_app.print_functions()
        if _min_info:
            tb_app.print(_min_info)
        tb_app.print(await tb_app.load_external_mods())

    if args.update:
        if args.update == "main":
            await tb_app.save_load("CloudM")
            tb_app.run_any("CloudM", "update_core")
            run_command("npm run build:tbjs && npm run build:web")
        else:
            res = await tb_app.a_run_any(
                "CloudM", "install", module_name=args.update, get_results=True
            )
            res.print()

    if args.background_application_runner:
        from toolboxv2.utils.extras.notification import quick_info

        quick_info(
            "Background Application",
            f"Starting background application {sys.argv}",
            timeout=12000,
        )
        daemon_app = await DaemonApp(
            tb_app, args.host, args.port if args.port != 5000 else 6587, t=False
        )
        tb_app.daemon_app = daemon_app
        args.live_application = False
    elif args.background_application:
        tb_app.sprint("Starting background application", not args.kill)
        if not args.kill:
            if args.background_application_runner:
                tb_app.sprint("Already in background runner mode, not spawning again")
            else:
                tb_app.sprint(f"Spawning background process...")
                start(args.name, sys.argv, filename=f"{info_folder}bg-{args.name}.pid")
                # NEU: Parent-Prozess sollte hier beenden
                tb_app.sprint(f"Background process spawned. Exiting parent.")
                sys.exit(0)
        else:
            if "-m " not in sys.argv:
                pid_file = f"{info_folder}bg-{args.name}.pid"
            try:
                _ = await ProxyApp(
                    tb_app,
                    args.host if args.host != "0.0.0.0" else "localhost",
                    args.port if args.port != 5000 else 6587,
                    timeout=4,
                )
                res = await _.verify()
                if await _.exit_main() != "No data look later":
                    stop(pid_file, args.name)
            except Exception:
                tb_app.sprint("Auto Stopping background application")
                stop(pid_file, args.name)
    elif args.live_application:
        try:
            tb_app = await a_get_proxy_app(
                tb_app,
                host=args.host if args.host != "0.0.0.0" else "localhost",
                port=args.port if args.port != 5000 else 6587,
                key=os.getenv("TB_R_KEY", "user@phfrase"),
            )
        except:
            import traceback

            print(traceback.format_exc())
            tb_app.sprint(
                "No bg instance found starting local, to run in the background use -bg"
            )

    return tb_app, args


async def command_runner(tb_app, command, **kwargs):
    if len(command) < 1:
        tb_app.print_functions()
        tb_app.print(
            "minimum command length is 2 {module_name} {function_name} optional args... Com^C to exit"
        )
        return await tb_app.a_idle()

    tb_app.print(f"Running command: {' '.join(command)} {kwargs}")
    call = CallingObject().empty()
    mod = tb_app.get_mod(command[0], spec="app")
    if hasattr(mod, "async_initialized") and not mod.async_initialized:
        await mod
    call.module_name = command[0]

    if len(command) < 2:
        tb_app.print_functions(command[0])
        tb_app.print(
            "minimum command length is 2 {module_name} {function_name} optional args..."
        )
        return

    call.function_name = command[1]
    call.args = command[2:]
    call.kwargs = kwargs

    if (
        "help" in call.kwargs
        and call.kwargs.get("help", False)
        or "h" in call.kwargs
        and call.kwargs.get("h", False)
    ):
        data = tb_app.get_function((call.module_name, call.function_name), metadata=True)
        pprint.pprint(data)
        return data

    spec = "app"  #  if not args.live_application else tb_app.id
    r = await tb_app.a_run_any(
        (call.module_name, call.function_name),
        tb_run_with_specification=spec,
        args_=call.args,
        get_results=True,
    )
    if asyncio.iscoroutine(r):
        r = await r
    if isinstance(r, asyncio.Task):
        r = await r

    tb_app.print("Running in", spec)
    if hasattr(r, "print"):
        r.print(full_data=True)
    else:
        tb_app.print(r)
    return r


async def main(App=TbApp, do_exit=True):
    """Console script for toolboxv2."""
    tb_app, args = await setup_app(App=App)
    __version__ = get_version_from_pyproject()

    abspath = os.path.dirname(os.path.abspath(__file__))
    info_folder = abspath + "\\.info\\pids\\"
    pid_file = f"{info_folder}{args.modi}-{args.name}.pid"

    async def log_in(app):
        res = await app.session.login()
        app.print(f"Logged in as: {app.session.username}") if res else None
        return res

    tb_app.run_bg_task_advanced(log_in, tb_app)

    if args.install:
        report = await tb_app.a_run_any(
            "CloudM", "install", module_name=args.install, get_results=True
        )
        report.print()

    if args.init == "main":
        from .setup_helper import setup_main

        setup_main()
    elif args.init == "config":
        from .utils.clis.config_wizard import run_config_wizard

        exit_code = run_config_wizard()
        await tb_app.a_exit()
        exit(exit_code)
    elif args.init is not None:
        tb_app.print(
            "No init action specified valid options are ['main', 'config', 'manifest']"
        )
        await tb_app.a_exit()
        exit(1)

    if args.lm:
        edit_logs()
        await tb_app.a_exit()
        exit(0)

    if (
        args.load_all_mod_in_files
        or args.save_function_enums_in_file
        or args.get_version
        or args.profiler
        or args.background_application_runner
        or args.test
    ):
        if args.save_function_enums_in_file:
            tb_app.save_registry_as_enums("utils\\system", "all_functions_enums.py")
            tb_app.alive = False
            await tb_app.a_exit()
            return 0
        if args.get_version:
            print(
                f"\n{' Version ':-^45}\n\n{Style.Bold(Style.CYAN(Style.ITALIC('RE'))) + Style.ITALIC('Simple') + 'ToolBox':<35}:{__version__:^10}\n"
            )
            for mod_name in tb_app.functions:
                if isinstance(tb_app.functions[mod_name].get("app_instance"), MainTool):
                    print(
                        f"{mod_name:^35}:{tb_app.functions[mod_name]['app_instance'].version:^10}"
                    )
                else:
                    try:
                        v = (
                            tb_app.functions[mod_name]
                            .get(list(tb_app.functions[mod_name].keys())[0])
                            .get("version", "unknown (functions only)")
                            .replace(f"{__version__}:", "")
                        )
                    except AttributeError:
                        v = "unknown"
                    print(f"{mod_name:^35}:{v:^10}")
            print("\n")
            tb_app.alive = False
            await tb_app.a_exit()
            return 0

    if _hook[0]:
        if asyncio.iscoroutinefunction(_hook[0]):
            await _hook[0](tb_app)
        elif callable(_hook[0]):
            _hook[0](tb_app)

    if args.profiler:
        profile_execute_all_functions(tb_app)
        tb_app.alive = False
        await tb_app.a_exit()
        return 0

    if (
        not args.kill
        and not args.docker
        and tb_app.alive
        and not args.background_application
        and "-m" in sys.argv
    ):
        tb_app.save_autocompletion_dict()
        if (
            args.background_application_runner
            and args.modi == "bg"
            and hasattr(tb_app, "daemon_app")
        ):
            await tb_app.daemon_app.online

        if args.remote:
            await tb_app.rrun_flows(args.modi, **args.kwargs[0])

        flows_dict = flows_dict_func(remote=False, flows_dict_=tb_app.flows)
        if args.modi not in flows_dict:
            flows_dict = {
                **flows_dict,
                **flows_dict_func(s=args.modi, remote=True, flows_dict_=tb_app.flows),
            }
        tb_app.set_flows(flows_dict)
        if args.modi not in flows_dict:
            print(
                f"Modi : [{args.modi}] not found on device installed modi : {list(flows_dict.keys())}"
            )
            exit(1)
        # open(f"./config/{args.modi}.pid", "w").write(app_pid)
        await tb_app.run_flows(args.modi, **args.kwargs[0])

    elif args.docker:
        flows_dict = flows_dict_func("docker")

        if "docker" not in flows_dict:
            print("No docker")
            return 1

        flows_dict["docker"](tb_app, args)

    elif args.kill and not args.background_application:
        if not os.path.exists(pid_file):
            print("You must first run the mode")
        else:
            try:
                tb_app.cluster_manager.stop_all()
            except Exception as e:
                print(Style.YELLOW(f"Error stopping cluster manager: {e}"))
            try:
                from toolboxv2.utils.clis.cli_worker_manager import WorkerManager
                from toolboxv2.utils.workers.config import load_config

                config = load_config(None)
                WorkerManager(config).stop_all()
            except Exception as e:
                print(Style.YELLOW(f"Error stopping workers manager: {e}"))
            try:
                from toolboxv2.utils.clis.tcm_p2p_cli import handle_stop

                _ = lambda: None
                _.names = None
                handle_stop(_)
            except Exception as e:
                print(Style.YELLOW(f"Error stopping workers manager: {e}"))

            with open(pid_file, encoding="utf8") as f:
                app_pid = f.read()
            print(f"Exit app {app_pid}")
            if system() == "Windows":
                os.system(f"taskkill /pid {app_pid} /F")
            else:
                os.system(f"kill -9 {app_pid}")

    if args.command and not args.background_application:
        for command in args.command:
            await command_runner(
                tb_app,
                command,
                **args.kwargs[
                    args.command.index(command)
                    if args.command.index(command) < len(args.kwargs) - 1
                    else 0
                ],
            )

    if args.live_application and args.debug:
        hide = tb_app.hide_console()
        if hide is not None:
            await hide

    if os.path.exists(pid_file):
        os.remove(pid_file)

    if do_exit and not tb_app.called_exit[0]:
        await tb_app.a_exit()
        return 0
    # print(
    #    f"\n\nPython-loc: {init_args[0]}\nCli-loc: {init_args[1]}\nargs: {tb_app.pretty_print(init_args[2:])}")
    return 0


def install_ipython():
    os.system("pip install ipython prompt_toolkit")


def tb_pre_ipy(app, eo):
    # print(f"In Data:  \n\t{eo.raw_cell}\n\t{eo.store_history}\n\t{eo.silent}\n\t{eo.shell_futures}\n\t{eo.cell_id}")
    # app.print(f"{eo.raw_cell=}{eo.raw_cell.split(' ')=}")
    if eo.raw_cell != "exit":
        eo.raw_cell = ""
    # start information getering


def tb_post_ipy(app, rest):
    # print(f"Out Data:  \n\t{rest.execution_count}\n\t{rest.error_before_exec}\n\t{rest.error_in_exec}
    # \n\t{rest.info.raw_cell}\n\t{rest.info.store_history}\n\t{rest.info.silent}\n\t{rest.info.shell_futures}
    # \n\t{rest.info.cell_id}\n\t{rest.result} ")
    # return information
    return ""


def line_magic_ipy(app, ipython, line):
    app.mod_online(line.split(" ")[0].strip(), True)
    if line.split(" ")[0].strip() in app.functions:
        async_test(command_runner)(app, line.split(" "))
    else:
        app.print_functions()


def configure_ipython(argv):
    from traitlets.config import Config

    c = Config()

    # Autocompletion with prompt_toolkit
    c.InteractiveShellCompleter.use_jedi = True
    c.InteractiveShell.automagic = True
    # Enable contextual help
    c.InteractiveShellApp.exec_lines = []

    c.TerminalInteractiveShell.editor = "nano"

    c.PrefilterManager.multi_line_specials = True

    c.InteractiveShell.colors = "LightBG"
    c.InteractiveShell.confirm_exit = True
    c.TerminalIPythonApp.display_banner = False
    c.AliasManager.user_aliases = [
        ("TB", "tb"),
        ("@", "!tb -c "),
    ]
    c.InteractiveShellApp.exec_lines.append(
        """import os
import sys
import toolboxv2 as tb
from toolboxv2.tests.a_util import async_test
from threading import Thread
# from toolboxv2.utils.system.ipy_completer import get_completer

from IPython.core.magic import register_line_magic, register_cell_magic
sys.argv = """
        + str(argv)
        + """
app, args = await tb.__main__.setup_app()
if hasattr(app, "daemon_app"):
    Thread(target=async_test(app.daemon_app.connect), args=(app,), daemon=True).start()


def pre_run_code_hook(eo):
    tb.__main__.tb_pre_ipy(app, eo)


def post_run_code_hook(result):
    tb.__main__.tb_post_ipy(app, result)


def load_ipython_extension(ipython):
    @register_line_magic
    def my_line_magic(line):
        parts = line.split(' ')
        f_name = "ipy_sessions/"+("tb_session" if len(parts) <= 1 else parts[-1])

        os.makedirs(f'{app.appdata}/ipy_sessions/',exist_ok=True)
        if "save" in parts[0]:
            do_inj = not os.path.exists(f'{app.appdata}/{f_name}.ipy')
            if do_inj:
                ipython.run_line_magic('save', f'{app.appdata}/{f_name}.ipy')
            else:
                ipython.run_line_magic('save', f'-r {app.appdata}/{f_name}.ipy')
        if "inject" in parts[0]:
                file_path = f'{app.appdata}/{f_name}.ipy'
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                # Insert lines after the first line
                lines[1:1] = [line + '\\n' for line in
                              ["import toolboxv2 as tb", "app, args = await tb.__main__.setup_app()"]]
                with open(file_path, 'w') as file:
                    file.writelines(lines)
        elif "loadX" in parts[0]:
            # ipython.run_line_magic('store', '-r')
            ipython.run_line_magic('run', f'{app.appdata}/{f_name}.ipy')
        elif "load" in parts[0]:
            # ipython.run_line_magic('store', '-r')
            ipython.run_line_magic('load', f'{app.appdata}/{f_name}.ipy')
        elif "open" in parts[0]:
            file_path = f'{app.appdata}/{f_name}.ipy'
            if os.path.exists(f'{app.appdata}/{f_name}.ipy'):
                l = "notebook" if not 'lab' in parts else 'labs'
                os.system(f"jupyter {l} {file_path}")
            else:
                print("Pleas save first")
        else:
            tb.__main__.line_magic_ipy(app, ipython, line)

    @register_cell_magic
    def my_cell_magic(line, cell):
        print(f"Custom cell magic {line} |CELL| {cell}")
        line = line + '\\n' + cell
        tb.__main__.line_magic_ipy(app, ipython, line)

    def apt_completers(self, event):
        return ['save', 'loadX', 'load', 'open', 'inject']

    ipython.set_hook('complete_command', apt_completers, re_key = '%tb')

    ipython.register_magic_function(my_line_magic, 'line', 'tb')
    ipython.register_magic_function(my_cell_magic, 'cell', 'tb')


load_ipython_extension(get_ipython())

# get_ipython().set_custom_completer(get_completer())
get_ipython().events.register("pre_run_cell", pre_run_code_hook)
get_ipython().events.register("post_run_cell", post_run_code_hook)

"""
    )
    ()
    return c


def start_ipython_session(argv):
    from IPython import start_ipython

    config = configure_ipython(argv)

    start_ipython(argv=None, config=config)


def runner_setup():
    def helper_gui():
        __import__("toolboxv2.utils.clis.tauri_cli", fromlist=["main"]).main()

    async def status_helper():
        print("🔍 ToolBoxV2 System Status")
        print("═" * 40)

        # Check login status
        try:
            from toolboxv2.utils.system.session import get_app

            app = get_app("status_check")

            # Check if user is logged in
            try:
                from toolboxv2.utils.security.cryp import Code
                from toolboxv2.utils.system.session import BlobFile

                username = app.get_username()

                with BlobFile(
                    f"claim/{username}/jwt.c", key=Code.DK()(), mode="r"
                ) as blob:
                    claim = blob.read()
                    if claim and claim != b"Error decoding":
                        print(f"🔐 Authentication: ✅ Logged in as {username}")

                        # Try to determine server
                        base_url = getattr(app, "base_url", None) or "Unknown"
                        print(f"🌐 Server: {base_url}")
                    else:
                        print("🔐 Authentication: ❌ Not logged in")
            except Exception:
                print("🔐 Authentication: ❌ Not logged in")

            from toolboxv2.mods.CloudM.mini import get_service_status

            str1 = get_service_status(app.info_dir.replace(app.id, ""))

            print(str1)

        except Exception as e:
            print(f"🔐 Authentication: ❌ Status check failed: {e}")

        print()

        # Existing status checks
        sys.argv = ["db", "status"]
        await cli_db_runner()
        print()
        print(Style.GREY("─" * 25))
        sys.argv = ["workers", "status"]
        cli_worker_manager()
        sys.argv = ["p2p", "status"]
        cli_tcm_runner()

    async def cli_web_login():
        """Enhanced CLI web login entry point with modern visual feedback"""
        import argparse

        # Clear screen for clean start
        print("\033[2J\033[H")

        parser = argparse.ArgumentParser(
            prog="tb login",
            description="🔐 Login to ToolBoxV2",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                            Login Options                                   ║
    ╠════════════════════════════════════════════════════════════════════════════╣
    ║                                                                            ║
    ║    $ tb login                                                              ║
    ║    $ tb login --status                                                     ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
            """,
        )

        parser.add_argument(
            "--status",
            help="Force remote login to SimpleCore Hub",
            action="store_true",
            default=False,
        )

        args = parser.parse_args()

        # Visual feedback
        print(
            "╔════════════════════════════════════════════════════════════════════════════╗"
        )
        print(
            "║                  🔐 ToolBoxV2 Authentication                               ║"
        )
        print(
            "╚════════════════════════════════════════════════════════════════════════════╝\n"
        )

        async def helper():
            app = get_app("CloudM.cli_web_login")
            if args.status:
                return await app.a_run_any("CloudM", "cli_status")
            res = await app.a_run_any(
                "CloudM",
                "cli_login",
            )
            return res

        return await helper()

    async def logout():
        app = get_app("CloudM.cli_web_login")
        return await app.a_run_any("CloudM", "cli_logout")

    async def run_flow_from_file_or_load_all_flows_and_mods_from_dir(app=None):
        from toolboxv2 import init_cwd

        if app is None:
            app = get_app("app.Flows")
        parser = argparse.ArgumentParser(
            prog="tb flow",
            description="Run flow from file or load all flows and mods from dir",
        )
        parser.add_argument("--flow", help="Run flow from file", default=None)
        parser.add_argument(
            "--remote", help="Force remote login", action="store_true", default=False
        )

        args = parser.parse_args()

        app.print(Style.VIOLET2(f"Loading externals from {init_cwd}"))
        await app.load_all_mods_in_file(init_cwd)
        app.flows = flows_dict_func(args.flow or ".py", args.remote, init_cwd, app.flows)
        if args.flow:
            await app.run_flows(args.flow)

    run_c = run_flow_from_file_or_load_all_flows_and_mods_from_dir

    async def mods_manager():
        app = get_app("CloudM.ModManager")
        await app.a_run_any("CloudM", "mods")

    async def registry():
        app = get_app("CloudM.RegistryServer")
        await app.a_run_any("CloudM.RegistryServer", "start")

    def _run_docksh():
        import sys

        from toolboxv2.Docksh import docksh
        argvs = sys.argv[1:]
        if len(argvs) == 0:
            print("tb docksh x  - quick reconnect")
        elif 'x' in argvs:
            docksh.cmd_connect([])

        if not argvs:
            docksh.cmd_connect([])
        elif argvs[0] == "setup":
            docksh.cmd_setup()
        elif argvs[0] == "connect":
            docksh.cmd_connect(argvs[1:])
        else:
            try:
                from docksh_srv import main as Docksh_srv

                Docksh_srv(docksh.docs)
            except ModuleNotFoundError:
                from pathlib import Path
                path = str(Path(__file__).parent.parent)+"/docksh_srv.py"
                if os.path.exists(path):
                    os.system(f"{sys.executable} {path} {' '.join(argvs)}")
                else:
                    print(f"{path=} not found")


    runner = {
        "venv": lambda: __import__(
            "toolboxv2.utils.system.venv_runner", fromlist=["main"]
        ).main(),
        "ipy": start_ipython_session,
        "db": cli_db_runner,
        "gui": helper_gui,
        "p2p": cli_tcm_runner,
        "status": status_helper,
        "browser": lambda: __import__(
            "toolboxv2.tb_browser.install", fromlist=["main"]
        ).main(),
        "mcp": cli_mcp_server,
        "login": cli_web_login,
        "logout": logout,
        "flow": run_c,
        "mods": mods_manager,
        "registry": registry,
        "run": cli_tbx_main,
        "user": minio_user_manager_main,
        "default": interactive_user_dashboard,
        "workers": cli_worker_manager,
        "session": cli_session,
        "broker": cli_event,
        "http_worker": cli_http_worker,
        "ws_worker": cli_ws_worker,
        "services": lambda: __import__(
            "toolboxv2.utils.clis.service_manager", fromlist=["cli_services"]
        ).cli_services(),
        "manifest": lambda: __import__(
            "toolboxv2.utils.clis.manifest_cli", fromlist=["cli_manifest_main"]
        ).cli_manifest_main(),
        "llm-gateway": lambda: __import__(
            "toolboxv2.utils.clis.llm_gateway_cli", fromlist=["cli_llm_gateway"]
        ).cli_llm_gateway(),
        "docksh": _run_docksh,
        "docker-image": lambda: __import__(
            "toolboxv2.utils.clis.docker_image_cli", fromlist=["main"]
        ).main(),
    }

    return runner


def main_runner():
    # The fuck is uv not PyO3 compatible
    sys.excepthook = sys.__excepthook__
    # Windows: Use SelectorEventLoop for ZMQ compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # IPython special case
    if "--ipy" in sys.argv:
        argv = sys.argv[1:]
        sys.argv = sys.argv[:1]
        start_ipython_session(argv)

    # Service Manager special case - start all auto-start services and exit
    elif "--sm" in sys.argv:
        from toolboxv2.utils.clis.service_manager import run_service_manager_startup
        if 'init' in sys.argv:
            if system() == "Linux":
                setup_service_linux()
            elif system()  == "Windows":
                asyncio.run(setup_service_windows())
            else:
                print(
                    f"Service manager not supported on this platform {system()}"
                )
        sys.exit(run_service_manager_startup())

    # Normale Main-App
    else:
        # Clear screen for clean start
        runner = runner_setup()
        runner_keys = list(RUNNER_KEYS)
        main_args, runner_name, runner_args = split_args_by_runner(
            sys.argv[1:], runner_keys
        )
        # Check for unknown runner (argument that looks like a runner but isn't in RUNNER_KEYS)
        # This catches cases like `tb xyz` where xyz is not a valid runner
        unknown_runner = None
        if runner_name is None and len(sys.argv) > 1:
            # Check if first non-flag argument could be an unknown runner
            continue_flag = False
            for arg in sys.argv[1:]:
                if arg in ["-c", "-v", "-m"]:
                    runner_name = ""
                    break
                if continue_flag:
                    continue_flag = False
                    continue
                if not arg.startswith("-") and "=" not in arg:
                    # This looks like a runner name but wasn't found in RUNNER_KEYS
                    unknown_runner = arg
                    break

                else:
                    continue_flag = True

        if unknown_runner:
            print(f"\n\033[1;31m❌ Unknown command: '{unknown_runner}'\033[0m")
            print(f"\n\033[1mAvailable commands:\033[0m")
            # Group runners by category for better display
            core_runners = ["user", "run", "db", "workers", "services", "registry"]
            util_runners = ["login", "logout", "status", "session", "manifest"]
            dev_runners = ["venv", "ipy", "mcp", "gui", "browser"]
            other_runners = [
                r
                for r in runner_keys
                if r not in core_runners + util_runners + dev_runners
            ]

            print(f"  \033[36mCore:\033[0m      {', '.join(core_runners)}")
            print(f"  \033[36mUtility:\033[0m   {', '.join(util_runners)}")
            print(f"  \033[36mDev:\033[0m       {', '.join(dev_runners)}")
            if other_runners:
                print(f"  \033[36mOther:\033[0m     {', '.join(other_runners)}")
            print(f"\n\033[2mUse 'tb <command> --help' for more information.\033[0m\n")
            sys.exit(1)

        try:
            loop = asyncio.new_event_loop()
            if runner_name == "flows":
                _hook[0] = runner["flows"]
                runner["flows"] = lambda: None
            if runner_name == "mcp":
                TbApp.print = lambda *a, **k: None

            async def main_helper(runner_name):
                # Default to interactive dashboard if no runner specified
                # This applies to: `tb`, `tb -l`, `tb --debug`, etc.
                if runner_name is None:
                    runner_name = "default"

                await main(TbApp, runner_name == "default")

                # Wenn Runner angegeben
                if runner_name:
                    # Setze sys.argv für Runner
                    sys.argv = [sys.argv[0]] + runner_args

                    try:
                        res = runner[runner_name]()
                        if asyncio.iscoroutine(res):
                            await res
                    except KeyboardInterrupt:
                        sys.exit(0)

            loop.run_until_complete(main_helper(runner_name))
        except KeyboardInterrupt:
            import traceback

            traceback.print_exc()
            pass


import ctypes


def get_real_python_executable():
    try:
        # Set the return type for the function call
        ctypes.pythonapi.Py_GetProgramFullPath.restype = ctypes.c_char_p
        exe_path = ctypes.pythonapi.Py_GetProgramFullPath()
        print(exe_path)
        if exe_path:
            return exe_path.decode("utf-8")
    except Exception as e:
        # If anything goes wrong, fall back to sys.executable
        print(f"Error detecting real executable: {e}")
    return sys.executable


def server_helper(instance_id: str = "main", db_mode=None):
    # real_exe = get_real_python_executable()
    from pathlib import Path

    sys.executable = str(Path(os.getenv("PYTHON_EXECUTABLE", sys.executable)))
    print("Using Python executable env:", sys.executable)
    loop = asyncio.new_event_loop()
    sys.argv.append("-l")
    sys.argv.append("--debug")
    app, _ = loop.run_until_complete(setup_app(instance_id))
    app.loop = loop
    if db_mode is None:
        db_mode = os.getenv("DB_MODE_KEY", "LC")
    app.is_server = True

    # db = app.get_mod("DB")
    # db.edit_cli(db_mode)
    # db.initialize_database()
    # execute all flows starting with server as bg tasks
    def task():
        flows_dict = flows_dict_func(remote=False)
        app.set_flows(flows_dict)
        for flow in flows_dict:
            if flow.startswith("server"):
                print(f"Starting server flow: {flow}")

                app.run_bg_task_advanced(app.run_flows, flow)

    # app.run_bg_task_advanced(task)
    return app


if __name__ == "__main__":
    # print("STARTED START FROM __main__")
    sys.exit(main_runner())
