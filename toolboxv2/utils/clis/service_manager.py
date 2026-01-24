"""
ToolBoxV2 Service Manager - Plattform-agnostischer Auto-Start

Verwendung:
    tb --sm              # Startet alle auto-start Services und beendet sich
    tb services start    # Interaktiver Service-Start

Features:
- Cross-Platform (Linux, Windows, macOS)
- PID-File basiertes Tracking
- Keine OS-spezifischen Service-APIs
- Subprocess-basiert für Isolation
"""

import os
import sys
import json
import time
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Plattform-Detection
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
IS_MACOS = sys.platform == "darwin"


@dataclass
class ServiceStartResult:
    """Result of a service start attempt"""
    name: str
    success: bool
    pid: Optional[int] = None
    error: Optional[str] = None


@dataclass
class ServiceDefinition:
    """Definition eines registrierten Services"""
    name: str
    description: str
    category: str  # core, infrastructure, extension
    module: str  # z.B. "toolboxv2.utils.clis.cli_worker_manager"
    entry_point: str  # z.B. "main"
    is_async: bool = False
    runner_key: Optional[str] = None  # Falls anders als name


class ServiceRegistry:
    """
    Registry für alle verfügbaren Services.

    Enthält Built-in Services und ermöglicht Discovery.
    """

    _instance: Optional["ServiceRegistry"] = None
    _services: Dict[str, ServiceDefinition] = {}

    def __new__(cls) -> "ServiceRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._services = {}
            cls._instance._register_builtin_services()
        return cls._instance

    def _register_builtin_services(self) -> None:
        """Registriere alle Built-in Services"""
        # Core Services
        self.register(ServiceDefinition(
            name="custom",
            description="Running a custom service",
            category="core",
            module="toolboxv2.__main__",
            entry_point="main_runner",
            runner_key="custom"
        ))
        self.register(ServiceDefinition(
            name="workers",
            description="Worker-Orchestrierung (HTTP, WS, Broker)",
            category="core",
            module="toolboxv2.utils.clis.cli_worker_manager",
            entry_point="main",
            runner_key="workers"
        ))
        self.register(ServiceDefinition(
            name="db",
            description="MinIO Blob Storage Management",
            category="core",
            module="toolboxv2.utils.clis.db_cli_manager",
            entry_point="cli_db_runner",
            is_async=True,
            runner_key="db"
        ))
        self.register(ServiceDefinition(
            name="isaaK",
            description="Discord & Telegram Kernel Agent",
            category="core",
            module="toolboxv2.mods.isaa.kernel.kernelin.run_unified_kernels",
            entry_point="run",
            is_async=True,
            runner_key="isaaK"
        ))
        #self.register(ServiceDefinition(
        #    name="user",
        #    description="Interaktives User Dashboard",
        #    category="core",
        #    module="toolboxv2.utils.clis.user_dashboard",
        #    entry_point="interactive_user_dashboard",
        #    is_async=True,
        #    runner_key="user"
        #))
        #self.register(ServiceDefinition(
        #    name="run",
        #    description="TB Language Build/Run/Compile",
        #    category="core",
        #    module="toolboxv2.utils.clis.tb_lang_cli",
        #    entry_point="cli_tbx_main",
        #    runner_key="run"
        #))

        # Infrastructure Services
        # self.register(ServiceDefinition(
        #     name="session",
        #     description="Session-Management für Workers",
        #     category="infrastructure",
        #     module="toolboxv2.utils.workers",
        #     entry_point="cli_session",
        #     runner_key="session"
        # ))
        self.register(ServiceDefinition(
            name="broker",
            description="Event-Management (ZMQ)",
            category="infrastructure",
            module="toolboxv2.utils.workers",
            entry_point="cli_event",
            runner_key="broker"
        ))
        self.register(ServiceDefinition(
            name="http_worker",
            description="HTTP Worker Server",
            category="infrastructure",
            module="toolboxv2.utils.workers",
            entry_point="cli_http_worker",
            runner_key="http_worker"
        ))
        self.register(ServiceDefinition(
            name="ws_worker",
            description="WebSocket Worker Server",
            category="infrastructure",
            module="toolboxv2.utils.workers",
            entry_point="cli_ws_worker",
            runner_key="ws_worker"
        ))

        # Extension Services
        self.register(ServiceDefinition(
            name="p2p",
            description="P2P Chat, File Transfer, Voice",
            category="extension",
            module="toolboxv2.utils.clis.tcm_p2p_cli",
            entry_point="cli_tcm_runner",
            runner_key="p2p"
        ))
        self.register(ServiceDefinition(
            name="mcp",
            description="MCP Server für AI Agents",
            category="extension",
            module="toolboxv2.utils.clis.mcp_server",
            entry_point="cli_mcp_server",
            runner_key="mcp"
        ))
        self.register(ServiceDefinition(
            name="gui",
            description="Graphical User Interface",
            category="extension",
            module="toolboxv2.__main__",
            entry_point="helper_gui",
            runner_key="gui"
        ))
        self.register(ServiceDefinition(
            name="llm-gateway",
            description="OpenAI-kompatible LLM API (llama.cpp)",
            category="extension",
            module="toolboxv2.utils.clis.llm_gateway_cli",
            entry_point="cli_llm_gateway",
            runner_key="llm-gateway"
        ))

    def register(self, service: ServiceDefinition) -> None:
        """Registriere einen Service"""
        self._services[service.name] = service

    def get(self, name: str) -> Optional[ServiceDefinition]:
        """Hole Service-Definition"""
        return self._services.get(name)

    def get_all(self) -> Dict[str, ServiceDefinition]:
        """Alle registrierten Services"""
        return self._services.copy()

    def get_by_category(self, category: str) -> List[ServiceDefinition]:
        """Services nach Kategorie"""
        return [s for s in self._services.values() if s.category == category]

    def list_names(self) -> List[str]:
        """Liste aller Service-Namen"""
        return list(self._services.keys())


class ServiceManager:
    """
    Plattform-agnostischer Service Manager

    Startet Services als Subprocesses und trackt sie via PID-Files.
    Kein Daemon-Mode - startet und beendet sich.
    """

    def __init__(self):
        from toolboxv2 import tb_root_dir
        self.pids_dir = tb_root_dir / ".info" / "pids"
        self.config_path = tb_root_dir / ".info" / "services.json"
        self.pids_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> Dict:
        """Lade Service-Konfiguration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {"services": {}}

    def save_config(self, config: Dict) -> None:
        """Speichere Service-Konfiguration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def get_auto_start_services(self) -> List[str]:
        """Liste aller Services mit auto_start=True"""
        config = self.load_config()
        return [
            name for name, cfg in config.get("services", {}).items()
            if cfg.get("auto_start", False)
        ]

    def is_service_running(self, name: str) -> Tuple[bool, Optional[int]]:
        """Prüfe ob Service läuft (via PID-File)"""
        pid_file = self.pids_dir / f"{name}.pid"

        if not pid_file.exists():
            return False, None

        try:
            pid = int(pid_file.read_text().strip())

            # Check if process exists
            if IS_WINDOWS:
                # Windows: tasklist check
                try:
                    result = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="ignore"
                    )
                    running = result.stdout and str(pid) in result.stdout
                except Exception:
                    running = False
            else:
                # Unix: kill -0 check (doesn't actually kill)
                try:
                    os.kill(pid, 0)
                    running = True
                except OSError:
                    running = False

            if not running:
                # Cleanup stale PID file
                pid_file.unlink(missing_ok=True)
                return False, None

            return True, pid

        except (ValueError, FileNotFoundError):
            return False, None

    def start_service(self, name: str, args: Optional[List[str]] = None,
                      save_args: bool = True) -> ServiceStartResult:
        """
        Starte einen Service als Subprocess

        Services werden über `tb <service_name> [args...]` gestartet.
        Der Prozess wird detached (läuft weiter nach Script-Ende).

        Args:
            name: Service-Name
            args: Argumente für den Service (None = gespeicherte Args verwenden)
            save_args: Wenn True und args übergeben, speichere Args für Restart
        """
        # Check if already running
        running, existing_pid = self.is_service_running(name)
        if running:
            return ServiceStartResult(
                name=name,
                success=True,
                pid=existing_pid,
                error="Already running"
            )

        # Determine args to use
        if args is None:
            # Use saved args from config
            args = self.get_service_args(name)
        elif save_args and args:
            # Save new args for future restarts
            self.configure_service(name, args=args)

        # Build command - use tb CLI for actual start
        if name != "custom":
            cmd = [sys.executable, "-m", "toolboxv2", name] + (args or [])
        else:
            cmd = [sys.executable, "-m", "toolboxv2"] + (args or [])

        if IS_WINDOWS:
            # Windows: CREATE_NO_WINDOW für headless
            creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
            kwargs = {"creationflags": creation_flags}
        else:
            # Unix: nohup-style detach
            kwargs = {
                "start_new_session": True,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }

        try:
            process = subprocess.Popen(cmd, **kwargs)
            pid = process.pid

            # Write PID file
            pid_file = self.pids_dir / f"{name}.pid"
            pid_file.write_text(str(pid))

            # Brief wait to check if it started successfully
            time.sleep(0.5)

            # Verify it's still running
            running, _ = self.is_service_running(name)

            return ServiceStartResult(
                name=name,
                success=running,
                pid=pid if running else None,
                error=None if running else "Process exited immediately"
            )

        except Exception as e:
            return ServiceStartResult(
                name=name,
                success=False,
                error=str(e)
            )

    def stop_service(self, name: str, graceful: bool = True) -> bool:
        """Stoppe einen Service via PID"""
        running, pid = self.is_service_running(name)

        if not running or pid is None:
            return True  # Already stopped

        try:
            if IS_WINDOWS:
                # Windows: taskkill
                flag = [] if graceful else ["/F"]
                subprocess.run(["taskkill", "/PID", str(pid)] + flag, check=True)
            else:
                # Unix: SIGTERM (graceful) or SIGKILL
                sig = signal.SIGTERM if graceful else signal.SIGKILL
                os.kill(pid, sig)

            # Wait for shutdown
            for _ in range(10):
                time.sleep(0.5)
                running, _ = self.is_service_running(name)
                if not running:
                    break

            # Cleanup PID file
            pid_file = self.pids_dir / f"{name}.pid"
            pid_file.unlink(missing_ok=True)

            return True

        except Exception:
            return False

    def get_all_status(self, include_registry: bool = True) -> Dict[str, dict]:
        """Status aller bekannten Services (inkl. Registry)"""
        config = self.load_config()
        status = {}

        # Configured services
        for name, cfg in config.get("services", {}).items():
            running, pid = self.is_service_running(name)
            registry = ServiceRegistry()
            svc_def = registry.get(name)
            status[name] = {
                "name": name,
                "running": running,
                "pid": pid,
                "auto_start": cfg.get("auto_start", False),
                "auto_restart": cfg.get("auto_restart", False),
                "category": svc_def.category if svc_def else "custom",
                "description": svc_def.description if svc_def else "",
                "registered": svc_def is not None,
            }

        # Add registry services not yet configured
        if include_registry:
            registry = ServiceRegistry()
            for name, svc_def in registry.get_all().items():
                if name not in status:
                    running, pid = self.is_service_running(name)
                    status[name] = {
                        "name": name,
                        "running": running,
                        "pid": pid,
                        "auto_start": False,
                        "auto_restart": False,
                        "category": svc_def.category,
                        "description": svc_def.description,
                        "registered": True,
                    }

        return status

    def get_service_info(self, name: str) -> Optional[dict]:
        """Detaillierte Info zu einem Service"""
        registry = ServiceRegistry()
        svc_def = registry.get(name)
        config = self.load_config()
        cfg = config.get("services", {}).get(name, {})
        running, pid = self.is_service_running(name)

        if not svc_def and not cfg:
            return None

        return {
            "name": name,
            "running": running,
            "pid": pid,
            "auto_start": cfg.get("auto_start", False),
            "auto_restart": cfg.get("auto_restart", False),
            "category": svc_def.category if svc_def else "custom",
            "description": svc_def.description if svc_def else "Custom service",
            "module": svc_def.module if svc_def else None,
            "entry_point": svc_def.entry_point if svc_def else None,
            "is_async": svc_def.is_async if svc_def else False,
            "runner_key": svc_def.runner_key if svc_def else name,
            "registered": svc_def is not None,
        }

    def configure_service(self, name: str, auto_start: Optional[bool] = None,
                          auto_restart: Optional[bool] = None,
                          args: Optional[List[str]] = None) -> None:
        """Konfiguriere einen Service"""
        config = self.load_config()
        if "services" not in config:
            config["services"] = {}
        if name not in config["services"]:
            config["services"][name] = {}

        if auto_start is not None:
            config["services"][name]["auto_start"] = auto_start
        if auto_restart is not None:
            config["services"][name]["auto_restart"] = auto_restart
        if args is not None:
            config["services"][name]["args"] = args

        self.save_config(config)

    def get_service_args(self, name: str) -> List[str]:
        """Hole gespeicherte Argumente für einen Service"""
        config = self.load_config()
        return config.get("services", {}).get(name, {}).get("args", [])


def run_service_manager_startup() -> int:
    """
    Entry Point für `tb --sm`

    Startet alle auto-start Services und beendet sich dann.
    Exit Code: 0 = alle erfolgreich, 1 = mindestens einer fehlgeschlagen
    """
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header, print_box_footer, print_status
    )

    print_box_header("ToolBoxV2 Service Manager", "🚀")

    manager = ServiceManager()
    auto_start = manager.get_auto_start_services()

    if not auto_start:
        print_status("No services configured for auto-start", "info")
        print_status("Configure with: tb services config <name> --auto-start=true", "info")
        print_box_footer()
        return 0

    print_status(f"Starting {len(auto_start)} service(s)...", "progress")
    print()

    results: List[ServiceStartResult] = []

    for name in auto_start:
        result = manager.start_service(name)
        results.append(result)

        if result.success:
            if result.error == "Already running":
                print_status(f"{name}: already running (PID {result.pid})", "info")
            else:
                print_status(f"{name}: started (PID {result.pid})", "success")
        else:
            print_status(f"{name}: failed - {result.error}", "error")

    print()

    # Summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    if failed == 0:
        print_status(f"All {successful} service(s) started successfully", "success")
    else:
        print_status(f"{successful} started, {failed} failed", "warning")

    print_box_footer()

    return 0 if failed == 0 else 1


def run_service_manager_status() -> None:
    """Zeige Status aller Services"""
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header, print_box_footer,
        print_table_header, print_table_row
    )

    manager = ServiceManager()
    status = manager.get_all_status()

    print_box_header("Service Status", "📊")

    if not status:
        print("  No services configured")
        print_box_footer()
        return

    columns = [("Service", 20), ("Status", 12), ("PID", 10), ("Auto", 8)]
    widths = [w for _, w in columns]
    print_table_header(columns, widths)

    for name, info in status.items():
        state = "running" if info["running"] else "stopped"
        state_color = "green" if info["running"] else "grey"
        auto = "✓" if info["auto_start"] else ""

        print_table_row(
            [name, state, str(info["pid"] or "-"), auto],
            widths,
            ["cyan", state_color, "grey", "green"]
        )

    print_box_footer()


def cli_services() -> None:
    """CLI Entry Point für `tb services` Command"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="tb services",
        description="🔧 ToolBoxV2 Service Manager"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status
    p_status = subparsers.add_parser("status", help="Show service status")
    p_status.add_argument("--name", "-n", help="Specific service name")

    # start
    p_start = subparsers.add_parser("start", help="Start service(s)")
    p_start.add_argument("name", nargs="?", help="Service name (all auto-start if omitted)")
    p_start.add_argument("--auto-start", action="store_true",
                         help="Enable auto-start for this service")
    p_start.add_argument("service_args", nargs="*", help="Arguments to pass to the service")

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop service(s)")
    p_stop.add_argument("name", nargs="?", help="Service name (all if omitted)")
    p_stop.add_argument("--force", "-f", action="store_true", help="Force stop")

    # config
    p_config = subparsers.add_parser("config", help="Configure services")
    p_config.add_argument("name", help="Service name")
    p_config.add_argument("--auto-start", choices=["true", "false"],
                          help="Enable/disable auto-start")
    p_config.add_argument("--auto-restart", choices=["true", "false"],
                          help="Enable/disable auto-restart")
    p_config.add_argument("--args", nargs="*", metavar="ARG",
                          help="Set default arguments for the service")
    p_config.add_argument("--clear-args", action="store_true",
                          help="Clear saved arguments")

    # list
    subparsers.add_parser("list", help="List all configured services")

    # info
    p_info = subparsers.add_parser("info", help="Show detailed service info")
    p_info.add_argument("name", help="Service name")

    # restart
    p_restart = subparsers.add_parser("restart", help="Restart service(s)")
    p_restart.add_argument("name", nargs="?", help="Service name (all running if omitted)")
    p_restart.add_argument("--force", "-f", action="store_true", help="Force restart")
    p_restart.add_argument("service_args", nargs="*", help="New arguments (overrides saved)")

    # registry
    subparsers.add_parser("registry", help="Show all registered (built-in) services")

    args = parser.parse_args()
    manager = ServiceManager()

    if args.command == "status":
        run_service_manager_status()
    elif args.command == "start":
        _cmd_start(manager, args)
    elif args.command == "stop":
        _cmd_stop(manager, args)
    elif args.command == "config":
        _cmd_config(manager, args)
    elif args.command == "list":
        _cmd_list(manager)
    elif args.command == "info":
        _cmd_info(manager, args)
    elif args.command == "restart":
        _cmd_restart(manager, args)
    elif args.command == "registry":
        _cmd_registry()
    else:
        parser.print_help()


def _cmd_start(manager: ServiceManager, args) -> None:
    """Start service(s)"""
    from toolboxv2.utils.clis.cli_printing import print_status

    if args.name:
        # Get service args from CLI (None if empty = use saved args)
        service_args = getattr(args, "service_args", None)
        if service_args is not None and len(service_args) == 0:
            service_args = None  # Empty list means use saved args

        # Handle --auto-start flag (configure, don't pass to service)
        if getattr(args, "auto_start", False):
            manager.configure_service(args.name, auto_start=True)
            print_status(f"Auto-start enabled for {args.name}", "info")

        print_status(f"Starting {args.name}...", "progress")
        if service_args:
            print_status(f"  Args: {' '.join(service_args)}", "info")

        result = manager.start_service(args.name, args=service_args)
        if result.success:
            print_status(f"{args.name} started (PID {result.pid})", "success")
        else:
            print_status(f"Failed to start {args.name}: {result.error}", "error")
    else:
        # Start all auto-start services (with their saved args)
        exit_code = run_service_manager_startup()
        if exit_code != 0:
            print_status("Some services failed to start", "warning")


def _cmd_stop(manager: ServiceManager, args) -> None:
    """Stop service(s)"""
    from toolboxv2.utils.clis.cli_printing import print_status

    graceful = not getattr(args, "force", False)

    if args.name:
        print_status(f"Stopping {args.name}...", "progress")
        success = manager.stop_service(args.name, graceful=graceful)
        if success:
            print_status(f"{args.name} stopped", "success")
        else:
            print_status(f"Failed to stop {args.name}", "error")
    else:
        # Stop all running services
        status = manager.get_all_status()
        for name, info in status.items():
            if info["running"]:
                print_status(f"Stopping {name}...", "progress")
                manager.stop_service(name, graceful=graceful)
        print_status("All services stopped", "success")


def _cmd_config(manager: ServiceManager, args) -> None:
    """Configure service"""
    from toolboxv2.utils.clis.cli_printing import print_status

    auto_start = None
    auto_restart = None
    service_args = None

    if args.auto_start:
        auto_start = args.auto_start == "true"
    if args.auto_restart:
        auto_restart = args.auto_restart == "true"

    # Handle args
    if getattr(args, "clear_args", False):
        service_args = []  # Empty list clears args
        print_status(f"Clearing saved args for {args.name}", "info")
    elif getattr(args, "args", None) is not None:
        service_args = args.args
        if service_args:
            print_status(f"Setting args: {' '.join(service_args)}", "info")

    manager.configure_service(args.name, auto_start=auto_start,
                              auto_restart=auto_restart, args=service_args)
    print_status(f"Configuration for {args.name} updated", "success")


def _cmd_list(manager: ServiceManager) -> None:
    """List all configured services"""
    from toolboxv2.utils.clis.cli_printing import print_box_header, print_box_footer

    config = manager.load_config()
    services = config.get("services", {})

    print_box_header("Configured Services", "📋")

    if not services:
        print("  No services configured")
        print("  Use: tb services config <name> --auto-start=true")
    else:
        for name, cfg in services.items():
            flags = []
            if cfg.get("auto_start"):
                flags.append("auto-start")
            if cfg.get("auto_restart"):
                flags.append("auto-restart")
            flag_str = f" [{', '.join(flags)}]" if flags else ""

            # Show saved args if any
            saved_args = cfg.get("args", [])
            args_str = f" -- {' '.join(saved_args)}" if saved_args else ""

            print(f"  • {name}{flag_str}{args_str}")

    print_box_footer()


def _cmd_info(manager: ServiceManager, args) -> None:
    """Show detailed service info"""
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header, print_box_footer, print_status
    )

    info = manager.get_service_info(args.name)

    if not info:
        print_status(f"Service '{args.name}' not found", "error")
        print_status("Use 'tb services registry' to see available services", "info")
        return

    print_box_header(f"Service: {args.name}", "ℹ")

    # Status
    status_str = "🟢 running" if info["running"] else "🔴 stopped"
    print(f"  Status:      {status_str}")
    if info["pid"]:
        print(f"  PID:         {info['pid']}")

    print()

    # Configuration
    print(f"  Category:    {info['category']}")
    print(f"  Description: {info['description']}")
    print(f"  Auto-Start:  {'✓' if info['auto_start'] else '✗'}")
    print(f"  Auto-Restart: {'✓' if info['auto_restart'] else '✗'}")

    # Saved args
    saved_args = manager.get_service_args(args.name)
    if saved_args:
        print(f"  Saved Args:  {' '.join(saved_args)}")
    else:
        print(f"  Saved Args:  (none)")

    if info.get("module"):
        print()
        print(f"  Module:      {info['module']}")
        print(f"  Entry Point: {info['entry_point']}")
        print(f"  Async:       {'Yes' if info['is_async'] else 'No'}")
        print(f"  Runner Key:  {info['runner_key']}")

    print_box_footer()


def _cmd_restart(manager: ServiceManager, args) -> None:
    """Restart service(s)"""
    from toolboxv2.utils.clis.cli_printing import print_status
    import time

    graceful = not getattr(args, "force", False)

    # Get new args if provided (overrides saved args)
    service_args = getattr(args, "service_args", None)
    if service_args is not None and len(service_args) == 0:
        service_args = None  # Empty = use saved args

    if args.name:
        # Restart single service
        print_status(f"Restarting {args.name}...", "progress")
        if service_args:
            print_status(f"  New args: {' '.join(service_args)}", "info")

        # Stop
        running, _ = manager.is_service_running(args.name)
        if running:
            manager.stop_service(args.name, graceful=graceful)
            time.sleep(1)

        # Start with new args (or saved args if None)
        result = manager.start_service(args.name, args=service_args)
        if result.success:
            print_status(f"{args.name} restarted (PID {result.pid})", "success")
        else:
            print_status(f"Failed to restart {args.name}: {result.error}", "error")
    else:
        # Restart all running services (with their saved args)
        status = manager.get_all_status(include_registry=False)
        running_services = [name for name, info in status.items() if info["running"]]

        if not running_services:
            print_status("No running services to restart", "info")
            return

        for name in running_services:
            print_status(f"Restarting {name}...", "progress")
            manager.stop_service(name, graceful=graceful)
            time.sleep(1)
            # Use saved args for each service
            result = manager.start_service(name, args=None)
            if result.success:
                print_status(f"{name} restarted (PID {result.pid})", "success")
            else:
                print_status(f"Failed to restart {name}: {result.error}", "error")


def _cmd_registry() -> None:
    """Show all registered (built-in) services"""
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header, print_box_footer,
        print_table_header, print_table_row
    )

    registry = ServiceRegistry()
    services = registry.get_all()

    print_box_header("Service Registry", "📦")

    # Group by category
    categories = {"core": [], "infrastructure": [], "extension": []}
    for name, svc in services.items():
        if svc.category in categories:
            categories[svc.category].append(svc)

    for category, svcs in categories.items():
        if not svcs:
            continue

        cat_icons = {"core": "🔷", "infrastructure": "🔧", "extension": "🔌"}
        print(f"\n  {cat_icons.get(category, '•')} {category.upper()}")
        print(f"  {'─' * 60}")

        for svc in svcs:
            print(f"    {svc.name:<15} {svc.description}")

    print()
    print_box_footer()

