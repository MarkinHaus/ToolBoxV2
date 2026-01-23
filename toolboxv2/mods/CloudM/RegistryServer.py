"""TB-Registry Server Module for ToolBoxV2.

This module provides an entrypoint to start the TB-Registry server
directly from ToolBoxV2.
"""

import asyncio
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from toolboxv2 import App, Result, get_app
from toolboxv2.utils.system.types import ToolBoxInterfaces

Name = "CloudM.RegistryServer"
version = "0.1.0"
export = get_app(f"{Name}.Export").tb

# Interactive CLI helper
HELPER = """
╔══════════════════════════════════════════════════════════════╗
║                    TB-Registry Server                        ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                   ║
║    start [--bg]     Start registry server                    ║
║    stop             Stop background server                   ║
║    status           Check server status                      ║
║    restart          Restart background server                ║
║                                                              ║
║  Options:                                                    ║
║    --host HOST      Bind address (default: 0.0.0.0)          ║
║    --port PORT      Port number (default: 4025)              ║
║    --debug          Enable debug mode                        ║
║    --bg             Run in background                        ║
╚══════════════════════════════════════════════════════════════╝
"""

# Default registry path relative to ToolBoxV2 root
DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent.parent.parent / "tb-registry"


def _find_registry_path() -> Optional[Path]:
    """Find the tb-registry installation path."""
    # Check environment variable first
    if env_path := os.environ.get("TB_REGISTRY_PATH"):
        path = Path(env_path)
        if path.exists():
            return path

    # Check default location
    if DEFAULT_REGISTRY_PATH.exists():
        return DEFAULT_REGISTRY_PATH

    # Check common locations
    for check_path in [
        Path.home() / "tb-registry",
        Path("/opt/tb-registry"),
        Path.cwd() / "tb-registry",
    ]:
        if check_path.exists() and (check_path / "registry").exists():
            return check_path

    return None


@export(mod_name=Name, name="start", version=version, test=False)
async def start_registry(
    app: App,
    host: str = "0.0.0.0",
    port: int = 4025,
    debug: bool = False,
    registry_path: Optional[str] = None,
    background: bool = False,
) -> Result:
    """Start the TB-Registry server.

    Args:
        app: ToolBoxV2 application instance.
        host: Host address to bind to.
        port: Port number to listen on.
        debug: Enable debug mode.
        registry_path: Path to tb-registry installation.
        background: Run in background process.

    Returns:
        Result with server status.
    """
    reg_path = Path(registry_path) if registry_path else _find_registry_path()

    if not reg_path or not reg_path.exists():
        return Result.custom_error(
            exec_code=-1,
            help_text="TB-Registry not found. Set TB_REGISTRY_PATH or install to default location.",
        )

    app.print(f"[RegistryServer] Starting from: {reg_path}")

    # Set environment variables
    env = os.environ.copy()
    env["SERVER_HOST"] = host
    env["SERVER_PORT"] = str(port)
    env["DEBUG"] = str(debug).lower()

    if background:
        return await _start_background(app, reg_path, env)
    else:
        return await _start_foreground(app, reg_path, env)


async def _start_foreground(app: App, reg_path: Path, env: dict) -> Result:
    """Start registry in foreground (blocking)."""
    app.print(f"[RegistryServer] Running on http://{env['SERVER_HOST']}:{env['SERVER_PORT']}")
    app.print("[RegistryServer] Press Ctrl+C to stop")

    try:
        # Use subprocess.Popen for cross-platform compatibility
        # asyncio.create_subprocess_exec doesn't work on Windows default event loop
        process = subprocess.Popen(
            ["uv", "run", "python", "-m", "registry"],
            cwd=str(reg_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Stream output in a thread to not block
        def stream_output():
            try:
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    app.print(f"[Registry] {line.rstrip()}")
            except Exception:
                pass

        output_thread = threading.Thread(target=stream_output, daemon=True)
        output_thread.start()

        # Wait for process (this blocks, but allows Ctrl+C)
        try:
            while process.poll() is None:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            process.terminate()
            process.wait(timeout=5)
            return Result.ok(info="Server stopped")

        return Result.ok(data={"exit_code": process.returncode})

    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=5)
        return Result.ok(info="Server stopped by user")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Result.custom_error(exec_code=-2, info=f"Failed to start: {e}")


async def _start_background(app: App, reg_path: Path, env: dict) -> Result:
    """Start registry as background process."""
    pid_file = reg_path / "data" / "registry.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if already running
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            return Result.custom_error(
                exec_code=-3,
                help_text=f"Registry already running (PID: {pid})",
            )
        except (ProcessLookupError, ValueError):
            pid_file.unlink(missing_ok=True)

    # Start background process
    if sys.platform == "win32":
        process = subprocess.Popen(
            ["uv", "run", "python", "-m", "registry"],
            cwd=str(reg_path),
            env=env,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        process = subprocess.Popen(
            ["uv", "run", "python", "-m", "registry"],
            cwd=str(reg_path),
            env=env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    pid_file.write_text(str(process.pid))
    app.print(f"[RegistryServer] Started in background (PID: {process.pid})")

    return Result.ok(
        data={"pid": process.pid, "host": env["SERVER_HOST"], "port": env["SERVER_PORT"]},
        info=f"Registry started on http://{env['SERVER_HOST']}:{env['SERVER_PORT']}",
    )


@export(mod_name=Name, name="stop", version=version, test=False)
async def stop_registry(
    app: App,
    registry_path: Optional[str] = None,
) -> Result:
    """Stop the TB-Registry background server.

    Args:
        app: ToolBoxV2 application instance.
        registry_path: Path to tb-registry installation.

    Returns:
        Result with stop status.
    """
    reg_path = Path(registry_path) if registry_path else _find_registry_path()

    if not reg_path:
        return Result.custom_error(exec_code=-1, help_text="TB-Registry not found")

    pid_file = reg_path / "data" / "registry.pid"

    if not pid_file.exists():
        return Result.custom_error(exec_code=-2, help_text="No running registry found")

    try:
        pid = int(pid_file.read_text().strip())

        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True)
        else:
            os.kill(pid, signal.SIGTERM)

        pid_file.unlink(missing_ok=True)
        app.print(f"[RegistryServer] Stopped (PID: {pid})")

        return Result.ok(data={"pid": pid}, info="Registry stopped")

    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return Result.ok(info="Registry was not running")
    except Exception as e:
        return Result.custom_error(exec_code=-3, help_text=f"Failed to stop: {e}")


@export(mod_name=Name, name="status", version=version, test=False)
async def registry_status(
    app: App,
    registry_path: Optional[str] = None,
) -> Result:
    """Check TB-Registry server status.

    Args:
        app: ToolBoxV2 application instance.
        registry_path: Path to tb-registry installation.

    Returns:
        Result with status information.
    """
    reg_path = Path(registry_path) if registry_path else _find_registry_path()

    if not reg_path:
        return Result.ok(data={"status": "not_installed"}, info="TB-Registry not found")

    pid_file = reg_path / "data" / "registry.pid"

    if not pid_file.exists():
        return Result.ok(data={"status": "stopped", "path": str(reg_path)}, info="Registry not running")

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # Check if process exists

        return Result.ok(
            data={"status": "running", "pid": pid, "path": str(reg_path)},
            info=f"Registry running (PID: {pid})",
        )
    except (ProcessLookupError, ValueError):
        pid_file.unlink(missing_ok=True)
        return Result.ok(data={"status": "stopped", "path": str(reg_path)}, info="Registry not running")


@export(mod_name=Name, name="restart", version=version, test=False)
async def restart_registry(
    app: App,
    host: str = "0.0.0.0",
    port: int = 4025,
    debug: bool = False,
    registry_path: Optional[str] = None,
) -> Result:
    """Restart the TB-Registry server.

    Args:
        app: ToolBoxV2 application instance.
        host: Host address to bind to.
        port: Port number to listen on.
        debug: Enable debug mode.
        registry_path: Path to tb-registry installation.

    Returns:
        Result with restart status.
    """
    # Stop first
    stop_result = await stop_registry(app, registry_path)
    if stop_result.is_error() and "not running" not in str(stop_result.info):
        return stop_result

    # Wait a moment
    await asyncio.sleep(1)

    # Start again
    return await start_registry(app, host, port, debug, registry_path, background=True)


@export(
    mod_name=Name,
    name="cli",
    version=version,
    test=False,
    helper=HELPER,
    api=False,
    samples=[
        ("RegistryServer", "cli", ["start"]),
        ("RegistryServer", "cli", ["start", "--bg"]),
        ("RegistryServer", "cli", ["stop"]),
        ("RegistryServer", "cli", ["status"]),
    ],
)
async def registry_cli(
    app: App,
    command: str = "status",
    host: str = "0.0.0.0",
    port: int = 4025,
    debug: bool = False,
    bg: bool = False,
) -> Result:
    """Interactive CLI for TB-Registry server management.

    Args:
        app: ToolBoxV2 application instance.
        command: Command to execute (start, stop, status, restart).
        host: Host address for start/restart.
        port: Port number for start/restart.
        debug: Enable debug mode.
        bg: Run in background (for start command).

    Returns:
        Result with command output.
    """
    command = command.lower().strip()

    if command == "start":
        return await start_registry(app, host, port, debug, background=bg)
    elif command == "stop":
        return await stop_registry(app)
    elif command == "status":
        return await registry_status(app)
    elif command == "restart":
        return await restart_registry(app, host, port, debug)
    elif command == "help":
        app.print(HELPER)
        return Result.ok()
    else:
        app.print(f"Unknown command: {command}")
        app.print(HELPER)
        return Result.custom_error(exec_code=-1, help_text=f"Unknown command: {command}")
