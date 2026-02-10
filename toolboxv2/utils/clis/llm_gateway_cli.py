"""
LLM Gateway CLI - Cross-Platform Management

Usage:
    tb llm-gateway setup     # Setup/Installation
    tb llm-gateway start     # Start server
    tb llm-gateway stop      # Stop server
    tb llm-gateway status    # Show status
    tb llm-gateway restart   # Restart server
    tb llm-gateway uninstall # Uninstall

Features:
- Cross-Platform (Linux, Windows, macOS)
- Automatic venv management
- PID-based process tracking
- Ollama backend integration
"""

import os
import sys
import json
import time
import signal
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple

# Platform detection
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
IS_MACOS = sys.platform == "darwin"


def get_gateway_dir() -> Path:
    """Get llm-gateway directory"""
    from toolboxv2 import tb_root_dir
    return tb_root_dir.parent / "llm-gateway"


def get_pids_dir() -> Path:
    """Get PIDs directory"""
    from toolboxv2 import tb_root_dir
    pids_dir = tb_root_dir / ".info" / "pids"
    pids_dir.mkdir(parents=True, exist_ok=True)
    return pids_dir


def get_venv_python(gateway_dir: Path) -> Path:
    """Get venv Python executable"""
    if IS_WINDOWS:
        return gateway_dir / "venv" / "Scripts" / "python.exe"
    return gateway_dir / "venv" / "bin" / "python"


def get_uvicorn(gateway_dir: Path) -> Path:
    """Get uvicorn executable"""
    if IS_WINDOWS:
        return gateway_dir / "venv" / "Scripts" / "uvicorn.exe"
    return gateway_dir / "venv" / "bin" / "uvicorn"


def is_running() -> Tuple[bool, Optional[int]]:
    """Check if gateway is running"""
    pid_file = get_pids_dir() / "llm-gateway.pid"

    if not pid_file.exists():
        return False, None

    try:
        pid = int(pid_file.read_text().strip())

        if IS_WINDOWS:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, encoding="utf-8", errors="ignore"
            )
            running = result.stdout and str(pid) in result.stdout
        else:
            try:
                os.kill(pid, 0)
                running = True
            except OSError:
                running = False

        if not running:
            pid_file.unlink(missing_ok=True)
            return False, None

        return True, pid
    except (ValueError, FileNotFoundError):
        return False, None


def check_ollama_installed() -> Tuple[bool, Optional[str]]:
    """Check if Ollama is installed and get version"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, version
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False, None


def check_ollama_running() -> bool:
    """Check if Ollama service is running"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def cmd_setup() -> int:
    """Setup llm-gateway environment"""
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header, print_box_footer, print_status
    )

    print_box_header("LLM Gateway Setup", "🔧")

    gateway_dir = get_gateway_dir()

    if not gateway_dir.exists():
        print_status(f"Gateway directory not found: {gateway_dir}", "error")
        print_status("Please clone the llm-gateway repository first", "info")
        print_box_footer()
        return 1

    print_status(f"Gateway directory: {gateway_dir}", "success")

    # Check Ollama installation
    ollama_installed, ollama_version = check_ollama_installed()
    if ollama_installed:
        print_status(f"Ollama: {ollama_version}", "success")

        # Check if Ollama is running
        if check_ollama_running():
            print_status("Ollama service: running", "success")
        else:
            print_status("Ollama service: not running", "warning")
            print_status("Start Ollama with: ollama serve", "info")
    else:
        print_status("Ollama: not installed", "warning")
        print_status("Install from: https://ollama.ai", "info")

    # Setup Python venv
    venv_dir = gateway_dir / "venv"
    if venv_dir.exists():
        print_status("Python venv: already exists", "info")
    else:
        print_status("Creating Python venv...", "progress")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            print_status("Python venv created", "success")
        except subprocess.CalledProcessError as e:
            print_status(f"Failed to create venv: {e}", "error")
            print_box_footer()
            return 1

    # Install requirements
    requirements = gateway_dir / "requirements.txt"
    if requirements.exists():
        print_status("Installing Python dependencies...", "progress")
        pip = venv_dir / ("Scripts" if IS_WINDOWS else "bin") / ("pip.exe" if IS_WINDOWS else "pip")

        try:
            subprocess.run(
                [str(pip), "install", "-r", str(requirements)],
                check=True,
                capture_output=True
            )
            print_status("Dependencies installed", "success")
        except subprocess.CalledProcessError as e:
            print_status(f"Failed to install dependencies: {e}", "error")
            print_box_footer()
            return 1
    else:
        print_status(f"requirements.txt not found", "warning")

    # Generate initial config if not exists
    config_dir = gateway_dir / "data"
    config_file = config_dir / "config.json"

    if not config_file.exists():
        print_status("Creating initial config.json...", "progress")
        config_dir.mkdir(parents=True, exist_ok=True)

        initial_config = {
            "ollama_base_url": "http://localhost:11434",
            "default_model": "llama2",
            "slots": {},
            "api_keys": [],
            "cors_origins": ["*"]
        }

        try:
            config_file.write_text(json.dumps(initial_config, indent=2))
            print_status("config.json created", "success")
        except Exception as e:
            print_status(f"Failed to create config: {e}", "warning")

    # Linux: offer systemd service
    if IS_LINUX:
        print_status("", "info")
        print_status("To enable auto-start on Linux, you can create a systemd service:", "info")
        print_status(f"  sudo systemctl enable --now llm-gateway.service", "info")

    print_status("", "info")
    print_status("Setup complete!", "success")
    print_box_footer()
    return 0


def cmd_start(port: int = 4000, host: str = "0.0.0.0", background: bool = True) -> int:
    """Start llm-gateway server"""
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header, print_box_footer, print_status
    )

    print_box_header("LLM Gateway Start", "🚀")

    running, pid = is_running()
    if running:
        print_status(f"Already running (PID {pid})", "info")
        print_box_footer()
        return 0

    gateway_dir = get_gateway_dir()
    uvicorn = get_uvicorn(gateway_dir)

    if not uvicorn.exists():
        print_status("uvicorn not found. Run 'tb llm-gateway setup' first.", "error")
        print_box_footer()
        return 1

    # Check Ollama
    if not check_ollama_running():
        print_status("Ollama is not running", "warning")
        print_status("The gateway will start but won't work until Ollama is running", "warning")
        print_status("Start Ollama with: ollama serve", "info")

    cmd = [
        str(uvicorn),
        "server:app",
        "--host", host,
        "--port", str(port),
        "--workers", "1"
    ]

    print_status(f"Starting on {host}:{port}...", "progress")

    if background:
        if IS_WINDOWS:
            creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
            process = subprocess.Popen(
                cmd, cwd=gateway_dir,
                creationflags=creation_flags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            process = subprocess.Popen(
                cmd, cwd=gateway_dir,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        # Save PID
        pid_file = get_pids_dir() / "llm-gateway.pid"
        pid_file.write_text(str(process.pid))

        time.sleep(2)
        running, _ = is_running()

        if running:
            print_status(f"Started (PID {process.pid})", "success")
            print_status("", "info")
            print_status(f"API: http://localhost:{port}/v1/", "info")
            print_status(f"Admin: http://localhost:{port}/admin/", "info")
            print_status(f"Docs: http://localhost:{port}/docs", "info")
        else:
            print_status("Failed to start", "error")
            print_box_footer()
            return 1
    else:
        # Foreground mode
        print_status("Running in foreground (Ctrl+C to stop)", "info")
        print_box_footer()
        subprocess.run(cmd, cwd=gateway_dir)
        return 0

    print_box_footer()
    return 0


def cmd_stop(force: bool = False) -> int:
    """Stop llm-gateway server"""
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header, print_box_footer, print_status
    )

    print_box_header("LLM Gateway Stop", "🛑")

    running, pid = is_running()
    if not running:
        print_status("Not running", "info")
        print_box_footer()
        return 0

    print_status(f"Stopping PID {pid}...", "progress")

    try:
        if IS_WINDOWS:
            flag = ["/F"] if force else []
            subprocess.run(["taskkill", "/PID", str(pid)] + flag, check=True)
        else:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)

        # Wait for shutdown
        for _ in range(10):
            time.sleep(0.5)
            running, _ = is_running()
            if not running:
                break

        # Cleanup PID file
        pid_file = get_pids_dir() / "llm-gateway.pid"
        pid_file.unlink(missing_ok=True)

        print_status("Stopped", "success")
    except Exception as e:
        print_status(f"Error: {e}", "error")
        print_box_footer()
        return 1

    print_box_footer()
    return 0


def cmd_status() -> int:
    """Show llm-gateway status"""
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header, print_box_footer, print_status
    )

    print_box_header("LLM Gateway Status", "📊")

    gateway_dir = get_gateway_dir()
    running, pid = is_running()

    # Directory check
    if gateway_dir.exists():
        print_status(f"Directory: {gateway_dir}", "success")
    else:
        print_status(f"Directory not found: {gateway_dir}", "error")
        print_box_footer()
        return 1

    # Venv check
    venv_python = get_venv_python(gateway_dir)
    if venv_python.exists():
        print_status("Python venv: installed", "success")
    else:
        print_status("Python venv: not installed", "warning")
        print_status("Run: tb llm-gateway setup", "info")

    # Ollama check
    ollama_installed, ollama_version = check_ollama_installed()
    if ollama_installed:
        print_status(f"Ollama: {ollama_version}", "success")

        if check_ollama_running():
            print_status("Ollama service: running", "success")
        else:
            print_status("Ollama service: not running", "warning")
    else:
        print_status("Ollama: not installed", "warning")

    # Gateway running status
    if running:
        print_status(f"Gateway: running (PID {pid})", "success")

        # Try to query health endpoint
        try:
            import httpx
            response = httpx.get("http://localhost:4000/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                model_count = len(data.get("models", []))
                print_status(f"Health check: OK ({model_count} models)", "success")
        except Exception:
            print_status("Health check: failed (gateway may still be starting)", "warning")
    else:
        print_status("Gateway: stopped", "info")

    # Config check
    config_file = gateway_dir / "data" / "config.json"
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
            ollama_url = config.get("ollama_base_url", "http://localhost:11434")
            default_model = config.get("default_model", "N/A")
            print_status(f"Config: {default_model} @ {ollama_url}", "info")
        except Exception:
            print_status("Config: exists but cannot parse", "warning")
    else:
        print_status("Config: not found", "warning")

    print_box_footer()
    return 0


def cmd_restart(port: int = 4000, host: str = "0.0.0.0") -> int:
    """Restart llm-gateway server"""
    from toolboxv2.utils.clis.cli_printing import (
        print_status
    )

    print_status("Restarting gateway...", "progress")
    cmd_stop()
    time.sleep(1)
    return cmd_start(port=port, host=host)


def cmd_uninstall(keep_models: bool = True) -> int:
    """Uninstall llm-gateway"""
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header, print_box_footer, print_status
    )

    print_box_header("LLM Gateway Uninstall", "🗑️")

    # Stop if running
    running, _ = is_running()
    if running:
        print_status("Stopping server...", "progress")
        cmd_stop(force=True)

    gateway_dir = get_gateway_dir()

    if not gateway_dir.exists():
        print_status("Not installed", "info")
        print_box_footer()
        return 0

    # Remove venv
    venv_dir = gateway_dir / "venv"
    if venv_dir.exists():
        print_status("Removing Python venv...", "progress")
        shutil.rmtree(venv_dir, ignore_errors=True)
        print_status("Venv removed", "success")

    # Remove build directory
    build_dir = gateway_dir / "build"
    if build_dir.exists():
        print_status("Removing build directory...", "progress")
        shutil.rmtree(build_dir, ignore_errors=True)

    # Optionally keep models
    if not keep_models:
        models_dir = gateway_dir / "data" / "models"
        if models_dir.exists():
            print_status("Removing models...", "progress")
            shutil.rmtree(models_dir, ignore_errors=True)
    else:
        print_status("Keeping data directory", "info")

    # Remove PID file
    pid_file = get_pids_dir() / "llm-gateway.pid"
    pid_file.unlink(missing_ok=True)

    print_status("", "info")
    print_status("Uninstall complete", "success")
    print_status("Note: Ollama was not uninstalled (uninstall separately if needed)", "info")
    print_box_footer()
    return 0


def cli_llm_gateway():
    """CLI Entry Point for `tb llm-gateway`"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="tb llm-gateway",
        description="🌐 LLM Gateway - OpenAI-compatible local LLM server with Ollama backend"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # setup
    subparsers.add_parser("setup", help="Setup/install llm-gateway")

    # start
    p_start = subparsers.add_parser("start", help="Start the gateway server")
    p_start.add_argument("--port", "-p", type=int, default=4000, help="Port (default: 4000)")
    p_start.add_argument("--host", "-H", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    p_start.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop the gateway server")
    p_stop.add_argument("--force", "-f", action="store_true", help="Force stop")

    # status
    subparsers.add_parser("status", help="Show gateway status")

    # restart
    p_restart = subparsers.add_parser("restart", help="Restart the gateway server")
    p_restart.add_argument("--port", "-p", type=int, default=4000, help="Port (default: 4000)")
    p_restart.add_argument("--host", "-H", default="0.0.0.0", help="Host (default: 0.0.0.0)")

    # uninstall
    p_uninstall = subparsers.add_parser("uninstall", help="Uninstall llm-gateway")
    p_uninstall.add_argument("--remove-models", action="store_true", help="Also remove data directory")

    args = parser.parse_args()

    if args.command == "setup":
        return cmd_setup()
    elif args.command == "start":
        return cmd_start(port=args.port, host=args.host, background=not args.foreground)
    elif args.command == "stop":
        return cmd_stop(force=args.force)
    elif args.command == "status":
        return cmd_status()
    elif args.command == "restart":
        return cmd_restart(port=args.port, host=args.host)
    elif args.command == "uninstall":
        return cmd_uninstall(keep_models=not args.remove_models)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(cli_llm_gateway())
