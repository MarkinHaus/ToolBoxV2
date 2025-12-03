# file: toolboxv2/api_manager.py
# A production-style, platform-agnostic Rust server manager with enhanced modern UI
# and optional POSIX zero-downtime update support.

import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

# --- Enhanced UI Imports ---
try:
    from toolboxv2.utils.extras.Style import Spinner, Style
except ImportError:
    try:
        from toolboxv2.extras.Style import Spinner, Style
    except ImportError:
        print("FATAL: UI utilities not found. Ensure 'toolboxv2/extras/Style.py' exists.")
        sys.exit(1)

# --- CLI Printing Utilities ---
from toolboxv2.utils.clis.cli_printing import (
    print_box_header,
    print_box_content,
    print_box_footer,
    print_status,
    print_separator,
    print_table_header,
    print_table_row,
    print_code_block,
    run_visual_test
)

# --- Configuration ---
try:
    import psutil
except ImportError:
    print("FATAL: Required library 'psutil' not found.")
    print("Please install it using: pip install psutil")
    sys.exit(1)

# Constants
SERVER_STATE_FILE = "server_state.json"
PERSISTENT_FD_FILE = "server_socket.fd"
DEFAULT_EXECUTABLE_NAME = "simple-core-server"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080
SOCKET_BACKLOG = 128

# =================== Helper Functions ===================

def get_executable_name_with_extension(base_name=DEFAULT_EXECUTABLE_NAME):
    """Get platform-specific executable name"""
    if platform.system().lower() == "windows":
        return f"{base_name}.exe"
    return base_name


def get_executable_path():
    """Find the release executable in standard locations"""
    exe_name = get_executable_name_with_extension()
    from toolboxv2 import tb_root_dir

    search_paths = [
        tb_root_dir / Path("bin") / exe_name,
        tb_root_dir / Path("src-core") / exe_name,
        tb_root_dir / exe_name,
        tb_root_dir / Path("src-core") / "target" / "release" / exe_name,
    ]

    for path in search_paths:
        if path.exists() and path.is_file():
            return path.resolve()

    return None


def read_server_state(state_file=SERVER_STATE_FILE):
    """Read server state from file"""
    try:
        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)
                return state.get('pid'), state.get('version'), state.get('executable_path')
        return None, None, None
    except Exception:
        return None, None, None


def write_server_state(pid, server_version, executable_path, state_file=SERVER_STATE_FILE):
    """Write server state to file"""
    if executable_path is None:
        executable_path = ''
    try:
        state = {
            'pid': pid,
            'version': server_version,
            'executable_path': str(Path(executable_path).resolve())
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print_status(f"Error writing server state: {e}", "error")


def is_process_running(pid):
    """Check if process is running"""
    if pid is None or psutil is None:
        return False
    try:
        return psutil.pid_exists(int(pid))
    except (ValueError, TypeError):
        return False


def stop_process(pid, timeout=10):
    """Stop process gracefully with timeout"""
    if not is_process_running(pid):
        print_status(f"Process {pid} not running", "warning")
        return True

    print_box_header(f"Stopping Process", "⏹️")
    print_box_content(f"PID: {pid}", "info")
    print_box_content(f"Timeout: {timeout}s", "info")
    print_box_footer()

    with Spinner(f"Stopping process {pid}", symbols="+", time_in_s=timeout, count_down=True) as s:
        try:
            proc = psutil.Process(int(pid))
            proc.terminate()
            proc.wait(timeout)
        except psutil.TimeoutExpired:
            s.message = f"Force killing process {pid}"
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            print()
            print_status(f"Error stopping process {pid}: {e}", "error")
            return False

    print()
    print_status(f"Process {pid} stopped successfully", "success")
    return True


# =================== Platform-Specific Socket Management ===================

def ensure_socket_and_fd_file_posix(host, port, backlog, fd_file_path):
    """Create socket and FD file for POSIX zero-downtime updates"""
    if os.path.exists(fd_file_path):
        print_status(f"Stale FD file found: {fd_file_path}", "warning")
        print_status("Removing to create new socket", "info")
        with contextlib.suppress(OSError):
            os.remove(fd_file_path)

    try:
        print_status("Creating listening socket", "progress")

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        fd_num = server_socket.fileno()

        # Make socket inheritable
        if hasattr(os, 'set_inheritable'):
            os.set_inheritable(fd_num, True)
        else:
            import fcntl
            flags = fcntl.fcntl(fd_num, fcntl.F_GETFD)
            fcntl.fcntl(fd_num, fcntl.F_SETFD, flags & ~fcntl.FD_CLOEXEC)

        server_socket.bind((host, port))
        server_socket.listen(backlog)

        # Save FD to file
        with open(fd_file_path, 'w') as f:
            f.write(str(fd_num))
        os.chmod(fd_file_path, 0o600)

        print_status(f"Socket created - FD {fd_num} saved to {fd_file_path}", "success")
        return server_socket, fd_num

    except Exception as e:
        print_status(f"Failed to create listening socket: {e}", "error")
        if 'server_socket' in locals():
            server_socket.close()
        return None, None


def start_rust_server_posix(executable_path: str, persistent_fd: int):
    """Start Rust server on POSIX with socket FD passing"""
    abs_path = Path(executable_path).resolve()

    env = os.environ.copy()
    env["PERSISTENT_LISTENER_FD"] = str(persistent_fd)

    print_status(f"Starting {abs_path.name} with FD {persistent_fd}", "server")

    try:
        return subprocess.Popen(
            [str(abs_path)],
            cwd=abs_path.parent,
            env=env,
            pass_fds=[persistent_fd]
        )
    except Exception as e:
        print_status(f"Failed to start server: {e}", "error")
        return None


def start_rust_server_windows(executable_path: str):
    """Start Rust server on Windows"""
    abs_path = Path(executable_path).resolve()

    print_status(f"Starting {abs_path.name}", "server")
    print_status(f"Working directory: {abs_path.parent}", "info")

    try:
        return subprocess.Popen([str(abs_path)], cwd=abs_path.parent)
    except Exception as e:
        print_status(f"Failed to start server: {e}", "error")
        return None


# =================== Server Management Functions ===================

def start_new_server(executable_path, version_str, use_posix_zdt):
    """Start a new server instance"""
    current_pid, _, _ = read_server_state()

    if is_process_running(current_pid):
        print_box_header("Server Already Running", "⚠")
        print_box_content(f"PID: {current_pid}", "warning")
        print_box_content("Use 'stop' or 'update' command", "info")
        print_box_footer()
        return

    print_box_header(f"Starting Server v{version_str}", "🚀")
    print_box_content(f"Executable: {executable_path}", "info")
    print_box_content(f"Host: {SERVER_HOST}:{SERVER_PORT}", "network")

    is_posix = platform.system().lower() != "windows"

    if is_posix and use_posix_zdt:
        print_box_content("Mode: POSIX Zero-Downtime", "info")
    else:
        print_box_content("Mode: Standard Start", "info")

    print_box_footer()

    process = None
    socket_obj = None

    with Spinner(f"Launching server", symbols="d") as s:
        if is_posix and use_posix_zdt:
            socket_obj, fd = ensure_socket_and_fd_file_posix(
                SERVER_HOST, SERVER_PORT, SOCKET_BACKLOG, PERSISTENT_FD_FILE
            )
            if fd is not None:
                process = start_rust_server_posix(executable_path, fd)
        else:
            process = start_rust_server_windows(executable_path)

        time.sleep(2)  # Stabilization period

    # Close parent's socket handle
    if socket_obj:
        socket_obj.close()

    print()

    if process and process.poll() is None:
        write_server_state(process.pid, version_str, executable_path)

        print_box_header("Server Started", "✓")
        print_box_content(f"Version: {version_str}", "success")
        print_box_content(f"PID: {process.pid}", "success")
        print_box_content(f"Port: {SERVER_PORT}", "success")
        print_box_footer()
    else:
        print_box_header("Server Failed to Start", "✗")
        print_box_content("Check logs for details", "error")
        print_box_footer()
        write_server_state(None, None, None)


def update_server_posix(new_executable_path: str, new_version: str):
    """Perform POSIX zero-downtime update"""
    print_box_header(f"Zero-Downtime Update to v{new_version}", "🔄")
    print_box_content("Method: POSIX Socket Passing", "info")
    print_box_footer()

    old_pid, old_version, _ = read_server_state()

    if not os.path.exists(PERSISTENT_FD_FILE):
        print_status(f"FD file '{PERSISTENT_FD_FILE}' not found", "error")
        print_status("Cannot perform zero-downtime update", "error")
        return False

    try:
        with open(PERSISTENT_FD_FILE) as f:
            persistent_fd = int(f.read().strip())
        print_status(f"Using FD {persistent_fd} for socket passing", "info")
    except Exception as e:
        print_status(f"Error reading FD from file: {e}", "error")
        return False

    # Step 1: Start new server
    print()
    print_separator("═")
    print("  PHASE 1: Starting New Server")
    print_separator("═")
    print()

    with Spinner(f"Starting new server v{new_version}", symbols="d") as s:
        new_process = start_rust_server_posix(new_executable_path, persistent_fd)
        time.sleep(3)

    print()

    if new_process is None or new_process.poll() is not None:
        print_status("New server process died on startup", "error")
        print_status("Update aborted", "error")
        return False

    print_status(f"New server started (PID: {new_process.pid})", "success")

    # Step 2: Stop old server
    print()
    print_separator("═")
    print("  PHASE 2: Stopping Old Server")
    print_separator("═")
    print()

    if stop_process(old_pid):
        write_server_state(new_process.pid, new_version, new_executable_path)

        print()
        print_box_header("Update Complete", "✓")
        print_box_content(f"Old Version: {old_version} (PID: {old_pid})", "info")
        print_box_content(f"New Version: {new_version} (PID: {new_process.pid})", "success")
        print_box_content("Zero downtime achieved!", "success")
        print_box_footer()
        return True
    else:
        print_status("Failed to stop old process", "error")
        print_status("Manual intervention may be required", "warning")
        stop_process(new_process.pid)
        return False


def update_server_graceful_restart(new_executable_path: str, new_version: str):
    """Perform graceful restart update"""
    print_box_header(f"Graceful Restart to v{new_version}", "🔄")
    print_box_content("Method: Stop & Start", "info")
    print_box_footer()

    old_pid, old_version, _ = read_server_state()

    # Step 1: Stop old server
    print_separator("═")
    print("  PHASE 1: Stopping Current Server")
    print_separator("═")
    print()

    if not stop_process(old_pid):
        print_status("Failed to stop old server", "error")
        print_status("Update aborted to prevent conflicts", "error")
        return False

    # Step 2: Start new server
    print()
    print_separator("═")
    print("  PHASE 2: Starting New Server")
    print_separator("═")
    print()

    start_new_server(new_executable_path, new_version, False)

    return True


def update_server(new_executable_path: str, new_version: str, use_posix_zdt: bool):
    """High-level update function"""
    is_posix = platform.system().lower() != "windows"

    if is_posix and use_posix_zdt:
        return update_server_posix(new_executable_path, new_version)
    else:
        if use_posix_zdt and not is_posix:
            print_status("--posix-zdt flag ignored on Windows", "warning")
            print_status("Using graceful restart instead", "info")
            print()
        return update_server_graceful_restart(new_executable_path, new_version)


def show_server_status(use_posix_zdt: bool):
    """Display current server status"""
    pid, ver, exe = read_server_state()

    print_box_header("Server Status", "🖥️")
    print()

    if is_process_running(pid):
        # Running - show full details
        columns = [
            ("Property", 15),
            ("Value", 50)
        ]
        widths = [w for _, w in columns]

        print_table_header(columns, widths)
        print_table_row(["Status", "RUNNING"], widths, ["white", "green"])
        print_table_row(["PID", str(pid)], widths, ["white", "grey"])
        print_table_row(["Version", ver or 'N/A'], widths, ["white", "yellow"])
        print_table_row(["Executable", str(exe) if exe else 'N/A'], widths, ["white", "cyan"])
        print_table_row(["Host", SERVER_HOST], widths, ["white", "blue"])
        print_table_row(["Port", str(SERVER_PORT)], widths, ["white", "blue"])

        # Check for POSIX ZDT
        is_posix = platform.system().lower() != "windows"
        if is_posix and os.path.exists(PERSISTENT_FD_FILE) and use_posix_zdt:
            try:
                with open(PERSISTENT_FD_FILE) as f:
                    fd_val = f.read().strip()
                print_table_row(["ZDT Mode", f"Active (FD: {fd_val})"], widths, ["white", "green"])
            except Exception:
                pass

        print()
        print_status("Server is healthy and running", "success")

    else:
        print_box_content("Status: STOPPED", "error")
        if pid:
            print_box_content(f"Stale PID in state: {pid}", "warning")
        print()
        print_status("Server is not running", "warning")

    print_box_footer()


def build_nuitka_module(tb_root_dir):
    """Build app_singleton.py with Nuitka"""
    print_separator("═")
    print("  Building Python Module with Nuitka")
    print_separator("═")
    print()

    src_core_path = tb_root_dir / "src-core"
    app_singleton_py = src_core_path / "app_singleton.py"
    build_dir = src_core_path / "build"

    if not app_singleton_py.exists():
        print_status(f"app_singleton.py not found at {app_singleton_py}", "warning")
        print_status("Skipping Nuitka build", "info")
        return False

    print_status(f"Source: {app_singleton_py}", "info")
    print_status(f"Output: {build_dir}", "info")
    print()

    try:
        with Spinner("Compiling with Nuitka", symbols="t", time_in_s=120) as s:
            result = subprocess.run(
                [
                    "python", "-m", "nuitka",
                    "--module",
                    "app_singleton.py",
                    "--output-dir=./build",
                    "--remove-output"
                ],
                cwd=src_core_path,
                check=True,
                capture_output=True,
                text=True
            )

        print()
        print_status("Nuitka compilation successful", "success")

        # Find the compiled .pyd file
        pyd_files = list(build_dir.glob("app_singleton*.pyd"))
        if pyd_files:
            pyd_file = pyd_files[0]
            print_status(f"Created: {pyd_file.name}", "success")

            # Copy to bin/build directory
            bin_build_dir = tb_root_dir / "bin" / "build"
            bin_build_dir.mkdir(parents=True, exist_ok=True)

            dest_pyd = bin_build_dir / pyd_file.name
            shutil.copy(pyd_file, dest_pyd)
            print_status(f"Copied to: {dest_pyd}", "success")

            # Also copy DLL files if they exist
            dll_files = list(build_dir.glob("*.dll"))
            for dll_file in dll_files:
                dest_dll = bin_build_dir / dll_file.name
                shutil.copy(dll_file, dest_dll)
                print_status(f"Copied DLL: {dll_file.name}", "info")

            return True
        else:
            print_status("Warning: No .pyd file found after compilation", "warning")
            return False

    except subprocess.CalledProcessError as e:
        print()
        print_status("Nuitka compilation failed", "error")
        if e.stderr:
            print("\nError output:")
            print(e.stderr)
        return False
    except FileNotFoundError:
        print()
        print_status("Nuitka not found - install with: pip install nuitka", "warning")
        print_status("Skipping Nuitka build", "info")
        return False


def handle_build(mode="release", skip_nuitka=False):
    """Build the Rust project

    Args:
        mode: Build mode - "debug" or "release" (default: "release")
        skip_nuitka: Skip Nuitka compilation step (default: False)
    """
    mode_display = mode.capitalize()
    print_box_header(f"Building Rust API Server ({mode_display})", "🔨")
    print_box_content("Compiler: Cargo (Rust)", "info")
    print_box_content(f"Mode: {mode_display}", "info")
    print_box_footer()

    from toolboxv2 import tb_root_dir

    # Step 1: Build Nuitka module (if not skipped)
    if not skip_nuitka:
        print()
        nuitka_success = build_nuitka_module(tb_root_dir)
        if nuitka_success:
            print()

    # Step 2: Build Rust server
    print_separator("═")
    print(f"  Building Rust Server ({mode_display})")
    print_separator("═")
    print()

    try:
        # Determine cargo command based on mode
        cargo_cmd = ["cargo", "build"]
        if mode == "release":
            cargo_cmd.append("--release")

        with Spinner(f"Compiling Rust project ({mode})", symbols="t", time_in_s=180) as s:
            result = subprocess.run(
                cargo_cmd,
                cwd=tb_root_dir / "src-core",
                check=True,
                capture_output=True,
                text=True
            )

        print()
        print_status("Build completed successfully", "success")

        # Copy executable
        src_core_path = tb_root_dir / "src-core"
        exe_name = get_executable_name_with_extension()

        # Determine source path based on mode
        if mode == "release":
            exe_source = src_core_path / "target" / "release" / exe_name
        else:
            exe_source = src_core_path / "target" / "debug" / exe_name

        if exe_source.exists():
            bin_dir = tb_root_dir / "bin"
            bin_dir.mkdir(exist_ok=True)

            try:
                dest_path = bin_dir / exe_name
                shutil.copy(exe_source, dest_path)
                print_status(f"Executable copied to: {dest_path}", "success")

                # Show file size
                size_mb = dest_path.stat().st_size / (1024 * 1024)
                print_status(f"Size: {size_mb:.2f} MB", "info")

            except Exception as e:
                print_status(f"Warning: Failed to copy to bin: {e}", "warning")

                # Fallback to ubin
                ubin_dir = tb_root_dir / "ubin"
                ubin_dir.mkdir(exist_ok=True)
                dest_path = ubin_dir / exe_name

                try:
                    shutil.copy(exe_source, dest_path)
                    print_status(f"Copied to fallback location: {dest_path}", "info")
                except Exception as e_ubin:
                    print_status(f"Error copying to ubin: {e_ubin}", "error")
        else:
            print_status(f"Warning: Executable not found at {exe_source}", "warning")

    except subprocess.CalledProcessError as e:
        print()
        print_box_header("Build Failed", "✗")
        print_box_content("Compilation errors occurred", "error")
        print_box_footer()
        print("\nError output:")
        print(e.stderr)

    except FileNotFoundError:
        print()
        print_box_header("Build Failed", "✗")
        print_box_content("'cargo' command not found", "error")
        print_box_content("Is Rust installed and in your PATH?", "info")
        print_box_footer()


def cleanup_build_files():
    """Clean build artifacts"""
    print_box_header("Cleaning Build Artifacts", "🧹")
    print_box_footer()

    from toolboxv2 import tb_root_dir
    src_core_path = tb_root_dir / "src-core"
    target_path = src_core_path / "target"

    if target_path.exists():
        try:
            with Spinner("Running cargo clean", symbols="+") as s:
                try:
                    subprocess.run(
                        ["cargo", "clean"],
                        cwd=src_core_path,
                        check=True,
                        capture_output=True
                    )
                except subprocess.CalledProcessError:
                    s.message = "Manually removing build directories"
                    for item in target_path.iterdir():
                        if item.is_dir() and item.name != ".rustc_info.json":
                            shutil.rmtree(item)

            print()
            print_status("Build artifacts cleaned successfully", "success")
            return True

        except Exception as e:
            print()
            print_status(f"Failed to clean build files: {e}", "error")
            return False
    else:
        print_status(f"Build directory not found: {target_path}", "warning")
        return True


def remove_release_executable():
    """Remove release executable"""
    print_box_header("Removing Release Executable", "🗑️")
    print_box_footer()

    from toolboxv2 import tb_root_dir
    src_core_path = tb_root_dir / "src-core"
    exe_name = get_executable_name_with_extension()

    paths_to_remove = [
        src_core_path / exe_name,
        src_core_path / "target" / "release" / exe_name
    ]

    removed_count = 0

    for path in paths_to_remove:
        if path.exists():
            try:
                path.unlink()
                print_status(f"Removed: {path}", "success")
                removed_count += 1
            except Exception as e:
                print_status(f"Failed to remove {path}: {e}", "error")

    if removed_count == 0:
        print_status("No executables found to remove", "warning")
    else:
        print()
        print_status(f"Removed {removed_count} executable(s)", "success")

    return True


def handle_debug():
    """Run server in debug mode with hot reload"""
    print_box_header("Debug Mode with Hot Reload", "🔥")
    print_box_content("Watching for file changes", "info")
    print_box_content("Press Ctrl+C to stop", "info")
    print_box_footer()

    from toolboxv2 import tb_root_dir
    src_core_path = tb_root_dir / "src-core"


    print()
    nuitka_success = build_nuitka_module(tb_root_dir)
    if nuitka_success:
        print()

    # Check if cargo-watch is installed
    try:
        subprocess.run(
            ["cargo", "watch", "--version"],
            check=True,
            cwd=src_core_path,
            capture_output=True
        )
    except Exception:
        print_status("cargo-watch not installed", "warning")
        print_status("Installing cargo-watch...", "progress")

        try:
            with Spinner("Installing cargo-watch", symbols="t"):
                subprocess.run(
                    ["cargo", "install", "cargo-watch"],
                    check=True,
                    cwd=src_core_path,
                    capture_output=True
                )
            print()
            print_status("cargo-watch installed successfully", "success")
        except subprocess.CalledProcessError as e:
            print()
            print_status("Failed to install cargo-watch", "error")
            print_status("Falling back to standard debug mode", "warning")
            print()

            # Fallback to standard debug
            print_separator("═")
            print("  Running in standard debug mode")
            print_separator("═")
            print()

            try:
                subprocess.run(["cargo", "run"], cwd=src_core_path)
            except KeyboardInterrupt:
                print()
                print_status("Debug mode stopped", "info")
            return

    # Run with hot reload
    print_separator("═")
    print("  Starting Hot Reload Session")
    print_separator("═")
    print()

    try:
        subprocess.run(["cargo", "watch", "-x", "run"], cwd=src_core_path)
    except KeyboardInterrupt:
        print()
        print_status("Hot reload stopped", "info")
    except subprocess.CalledProcessError as e:
        print_status(f"Hot reload failed: {e}", "error")


def manage_server(action: str, executable_path: str = None, version_str: str = "unknown", use_posix_zdt: bool = False):
    """Main server management dispatcher"""

    if action == "start":
        if not executable_path:
            executable_path = get_executable_path()

        if not executable_path:
            print_box_header("Executable Not Found", "✗")
            print_box_content("No compiled executable found", "error")
            print_box_content("Build first with: tb api build", "info")
            print_box_footer()
            return False

        start_new_server(executable_path, version_str, use_posix_zdt)

    elif action == "stop":
        pid, _, _ = read_server_state()
        if stop_process(pid):
            write_server_state(None, None, None)

            # Clean up FD file if exists
            is_posix = platform.system().lower() != "windows"
            if is_posix and os.path.exists(PERSISTENT_FD_FILE):
                print()
                print_status(f"Note: Stale FD file exists: {PERSISTENT_FD_FILE}", "warning")
                print_status("Consider removing it before next start", "info")

    elif action == "update":
        if not executable_path:
            print_box_header("Update Failed", "✗_")
            print_box_content("New executable path required (--exe)", "error")
            print_box_footer()
            return False

        if not version_str or version_str == "unknown":
            print_box_header("Update Failed", "✗_")
            print_box_content("Version string required (--version)", "error")
            print_box_footer()
            return False

        update_server(executable_path, version_str, use_posix_zdt)

    elif action == "status":
        show_server_status(use_posix_zdt)

    return True


# =================== CLI Entry Point ===================

def cli_api_runner():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='tb api',
        description='🚀 Platform-Agnostic Rust API Server Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔════════════════════════════════════════════════════════════════════════════╗
║                           Command Examples                                 ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Build & Development:                                                      ║
║    $ tb api build                    # Build release (with Nuitka)         ║
║    $ tb api build --mode debug       # Build debug version                 ║
║    $ tb api build --skip-nuitka      # Build without Nuitka step           ║
║    $ tb api debug                    # Run with hot reload                 ║
║    $ tb api clean                    # Clean build artifacts               ║
║    $ tb api remove-exe               # Remove compiled executable          ║
║    $ tb api visual-test              # Run visual UI component test        ║
║                                                                            ║
║  Server Management:                                                        ║
║    $ tb api start                    # Start server                        ║
║    $ tb api stop                     # Stop server                         ║
║    $ tb api status                   # Check server status                 ║
║                                                                            ║
║  Advanced Options:                                                         ║
║    $ tb api start --posix-zdt        # Start with zero-downtime (Linux)    ║
║    $ tb api start --exe /path --version 1.0.0                              ║
║                                                                            ║
║  Zero-Downtime Updates (Linux/macOS):                                      ║
║    $ tb api update --exe /new/path --version 1.1.0 --posix-zdt             ║
║                                                                            ║
║  Graceful Updates (All Platforms):                                         ║
║    $ tb api update --exe /new/path --version 1.1.0                         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Server Configuration:
  Host: {SERVER_HOST}
  Port: {SERVER_PORT}

Zero-Downtime Updates:
  Available on Linux and macOS using --posix-zdt flag
  Uses socket file descriptor passing for seamless updates
  Windows uses graceful restart (brief downtime)
        """
    )

    subparsers = parser.add_subparsers(dest="action", required=False, help="Available actions")

    # Build commands
    build_parser = subparsers.add_parser('build', help='Build the Rust project')
    build_parser.add_argument(
        '--mode',
        type=str,
        choices=['debug', 'release'],
        default='release',
        help='Build mode: debug or release (default: release)'
    )
    build_parser.add_argument(
        '--skip-nuitka',
        action='store_true',
        help='Skip Nuitka compilation of Python modules'
    )

    subparsers.add_parser('debug', help='Run in debug mode with hot reload')
    subparsers.add_parser('clean', help='Clean build artifacts')
    subparsers.add_parser('remove-exe', help='Remove release executable')
    subparsers.add_parser('visual-test', help='Run visual test for UI components')

    # Server management commands
    actions = {
        'start': 'Start the API server',
        'stop': 'Stop the running server',
        'update': 'Update server to new version',
        'status': 'Display server status'
    }

    for action, help_text in actions.items():
        p = subparsers.add_parser(action, help=help_text)

        if action in ['start', 'update', 'status']:
            p.add_argument(
                '--posix-zdt',
                action='store_true',
                help='(Linux/macOS) Enable POSIX zero-downtime updates via socket passing'
            )

        if action in ['start', 'update']:
            p.add_argument(
                '--exe',
                type=str,
                help='Path to server executable'
            )
            p.add_argument(
                '--version',
                type=str,
                default='unknown',
                help='Version string for the server'
            )

    args = parser.parse_args()

    # Handle simple actions
    if args.action == 'build':
        mode = getattr(args, 'mode', 'release')
        skip_nuitka = getattr(args, 'skip_nuitka', False)
        handle_build(mode=mode, skip_nuitka=skip_nuitka)
        return

    if args.action == 'clean':
        cleanup_build_files()
        return

    if args.action == 'remove-exe':
        remove_release_executable()
        return

    if args.action == 'debug':
        handle_debug()
        return

    if args.action == 'visual-test':
        run_visual_test()
        return

    if not args.action:
        # Default to status if no action provided
        show_server_status(False)
        return

    # Handle server management
    manage_server(
        action=args.action,
        executable_path=getattr(args, 'exe', None),
        version_str=getattr(args, 'version', 'unknown'),
        use_posix_zdt=getattr(args, 'posix_zdt', False)
    )


if __name__ == "__main__":
    cli_api_runner()
