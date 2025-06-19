import re
from packaging import version
from toolboxv2 import tb_root_dir


try:
    import psutil
except ImportError:
    print("Warning: 'psutil' library not found. Process management features will be limited.")
    psutil = None

def find_highest_zip_version_entry(name, target_app_version=None, filepath='tbState.yaml'):
    """
    Findet den Eintrag mit der höchsten ZIP-Version für einen gegebenen Namen und eine optionale Ziel-App-Version in einer YAML-Datei.

    :param name: Der Name des gesuchten Eintrags.
    :param target_app_version: Die Zielversion der App als String (optional).
    :param filepath: Der Pfad zur YAML-Datei.
    :return: Den Eintrag mit der höchsten ZIP-Version innerhalb der Ziel-App-Version oder None, falls nicht gefunden.
    """
    import yaml
    highest_zip_ver = None
    highest_entry = {}

    with open(filepath) as file:
        data = yaml.safe_load(file)
        # print(data)
        app_ver_h = None
        for key, value in list(data.get('installable', {}).items())[::-1]:
            # Prüfe, ob der Name im Schlüssel enthalten ist

            if name in key:
                v = value['version']
                if len(v) == 1:
                    app_ver = v[0].split('v')[-1]
                    zip_ver = "0.0.0"
                else:
                    app_ver, zip_ver = v
                    app_ver = app_ver.split('v')[-1]
                app_ver = version.parse(app_ver)
                # Wenn eine Ziel-App-Version angegeben ist, vergleiche sie
                if target_app_version is None or app_ver == version.parse(target_app_version):
                    current_zip_ver = version.parse(zip_ver)
                    # print(current_zip_ver, highest_zip_ver)

                    if highest_zip_ver is None or current_zip_ver > highest_zip_ver:
                        highest_zip_ver = current_zip_ver
                        highest_entry = value

                    if app_ver_h is None or app_ver > app_ver_h:
                        app_ver_h = app_ver
                        highest_zip_ver = current_zip_ver
                        highest_entry = value
    return highest_entry


def find_highest_zip_version(name_filter: str, app_version: str = None, root_dir: str = "mods_sto", version_only=False) -> str:
    """
    Findet die höchste verfügbare ZIP-Version in einem Verzeichnis basierend auf einem Namensfilter.

    Args:
        root_dir (str): Wurzelverzeichnis für die Suche
        name_filter (str): Namensfilter für die ZIP-Dateien
        app_version (str, optional): Aktuelle App-Version für Kompatibilitätsprüfung

    Returns:
        str: Pfad zur ZIP-Datei mit der höchsten Version oder None wenn keine gefunden
    """

    # Kompiliere den Regex-Pattern für die Dateinamen
    pattern = fr"{name_filter}&v[0-9.]+§([0-9.]+)\.zip$"

    highest_version = None
    highest_version_file = None

    # Durchsuche das Verzeichnis
    root_path = Path(root_dir)
    for file_path in root_path.rglob("*.zip"):
        if "RST$"+name_filter not in str(file_path):
            continue
        match = re.search(pattern, str(file_path).split("RST$")[-1].strip())
        if match:
            zip_version = match.group(1)

            # Prüfe App-Version Kompatibilität falls angegeben
            if app_version:
                file_app_version = re.search(r"&v([0-9.]+)§", str(file_path)).group(1)
                if version.parse(file_app_version) > version.parse(app_version):
                    continue

            # Vergleiche Versionen
            current_version = version.parse(zip_version)
            if highest_version is None or current_version > highest_version:
                highest_version = current_version
                highest_version_file = str(file_path)
    if version_only:
        return str(highest_version)
    return highest_version_file


def detect_os_and_arch():
    """Detect the current operating system and architecture."""
    current_os = platform.system().lower()  # e.g., 'windows', 'linux', 'darwin'
    machine = platform.machine().lower()  # e.g., 'x86_64', 'amd64'
    return current_os, machine


def query_executable_url(current_os, machine):
    """
    Query a remote URL for a matching executable based on OS and architecture.
    The file name is built dynamically based on parameters.
    """
    base_url = "https://example.com/downloads"  # Replace with the actual URL
    # Windows executables have .exe extension
    if current_os == "windows":
        file_name = f"server_{current_os}_{machine}.exe"
    else:
        file_name = f"server_{current_os}_{machine}"
    full_url = f"{base_url}/{file_name}"
    return full_url, file_name


def download_executable(url, file_name):
    """Attempt to download the executable from the provided URL."""
    try:
        import requests
    except ImportError:
        print("The 'requests' library is required. Please install it via pip install requests")
        sys.exit(1)

    print(f"Attempting to download executable from {url}...")
    try:
        response = requests.get(url, stream=True)
    except Exception as e:
        print(f"Download error: {e}")
        return None

    if response.status_code == 200:
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        # Make the file executable on non-Windows systems
        if platform.system().lower() != "windows":
            os.chmod(file_name, 0o755)
        return file_name
    else:
        print("Download failed. Status code:", response.status_code)
        return None


def run_executable(file_path):
    """Run the executable file."""
    try:
        print("Running it.")
        subprocess.run([os.path.abspath(file_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute {file_path}: {e}")
    except KeyboardInterrupt:
        print("Exiting call from:", file_path)


def check_cargo_installed():
    """Check if Cargo (Rust package manager) is installed on the system."""
    try:
        subprocess.run(["cargo", "--version"], check=True, capture_output=True)
        return True
    except Exception:
        return False


def build_cargo_project(debug=False):
    """Build the Cargo project, optionally in debug mode."""
    mode = "debug" if debug else "release"
    args = ["cargo", "build"]
    if not debug:
        args.append("--release")

    print(f"Building in {mode} mode...")
    try:
        subprocess.run(args, cwd=os.path.join(".", "src-core"), check=True)
        exe_path = get_executable_name_with_extension()
        if exe_path:
            bin_dir = tb_root_dir / "bin"
            bin_dir.mkdir(exist_ok=True)
            exe_path = Path(exe_path)
            try:
                shutil.copy(exe_path, bin_dir / exe_path.name)
            except Exception as e:
                bin_dir = tb_root_dir / "ubin"
                bin_dir.mkdir(exist_ok=True)
                (bin_dir / exe_path.name).unlink(missing_ok=True)
                try:
                    shutil.copy(exe_path, bin_dir / exe_path.name)
                except Exception as e:
                    print(f"Failed to copy executable: {e}")
            print(f"Copied executable to '{bin_dir.resolve()}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Cargo build failed: {e}")
        return False


def run_with_hot_reload():
    """Run the Cargo project with hot reloading."""
    src_core_path = os.path.join(".", "src-core")

    # Check if cargo-watch is installed
    try:
        subprocess.run(["cargo", "watch", "--version"], check=True, capture_output=True)
    except Exception:
        print("cargo-watch is not installed. Installing now...")
        try:
            subprocess.run(["cargo", "install", "cargo-watch"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install cargo-watch: {e}")
            print("Running without hot reload")
            return run_in_debug_mode()

    print("Running with hot reload in debug mode...")
    try:
        subprocess.run(["cargo", "watch", "-x", "run"], cwd=src_core_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Hot reload execution failed: {e}")
        return False


def run_in_debug_mode():
    """Run the Cargo project in debug mode."""
    src_core_path = os.path.join(".", "src-core")
    print("Running in debug mode...")
    try:
        subprocess.run(["cargo", "run"], cwd=src_core_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Debug execution failed: {e}")
        return False


def remove_release_executable():
    """Removes the release executable."""
    src_core_path = os.path.join(".", "src-core")
    expected_name = "simple-core-server.exe" if platform.system().lower() == "windows" else "simple-core-server"

    # Remove from src-core root
    direct_path = os.path.join(src_core_path, expected_name)
    if os.path.exists(direct_path):
        try:
            os.remove(direct_path)
            print(f"Removed release executable: {direct_path}")
        except Exception as e:
            print(f"Failed to remove {direct_path}: {e}")

    # Remove from target/release
    release_path = os.path.join(src_core_path, "target", "release", expected_name)
    if os.path.exists(release_path):
        try:
            os.remove(release_path)
            print(f"Removed release executable: {release_path}")
        except Exception as e:
            print(f"Failed to remove {release_path}: {e}")

    return True


def cleanup_build_files():
    """Cleans up build files."""
    src_core_path = os.path.join(".", "src-core")
    target_path = os.path.join(src_core_path, "target")

    if os.path.exists(target_path):
        try:
            print(f"Cleaning up build files in {target_path}...")
            # First try using cargo clean
            try:
                subprocess.run(["cargo", "clean"], cwd=src_core_path, check=True)
                print("Successfully cleaned up build files with cargo clean")
            except subprocess.CalledProcessError:
                # If cargo clean fails, manually remove directories
                print("Cargo clean failed, manually removing build directories...")
                for item in os.listdir(target_path):
                    item_path = os.path.join(target_path, item)
                    if os.path.isdir(item_path) and item != ".rustc_info.json":
                        shutil.rmtree(item_path)
                        print(f"Removed {item_path}")
            return True
        except Exception as e:
            print(f"Failed to clean up build files: {e}")
            return False
    else:
        print(f"Build directory {target_path} not found")
        return True



# file: toolboxv2/api_manager.py
# A production-style, platform-agnostic Rust server manager with an enhanced UI
# and optional POSIX zero-downtime update support.

import argparse
import contextlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import textwrap
from pathlib import Path

# --- Enhanced UI Imports ---
try:
    from ..extras.Style import Style, Spinner
except ImportError:
    # Fallback for different execution contexts
    try:
        from toolboxv2.extras.Style import Style, Spinner
    except ImportError:
        print("FATAL: UI utilities not found. Ensure 'toolboxv2/extras/Style.py' exists.")
        sys.exit(1)

# --- Configuration ---
try:
    import psutil
except ImportError:
    print(Style.RED("FATAL: Required library 'psutil' not found."))
    print(Style.YELLOW("Please install it using: pip install psutil"))
    sys.exit(1)

# These constants should be in a shared config or directly here
SERVER_STATE_FILE = "server_state.json"
PERSISTENT_FD_FILE = "server_socket.fd"
DEFAULT_EXECUTABLE_NAME = "simple-core-server"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080
SOCKET_BACKLOG = 128


# --- Helper Functions (Functionality 1-to-1) ---

def get_executable_name_with_extension(base_name=DEFAULT_EXECUTABLE_NAME):
    if platform.system().lower() == "windows":
        return f"{base_name}.exe"
    return base_name


def get_executable_path():
    """Find the release executable in standard locations."""
    # This function is simplified from your example to match this script's scope
    exe_name = get_executable_name_with_extension()
    from toolboxv2 import tb_root_dir
    search_paths = [
        tb_root_dir / Path("bin") / exe_name,
        tb_root_dir / Path("src-core") / exe_name,
        tb_root_dir / exe_name,
        tb_root_dir / Path("src-core") / "target" / "release" / exe_name,
    ]
    for path in search_paths:
        print(path)
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def read_server_state(state_file=SERVER_STATE_FILE):
    try:
        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)
                return state.get('pid'), state.get('version'), state.get('executable_path')
        return None, None, None
    except Exception:
        return None, None, None


def write_server_state(pid, server_version, executable_path, state_file=SERVER_STATE_FILE):
    if executable_path is None:
        executable_path = ''
    try:
        state = {'pid': pid, 'version': server_version, 'executable_path': str(Path(executable_path).resolve())}
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(Style.RED(f"Error writing server state: {e}"))


def is_process_running(pid):
    if pid is None or psutil is None: return False
    try:
        return psutil.pid_exists(int(pid))
    except (ValueError, TypeError):
        return False


def stop_process(pid, timeout=10):
    if not is_process_running(pid):
        print(Style.YELLOW(f"Process {pid} not running or psutil unavailable."))
        return True

    with Spinner(f"Stopping process {pid}", symbols="+", time_in_s=timeout, count_down=True) as s:
        try:
            proc = psutil.Process(int(pid))
            proc.terminate()
            proc.wait(timeout)
        except psutil.TimeoutExpired:
            s.message = f"Force killing process {pid}"
            proc.kill()
        except psutil.NoSuchProcess:
            pass  # Already gone
        except Exception as e:
            print(f"\n{Style.RED2('Error stopping process')} {pid}: {e}")
            return False

    print(f"\n{Style.VIOLET2('Process')} {pid} {Style.VIOLET2('stopped.')}")
    return True


# --- Platform-Specific Logic (Functionality 1-to-1) ---

def ensure_socket_and_fd_file_posix(host, port, backlog, fd_file_path) -> tuple[socket.socket | None, int | None]:
    if os.path.exists(fd_file_path):
        print(Style.YELLOW(f"[POSIX] Stale FD file found: {fd_file_path}. Removing to create a new socket."))
        with contextlib.suppress(OSError):
            os.remove(fd_file_path)

    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        fd_num = server_socket.fileno()
        if hasattr(os, 'set_inheritable'):
            os.set_inheritable(fd_num, True)
        else:
            import fcntl
            flags = fcntl.fcntl(fd_num, fcntl.F_GETFD)
            fcntl.fcntl(fd_num, fcntl.F_SETFD, flags & ~fcntl.FD_CLOEXEC)

        server_socket.bind((host, port))
        server_socket.listen(backlog)
        with open(fd_file_path, 'w') as f:
            f.write(str(fd_num))
        os.chmod(fd_file_path, 0o600)
        print(Style.GREEN(f"[POSIX] Created new socket. FD {fd_num} saved to {fd_file_path}."))
        return server_socket, fd_num
    except Exception as e:
        print(Style.RED(f"[POSIX] Fatal: Could not create listening socket FD: {e}"))
        if 'server_socket' in locals():
            server_socket.close()
        return None, None


def start_rust_server_posix(executable_path: str, persistent_fd: int):
    abs_path = Path(executable_path).resolve()
    env = os.environ.copy()
    env["PERSISTENT_LISTENER_FD"] = str(persistent_fd)
    print(Style.CYAN(f"[POSIX] Starting Rust server {abs_path.name} using FD {persistent_fd}"))
    try:
        return subprocess.Popen([str(abs_path)], cwd=abs_path.parent, env=env, pass_fds=[persistent_fd])
    except Exception as e:
        print(Style.RED(f"[POSIX] Failed to start Rust server: {e}"))
        return None


def start_rust_server_windows(executable_path: str):
    abs_path = Path(executable_path).resolve()
    print(Style.CYAN(f"[WINDOWS] Starting Rust server {abs_path.name} (will bind its own socket). in {abs_path.parent}"))
    try:
        return subprocess.Popen([str(abs_path)], cwd=abs_path.parent)
    except Exception as e:
        print(Style.RED(f"[WINDOWS] Failed to start Rust server: {e}"))
        return None


# --- Main Management Logic with UI Enhancements ---

def update_server(new_executable_path: str, new_version: str, use_posix_zdt: bool):
    """High-level update function, calls platform-specific logic."""
    # Only use POSIX ZDT if flag is set AND on a non-windows system
    is_posix = platform.system().lower() != "windows"
    if is_posix and use_posix_zdt:
        return update_server_posix(new_executable_path, new_version)
    else:
        if use_posix_zdt and not is_posix:
            print(Style.YELLOW("Warning: --posix-zdt flag ignored on Windows. Using graceful restart."))
        return update_server_graceful_restart(new_executable_path, new_version)


def update_server_posix(new_executable_path: str, new_version: str):
    header = f"--- [POSIX] Starting Zero-Downtime Update to {Style.YELLOW(new_version)} ---"
    print(Style.Bold(header))
    if not psutil: return False
    old_pid, old_version, _ = read_server_state()

    if not os.path.exists(PERSISTENT_FD_FILE):
        print(Style.RED(f"[POSIX] Error: FD file '{PERSISTENT_FD_FILE}' not found. Cannot perform ZDT update."))
        return False
    try:
        with open(PERSISTENT_FD_FILE) as f:
            persistent_fd = int(f.read().strip())
    except Exception as e:
        print(Style.RED(f"[POSIX] Error reading FD from file: {e}"))
        return False

    with Spinner(f"Starting new server v{new_version}", symbols="d") as s:
        new_process = start_rust_server_posix(new_executable_path, persistent_fd)
        time.sleep(3)  # Allow time to initialize

    if new_process is None or new_process.poll() is not None:
        print(f"\n{Style.RED2('Update failed:')} New server process died on startup.")
        return False
    print(f"\n{Style.GREEN('New server started')} (PID: {new_process.pid}).")

    if stop_process(old_pid):
        write_server_state(new_process.pid, new_version, new_executable_path)
        print(f"--- {Style.GREEN2('Update Complete.')} New PID: {new_process.pid} ---")
        return True
    else:
        print(Style.RED2("Failed to stop the old process. Manual intervention may be required."))
        # You might want to stop the new process here to avoid two running instances
        stop_process(new_process.pid)
        return False


def update_server_graceful_restart(new_executable_path: str, new_version: str):
    header = f"--- Starting Graceful Restart to {Style.YELLOW(new_version)} ---"
    print(Style.Bold(header))
    if not psutil: return False
    old_pid, _, _ = read_server_state()

    if not stop_process(old_pid):
        print(Style.RED("Failed to stop old server. Update aborted to prevent conflicts."))
        return False

    # After stopping, start the new server
    # We use a sub-function to avoid code duplication from `manage_server('start', ...)`
    start_new_server(new_executable_path, new_version, False)


def start_new_server(executable_path, version_str, use_posix_zdt):
    current_pid, _, _ = read_server_state()
    if is_process_running(current_pid):
        print(Style.YELLOW(f"Server already running (PID {current_pid}). Use 'stop' or 'update'."))
        return

    is_posix = platform.system().lower() != "windows"
    process = None
    socket_obj = None

    with Spinner(f"Starting server v{version_str}", symbols="d") as s:
        if is_posix and use_posix_zdt:
            socket_obj, fd = ensure_socket_and_fd_file_posix(SERVER_HOST, SERVER_PORT, SOCKET_BACKLOG,
                                                             PERSISTENT_FD_FILE)
            if fd is not None:
                process = start_rust_server_posix(executable_path, fd)
        else:  # Windows or non-ZDT start
            process = start_rust_server_windows(executable_path)

        time.sleep(2)  # Stabilize

    if socket_obj:
        # The parent can close its handle to the socket. The child now owns it.
        socket_obj.close()

    if process and process.poll() is None:
        write_server_state(process.pid, version_str, executable_path)
        print(
            f"\n{Style.GREEN2('✅ Server started.')} Version: {Style.YELLOW(version_str)}, PID: {Style.GREY(process.pid)}")
    else:
        print(f"\n{Style.RED2('❌ Server failed to start.')} Check logs for details.")
        write_server_state(None, None, None)  # Clean up state


def manage_server(action: str, executable_path: str = None, version_str: str = "unknown", use_posix_zdt: bool = False):
    if action == "start":
        if not executable_path:
            executable_path = get_executable_path()
        if not executable_path:
            print(Style.RED("Executable not found. Build with 'build' action or provide --exe path."))
            return
        start_new_server(executable_path, version_str, use_posix_zdt)

    elif action == "stop":
        pid, _, _ = read_server_state()
        if stop_process(pid):
            write_server_state(None, None, None)
            if platform.system().lower() != "windows" and os.path.exists(PERSISTENT_FD_FILE):
                print(Style.YELLOW(f"Note: Server stopped. Consider removing stale FD file: {PERSISTENT_FD_FILE}"))

    elif action == "update":
        if not executable_path:
            print(Style.RED("Error: Path to new executable is required for update (--exe)."))
            return
        if not version_str or version_str == "unknown":
            print(Style.RED("Error: Version string is required for update (--version)."))
            return
        update_server(executable_path, version_str, use_posix_zdt)

    elif action == "status":
        pid, ver, exe = read_server_state()
        header = f"--- {Style.Bold('Server Status')} ---"
        print(header)
        if is_process_running(pid):
            print(f"  {Style.GREEN2('✅ RUNNING')}")
            print(f"    {Style.WHITE('PID:')}        {Style.GREY(pid)}")
            print(f"    {Style.WHITE('Version:')}    {Style.YELLOW(ver)}")
            print(f"    {Style.WHITE('Executable:')} {Style.GREY(exe)}")
            if platform.system().lower() != "windows" and os.path.exists(PERSISTENT_FD_FILE) and use_posix_zdt:
                try:
                    with open(PERSISTENT_FD_FILE) as f:
                        fd_val = f.read().strip()
                    print(f"    {Style.WHITE('Listening FD:')} {Style.BLUE2(fd_val)} (POSIX ZDT Active)")
                except Exception:
                    pass
        else:
            print(f"  {Style.RED2('❌ STOPPED')}")
            if pid: print(f"    {Style.YELLOW('Stale PID in state:')} {pid}")


def handle_build():
    print(Style.CYAN("Building Rust project in release mode..."))
    from toolboxv2 import tb_root_dir  # Assuming this is available
    try:
        with Spinner("Compiling with cargo", symbols="t") as s:
            # Assume simple-core-server is in a subdirectory `src-core`
            result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=tb_root_dir / "src-core",
                check=True, capture_output=True, text=True
            )
        print(f"\n{Style.GREEN2('✅ Build successful.')}")
    except subprocess.CalledProcessError as e:
        print(f"\n{Style.RED2('❌ Build failed:')}")
        print(Style.GREY(e.stderr))
    except FileNotFoundError:
        print(f"\n{Style.RED2('❌ Build failed:')} 'cargo' command not found.")


def cli_api_runner():
    parser = argparse.ArgumentParser(
        description=f"🚀 {Style.Bold('Platform-Agnostic Rust Server Manager')}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    # Add actions with shared arguments
    actions = {
        'start': 'Start the server.',
        'debug': 'debug the server.',
        'stop': 'Stop the running server.',
        'update': 'Update the server.',
        'status': 'Check server status.',
        'build': 'Build the Rust project.',
        'clean': 'Clean build artifacts.',
        'remove-exe': 'Clean build artifacts.',
    }

    for action, help_text in actions.items():
        p = subparsers.add_parser(action, help=help_text)
        if action in ['start', 'update', 'status']:
            p.add_argument('--posix-zdt', action='store_true',
                           help='(Linux/macOS only) Use POSIX zero-downtime restarts via socket passing.')
        if action in ['start', 'update']:
            p.add_argument('--exe', type=str, help='Path to the server executable.')
            p.add_argument('--version', type=str, default='unknown', help='Version string for the build.')

    args = parser.parse_args()

    # Handle simple actions first
    if args.action == 'build':
        handle_build()
        # You can add clean handling here too if desired
        return

    if args.action == 'clean':
        cleanup_build_files()
        return

    if args.action == 'remove-exe':
        remove_release_executable()
        return

    if args.action == 'debug':
        print("Starting in DEBUG mode with hot reloading enabled...")
        if check_cargo_installed():
            run_with_hot_reload()
        else:
            print("Cargo is not installed. Hot reloading requires Cargo.")
        return

    manage_server(
        action=args.action,
        executable_path=getattr(args, 'exe', None),
        version_str=getattr(args, 'version', 'unknown'),
        use_posix_zdt=getattr(args, 'posix_zdt', False)
    )


if __name__ == "__main__":
    cli_api_runner()
