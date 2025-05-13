import contextlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from packaging import version

SERVER_STATE_FILE = "server_state.json"
PERSISTENT_FD_FILE = "server_socket.fd" # Used on POSIX
DEFAULT_EXECUTABLE_NAME = "simple-core-server" # Adjust if your exe name is different
SERVER_HOST = "0.0.0.0" # Must match Rust config
SERVER_PORT = 8080       # Must match Rust config
SOCKET_BACKLOG = 128


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


def check_and_run_local_release(do_run=True):
    """Search for a pre-built release executable in the src-core folder and run it if found."""
    src_core_path = os.path.join(".", "src-core")
    if os.path.isdir(src_core_path):
        # Define the path to the expected release executable, assuming a Cargo project structure
        expected_name = "simple-core-server.exe" if platform.system().lower() == "windows" else "simple-core-server"
        release_path = os.path.join(src_core_path, expected_name)
        if os.path.isfile(release_path):
            print("Found pre-built release executable.")
            run_executable(release_path)
            return True
        release_path = os.path.join(src_core_path, "target", "release", expected_name)
        if os.path.isfile(release_path):
            print("Found pre-built release executable.")
            # Move the executable from target/release to src_core_path for easier access next time
            dest_path = os.path.join(src_core_path, expected_name)
            try:
                import shutil
                shutil.copy2(release_path, dest_path)
                print(f"Copied executable to {dest_path} for easier access next time")
            except Exception as e:
                print(f"Failed to copy executable: {e}")
                return False
            if do_run:
                run_executable(dest_path)
            else:
                return dest_path
            return True
    return False


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


def is_uv_installed():
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False

def get_uv_site_packages():
    """Find the site-packages directory for a uv-managed virtual environment."""
    try:
        site_packages = subprocess.check_output(["uv", "info", "--json"], text=True)
        import json
        data = json.loads(site_packages)
        return data["venv"]["site_packages"]
    except Exception as e:
        print(f"Error finding uv site-packages: {e}")
        return None

def create_dill_archive(site_packages, output_file="python312.dill"):
    """Package dill and all dependencies into a single .dill archive."""
    try:
        temp_dir = "/tmp/dill_package"
        os.makedirs(temp_dir, exist_ok=True)

        # Copy only necessary packages
        packages = ["dill"]
        for package in packages:
            package_path = os.path.join(site_packages, package)
            if os.path.exists(package_path):
                shutil.copytree(package_path, os.path.join(temp_dir, package), dirs_exist_ok=True)
            else:
                print(f"Warning: {package} not found in site-packages.")

        # Create the .dill archive
        with tarfile.open(output_file, "w:gz") as tar:
            tar.add(temp_dir, arcname=".")

        print(f"Successfully created {output_file}")

        # Clean up
        shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"Error creating .dill archive: {e}")

def add_py_dill():
    if not is_uv_installed():
        print("uv is not installed. Please install uv before running this script.")
        return
    print(f"VIRTUAL_ENV=$ {os.getenv('VIRTUAL_ENV')}")
    site_packages = os.getenv("PY_SITE_PACKAGES")
    if not site_packages:
        print("Could not determine site-packages path. Is this a uv environment?")
        return

    print(f"Packaging dill from {site_packages}...")
    create_dill_archive(site_packages, output_file=os.getenv("PY_DILL"))


def main_api_runner(debug=False, run=True):
    """
    Main function to run the API server.
    When debug=True, enables hot reloading and runs in debug mode.

    Non blocking!
    """
    if not os.path.exists(os.getenv("PY_DILL", '.')):
        add_py_dill()
    if is_uv_installed():
        print(f"VIRTUAL_ENV=$ {os.getenv('VIRTUAL_ENV')} {os.getenv('PY_SITE_PACKAGES')}")
        os.environ["VIRTUAL_ENV"] = os.getenv('UV_BASE_ENV', os.getenv('VIRTUAL_ENV'))
        # os.environ["PY_SITE_PACKAGES"] = os.getenv('PY_SITE_PACKAGES')
    if debug:
        print("Starting in DEBUG mode with hot reloading enabled...")
        if check_cargo_installed():
            run_with_hot_reload()
        else:
            print("Cargo is not installed. Hot reloading requires Cargo.")
        return None

    # Release mode flow
    if exe := check_and_run_local_release(run):
        return exe

    # Step 1: Detect current OS and machine architecture
    current_os, machine = detect_os_and_arch()
    print(f"Detected OS: {current_os}, Architecture: {machine}")

    # Step 2: Attempt to download executable from remote URL
    url, file_name = query_executable_url(current_os, machine)
    downloaded_exe = download_executable(url, file_name)

    if downloaded_exe and run:
        print("Downloaded executable. Executing it...")
        run_executable(downloaded_exe)
        return None

    if downloaded_exe and not run:
        return downloaded_exe

    # Step 3: Fallback: Check for local pre-built release executable in src-core folder
    print("Remote executable not found. Searching local 'src-core' folder...")
    if exe := check_and_run_local_release():
        return exe
    else:
        print("Pre-built release executable not found locally.")

        # Step 4: If executable not found locally, check if Cargo is installed
        if not check_cargo_installed():

            print("Cargo is not installed. Please install Cargo to build the project.")
            return None

        print("Cargo is installed. Proceeding with build.")
        if not build_cargo_project(debug=False):

            print("Failed to build the Cargo project.")
            return None

        # After successful build, try running the release executable again
        if exe :=  check_and_run_local_release(run):
            return exe

        print("Release executable missing even after build.")
        return None


# --- Zoro downtime unix manager and windows quick restart ---

def get_executable_name_with_extension(base_name=DEFAULT_EXECUTABLE_NAME):
    if platform.system().lower() == "windows":
        return f"{base_name}.exe"
    return base_name

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
    try:
        state = {'pid': pid, 'version': server_version, 'executable_path': str(Path(executable_path).resolve())} # Store absolute path
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Error writing server state: {e}")

def is_process_running(pid):
    if pid is None or psutil is None: return False
    try:
        return psutil.pid_exists(int(pid))
    except ValueError: # Handle if pid is not an int
        return False


def stop_process(pid, timeout=10):
    if pid is None or not psutil or not is_process_running(pid):
        print(f"Process {pid} not running or psutil unavailable.")
        return True
    try:
        proc = psutil.Process(int(pid))
        print(f"Sending SIGTERM (or equivalent) to process {pid}...")
        proc.terminate() # Cross-platform terminate
        proc.wait(timeout)
        print(f"Process {pid} terminated.")
        return True
    except psutil.TimeoutExpired:
        print(f"Process {pid} did not terminate gracefully. Force killing...")
        try:
            proc.kill() # Cross-platform kill
            proc.wait(2)
            print(f"Process {pid} killed.")
        except Exception as e_kill:
            print(f"Error killing process {pid}: {e_kill}")
            return False
        return True
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found (already stopped?).")
        return True
    except Exception as e:
        print(f"Error stopping process {pid}: {e}")
        return False

# --- Platform-Specific Socket and Process Starting ---

def ensure_socket_and_fd_file_posix(host, port, backlog, fd_file_path) -> tuple[socket.socket | None, int | None]:
    """POSIX: Ensures a listening socket exists and its FD is in the fd_file."""
    if os.path.exists(fd_file_path):
        try:
            with open(fd_file_path) as f:
                fd = int(f.read().strip())
            # Basic check (less reliable than on-demand creation for ensuring liveness)
            # This check is mostly to see if the FD *number* is plausible.
            # The real test is when the Rust server tries to use it.
            print(f"[POSIX] Found existing persistent FD {fd} in {fd_file_path}.")
            # We don't return a socket object if FD exists, Rust will use the FD directly.
            return None, fd
        except Exception as e:
            print(f"[POSIX] Persistent FD file {fd_file_path} exists but FD invalid/unreadable: {e}. Will create new.")
            with contextlib.suppress(OSError): os.remove(fd_file_path)

    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(os, 'set_inheritable'): # Python 3.4+
            os.set_inheritable(server_socket.fileno(), True)
        else: # POSIX, Python < 3.4 (fcntl not on Windows)
            import fcntl
            fd_num = server_socket.fileno()
            flags = fcntl.fcntl(fd_num, fcntl.F_GETFD)
            fcntl.fcntl(fd_num, fcntl.F_SETFD, flags & ~fcntl.FD_CLOEXEC)

        server_socket.bind((host, port))
        server_socket.listen(backlog)
        fd = server_socket.fileno()
        with open(fd_file_path, 'w') as f: f.write(str(fd))
        os.chmod(fd_file_path, 0o600)
        print(f"[POSIX] Created new socket. FD {fd} saved to {fd_file_path}.")
        return server_socket, fd # Return the socket object for the initial creator
    except Exception as e:
        print(f"[POSIX] Fatal: Could not create and save listening socket FD: {e}")
        return None, None

def start_rust_server_posix(executable_path: str, persistent_fd: int):
    """POSIX: Starts Rust server passing the persistent_fd."""
    abs_executable_path = Path(executable_path).resolve()
    env = os.environ.copy()
    env["PERSISTENT_LISTENER_FD"] = str(persistent_fd)
    env["LISTEN_FDS"] = str(persistent_fd) # Also set for listenfd standard mechanism
    env["LISTEN_PID"] = str(os.getpid())
    print(f"[POSIX] Starting Rust server: {abs_executable_path} using FD {persistent_fd}")
    try:
        process = subprocess.Popen(
            [str(abs_executable_path)],
            cwd=abs_executable_path.parent,
            env=env,
        )
        return process
    except Exception as e:
        print(f"[POSIX] Failed to start Rust server {abs_executable_path}: {e}")
        return None

def start_rust_server_windows(executable_path: str):
    """WINDOWS: Starts Rust server normally. It will bind its own socket."""
    abs_executable_path = Path(executable_path).resolve()
    print(f"[WINDOWS] Starting Rust server: {abs_executable_path}. It will bind its own socket.")
    try:
        process = subprocess.Popen(
            [str(abs_executable_path)],
            cwd=abs_executable_path.parent,
            # No special env vars for socket needed for Windows fallback
        )
        return process
    except Exception as e:
        print(f"[WINDOWS] Failed to start Rust server {abs_executable_path}: {e}")
        return None

# --- Main Management Logic ---

def update_server(new_executable_path: str, new_version: str):
    """High-level update function, calls platform-specific logic."""
    if platform.system().lower() == "windows":
        return update_server_windows(new_executable_path, new_version)
    else: # POSIX
        return update_server_posix(new_executable_path, new_version)

def update_server_posix(new_executable_path: str, new_version: str):
    """POSIX: Zero-downtime update using persistent FD."""
    if not psutil: return False
    print(f"--- [POSIX] Starting Update to {new_version} ---")
    old_pid, old_version, old_exe_path = read_server_state()

    if not os.path.exists(PERSISTENT_FD_FILE):
        print(f"[POSIX] Error: FD file '{PERSISTENT_FD_FILE}' not found. Cannot update.")
        return False
    try:
        with open(PERSISTENT_FD_FILE) as f: persistent_fd = int(f.read().strip())
        print(f"[POSIX] Using persistent listener FD: {persistent_fd}")
    except Exception as e:
        print(f"[POSIX] Error reading FD from '{PERSISTENT_FD_FILE}': {e}")
        return False

    new_process = start_rust_server_posix(new_executable_path, persistent_fd)
    if new_process is None: return False
    time.sleep(5) # Allow new server to init
    if new_process.poll() is not None:
        print(f"[POSIX] New server (PID {new_process.pid}) died. Exit: {new_process.poll()}. Update failed.")
        return False
    print(f"[POSIX] New server (v{new_version}, PID {new_process.pid}) started.")

    if old_pid and is_process_running(old_pid):
        print(f"[POSIX] Stopping old server (v{old_version}, PID {old_pid})...")
        if not stop_process(old_pid):
            print(f"[POSIX] Warning: Failed to stop old server PID {old_pid}.")
    else:
        print("[POSIX] No old server or PID was stale.")

    write_server_state(new_process.pid, new_version, new_executable_path)
    print(f"--- [POSIX] Update to {new_version} complete. New PID: {new_process.pid} ---")
    return True

def update_server_windows(new_executable_path: str, new_version: str):
    """WINDOWS: Graceful restart (stop old, start new)."""
    if not psutil: return False
    print(f"--- [WINDOWS] Starting Update (Graceful Restart) to {new_version} ---")
    old_pid, old_version, old_exe_path = read_server_state()

    if old_pid and is_process_running(old_pid):
        print(f"[WINDOWS] Stopping old server (v{old_version}, PID {old_pid})...")
        if not stop_process(old_pid):
            print(f"[WINDOWS] Failed to stop old server PID {old_pid}. Update aborted to prevent conflicts.")
            return False
        print("[WINDOWS] Old server stopped.")
        time.sleep(2) # Give OS time to release port
    else:
        print("[WINDOWS] No old server running or PID was stale.")

    new_process = start_rust_server_windows(new_executable_path)
    if new_process is None: return False
    time.sleep(3) # Allow new server to init
    if new_process.poll() is not None:
        print(f"[WINDOWS] New server (PID {new_process.pid}) died. Exit: {new_process.poll()}. Update failed.")
        return False
    print(f"[WINDOWS] New server (v{new_version}, PID {new_process.pid}) started.")

    write_server_state(new_process.pid, new_version, new_executable_path)
    print(f"--- [WINDOWS] Update to {new_version} complete. New PID: {new_process.pid} ---")
    return True


def manage_server(action: str, executable_path: str = None, version_str: str = "unknown"):
    if action == "start":
        current_pid, _, _ = read_server_state()
        if current_pid and is_process_running(current_pid):
            print(f"Server already running (PID {current_pid}). Use 'stop' first or 'update'.")
            return

        if not executable_path: # Determine executable path
            # Check in target/release first, then src-core root
            exe_name = get_executable_name_with_extension()
            path_options = [
                Path("src-core") / "target" / "release" / exe_name,
                Path("src-core") / exe_name,
                Path(".") / exe_name # Current dir
            ]
            for p_opt in path_options:
                if p_opt.exists():
                    executable_path = str(p_opt)
                    break
            if not executable_path:
                print(f"Executable '{exe_name}' not found in standard locations. Build or provide --exe.")
                return

        print(f"Resolved executable path: {executable_path}")


        if platform.system().lower() == "windows":
            process = start_rust_server_windows(executable_path)
        else: # POSIX
            # This script instance creates the socket and FD file if they don't exist
            # It can then exit. The Rust server keeps using the FD.
            server_socket_obj, persistent_fd = ensure_socket_and_fd_file_posix(
                SERVER_HOST, SERVER_PORT, SOCKET_BACKLOG, PERSISTENT_FD_FILE
            )
            if persistent_fd is None:
                print("[POSIX] Failed to ensure server socket for start. Aborting.")
                return
            process = start_rust_server_posix(executable_path, persistent_fd)
            if process and server_socket_obj:
                # Initial creator of the socket can close its handle
                # The FD is now managed by the kernel and used by the Rust child
                print("[POSIX] Initial Python starter closing its socket object handle.")
                server_socket_obj.close()
            elif not process and server_socket_obj: # Failed to start rust server
                print("[POSIX] Rust server failed to start, cleaning up socket and FD file.")
                server_socket_obj.close()
                if os.path.exists(PERSISTENT_FD_FILE):
                    with contextlib.suppress(OSError): os.remove(PERSISTENT_FD_FILE)


        if process:
            # Wait briefly to see if it dies immediately
            time.sleep(2)
            if process.poll() is None: # Still running
                write_server_state(process.pid, version_str, executable_path)
                print(f"Server (v{version_str}) started. PID: {process.pid}.")
                print("Python manager can now exit.")
            else:
                print(f"Server process (PID {process.pid}) terminated quickly (exit code {process.poll()}). Check server logs.")
                # If start failed, clear any potentially written state
                pid_check, _, _ = read_server_state()
                if pid_check == process.pid:
                    write_server_state(None, None, None)
        else:
            print("Failed to start Rust server process.")

    elif action == "stop":
        pid, _, _ = read_server_state()
        if stop_process(pid):
            write_server_state(None, None, None) # Clear state
            # On POSIX, if we stop the last server, the FD file becomes stale.
            # Optionally remove it. This means 'start' will always create a new socket.
            if platform.system().lower() != "windows" and os.path.exists(PERSISTENT_FD_FILE):
                print(f"Server stopped. Removing persistent FD file: {PERSISTENT_FD_FILE}")
                try: os.remove(PERSISTENT_FD_FILE)
                except OSError as e: print(f"Could not remove FD file: {e}")
        else:
            print("Failed to stop server or server not running.")

    elif action == "update":
        if not executable_path:
            print("Error: Path to new executable is required for update (--exe).")
            return
        if not version_str or version_str == "unknown":
            print("Error: Version string for the new executable is required (--version).")
            return
        update_server(executable_path, version_str)

    elif action == "status":
        pid, ver, exe = read_server_state()
        if pid and is_process_running(pid):
            print("Server is RUNNING.")
            print(f"  PID: {pid}\n  Version: {ver}\n  Executable: {exe}")
            if platform.system().lower() != "windows" and os.path.exists(PERSISTENT_FD_FILE):
                try:
                    with open(PERSISTENT_FD_FILE) as f_fd: fd_val = f_fd.read().strip()
                    print(f"  Listening FD (from file, POSIX only): {fd_val}")
                except Exception: pass
        else:
            print("Server is STOPPED (or state inconsistent).")
            if pid: print(f"  Stale PID in state: {pid}")
            if platform.system().lower() != "windows" and os.path.exists(PERSISTENT_FD_FILE):
                 print(f"  Warning (POSIX): {PERSISTENT_FD_FILE} exists, but server not found.")
    else:
        print(f"Unknown action: {action}.")


def api_manager(action: str, debug, exe=None, version="v0.1"):
    if action not in ['start', 'stop', 'update', 'status', 'build', 'clean', 'remove-exe']:
        return f"invalid action {action} valid ar ['start', 'stop', 'update', 'status', 'build', 'clean', 'remove-exe']"

    if action == 'build':
        print("Build action placeholder...")  # Replace with actual build_cargo_project() call
        return None
    if action == 'clean':
        cleanup_build_files()
        return None
    if action == 'remove-exe':
        remove_release_executable()
        return None

    if not psutil and action in ['start', 'stop', 'update', 'status']:
        sys.exit("Error: 'psutil' library is required. pip install psutil")

    if exe is None:
        exe = main_api_runner(debug, False)

    manage_server(action, exe, version)
    return None


def cli_api_runner():
    import argparse
    import textwrap

    class CustomFormatter(argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
            🚀 Platform-Agnostic Rust Server Manager

            Manage your Rust-based server across platforms with ease.

            Available actions:
              start        Start the server (optional --exe)
              stop         Stop the running server
              update       Stop, replace binary, and restart the server (optional --exe)
              status       Check if the server is currently running
              build        Build the server
              clean        Clean build artifacts
              remove-exe   Remove the current server executable

            Examples:
              tb api start
              tb api update --version 1.1.0
              tb api stop
        """),
        formatter_class=CustomFormatter
    )

    parser.add_argument(
        'action',
        choices=['start', 'stop', 'update', 'status', 'build', 'clean', 'remove-exe', 'help'],
        help="Action to perform. See the list above for details.",
        default='help'
    )
    parser.add_argument(
        '--exe',
        type=str,
        help="Path to the server executable",
        default=None,
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug mode with hot reloading support."
    )
    parser.add_argument(
        '--version',
        type=str,
        default="unknown",
        help="Version string for logging and update tracking."
    )
    print("hey")
    if 'tb' in sys.argv[0] and len(sys.argv) < 2:
        sys.argv.append("help")
    args = parser.parse_args()
    print(args.action, "sdad")
    if args.action == "help":
        parser.print_help()
        return
    api_manager(args.action, args.debug, args.exe, args.version)

if __name__ == "__main__":
    cli_api_runner()
