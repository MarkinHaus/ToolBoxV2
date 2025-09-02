# file: r_blob_db/db_cli.py
# A Production-Ready Manager for r_blob_db Instances and Clusters
# Functionality is 1-to-1 with the original, with an enhanced UI.

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- Enhanced UI Imports ---
try:
    # This path is relative to the execution context of the script
    from ..extras.Style import Spinner, Style
except ImportError:
    # Fallback for direct execution or if the structure is different
    try:
        from toolboxv2.extras.Style import Spinner, Style
    except ImportError:
        print("FATAL: UI utilities not found. Ensure 'toolboxv2/extras/Style.py' exists.")
        sys.exit(1)

# --- Configuration ---
try:
    import psutil
    import requests
except ImportError:
    print(Style.RED("FATAL: Required libraries 'psutil' and 'requests' not found."))
    print(Style.YELLOW("Please install them using: pip install psutil requests"))
    sys.exit(1)

# Default configuration file name
CLUSTER_CONFIG_FILE = "cluster_config.json"
# The base name of the Rust executable
EXECUTABLE_NAME = "r_blob_db"


# --- Helper Functions ---
def get_executable_path(base_name: str = EXECUTABLE_NAME, update=False) -> Path | None:
    """Finds the release executable in standard locations."""
    name_with_ext = f"{base_name}.exe" if platform.system() == "Windows" else base_name
    from toolboxv2 import tb_root_dir
    search_paths = [
        tb_root_dir / "bin" / name_with_ext,
        tb_root_dir / "r_blob_db" / "target" / "release" / name_with_ext,
    ]
    if update:
        search_paths = search_paths[::-1]
    for path in search_paths:
        if path.is_file():
            return path.resolve()
    return None


# --- Core Management Classes ---

class DBInstanceManager:
    """Manages a single r_blob_db instance."""

    def __init__(self, instance_id: str, config: dict):
        self.id = instance_id
        self.port = config['port']
        self.host = config.get('host', '127.0.0.1')
        self.data_dir = Path(config['data_dir'])
        self.state_file = self.data_dir / "instance_state.json"
        self.log_file = self.data_dir / "instance.log"  # Added for better logging info

    def read_state(self) -> tuple[int | None, str | None]:
        """Reads the PID and version from the instance's state file."""
        if not self.state_file.exists():
            return None, None
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            return state.get('pid'), state.get('version')
        except (json.JSONDecodeError, ValueError, FileNotFoundError):
            return None, None

    def write_state(self, pid: int | None, version: str | None):
        """Writes the PID and version to the state file."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        state = {'pid': pid, 'version': version}
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=4)

    def is_running(self) -> bool:
        """Checks if the process associated with this instance is running."""
        pid, _ = self.read_state()
        return psutil.pid_exists(pid) if pid else False

    def start(self, executable_path: Path, version: str) -> bool:
        """Starts the instance process and detaches, redirecting output to a log file."""
        if self.is_running():
            print(Style.YELLOW(f"Instance '{self.id}' is already running."))
            return True

        print(Style.CYAN(f"🚀 Starting instance '{self.id}' on port {self.port}..."))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        log_handle = open(self.log_file, 'a')

        env = os.environ.copy()
        env["R_BLOB_DB_CLEAN"] = os.getenv("R_BLOB_DB_CLEAN", "false")
        env["R_BLOB_DB_PORT"] = str(self.port)
        env["R_BLOB_DB_DATA_DIR"] = str(self.data_dir.resolve())
        env["RUST_LOG"] = "info,tower_http=debug" # "error"

        try:
            if executable_path is None:
                raise ValueError(f"\n{Style.RED2('❌ ERROR:')} Executable not found. Build it first.")
            with Spinner(f"Launching process for '{self.id}'", symbols="d"):
                process = subprocess.Popen(
                    [str(executable_path.resolve())],
                    env=env,
                    stdout=log_handle,
                    stderr=log_handle,
                    creationflags=subprocess.DETACHED_PROCESS if platform.system() == "Windows" else 0
                )
                time.sleep(1.5)

            if process.poll() is not None:
                print(f"\n{Style.RED2('❌ ERROR:')} Instance '{self.id}' failed to start. Check logs:")
                print(f"    {Style.GREY(self.log_file)}")
                return False

            self.write_state(process.pid, version)
            print(
                f"\n{Style.GREEN2('✅ Instance')} '{Style.Bold(self.id)}' {Style.GREEN2('started successfully.')} {Style.GREY(f'(PID: {process.pid})')}")
            print(f"   {Style.BLUE('Logging to:')} {Style.GREY(self.log_file)}")
            return True
        except Exception as e:
            print(f"\n{Style.RED2('❌ ERROR:')} Failed to launch instance '{self.id}': {e}")
            log_handle.close()
            return False

    def stop(self, timeout: int = 10) -> bool:
        """Stops the instance process gracefully."""
        if not self.is_running():
            print(Style.YELLOW(f"Instance '{self.id}' is not running."))
            self.write_state(None, None)
            return True

        pid, _ = self.read_state()
        with Spinner(f"Stopping '{self.id}' (PID: {pid})", symbols="+", time_in_s=timeout, count_down=True) as s:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                proc.wait(timeout)
            except psutil.TimeoutExpired:
                s.message = f"Force killing '{self.id}'"
                proc.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                print(f"\n{Style.RED2('❌ ERROR:')} Failed to stop instance '{self.id}': {e}")
                return False

        self.write_state(None, None)
        print(f"\n{Style.VIOLET2('⏹️  Instance')} '{Style.Bold(self.id)}' {Style.VIOLET2('stopped.')}")
        return True

    def get_health(self) -> dict:
        """Performs a health check on the running instance."""
        if not self.is_running():
            return {'id': self.id, 'status': 'STOPPED', 'error': 'Process not running'}

        pid, version = self.read_state()
        health_url = f"http://{self.host}:{self.port}/health"
        start_time = time.monotonic()
        try:
            response = requests.get(health_url, timeout=2)
            latency_ms = (time.monotonic() - start_time) * 1000
            response.raise_for_status()
            health_data = response.json()
            health_data.update({
                'id': self.id, 'pid': pid, 'latency_ms': round(latency_ms),
                'server_version': health_data.pop('version', 'unknown'),
                'manager_known_version': version
            })
            return health_data
        except requests.exceptions.RequestException as e:
            return {'id': self.id, 'status': 'UNREACHABLE', 'pid': pid, 'error': str(e)}
        except Exception as e:
            return {'id': self.id, 'status': 'ERROR', 'pid': pid, 'error': f'Failed to parse health response: {e}'}


class ClusterManager:
    """Manages a cluster of r_blob_db instances defined in a config file."""

    def __init__(self, config_path: str = CLUSTER_CONFIG_FILE):
        self.config_path = Path(config_path)
        self.instances: dict[str, DBInstanceManager] = self._load_config()

    def _load_config(self) -> dict[str, DBInstanceManager]:
        """Loads and validates the cluster configuration."""
        from toolboxv2 import tb_root_dir
        if not self.config_path.is_absolute():
            self.config_path = tb_root_dir / self.config_path

        default_config_dir = (tb_root_dir / ".data/db_data/").resolve()
        default_config = {
            "instance-01": {"port": 3001, "data_dir": str(default_config_dir / "01")},
            "instance-02": {"port": 3002, "data_dir": str(default_config_dir / "02")},
        }

        if not self.config_path.exists():
            print(Style.YELLOW(f"Warning: Cluster config '{self.config_path}' not found. Creating a default example."))

            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            config_data = default_config
        else:
            try:
                with open(self.config_path) as f:
                    config_data = json.load(f)
            except json.JSONDecodeError:
                print(Style.RED(f"Error: Cluster config '{self.config_path}' is not valid JSON. using default config."))
                config_data = default_config

        return {id: DBInstanceManager(id, cfg) for id, cfg in config_data.items()}

    def get_instances(self, instance_id: str | None = None) -> list[DBInstanceManager]:
        """Returns a list of instances to operate on."""
        if instance_id:
            if instance_id not in self.instances:
                raise ValueError(f"Instance ID '{instance_id}' not found in '{self.config_path}'.")
            return [self.instances[instance_id]]
        return list(self.instances.values())

    def start_all(self, executable_path: Path, version: str, instance_id: str | None = None):
        for instance in self.get_instances(instance_id):
            instance.start(executable_path, version)

    def stop_all(self, instance_id: str | None = None):
        for instance in self.get_instances(instance_id):
            instance.stop()

    def status_all(self, instance_id: str | None = None, silent=False):
        if not silent:
            header = f"--- {Style.Bold('Cluster Status')} ---"
            print(header)
            print(
                f"{Style.Underline('INSTANCE ID'):<18} {Style.Underline('STATUS'):<20} {Style.Underline('PID'):<8} {Style.Underline('VERSION'):<12} {Style.Underline('PORT')}")

        services_online = 0
        server_list = []
        for instance in self.get_instances(instance_id):
            pid, version = instance.read_state()
            is_running = instance.is_running()
            if is_running:
                server_list.append(f"http://{instance.host}:{instance.port}")
                services_online += 1
            if not silent:
                status_str = "✅ RUNNING" if is_running else "❌ STOPPED"
                status_color = Style.GREEN2 if is_running else Style.RED2
                print(
                    f"  {Style.WHITE(instance.id):<16} {status_color(status_str):<20} {Style.GREY(str(pid or 'N/A')):<8} {Style.BLUE2(version or 'N/A'):<12} {Style.YELLOW(str(instance.port))}"
                )
        if not silent:
            print("-" * len(header))
        return services_online, server_list

    def health_check_all(self, instance_id: str | None = None):
        header = f"--- {Style.Bold('Cluster Health Check')} ---"
        print(header)
        print(
            f"{Style.Underline('INSTANCE ID'):<18} {Style.Underline('STATUS'):<22} {Style.Underline('PID'):<8} {Style.Underline('LATENCY'):<12} {Style.Underline('DETAILS')}")

        for instance in self.get_instances(instance_id):
            health = instance.get_health()
            status = health.get('status', 'UNKNOWN')
            pid = health.get('pid', 'N/A')
            details = ""

            if status == 'OK':
                status_str, color = "✅ OK", Style.GREEN2
                latency = f"{health['latency_ms']}ms"
                details = f"Blobs: {Style.YELLOW(str(health['blobs_managed']))} | Version: {Style.BLUE2(health['server_version'])}"
            elif status == 'STOPPED':
                status_str, color = "❌ STOPPED", Style.RED2
                latency = "N/A"
            else:
                status_str, color = f"🔥 {status}", Style.RED
                latency = "N/A"
                details = Style.GREY(str(health.get('error', 'N/A')))

            print(
                f"  {Style.WHITE(instance.id):<16} {color(status_str):<22} {Style.GREY(str(pid)):<8} {Style.GREEN(latency):<12} {details}")
        print("-" * len(header))

    def update_all_rolling(self, new_executable_path: Path, new_version: str, instance_id: str | None = None):
        """Performs a zero-downtime rolling update of the cluster."""
        print(f"--- {Style.Bold(f'Starting Rolling Update to Version {Style.YELLOW(new_version)}')} ---")
        instances_to_update = self.get_instances(instance_id)
        for i, instance in enumerate(instances_to_update):
            print(
                f"\n{Style.CYAN(f'[{i + 1}/{len(instances_to_update)}] Updating instance')} '{Style.WHITE(instance.id)}'...")

            if not instance.stop():
                print(Style.RED2(f"CRITICAL: Failed to stop old instance '{instance.id}'. Aborting update."))
                return

            if not instance.start(new_executable_path, new_version):
                print(Style.RED2(f"CRITICAL: Failed to start new version for '{instance.id}'. Update halted."))
                print(Style.YELLOW("The cluster might be in a partially updated state. Please investigate."))
                return

            with Spinner(f"Waiting for '{instance.id}' to become healthy", symbols="t") as s:
                for attempt in range(5):
                    s.message = f"Waiting for '{instance.id}' to become healthy (attempt {attempt + 1}/5)"
                    time.sleep(2)
                    health = instance.get_health()
                    if health.get('status') == 'OK':
                        print(
                            f"\n{Style.GREEN('✅ Instance')} '{instance.id}' {Style.GREEN('is healthy with new version.')}")
                        break
                else:
                    print(
                        f"\n{Style.RED2('CRITICAL:')} Instance '{instance.id}' did not become healthy after update. Update halted.")
                    return

        print(f"\n--- {Style.GREEN2('Rolling Update Complete')} ---")


# --- CLI Command Handlers ---
def handle_build():
    print(Style.CYAN("Building Rust project in release mode..."))
    from toolboxv2 import tb_root_dir
    try:
        with Spinner("Compiling with Cargo", symbols='t'):
            result = subprocess.run(
                ["cargo", "build", "--release", "--package", "r_blob_db"],
                check=True, cwd=tb_root_dir / "r_blob_db",
                capture_output=True, text=True
            )

        print(f"\n{Style.GREEN2('✅ Build successful.')}")
        exe_path = get_executable_path()
        if exe_path:
            bin_dir = tb_root_dir / "bin"
            bin_dir.mkdir(exist_ok=True)
            try:
                dest_path = bin_dir / exe_path.name
                shutil.copy(exe_path, dest_path)
                print(f"   {Style.BLUE('Copied executable to:')} {Style.GREY(dest_path)}")
            except Exception as e:
                print(Style.YELLOW(f"Warning: Failed to copy executable to bin directory: {e}"))
                # Fallback to ubin
                ubin_dir = tb_root_dir / "ubin"
                ubin_dir.mkdir(exist_ok=True)
                dest_path = ubin_dir / exe_path.name
                try:
                    shutil.copy(exe_path, dest_path)
                    print(f"   {Style.BLUE('Copied executable to fallback:')} {Style.GREY(dest_path)}")
                except Exception as e_ubin:
                    print(Style.RED(f"Error copying to ubin as well: {e_ubin}"))


    except subprocess.CalledProcessError as e:
        print(f"\n{Style.RED2('❌ Build failed:')}")
        print(Style.GREY(e.stderr))
    except FileNotFoundError:
        print(f"\n{Style.RED2('❌ Build failed:')} 'cargo' command not found. Is Rust installed and in your PATH?")


def handle_clean():
    print(Style.CYAN("Cleaning build artifacts..."))
    try:
        with Spinner("Running cargo clean", symbols='+'):
            subprocess.run(["cargo", "clean"], check=True, capture_output=True)
        print(f"\n{Style.GREEN2('✅ Clean successful.')}")
    except Exception as e:
        print(f"\n{Style.RED2('❌ Clean failed:')} {e}")


def cli_db_runner():
    """The main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description=f"🚀 {Style.Bold('A manager for r_blob_db instances and clusters.')}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="action", required=True, help="Available actions")

    # Define common arguments
    instance_arg = {'name_or_flags': ['--instance-id'], 'type': str,
                    'help': 'Target a specific instance ID. If omitted, action applies to the whole cluster.',
                    'default': None}
    version_arg = {'name_or_flags': ['--version'], 'type': str,
                   'help': 'Specify a version string for the executable (e.g., "1.2.0").', 'default': 'dev'}

    # --- Define Commands ---
    p_start = subparsers.add_parser('start', help='Start instance(s).')
    p_start.add_argument(*instance_arg['name_or_flags'],
                         **{k: v for k, v in instance_arg.items() if k != 'name_or_flags'})
    p_start.add_argument(*version_arg['name_or_flags'],
                         **{k: v for k, v in version_arg.items() if k != 'name_or_flags'})

    p_stop = subparsers.add_parser('stop', help='Stop instance(s).')
    p_stop.add_argument(*instance_arg['name_or_flags'],
                        **{k: v for k, v in instance_arg.items() if k != 'name_or_flags'})

    p_status = subparsers.add_parser('status', help='Show the running status of instance(s).')
    p_status.add_argument(*instance_arg['name_or_flags'],
                          **{k: v for k, v in instance_arg.items() if k != 'name_or_flags'})

    p_health = subparsers.add_parser('health', help='Perform a health check on instance(s).')
    p_health.add_argument(*instance_arg['name_or_flags'],
                          **{k: v for k, v in instance_arg.items() if k != 'name_or_flags'})

    p_update = subparsers.add_parser('update', help='Perform a rolling update on the cluster.')
    p_update.add_argument(*instance_arg['name_or_flags'],
                          **{k: v for k, v in instance_arg.items() if k != 'name_or_flags'})
    version_arg_update = {**version_arg, 'required': True}
    p_update.add_argument(*version_arg_update['name_or_flags'],
                          **{k: v for k, v in version_arg_update.items() if k != 'name_or_flags'})

    subparsers.add_parser('build', help='Build the Rust executable from source.')
    subparsers.add_parser('clean', help='Clean the Rust build artifacts.')

    # --- Execute Command ---
    args = parser.parse_args()

    if args.action == 'build':
        handle_build()
        return
    if args.action == 'clean':
        handle_clean()
        return

    manager = ClusterManager()

    if args.action in ['start', 'update']:
        executable_path = get_executable_path(update=(args.action == 'update'))
        if not executable_path:
            print(Style.RED(f"ERROR: Could not find the {EXECUTABLE_NAME} executable."))
            print(Style.YELLOW("Please build it first with: python -m toolboxv2.r_blob_db.db_cli build"))
            return

    if args.action == 'start':
        manager.start_all(executable_path, args.version, args.instance_id)
    elif args.action == 'stop':
        manager.stop_all(args.instance_id)
    elif args.action == 'status':
        manager.status_all(args.instance_id)
    elif args.action == 'health':
        manager.health_check_all(args.instance_id)
    elif args.action == 'update':
        manager.update_all_rolling(executable_path, args.version, args.instance_id)


if __name__ == "__main__":
    cli_db_runner()
