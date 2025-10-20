# file: r_blob_db/db_cli.py
# A Production-Ready Manager for r_blob_db Instances and Clusters
# Enhanced with Modern UI and Interactive Data Discovery

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from toolboxv2.tb_browser.install import detect_shell

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
    print_table_row
)

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

# =================== Helper Functions ===================

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


# =================== Core Management Classes ===================

class DBInstanceManager:
    """Manages a single r_blob_db instance."""

    def __init__(self, instance_id: str, config: dict):
        self.id = instance_id
        self.port = config['port']
        self.host = config.get('host', '127.0.0.1')
        self.data_dir = Path(config['data_dir'])
        self.state_file = self.data_dir / "instance_state.json"
        self.log_file = self.data_dir / "instance.log"

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
            print_status(f"Instance '{self.id}' is already running", "warning")
            return True

        print_box_header(f"Starting Instance: {self.id}", "🚀")
        print_box_content(f"Port: {self.port}", "info")
        print_box_content(f"Data Directory: {str(self.data_dir)[:15]}...{str(self.data_dir)[-15:]}", "info")
        print_box_footer()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        log_handle = open(self.log_file, 'a')

        env = os.environ.copy()
        env["R_BLOB_DB_CLEAN"] = os.getenv("R_BLOB_DB_CLEAN", "false")
        env["R_BLOB_DB_PORT"] = str(self.port)
        env["R_BLOB_DB_DATA_DIR"] = str(self.data_dir.resolve())
        env["RUST_LOG"] = "info,tower_http=debug"

        try:
            if executable_path is None:
                raise ValueError(f"Executable not found. Build it first.")

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
                print_status(f"Instance '{self.id}' failed to start. Check logs:", "error")
                print(f"    {Style.GREY(str(self.log_file))}")
                return False

            self.write_state(process.pid, version)
            print_status(f"Instance '{self.id}' started successfully (PID: {process.pid})", "success")
            print_status(f"Logging to: {self.log_file}", "info")
            return True

        except Exception as e:
            print_status(f"Failed to launch instance '{self.id}': {e}", "error")
            log_handle.close()
            return False

    def stop(self, timeout: int = 10) -> bool:
        """Stops the instance process gracefully."""
        if not self.is_running():
            print_status(f"Instance '{self.id}' is not running", "warning")
            self.write_state(None, None)
            return True

        pid, _ = self.read_state()

        print_box_header(f"Stopping Instance: {self.id}", "⏹️")
        print_box_content(f"PID: {pid}", "info")
        print_box_content(f"Timeout: {timeout}s", "info")
        print_box_footer()

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
                print_status(f"Failed to stop instance '{self.id}': {e}", "error")
                return False

        self.write_state(None, None)
        print_status(f"Instance '{self.id}' stopped", "success")
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
                'id': self.id,
                'pid': pid,
                'latency_ms': round(latency_ms),
                'server_version': health_data.pop('version', 'unknown'),
                'manager_known_version': version
            })
            return health_data
        except requests.exceptions.RequestException as e:
            return {'id': self.id, 'status': 'UNREACHABLE', 'pid': pid, 'error': str(e)}
        except Exception as e:
            return {'id': self.id, 'status': 'ERROR', 'pid': pid, 'error': f'Failed to parse health response: {e}'}

    def get_blob_list(self) -> List[Dict[str, Any]]:
        """Get list of all blobs from this instance"""
        if not self.is_running():
            return []

        # Try API endpoint first
        try:
            response = requests.get(f"http://{self.host}:{self.port}/blobs", timeout=5)
            response.raise_for_status()
            return response.json().get('blobs', [])
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Endpoint doesn't exist, fallback to scanning data directory
                print_status("API endpoint /blobs not available, scanning data directory...", "warning")
                return self._scan_data_directory()
            return []
        except Exception as e:
            print_status(f"Error fetching blob list: {e}", "warning")
            # Fallback to scanning data directory
            return self._scan_data_directory()

    def _scan_data_directory(self) -> List[Dict[str, Any]]:
        """Scan the data directory to find blobs (fallback method)"""
        blobs = []

        if not self.data_dir.exists():
            return []

        try:
            # Look for blob files in the data directory
            # Adjust patterns based on your Rust server's storage structure
            for item in self.data_dir.rglob('*'):
                if item.is_file() and not item.name.startswith('.'):
                    # Skip metadata files
                    if item.name in ['instance_state.json', 'instance.log']:
                        continue

                    # Get file stats
                    stat = item.stat()
                    blob_info = {
                        'id': item.stem if item.suffix == '.blob' else item.name,
                        'size': stat.st_size,
                        'created_at': time.strftime('%Y-%m-%d %H:%M:%S',
                                                    time.localtime(stat.st_ctime)),
                        'modified_at': time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(stat.st_mtime)),
                        'path': str(item.relative_to(self.data_dir))
                    }
                    blobs.append(blob_info)

            return sorted(blobs, key=lambda x: x['created_at'], reverse=True)

        except Exception as e:
            print_status(f"Error scanning data directory: {e}", "error")
            return []

    def get_blob_data(self, blob_id: str) -> Optional[bytes]:
        """Get data from a specific blob"""
        if not self.is_running():
            # Try to read from disk if server is not running
            return self._read_blob_from_disk(blob_id)

        try:
            response = requests.get(f"http://{self.host}:{self.port}/blob/{blob_id}", timeout=5)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print_status(f"Error fetching blob via API: {e}, trying disk...", "warning")
            return self._read_blob_from_disk(blob_id)

    def _read_blob_from_disk(self, blob_id: str) -> Optional[bytes]:
        """Read blob directly from disk (fallback method)"""
        try:
            # Try different possible file patterns
            possible_paths = [
                self.data_dir / blob_id,
                self.data_dir / f"{blob_id}.blob",
                self.data_dir / "blobs" / blob_id,
                self.data_dir / "blobs" / f"{blob_id}.blob",
            ]

            for path in possible_paths:
                if path.exists() and path.is_file():
                    with open(path, 'rb') as f:
                        return f.read()

            return None
        except Exception as e:
            print_status(f"Error reading blob from disk: {e}", "error")
            return None

    def delete_blob(self, blob_id: str) -> bool:
        """Delete a specific blob"""
        if not self.is_running():
            print_status("Instance is not running, cannot delete blob via API", "error")
            return False

        try:
            response = requests.delete(f"http://{self.host}:{self.port}/blob/{blob_id}", timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            print_status(f"Error deleting blob: {e}", "error")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics from this instance"""
        if not self.is_running():
            # Return local stats if server not running
            return self._get_local_stats()

        try:
            # Try stats endpoint
            response = requests.get(f"http://{self.host}:{self.port}/stats", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Stats endpoint doesn't exist, use health endpoint
                health = self.get_health()
                if health.get('status') == 'OK':
                    return {
                        'blob_count': health.get('blobs_managed', 0),
                        'total_size': 'N/A',
                        'status': 'OK'
                    }
            return self._get_local_stats()
        except Exception:
            return self._get_local_stats()

    def _get_local_stats(self) -> Dict[str, Any]:
        """Calculate statistics from local data directory"""
        try:
            if not self.data_dir.exists():
                return {'blob_count': 0, 'total_size': 0}

            total_size = 0
            blob_count = 0

            for item in self.data_dir.rglob('*'):
                if item.is_file() and not item.name.startswith('.'):
                    if item.name not in ['instance_state.json', 'instance.log']:
                        total_size += item.stat().st_size
                        blob_count += 1

            return {
                'blob_count': blob_count,
                'total_size': f"{total_size / 1024:.2f} KB" if total_size < 1024 * 1024
                else f"{total_size / (1024 * 1024):.2f} MB",
                'total_size_bytes': total_size,
                'status': 'offline' if not self.is_running() else 'unknown'
            }
        except Exception as e:
            return {'blob_count': 0, 'total_size': 0, 'error': str(e)}


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
            print_status(f"Cluster config '{self.config_path}' not found. Creating default", "warning")

            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            config_data = default_config
        else:
            try:
                with open(self.config_path) as f:
                    config_data = json.load(f)
            except json.JSONDecodeError:
                print_status(f"Cluster config '{self.config_path}' is not valid JSON. Using default", "error")
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
        """Start all instances"""
        instances = self.get_instances(instance_id)

        if len(instances) > 1:
            print_box_header("Starting Multiple Instances", "🚀")
            print_box_content(f"Total instances: {len(instances)}", "info")
            print_box_footer()

        for instance in instances:
            instance.start(executable_path, version)
            if len(instances) > 1:
                print()

    def stop_all(self, instance_id: str | None = None):
        """Stop all instances"""
        instances = self.get_instances(instance_id)

        if len(instances) > 1:
            print_box_header("Stopping Multiple Instances", "⏹️")
            print_box_content(f"Total instances: {len(instances)}", "info")
            print_box_footer()

        for instance in instances:
            instance.stop()
            if len(instances) > 1:
                print()

    def status_all(self, instance_id: str | None = None, silent=False):
        """Show status of all instances"""
        instances = self.get_instances(instance_id)

        if not silent:

            # Table header
            columns = [
                ("INSTANCE ID", 18),
                ("STATUS", 12),
                ("PID", 8),
                ("VERSION", 12),
                ("PORT", 6),
                ("HOST", 15)
            ]
            widths = [w for _, w in columns]

            print("🖥️ Cluster Status\n")
            print_table_header(columns, widths)

        services_online = 0
        server_list = []

        for instance in instances:
            pid, version = instance.read_state()
            is_running = instance.is_running()

            if is_running:
                server_list.append(f"http://{instance.host}:{instance.port}")
                services_online += 1

            if not silent:
                status_str = " RUNNING" if is_running else " STOPPED"
                status_style = "green" if is_running else "red"

                print_table_row(
                    [
                        instance.id,
                        status_str,
                        str(pid or 'N/A'),
                        version or 'N/A',
                        str(instance.port),
                        instance.host
                    ],
                    widths,
                    ["white", status_style, "grey", "blue", "yellow", "cyan"]
                )

        if not silent:
            print()
            print_status(f"Services online: {services_online}/{len(instances)}", "info")

        return services_online, server_list

    def health_check_all(self, instance_id: str | None = None):
        """Perform health check on all instances"""
        instances = self.get_instances(instance_id)

        print("🏥 Cluster Health Check\n")

        columns = [
            ("INSTANCE ID", 18),
            ("STATUS", 12),
            ("PID", 8),
            ("LATENCY", 10),
            ("BLOBS", 8),
            ("VERSION", 12)
        ]
        widths = [w for _, w in columns]
        print_table_header(columns, widths)

        healthy_count = 0

        for instance in instances:
            health = instance.get_health()
            status = health.get('status', 'UNKNOWN')
            pid = health.get('pid', 'N/A')

            if status == 'OK':
                healthy_count += 1
                status_str, style = " OK", "green"
                latency = f"{health['latency_ms']}ms"
                blobs = str(health.get('blobs_managed', 'N/A'))
                version = health.get('server_version', 'N/A')
            elif status == 'STOPPED':
                status_str, style = "❌ STOPPED", "red"
                latency = blobs = version = "N/A"
            else:
                status_str, style = f"🔥 {status}", "red"
                latency = blobs = version = "N/A"

            print_table_row(
                [instance.id, status_str, str(pid), latency, blobs, version],
                widths,
                ["white", style, "grey", "green", "yellow", "blue"]
            )

        print()
        print_status(f"Healthy instances: {healthy_count}/{len(instances)}",
                     "success" if healthy_count == len(instances) else "warning")

    def update_all_rolling(self, new_executable_path: Path, new_version: str, instance_id: str | None = None):
        """Performs a zero-downtime rolling update of the cluster."""
        instances_to_update = self.get_instances(instance_id)

        print_box_header(f"Rolling Update to Version {new_version}", "🔄")
        print_box_content(f"Instances to update: {len(instances_to_update)}", "info")
        print_box_content(f"Executable: {new_executable_path}", "info")
        print_box_footer()

        for i, instance in enumerate(instances_to_update):
            print_separator("═")
            print(f"  [{i + 1}/{len(instances_to_update)}] Updating instance '{instance.id}'")
            print_separator("═")
            print()

            if not instance.stop():
                print_status(f"CRITICAL: Failed to stop old instance '{instance.id}'. Aborting", "error")
                return

            if not instance.start(new_executable_path, new_version):
                print_status(f"CRITICAL: Failed to start new version for '{instance.id}'", "error")
                print_status("The cluster might be in a partially updated state", "warning")
                return

            # Health check with retries
            print()
            with Spinner(f"Waiting for '{instance.id}' to become healthy", symbols="t") as s:
                for attempt in range(5):
                    s.message = f"Waiting for '{instance.id}' to become healthy (attempt {attempt + 1}/5)"
                    time.sleep(2)
                    health = instance.get_health()
                    if health.get('status') == 'OK':
                        print()
                        print_status(f"Instance '{instance.id}' is healthy with new version", "success")
                        break
                else:
                    print()
                    print_status(f"Instance '{instance.id}' did not become healthy. Update halted", "error")
                    return
            print()

        print_separator("═")
        print_status("Rolling Update Complete!", "success")
        print_separator("═")


# =================== Interactive Data Discovery ===================

class DataDiscovery:
    """Interactive data discovery and manipulation interface"""

    def __init__(self, manager: ClusterManager):
        self.manager = manager
        self.selected_instance: Optional[DBInstanceManager] = None
        self.current_view = "instances"  # instances, blobs, blob_detail
        self.selected_index = 0
        self.blob_list = []
        self.current_blob = None

    async def run(self):
        """Run interactive discovery session"""
        print('\033[2J\033[H')  # Clear screen

        print_box_header("Interactive Data Discovery & Manipulation", "🔍")
        print_box_content("Navigate with ↑↓ or w/s, Enter to select, q to quit", "info")
        print_box_footer()

        while True:
            if self.current_view == "instances":
                action = self.show_instances()
            elif self.current_view == "blobs":
                action = self.show_blobs()
            elif self.current_view == "blob_detail":
                action = self.show_blob_detail()

            if action == "quit":
                break
            elif action == "back":
                self.go_back()

    def show_instances(self):
        """Show instance selection menu with keyboard navigation"""
        import sys

        def get_key():
            """Get single keypress (cross-platform)"""
            if platform.system() == "Windows":
                import msvcrt
                key = msvcrt.getch()
                if key == b'\xe0':  # Arrow key prefix on Windows
                    key = msvcrt.getch()
                    if key == b'H':
                        return 'up'
                    elif key == b'P':
                        return 'down'
                elif key == b'\r':
                    return 'enter'
                elif key in (b'q', b'Q'):
                    return 'quit'
                elif key in (b'w', b'W'):
                    return 'up'
                elif key in (b's', b'S'):
                    return 'down'
                return None
            else:
                import tty
                import termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':  # ESC sequence
                        next_chars = sys.stdin.read(2)
                        if next_chars == '[A':
                            return 'up'
                        elif next_chars == '[B':
                            return 'down'
                    elif ch in ('\r', '\n'):
                        return 'enter'
                    elif ch in ('q', 'Q', '\x03'):  # q or Ctrl+C
                        return 'quit'
                    elif ch in ('w', 'W'):
                        return 'up'
                    elif ch in ('s', 'S'):
                        return 'down'
                    return None
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        while True:
            print('\033[2J\033[H')  # Clear screen

            print_box_header("Select Database Instance", "🗄️")
            print()

            instances = self.manager.get_instances()

            for i, instance in enumerate(instances):
                is_selected = i == self.selected_index
                is_running = instance.is_running()

                status_icon = "✅" if is_running else "❌"
                arrow = "▶" if is_selected else " "

                if is_selected:
                    print(f"  {arrow} \033[1;96m{status_icon} {instance.id:<20} Port: {instance.port:<6}\033[0m")
                else:
                    print(f"  {arrow} {status_icon} {instance.id:<20} Port: {instance.port:<6}")

            print()
            print_box_footer()
            print_status("↑↓/w/s: Navigate | Enter: Select | q: Quit", "info")

            # Get single keypress
            key = get_key()

            if key == 'quit':
                return "quit"
            elif key == 'up':
                self.selected_index = max(0, self.selected_index - 1)
            elif key == 'down':
                self.selected_index = min(len(instances) - 1, self.selected_index + 1)
            elif key == 'enter':
                self.selected_instance = instances[self.selected_index]
                if not self.selected_instance.is_running():
                    print()
                    print_status("Instance is not running! Please start it first.", "error")
                    time.sleep(2)
                else:
                    self.current_view = "blobs"
                    self.selected_index = 0
                    self.load_blobs()
                    return "continue"

    def load_blobs(self):
        """Load blob list from selected instance"""
        if not self.selected_instance:
            return

        print_status("Loading blobs...", "progress")
        self.blob_list = self.selected_instance.get_blob_list()

    def show_blobs(self):
        """Show blob list with keyboard navigation"""
        import sys

        def get_key():
            """Get single keypress (cross-platform)"""
            if platform.system() == "Windows":
                import msvcrt
                key = msvcrt.getch()
                if key == b'\xe0':  # Arrow key prefix on Windows
                    key = msvcrt.getch()
                    if key == b'H':
                        return 'up'
                    elif key == b'P':
                        return 'down'
                elif key == b'\r':
                    return 'enter'
                elif key in (b'q', b'Q'):
                    return 'quit'
                elif key in (b'w', b'W'):
                    return 'up'
                elif key in (b's', b'S'):
                    return 'down'
                elif key in (b'd', b'D'):
                    return 'delete'
                elif key in (b'r', b'R'):
                    return 'refresh'
                elif key in (b'b', b'B'):
                    return 'back'
                return None
            else:
                import tty
                import termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':  # ESC sequence
                        next_chars = sys.stdin.read(2)
                        if next_chars == '[A':
                            return 'up'
                        elif next_chars == '[B':
                            return 'down'
                    elif ch in ('\r', '\n'):
                        return 'enter'
                    elif ch in ('q', 'Q', '\x03'):  # q or Ctrl+C
                        return 'quit'
                    elif ch in ('w', 'W'):
                        return 'up'
                    elif ch in ('s', 'S'):
                        return 'down'
                    elif ch in ('d', 'D'):
                        return 'delete'
                    elif ch in ('r', 'R'):
                        return 'refresh'
                    elif ch in ('b', 'B'):
                        return 'back'
                    return None
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        while True:
            print('\033[2J\033[H')  # Clear screen

            print_box_header(f"Blobs in {self.selected_instance.id}", "📦")
            print()

            if not self.blob_list:
                print_box_content("No blobs found in this instance", "warning")
                print_box_footer()
                print_status("Press 'b' to go back or 'r' to refresh...", "info")

                key = get_key()
                if key in ('back', 'quit'):
                    return "back"
                elif key == 'refresh':
                    self.load_blobs()
                continue

            # Show stats
            stats = self.selected_instance.get_stats()
            if stats:
                print(f"  Total Blobs: {len(self.blob_list)}")
                if 'total_size' in stats:
                    print(f"  Total Size: {stats.get('total_size', 'N/A')}")
                print()

            # Show blob list
            print_separator()

            # Calculate visible range (show 15 items at a time)
            visible_count = 15
            start_idx = max(0, self.selected_index - visible_count // 2)
            end_idx = min(len(self.blob_list), start_idx + visible_count)

            # Adjust start if we're near the end
            if end_idx - start_idx < visible_count:
                start_idx = max(0, end_idx - visible_count)

            for i in range(start_idx, end_idx):
                blob = self.blob_list[i]
                is_selected = i == self.selected_index
                arrow = "▶" if is_selected else " "

                blob_id = blob.get('id', 'N/A')
                blob_id_display = (blob_id[:37] + '...') if len(blob_id) > 40 else blob_id
                blob_size = blob.get('size', 0)
                size_str = self.format_size(blob_size)

                if is_selected:
                    print(f"  {arrow} \033[1;96m{blob_id_display:<42} {size_str:>10}\033[0m")
                else:
                    print(f"  {arrow} {blob_id_display:<42} {size_str:>10}")

            if len(self.blob_list) > visible_count:
                print(f"\n  Showing {start_idx + 1}-{end_idx} of {len(self.blob_list)}")

            print()
            print_box_footer()
            print_status("↑↓/w/s: Navigate | Enter: View | d: Delete | r: Refresh | b: Back | q: Quit", "info")

            # Get user input
            key = get_key()

            if key == 'quit':
                return "quit"
            elif key == 'back':
                return "back"
            elif key == 'refresh':
                print()
                print_status("Refreshing blob list...", "progress")
                self.load_blobs()
                self.selected_index = min(self.selected_index, len(self.blob_list) - 1)
                time.sleep(0.5)
            elif key == 'up':
                self.selected_index = max(0, self.selected_index - 1)
            elif key == 'down':
                self.selected_index = min(len(self.blob_list) - 1, self.selected_index + 1)
            elif key == 'delete':
                if 0 <= self.selected_index < len(self.blob_list):
                    if self.confirm_delete_blob():
                        self.delete_current_blob()
                        # Adjust selected index if needed
                        if self.selected_index >= len(self.blob_list):
                            self.selected_index = max(0, len(self.blob_list) - 1)
            elif key == 'enter':
                if 0 <= self.selected_index < len(self.blob_list):
                    self.current_blob = self.blob_list[self.selected_index]
                    self.current_view = "blob_detail"
                    return "continue"

    def show_blob_detail(self):
        """Show detailed view of a blob with keyboard navigation"""
        import sys

        def get_key():
            """Get single keypress (cross-platform)"""
            if platform.system() == "Windows":
                import msvcrt
                key = msvcrt.getch()
                if key == b'\xe0':  # Arrow key prefix on Windows
                    key = msvcrt.getch()
                elif key in (b'q', b'Q'):
                    return 'quit'
                elif key in (b'e', b'E'):
                    return 'export'
                elif key in (b'd', b'D'):
                    return 'delete'
                elif key in (b'b', b'B'):
                    return 'back'
                return None
            else:
                import tty
                import termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':  # ESC sequence
                        sys.stdin.read(2)  # Consume arrow key codes
                    elif ch in ('q', 'Q', '\x03'):  # q or Ctrl+C
                        return 'quit'
                    elif ch in ('e', 'E'):
                        return 'export'
                    elif ch in ('d', 'D'):
                        return 'delete'
                    elif ch in ('b', 'B'):
                        return 'back'
                    return None
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        while True:
            print('\033[2J\033[H')  # Clear screen

            blob_id = self.current_blob.get('id', 'N/A')

            print_box_header(f"Blob Details", "📄")
            print()
            print(f"  ID: {blob_id}")
            print(f"  Size: {self.format_size(self.current_blob.get('size', 0))}")

            if 'created_at' in self.current_blob:
                print(f"  Created: {self.current_blob.get('created_at', 'N/A')}")
            if 'modified_at' in self.current_blob:
                print(f"  Modified: {self.current_blob.get('modified_at', 'N/A')}")
            if 'path' in self.current_blob:
                print(f"  Path: {self.current_blob.get('path', 'N/A')}")

            print()

            # Try to load and preview data
            print_status("Loading blob data...", "progress")
            data = self.selected_instance.get_blob_data(blob_id)

            if data:
                print()
                print_separator()
                print("  Data Preview (first 500 bytes):")
                print_separator()

                try:
                    # Try to decode as text
                    text_preview = data[:500].decode('utf-8', errors='ignore')
                    # Replace control characters except newlines
                    text_preview = ''.join(
                        char if char == '\n' or (32 <= ord(char) < 127) else '.'
                        for char in text_preview
                    )
                    print(f"\n{text_preview}\n")
                except:
                    # Show hex dump
                    hex_preview = data[:200].hex()
                    # Format as rows of 32 hex chars
                    for i in range(0, len(hex_preview), 32):
                        print(f"  {hex_preview[i:i + 32]}")
                    print()

                print_separator()
            else:
                print()
                print_status("Could not load blob data", "error")

            print()
            print_box_footer()
            print_status("e: Export | d: Delete | b: Back | q: Quit", "info")

            # Get user input
            key = get_key()

            if key == 'quit':
                return "quit"
            elif key == 'back':
                return "back"
            elif key == 'export':
                if data:
                    # Temporarily restore terminal for input
                    if platform.system() != "Windows":
                        import tty
                        import termios
                        fd = sys.stdin.fileno()
                        old_settings = termios.tcgetattr(fd)
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                    self.export_current_blob(data)

                    # Wait for keypress to continue
                    print()
                    print_status("Press any key to continue...", "info")
                    get_key()
                else:
                    print()
                    print_status("No data to export", "error")
                    time.sleep(1)
            elif key == 'delete':
                # Temporarily restore terminal for confirmation input
                if platform.system() != "Windows":
                    import tty
                    import termios
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                if self.confirm_delete_blob():
                    self.delete_current_blob()
                    return "back"

    def confirm_delete_blob(self) -> bool:
        """Confirm blob deletion with user"""
        if self.current_view == "blob_detail":
            blob = self.current_blob
        elif self.current_view == "blobs":
            if 0 <= self.selected_index < len(self.blob_list):
                blob = self.blob_list[self.selected_index]
            else:
                return False
        else:
            return False

        blob_id = blob.get('id', 'N/A')
        blob_id_short = (blob_id[:20] + '...') if len(blob_id) > 23 else blob_id

        print()
        print_status(f"Really delete blob {blob_id_short}?", "warning")
        print("  Type 'yes' to confirm: ", end='', flush=True)

        # Restore normal input temporarily
        if platform.system() != "Windows":
            import tty
            import termios
            import sys
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        confirm = input().strip().lower()
        return confirm == 'yes'

    def delete_current_blob(self):
        """Delete the currently selected blob"""
        if self.current_view == "blob_detail":
            blob = self.current_blob
        elif self.current_view == "blobs":
            if 0 <= self.selected_index < len(self.blob_list):
                blob = self.blob_list[self.selected_index]
            else:
                return
        else:
            return

        blob_id = blob.get('id', 'N/A')

        print()
        print_status(f"Deleting blob...", "progress")

        if self.selected_instance.delete_blob(blob_id):
            print_status("Blob deleted successfully", "success")
            self.load_blobs()  # Refresh list
        else:
            print_status("Failed to delete blob", "error")

        time.sleep(1)

    def export_current_blob(self, data: bytes):
        """Export blob data to file"""
        print()
        print("  Enter filename to export to: ", end='', flush=True)

        filename = input().strip()

        if filename:
            try:
                with open(filename, 'wb') as f:
                    f.write(data)
                print_status(f"Blob exported to {filename}", "success")
            except Exception as e:
                print_status(f"Export failed: {e}", "error")
        else:
            print_status("Export cancelled", "warning")

    def go_back(self):
        """Navigate back in the view hierarchy"""
        if self.current_view == "blob_detail":
            self.current_view = "blobs"
            self.current_blob = None
        elif self.current_view == "blobs":
            self.current_view = "instances"
            self.selected_instance = None
            self.blob_list = []
            self.selected_index = 0

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format byte size to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"


# =================== CLI Command Handlers ===================

def handle_build():
    """Build the Rust project"""
    print_box_header("Building r_blob_db", "🔨")
    print_box_content("Compiling Rust project in release mode", "info")
    print_box_footer()

    from toolboxv2 import tb_root_dir

    try:
        with Spinner("Compiling with Cargo", symbols='t'):
            a,b = detect_shell()
            result = subprocess.run(
                [a,b,"cargo", "build", "--release", "--package", "r_blob_db"],
                check=True,
                cwd=tb_root_dir / "r_blob_db",
                capture_output=True,
                text=True
            )

        print_status("Build successful!", "success")

        exe_path = get_executable_path()
        if exe_path:
            bin_dir = tb_root_dir / "bin"
            bin_dir.mkdir(exist_ok=True)
            try:
                dest_path = bin_dir / exe_path.name
                shutil.copy(exe_path, dest_path)
                print_status(f"Executable copied to: {dest_path}", "info")
            except Exception as e:
                print_status(f"Warning: Failed to copy to bin: {e}", "warning")
                # Fallback to ubin
                ubin_dir = tb_root_dir / "ubin"
                ubin_dir.mkdir(exist_ok=True)
                dest_path = ubin_dir / exe_path.name
                try:
                    shutil.copy(exe_path, dest_path)
                    print_status(f"Copied to fallback location: {dest_path}", "info")
                except Exception as e_ubin:
                    print_status(f"Error copying to ubin: {e_ubin}", "error")

    except subprocess.CalledProcessError as e:
        print_status("Build failed!", "error")
        print(Style.GREY(e.stderr))
    except FileNotFoundError:
        print_status("Build failed: 'cargo' command not found", "error")
        print_status("Is Rust installed and in your PATH?", "info")


def handle_clean():
    """Clean build artifacts"""
    print_box_header("Cleaning Build Artifacts", "🧹")
    print_box_footer()

    try:
        with Spinner("Running cargo clean", symbols='+'):
            a,b=detect_shell()
            from toolboxv2 import tb_root_dir
            subprocess.run([a,b,"cargo", "clean"], check=True, capture_output=True, cwd=tb_root_dir/"r_blob_db")
        print_status("Clean successful!", "success")
    except Exception as e:
        print_status(f"Clean failed: {e}", "error")


async def handle_discover(manager: ClusterManager):
    """Handle interactive data discovery"""
    discovery = DataDiscovery(manager)
    await discovery.run()


# =================== CLI Entry Point ===================

async def cli_db_runner():
    """The main entry point for the CLI application."""

    parser = argparse.ArgumentParser(
        description="🗄️  r_blob_db Cluster Manager - Interactive Database Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='tb db',
        epilog="""
╔════════════════════════════════════════════════════════════════════════════╗
║                           Command Examples                                 ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Build & Setup:                                                            ║
║    $ tb db build                     # Build Rust executable               ║
║    $ tb db clean                     # Clean build artifacts               ║
║                                                                            ║
║  Instance Management:                                                      ║
║    $ tb db start                     # Start all instances                 ║
║    $ tb db start --instance-id i1    # Start specific instance             ║
║    $ tb db stop                      # Stop all instances                  ║
║    $ tb db status                    # Show instance status                ║
║                                                                            ║
║  Health & Monitoring:                                                      ║
║    $ tb db health                    # Health check all instances          ║
║    $ tb db health --instance-id i1   # Check specific instance             ║
║                                                                            ║
║  Data Discovery:                                                           ║
║    $ tb db discover                  # Interactive data browser            ║
║                                                                            ║
║  Updates:                                                                  ║
║    $ tb db update --version 1.2.0    # Rolling cluster update              ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
        """
    )

    subparsers = parser.add_subparsers(dest="action", required=False, help="Available actions")

    # Define common arguments
    instance_arg = {
        'name_or_flags': ['--instance-id'],
        'type': str,
        'help': 'Target a specific instance ID. If omitted, action applies to the whole cluster.',
        'default': None
    }
    version_arg = {
        'name_or_flags': ['--version'],
        'type': str,
        'help': 'Specify a version string for the executable (e.g., "1.2.0").',
        'default': 'dev'
    }

    # --- Define Commands ---
    p_start = subparsers.add_parser('start', help='Start database instance(s)')
    p_start.add_argument(*instance_arg['name_or_flags'],
                         **{k: v for k, v in instance_arg.items() if k != 'name_or_flags'})
    p_start.add_argument(*version_arg['name_or_flags'],
                         **{k: v for k, v in version_arg.items() if k != 'name_or_flags'})

    p_stop = subparsers.add_parser('stop', help='Stop database instance(s)')
    p_stop.add_argument(*instance_arg['name_or_flags'],
                        **{k: v for k, v in instance_arg.items() if k != 'name_or_flags'})

    p_status = subparsers.add_parser('status', help='Show running status of instance(s)')
    p_status.add_argument(*instance_arg['name_or_flags'],
                          **{k: v for k, v in instance_arg.items() if k != 'name_or_flags'})

    p_health = subparsers.add_parser('health', help='Perform health check on instance(s)')
    p_health.add_argument(*instance_arg['name_or_flags'],
                          **{k: v for k, v in instance_arg.items() if k != 'name_or_flags'})

    p_update = subparsers.add_parser('update', help='Perform rolling update on cluster')
    p_update.add_argument(*instance_arg['name_or_flags'],
                          **{k: v for k, v in instance_arg.items() if k != 'name_or_flags'})
    version_arg_update = {**version_arg, 'required': True}
    p_update.add_argument(*version_arg_update['name_or_flags'],
                          **{k: v for k, v in version_arg_update.items() if k != 'name_or_flags'})

    subparsers.add_parser('build', help='Build the Rust executable from source')
    subparsers.add_parser('clean', help='Clean the Rust build artifacts')
    subparsers.add_parser('discover', help='Interactive data discovery and manipulation')

    # --- Execute Command ---
    args = parser.parse_args()

    if args.action == 'build':
        handle_build()
        return
    if args.action == 'clean':
        handle_clean()
        return

    manager = ClusterManager()

    if args.action == 'discover':
        await handle_discover(manager)
        return
    executable_path = None
    if args.action in ['start', 'update']:
        executable_path = get_executable_path(update=(args.action == 'update'))
        if not executable_path:
            print_status(f"Could not find the {EXECUTABLE_NAME} executable", "error")
            print_status("Please build it first with: tb db build", "info")
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
    else:
        import asyncio
        await handle_discover(manager)


if __name__ == "__main__":
    import asyncio
    asyncio.run(cli_db_runner())
