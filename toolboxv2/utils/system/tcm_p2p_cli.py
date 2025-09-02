# manager.py
# An advanced, production-style manager for the P2P application.
# Now with enhanced, colorful, and dynamic CLI output.

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- Enhanced UI Imports ---
# Assuming your style utilities are in 'utils/style.py'
try:
    from ..extras.Style import Spinner, Style
except ImportError:
    print("FATAL: UI utilities not found. Ensure 'utils/style.py' exists.")
    sys.exit(1)

# --- Configuration ---
try:
    import psutil
except ImportError:
    print(Style.RED("FATAL: Required library 'psutil' not found."))
    print(Style.YELLOW("Please install it using: pip install psutil"))
    sys.exit(1)

# The name of the Rust executable
EXECUTABLE_NAME = "tcm.exe" if platform.system() == "Windows" else "tcm"
# Base directory for all managed instances and their state

from ... import tb_root_dir

INSTANCES_ROOT_DIR = tb_root_dir / ".info" / "p2p_instances"


# --- Helper Functions ---

def get_executable_path(update=False) -> Path | None:
    """Finds the release executable in standard locations."""
    # Look in a dedicated 'bin' folder first, then cargo's default
    from toolboxv2 import tb_root_dir
    search_paths = [
        tb_root_dir /"bin" / EXECUTABLE_NAME,
        tb_root_dir / "tcm"/ "target" / "release" / EXECUTABLE_NAME,
    ]
    if update:
        search_paths = search_paths[::-1]
    for path in search_paths:
        print(path)
        if path.is_file():
            return path.resolve()
    return None



def find_instances() -> list['InstanceManager']:
    """Discovers all managed instances by scanning the instances directory."""
    if not INSTANCES_ROOT_DIR.is_dir():
        return []

    instance_managers = []
    for instance_dir in INSTANCES_ROOT_DIR.iterdir():
        if instance_dir.is_dir():
            instance_managers.append(InstanceManager(instance_dir.name))
    return instance_managers


# --- Core Management Class ---

class InstanceManager:
    """Manages a single named instance (relay or peer) of the P2P application."""

    def __init__(self, name: str):
        self.name = name
        self.instance_dir = INSTANCES_ROOT_DIR / self.name
        self.state_file = self.instance_dir / "state.json"
        self.config_file = self.instance_dir / "config.toml"
        self.log_file = self.instance_dir / "instance.log"

    def read_state(self) -> dict:
        """Reads the instance's state (pid, mode, etc.) from its state file."""
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def write_state(self, state_data: dict):
        """Writes the instance's state to its state file."""
        self.instance_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

    def is_running(self) -> bool:
        """Checks if the process associated with this instance is active."""
        pid = self.read_state().get('pid')
        return psutil.pid_exists(pid) if pid else False

    def generate_config(self, mode: str, config_data: dict):
        """Generates the config.toml file for this specific instance."""
        content = f'mode = "{mode}"\n\n'

        if mode == "relay":
            content += "[relay]\n"
            content += f'bind_address = "{config_data.get("bind_address", "0.0.0.0:9000")}"\n'
            content += f'password = "{config_data.get("password", "")}"\n'

        elif mode == "peer":
            content += "[peer]\n"
            content += f'relay_address = "{config_data.get("relay_address", "127.0.0.1:9000")}"\n'
            content += f'relay_password = "{config_data.get("relay_password", "")}"\n'
            content += f'peer_id = "{config_data.get("peer_id", "default-peer")}"\n'
            content += f'listen_address = "{config_data.get("listen_address", "127.0.0.1:8000")}"\n'
            content += f'forward_to_address = "{config_data.get("forward_to_address", "127.0.0.1:3000")}"\n'
            if config_data.get("target_peer_id"):
                content += f'target_peer_id = "{config_data.get("target_peer_id")}"\n'

        self.instance_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w") as f:
            f.write(content)
        print(f"    {Style.GREEN('Generated config:')} {Style.GREY(str(self.config_file))}")

    def start(self, executable_path: Path, mode: str, config_data: dict) -> bool:
        """Starts the instance process, detaches it, and logs its state."""
        if self.is_running():
            print(Style.YELLOW(f"Instance '{self.name}' is already running."))
            return True

        print(Style.CYAN(f"🚀 Starting instance '{self.name}'..."))
        self.generate_config(mode, config_data)
        log_handle = open(self.log_file, 'a')

        try:
            with Spinner(f"Launching process for '{self.name}'", symbols="d"):
                process = subprocess.Popen(
                    [str(executable_path)],
                    cwd=str(self.instance_dir),
                    stdout=log_handle,
                    stderr=log_handle,
                    creationflags=subprocess.DETACHED_PROCESS if platform.system() == "Windows" else 0
                )
                time.sleep(1.5)  # Give it a moment to stabilize or crash

            if process.poll() is not None:
                print(f"\n{Style.RED2('❌ ERROR:')} Instance '{self.name}' failed to start. Check logs for details:")
                print(f"    {Style.GREY(self.log_file)}")
                return False

            state = {'pid': process.pid, 'mode': mode, 'config': config_data}
            self.write_state(state)
            print(
                f"\n{Style.GREEN2('✅ Instance')} '{Style.Bold(self.name)}' {Style.GREEN2('started successfully.')} {Style.GREY(f'(PID: {process.pid})')}")
            print(f"   {Style.BLUE('Logging to:')} {Style.GREY(self.log_file)}")
            return True
        except Exception as e:
            print(f"\n{Style.RED2('❌ ERROR:')} Failed to launch instance '{self.name}': {e}")
            log_handle.close()
            return False

    def stop(self, timeout: int = 10) -> bool:
        """Stops the instance process gracefully with a forced kill fallback."""
        if not self.is_running():
            print(Style.YELLOW(f"Instance '{self.name}' is not running."))
            self.write_state({})
            return True

        pid = self.read_state().get('pid')
        with Spinner(f"Stopping '{self.name}' (PID: {pid})", symbols="+", time_in_s=timeout, count_down=True) as s:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                proc.wait(timeout)
            except psutil.TimeoutExpired:
                s.message = f"Force killing '{self.name}'"
                proc.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                print(f"\n{Style.RED2('❌ ERROR:')} Failed to stop instance '{self.name}': {e}")
                return False

        self.write_state({})
        print(f"\n{Style.VIOLET2('⏹️  Instance')} '{Style.Bold(self.name)}' {Style.VIOLET2('stopped.')}")
        return True


# --- CLI Command Handlers ---

def handle_start_relay(args):
    executable_path = get_executable_path()
    if not executable_path:
        print(Style.RED2("❌ ERROR: Executable not found. Please run 'tb p2p build' first."))
        return

    manager = InstanceManager(args.name)
    config = {
        "bind_address": args.bind,
        "password": args.password,
    }
    manager.start(executable_path, "relay", config)


def handle_start_peer(args):
    executable_path = get_executable_path()
    if not executable_path:
        print(Style.RED2("❌ ERROR: Executable not found. Please run 'python manager.py build' first."))
        return

    manager = InstanceManager(args.name)
    config = {
        "relay_address": args.relay_addr,
        "relay_password": args.relay_pass,
        "peer_id": args.peer_id or args.name,
        "listen_address": args.listen,
        "forward_to_address": args.forward,
        "target_peer_id": args.target
    }

    if args.forward:
        import asyncio

        from toolboxv2 import get_app
        from toolboxv2.mods.P2PRPCServer import start_rpc_server
        app = get_app("P2P_RPC_Server_Starter")
        host, port = args.forward.split(':')
        asyncio.create_task(start_rpc_server(app, host, int(port)))
        print(f"Starting P2P RPC Server on {args.forward}")

    manager.start(executable_path, "peer", config)


def handle_stop(args):
    if not args.names:
        instances_to_stop = find_instances()
        if not instances_to_stop:
            print(Style.YELLOW("No active instances found to stop."))
            return
    else:
        instances_to_stop = [InstanceManager(name) for name in args.names]

    print(Style.CYAN(f"Attempting to stop {len(instances_to_stop)} instance(s)..."))
    for manager in instances_to_stop:
        manager.stop()


def handle_status(args):
    header = f"--- {Style.Bold('Instance Status')} (Source: {Style.GREY(str(INSTANCES_ROOT_DIR))}) ---"
    print(header)
    instances = find_instances()
    if not instances:
        print(Style.YELLOW("No instances are currently being managed."))
        return

    print(
        f"{Style.Underline('NAME'):<18} {Style.Underline('PID'):<8} {Style.Underline('STATUS'):<20} {Style.Underline('MODE'):<10} {Style.Underline('ROLE')}")

    for manager in sorted(instances, key=lambda i: i.name):
        state = manager.read_state()
        pid = state.get('pid')
        mode = state.get('mode', 'N/A')
        config = state.get('config', {})

        status_str = "✅ RUNNING" if manager.is_running() else "❌ STOPPED"
        status_color = Style.GREEN2 if "RUNNING" in status_str else Style.RED2

        role = ""
        if mode == "peer":
            if config.get("forward_to_address"):
                role = f"Provider (forwards to {Style.YELLOW(config['forward_to_address'])})"
            elif config.get("target_peer_id"):
                role = f"Consumer (targets {Style.CYAN(config['target_peer_id'])})"
            else:
                role = "Idle (listener)"
        elif mode == "relay":
            role = f"Relay Server (on {Style.YELLOW(config.get('bind_address', 'N/A'))})"

        print(
            f"{Style.WHITE(manager.name):<18} {Style.GREY(str(pid or 'N/A')):<8} {status_color(status_str):<20} {Style.BLUE2(mode):<10} {role}")
    print("-" * len(header))


def handle_logs(args):
    manager = InstanceManager(args.name)
    if not manager.log_file.exists():
        print(f"{Style.RED2('❌ ERROR:')} Log file for instance '{args.name}' not found at:")
        print(f"    {Style.GREY(manager.log_file)}")
        return

    header = f"--- Tailing last {args.lines} lines of log for '{Style.Bold(args.name)}' ---"
    print(header)
    with open(manager.log_file) as f:
        lines = f.readlines()
        for line in lines[-args.lines:]:
            print(Style.GREY(line.strip()))
    print("-" * len(header))


def handle_build(args):
    print(Style.CYAN("Building Rust project in release mode..."))
    try:
        with Spinner("Compiling with Cargo", symbols="t"):
            # Redirect stdout to DEVNULL to not clutter spinner output
            process = subprocess.run(
                ["cargo", "build", "--release"],
                check=True,
                cwd=tb_root_dir/"tcm",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
        print(f"\n{Style.GREEN2('✅ Build successful.')} Executable is in './target/release/'.")
    except subprocess.CalledProcessError as e:
        print(f"\n{Style.RED2('❌ Build failed:')}")
        # Decode and print stderr from the failed process
        print(Style.GREY(e.stderr.decode()))
    except FileNotFoundError:
        print(f"\n{Style.RED2('❌ Build failed:')} 'cargo' command not found. Is Rust installed and in your PATH?")


def handle_cleanup(args):
    print(Style.YELLOW2(Style.Bold("This will stop ALL running instances and DELETE the entire management directory.")))
    if not args.yes:
        confirm = input(f"Are you sure you want to delete '{INSTANCES_ROOT_DIR}'? (y/N): ")
        if confirm.lower() != 'y':
            print(Style.GREEN("Cleanup aborted."))
            return

    handle_stop(argparse.Namespace(names=[]))

    if INSTANCES_ROOT_DIR.exists():
        print(Style.RED(f"Removing management directory: {INSTANCES_ROOT_DIR}"))
        shutil.rmtree(INSTANCES_ROOT_DIR)

    print(Style.GREEN2("✅ Cleanup complete."))


# --- Main CLI Setup ---

def cli_tcm_runner():
    parser = argparse.ArgumentParser(
        description=f"🚀 {Style.Bold('A production-style manager for the P2P application.')}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ... (Rest of the parsers are the same)
    p_build = subparsers.add_parser('build', help=f'Compile the Rust application ({Style.YELLOW("release mode")}).')
    p_build.set_defaults(func=handle_build)

    p_start_relay = subparsers.add_parser("start-relay", help="Start a relay server instance.")
    p_start_relay.add_argument("name", help="A unique name for this relay instance.")
    p_start_relay.add_argument("--bind", default="0.0.0.0:9000", help="Address and port for the relay to bind to.")
    p_start_relay.add_argument("--password", required=True, help="Password to protect the relay server.")
    p_start_relay.set_defaults(func=handle_start_relay)

    p_start_peer = subparsers.add_parser("start-peer", help="Start a peer client instance.")
    p_start_peer.add_argument("name", help="A unique name for this peer instance (e.g., 'peer-A').")
    p_start_peer.add_argument("--peer-id", help="The peer's public ID (defaults to its instance name).")
    p_start_peer.add_argument("--relay-addr", required=True, help="Address of the relay server (e.g., 127.0.0.1:9000).")
    p_start_peer.add_argument("--relay-pass", required=True, help="Password for the relay server.")
    p_start_peer.add_argument("--listen", default="127.0.0.1:8000", help="Local address for proxying connections.")
    group = p_start_peer.add_mutually_exclusive_group()
    group.add_argument("--forward", help=f'{Style.GREEN("PROVIDER role:")} Forwards connections to this local address.')
    group.add_argument("--target", help=f'{Style.CYAN("CONSUMER role:")} Connects to the peer with this ID.')
    p_start_peer.set_defaults(func=handle_start_peer)

    p_stop = subparsers.add_parser("stop", help="Stop one or more running instances.")
    p_stop.add_argument("names", nargs='*', help="Names of instances to stop. If empty, stops all.")
    p_stop.set_defaults(func=handle_stop)

    p_status = subparsers.add_parser("status", help="Show the status of all managed instances.")
    p_status.set_defaults(func=handle_status)

    p_logs = subparsers.add_parser("logs", help="View the tail of the log for a specific instance.")
    p_logs.add_argument("name", help="The name of the instance to view logs for.")
    p_logs.add_argument("-n", "--lines", type=int, default=20, help="Number of lines to show.")
    p_logs.set_defaults(func=handle_logs)

    p_cleanup = subparsers.add_parser("cleanup",
                                      help=f'Stop ALL instances and {Style.RED("permanently delete")} the management directory.')
    p_cleanup.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt.")
    p_cleanup.set_defaults(func=handle_cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli_tcm_runner()
