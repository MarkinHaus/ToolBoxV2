"""
Modern Package Manager Runner - Supporting conda, uv, and native Python
"""

import argparse
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import textwrap

from tqdm import tqdm
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.shortcuts import radiolist_dialog, yes_no_dialog, input_dialog, message_dialog
from prompt_toolkit.styles import Style


# =================== Constants & Enums ===================

class PackageManager(Enum):
    """Supported package managers."""
    CONDA = "conda"
    UV = "uv"
    NATIVE = "native"  # pip/venv


MODERN_STYLE = Style.from_dict({
    'success': '#50fa7b bold',
    'error': '#ff5555 bold',
    'warning': '#ffb86c bold',
    'info': '#8be9fd',
    'prompt': '#ff79c6 bold',
    'header': '#bd93f9 bold',
})


# =================== Helper Functions ===================

def print_status(status: str, message: str):
    """Print colored status message."""
    icons = {
        'success': '✓',
        'error': '✗',
        'warning': '⚠',
        'info': 'ℹ',
        'running': '⟳'
    }
    icon = icons.get(status, '•')
    print_formatted_text(HTML(f'<{status}>{icon} {message}</{status}>'), style=MODERN_STYLE)


def print_header(title: str):
    """Print section header."""
    width = 78
    print_formatted_text(HTML(f'\n<header>{"─" * width}</header>'))
    print_formatted_text(HTML(f'<header>{title.center(width)}</header>'))
    print_formatted_text(HTML(f'<header>{"─" * width}</header>\n'))


def detect_package_manager() -> PackageManager:
    """Auto-detect available package manager."""
    if shutil.which("uv"):
        return PackageManager.UV
    elif shutil.which("conda"):
        return PackageManager.CONDA
    else:
        return PackageManager.NATIVE


def get_encoding():
    """Get system encoding with fallback."""
    try:
        return sys.stdout.encoding or 'utf-8'
    except:
        return 'utf-8'


def discover_environments() -> Dict[str, List[Dict[str, str]]]:
    """Discover existing environments from all package managers."""
    from toolboxv2 import init_cwd
    discovered = {
        'conda': [],
        'uv': [],
        'native': []
    }

    # Discover Conda environments
    if shutil.which("conda"):
        try:
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True, text=True, check=True
            )
            data = json.loads(result.stdout)
            for env_path in data.get('envs', []):
                env_name = Path(env_path).name
                if env_name != 'base':  # Skip base environment
                    discovered['conda'].append({
                        'name': env_name,
                        'path': env_path,
                        'manager': 'conda'
                    })
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass

    # Discover UV environments
    if shutil.which("uv"):
        uv_envs_path = Path.home() / ".uv" / "envs"
        if uv_envs_path.exists():
            for env_dir in uv_envs_path.iterdir():
                if env_dir.is_dir():
                    discovered['uv'].append({
                        'name': env_dir.name,
                        'path': str(env_dir),
                        'manager': 'uv'
                    })

    # Discover Native Python environments
    native_paths = [
        Path.home() / "python_env",
        Path.home() / ".python_envs",
        Path.cwd() / "venv",
        Path.cwd() / ".venv",
        Path.cwd() / "env",
        init_cwd / "python_env",
        init_cwd / ".python_envs",
        init_cwd/ "venv",
        init_cwd/ ".venv",
        init_cwd/ "env"
    ]

    for base_path in native_paths:
        if base_path.exists():
            if base_path.name in ['venv', '.venv', 'env']:
                # Single environment in current directory
                if _is_valid_venv(base_path):
                    discovered['native'].append({
                        'name': f"local-{base_path.name}",
                        'path': str(base_path),
                        'manager': 'native'
                    })
            else:
                # Multiple environments in directory
                for env_dir in base_path.iterdir():
                    if env_dir.is_dir() and _is_valid_venv(env_dir):
                        discovered['native'].append({
                            'name': env_dir.name,
                            'path': str(env_dir),
                            'manager': 'native'
                        })

    return discovered


def _is_valid_venv(path: Path) -> bool:
    """Check if a path contains a valid Python virtual environment."""
    # Check for Unix-like systems (Linux, macOS)
    unix_python = path / "bin" / "python"
    unix_python3 = path / "bin" / "python3"

    # Check for Windows
    win_python = path / "Scripts" / "python.exe"
    win_python_2 = path / "python.exe"

    # Check for pyvenv.cfg which is created by venv and virtualenv
    pyvenv_cfg = path / "pyvenv.cfg"

    return (unix_python.exists() or
            unix_python3.exists() or
            win_python.exists() or
            win_python_2.exists() or
            pyvenv_cfg.exists())

def save_discovered_environments(discovered: Dict[str, List[Dict[str, str]]]) -> Path:
    """Save discovered environments to registry file."""
    registry_file = Path.home() / ".toolbox_env_registry.json"

    # Load existing registry or create new
    existing_registry = {}
    if registry_file.exists():
        try:
            with open(registry_file, 'r') as f:
                existing_registry = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Merge discovered environments
    for manager, envs in discovered.items():
        if manager not in existing_registry:
            existing_registry[manager] = []

        # Add new environments (avoid duplicates)
        existing_names = {env['name'] for env in existing_registry[manager]}
        for env in envs:
            if env['name'] not in existing_names:
                existing_registry[manager].append(env)

    # Save updated registry
    try:
        with open(registry_file, 'w') as f:
            json.dump(existing_registry, f, indent=2)
        return registry_file
    except IOError as e:
        raise Exception(f"Failed to save registry: {e}")


# =================== Command Execution ===================

class CommandRunner:
    """Enhanced command runner with better output handling."""

    def __init__(self, package_manager: PackageManager):
        self.pm = package_manager
        self.encoding = get_encoding()

    def run(self, command: str, live: bool = True, capture: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Execute command with optional live output.

        Args:
            command: Command to execute
            live: Stream output in real-time
            capture: Capture and return output

        Returns:
            Tuple of (success, output)
        """
        print_status('running', f'Executing: {command}')

        if live and not capture:
            # Stream output live
            try:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    text=True,
                    encoding=self.encoding,
                    errors='replace'
                )
                process.communicate()
                success = process.returncode == 0

                if success:
                    print_status('success', 'Command completed successfully')
                else:
                    print_status('error', f'Command failed with code {process.returncode}')

                return success, None
            except Exception as e:
                print_status('error', f'Execution error: {e}')
                return False, None

        else:
            # Capture output
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    text=True,
                    capture_output=True,
                    encoding=self.encoding,
                    errors='replace'
                )
                print_status('success', 'Command completed')
                return True, result.stdout

            except subprocess.CalledProcessError as e:
                print_status('error', 'Command failed')
                if e.stdout:
                    print(f"\nOutput:\n{e.stdout}")
                if e.stderr:
                    print(f"\nError:\n{e.stderr}")
                return False, None

            except Exception as e:
                print_status('error', f'Execution error: {e}')
                return False, None


# =================== Package Manager Implementations ===================

class BasePackageManager:
    """Base class for package managers."""

    def __init__(self, runner: CommandRunner):
        self.runner = runner

    def create_env(self, env_name: str, python_version: str = "3.11") -> bool:
        raise NotImplementedError

    def delete_env(self, env_name: str) -> bool:
        raise NotImplementedError

    def list_envs(self) -> List[str]:
        raise NotImplementedError

    def install_package(self, env_name: str, package: str) -> bool:
        raise NotImplementedError

    def update_package(self, env_name: str, package: str) -> bool:
        raise NotImplementedError

    def list_packages(self, env_name: str) -> List[Dict[str, str]]:
        raise NotImplementedError

    def run_script(self, env_name: str, script: str, args: List[str], python: bool = True) -> bool:
        raise NotImplementedError


class CondaManager(BasePackageManager):
    """Conda package manager implementation."""

    def create_env(self, env_name: str, python_version: str = "3.11") -> bool:
        command = f"conda create -n {env_name} python={python_version} -y"
        return self.runner.run(command)[0]

    def delete_env(self, env_name: str) -> bool:
        command = f"conda env remove -n {env_name} -y"
        success = self.runner.run(command)[0]

        # Clean up registry
        registry_file = Path(f"{env_name}_registry.json")
        if registry_file.exists():
            registry_file.unlink()
            print_status('info', f'Removed registry file: {registry_file}')

        return success

    def list_envs(self) -> List[str]:
        command = "conda env list --json"
        success, output = self.runner.run(command, live=False, capture=True)

        if success and output:
            try:
                data = json.loads(output)
                envs = [Path(env).name for env in data.get('envs', [])]
                return envs
            except json.JSONDecodeError:
                print_status('error', 'Failed to parse environment list')

        return []

    def install_package(self, env_name: str, package: str) -> bool:
        command = f"conda install -n {env_name} {package} -y"
        success = self.runner.run(command)[0]

        if success:
            self._update_registry(env_name, package)

        return success

    def update_package(self, env_name: str, package: str) -> bool:
        command = f"conda update -n {env_name} {package} -y"
        return self.runner.run(command)[0]

    def list_packages(self, env_name: str) -> List[Dict[str, str]]:
        command = f"conda list -n {env_name} --json"
        success, output = self.runner.run(command, live=False, capture=True)

        if success and output:
            try:
                packages = json.loads(output)
                return [{"name": pkg["name"], "version": pkg["version"]} for pkg in packages]
            except json.JSONDecodeError:
                print_status('error', 'Failed to parse package list')

        return []

    def run_script(self, env_name: str, script: str, args: List[str], python: bool = True) -> bool:
        if python:
            command = f"conda run -v --no-capture-output -n {env_name} python {script} {' '.join(args)}"
        else:
            command = f"conda run -v --no-capture-output -n {env_name} {script} {' '.join(args)}"

        return self.runner.run(command)[0]

    def _update_registry(self, env_name: str, package: str):
        """Update package registry."""
        registry_file = Path(f"{env_name}_registry.json")

        try:
            if registry_file.exists():
                with open(registry_file) as f:
                    registry = json.load(f)
            else:
                registry = []

            if package not in registry:
                registry.append(package)

            with open(registry_file, 'w') as f:
                json.dump(registry, f, indent=2)

            print_status('info', f'Updated registry: {registry_file}')

        except Exception as e:
            print_status('warning', f'Failed to update registry: {e}')


class UVManager(BasePackageManager):
    """UV package manager implementation."""

    def create_env(self, env_name: str, python_version: str = "3.11") -> bool:
        env_path = Path.home() / ".uv" / "envs" / env_name
        command = f"uv venv {env_path} --python {python_version}"
        return self.runner.run(command)[0]

    def delete_env(self, env_name: str) -> bool:
        env_path = Path.home() / ".uv" / "envs" / env_name

        if env_path.exists():
            try:
                shutil.rmtree(env_path)
                print_status('success', f'Removed environment: {env_path}')
                return True
            except Exception as e:
                print_status('error', f'Failed to remove environment: {e}')
                return False
        else:
            print_status('warning', f'Environment not found: {env_path}')
            return False

    def list_envs(self) -> List[str]:
        envs_path = Path.home() / ".uv" / "envs"

        if envs_path.exists():
            return [d.name for d in envs_path.iterdir() if d.is_dir()]

        return []

    def install_package(self, env_name: str, package: str) -> bool:
        env_path = Path.home() / ".uv" / "envs" / env_name
        command = f"uv pip install --python {env_path}/bin/python {package}"
        return self.runner.run(command)[0]

    def update_package(self, env_name: str, package: str) -> bool:
        env_path = Path.home() / ".uv" / "envs" / env_name
        command = f"uv pip install --upgrade --python {env_path}/bin/python {package}"
        return self.runner.run(command)[0]

    def list_packages(self, env_name: str) -> List[Dict[str, str]]:
        env_path = Path.home() / ".uv" / "envs" / env_name
        command = f"uv pip list --python {env_path}/bin/python --format json"
        success, output = self.runner.run(command, live=False, capture=True)

        if success and output:
            try:
                packages = json.loads(output)
                return [{"name": pkg["name"], "version": pkg["version"]} for pkg in packages]
            except json.JSONDecodeError:
                print_status('error', 'Failed to parse package list')

        return []

    def run_script(self, env_name: str, script: str, args: List[str], python: bool = True) -> bool:
        env_path = Path.home() / ".uv" / "envs" / env_name
        python_bin = env_path / "bin" / "python"

        if python:
            command = f"{python_bin} {script} {' '.join(args)}"
        else:
            command = f"{script} {' '.join(args)}"

        return self.runner.run(command)[0]


class NativeManager(BasePackageManager):
    """Native Python (venv + pip) manager implementation."""

    def __init__(self, runner: CommandRunner):
        super().__init__(runner)
        self.envs_base = Path.home() / ".python_envs"
        self.envs_base.mkdir(exist_ok=True)

    def create_env(self, env_name: str, python_version: str = "3.11") -> bool:
        env_path = self.envs_base / env_name
        command = f"python{python_version} -m venv {env_path}"

        # Fallback to default python if version not available
        if not shutil.which(f"python{python_version}"):
            print_status('warning', f'Python {python_version} not found, using default python')
            command = f"python -m venv {env_path}"

        return self.runner.run(command)[0]

    def delete_env(self, env_name: str) -> bool:
        env_path = self.envs_base / env_name

        if env_path.exists():
            try:
                shutil.rmtree(env_path)
                print_status('success', f'Removed environment: {env_path}')
                return True
            except Exception as e:
                print_status('error', f'Failed to remove environment: {e}')
                return False
        else:
            print_status('warning', f'Environment not found: {env_path}')
            return False

    def list_envs(self) -> List[str]:
        if self.envs_base.exists():
            return [d.name for d in self.envs_base.iterdir() if d.is_dir() and (d / "bin" / "python").exists()]
        return []

    def install_package(self, env_name: str, package: str) -> bool:
        env_path = self.envs_base / env_name
        pip_bin = env_path / "bin" / "pip"

        if sys.platform == "win32":
            pip_bin = env_path / "Scripts" / "pip.exe"

        command = f"{pip_bin} install {package}"
        return self.runner.run(command)[0]

    def update_package(self, env_name: str, package: str) -> bool:
        env_path = self.envs_base / env_name
        pip_bin = env_path / "bin" / "pip"

        if sys.platform == "win32":
            pip_bin = env_path / "Scripts" / "pip.exe"

        command = f"{pip_bin} install --upgrade {package}"
        return self.runner.run(command)[0]

    def list_packages(self, env_name: str) -> List[Dict[str, str]]:
        env_path = self.envs_base / env_name
        pip_bin = env_path / "bin" / "pip"

        if sys.platform == "win32":
            pip_bin = env_path / "Scripts" / "pip.exe"

        command = f"{pip_bin} list --format json"
        success, output = self.runner.run(command, live=False, capture=True)

        if success and output:
            try:
                packages = json.loads(output)
                return [{"name": pkg["name"], "version": pkg["version"]} for pkg in packages]
            except json.JSONDecodeError:
                print_status('error', 'Failed to parse package list')

        return []

    def run_script(self, env_name: str, script: str, args: List[str], python: bool = True) -> bool:
        env_path = self.envs_base / env_name
        python_bin = env_path / "bin" / "python"

        if sys.platform == "win32":
            python_bin = env_path / "Scripts" / "python.exe"

        if python:
            command = f"{python_bin} {script} {' '.join(args)}"
        else:
            command = f"{script} {' '.join(args)}"

        return self.runner.run(command)[0]


# =================== Manager Factory ===================

def create_manager(pm_type: PackageManager) -> BasePackageManager:
    """Create appropriate package manager."""
    runner = CommandRunner(pm_type)

    if pm_type == PackageManager.CONDA:
        return CondaManager(runner)
    elif pm_type == PackageManager.UV:
        return UVManager(runner)
    else:
        return NativeManager(runner)


def handle_discover(args, manager: BasePackageManager):
    """Handle environment discovery."""
    print_header('Discovering Environments')

    discovered = discover_environments()

    total_found = sum(len(envs) for envs in discovered.values())

    if total_found == 0:
        print_status('warning', 'No environments discovered')
        return

    if args.json:
        print(json.dumps(discovered, indent=2))
        return

    # Display discovered environments
    for manager_name, envs in discovered.items():
        if envs:
            print(f"\n📦 {manager_name.upper()} Environments ({len(envs)} found):")
            print('─' * 60)

            for i, env in enumerate(envs, 1):
                print(f"  {i:>2}. {env['name']:<25} → {env['path']}")

    print(f"\n🔍 Total discovered: {total_found} environment(s)")

    # Save to registry if requested
    if args.save:
        try:
            registry_file = save_discovered_environments(discovered)
            print_status('success', f'Environments saved to registry: {registry_file}')
        except Exception as e:
            print_status('error', f'Failed to save registry: {e}')
# =================== CLI Interface ===================

def create_parser() -> argparse.ArgumentParser:
    """Create modern CLI parser."""

    parser = argparse.ArgumentParser(
        prog='tb venv',
        description=textwrap.dedent("""
        ╔════════════════════════════════════════════════════════════════════╗
        ║          🐍 Modern Python Environment Manager 🐍                   ║
        ╚════════════════════════════════════════════════════════════════════╝

        Unified interface for conda, uv, and native Python environments.

        """),
        epilog=textwrap.dedent("""
        ┌─ EXAMPLES ─────────────────────────────────────────────────────────┐
        │                                                                    │
        │  Environment Management:                                           │
        │    $ tb venv create myenv                  # Create environment    │
        │    $ tb venv list                          # List environments     │
        │    $ tb venv delete myenv                  # Delete environment    │
        │                                                                    │
        │  Package Management:                                               │
        │    $ tb venv install myenv numpy           # Install package       │
        │    $ tb venv update myenv numpy            # Update package        │
        │    $ tb venv packages myenv                # List packages         │
        │                                                                    │
        │  Script Execution:                                                 │
        │    $ tb venv run myenv script.py arg1      # Run Python script     │
        │    $ tb venv exec myenv command args       # Run command           │
        │                                                                    │
        │  Advanced:                                                         │
        │    $ tb venv registry myenv                # Create registry       │
        │    $ tb venv update-all myenv              # Update all packages   │
        │    $ tb venv --manager uv create myenv     # Use specific PM       │
        │                                                                    │
        └────────────────────────────────────────────────────────────────────┘
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global options
    parser.add_argument('--manager', '-m',
                        choices=['conda', 'uv', 'native'],
                        help='Package manager to use (auto-detect if not specified)')

    parser.add_argument('--python', '-py',
                        default='3.11',
                        help='Python version (default: 3.11)')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # =================== ENVIRONMENT COMMANDS ===================

    # Create environment
    create_parser = subparsers.add_parser('create', help='Create new environment')
    create_parser.add_argument('env_name', help='Environment name')
    create_parser.add_argument('--python', '-py', help='Python version (default: 3.11)')

    # Delete environment
    delete_parser = subparsers.add_parser('delete', help='Delete environment')
    delete_parser.add_argument('env_name', help='Environment name')
    delete_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')

    # List environments
    list_parser = subparsers.add_parser('list', help='List all environments')

    # =================== PACKAGE COMMANDS ===================

    # Install package
    install_parser = subparsers.add_parser('install', help='Install package')
    install_parser.add_argument('env_name', help='Environment name')
    install_parser.add_argument('packages', nargs='+', help='Package(s) to install')
    install_parser.add_argument('--save', '-s', action='store_true', help='Save to registry')

    # Update package
    update_parser = subparsers.add_parser('update', help='Update package')
    update_parser.add_argument('env_name', help='Environment name')
    update_parser.add_argument('package', nargs='?', help='Package to update (all if not specified)')

    # List packages
    packages_parser = subparsers.add_parser('packages', help='List installed packages')
    packages_parser.add_argument('env_name', help='Environment name')
    packages_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # =================== EXECUTION COMMANDS ===================

    # Run Python script
    run_parser = subparsers.add_parser('run', help='Run Python script in environment')
    run_parser.add_argument('env_name', help='Environment name')
    run_parser.add_argument('script', help='Script to run')
    run_parser.add_argument('args', nargs='*', help='Script arguments')

    # Execute command
    exec_parser = subparsers.add_parser('exec', help='Execute command in environment')
    exec_parser.add_argument('env_name', help='Environment name')
    exec_parser.add_argument('command', help='Command to execute')
    exec_parser.add_argument('args', nargs='*', help='Command arguments')

    # =================== UTILITY COMMANDS ===================

    # Create registry
    registry_parser = subparsers.add_parser('registry', help='Create package registry')
    registry_parser.add_argument('env_name', help='Environment name')

    # Update all packages
    update_all_parser = subparsers.add_parser('update-all', help='Update all packages')
    update_all_parser.add_argument('env_name', help='Environment name')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show environment information')
    info_parser.add_argument('env_name', nargs='?', help='Environment name (current if not specified)')
    # Discover environments
    discover_parser = subparsers.add_parser('discover', help='Discover existing environments from all managers')
    discover_parser.add_argument('--save', '-s', action='store_true', help='Save discovered environments to registry')
    discover_parser.add_argument('--json', action='store_true', help='Output as JSON')
    return parser


# =================== Command Handlers ===================

def handle_create(args, manager: BasePackageManager):
    """Handle environment creation."""
    print_header(f'Creating Environment: {args.env_name}')

    python_version = args.python or "3.11"

    if manager.create_env(args.env_name, python_version):
        print_status('success', f'Environment "{args.env_name}" created successfully!')
    else:
        print_status('error', f'Failed to create environment "{args.env_name}"')
        sys.exit(1)


def handle_delete(args, manager: BasePackageManager):
    """Handle environment deletion."""
    if not args.force:
        # Confirm deletion
        result = yes_no_dialog(
            title='Confirm Deletion',
            text=f'Really delete environment "{args.env_name}"?\n\nThis action cannot be undone.',
            style=MODERN_STYLE
        ).run()

        if not result:
            print_status('info', 'Deletion cancelled')
            return

    print_header(f'Deleting Environment: {args.env_name}')

    if manager.delete_env(args.env_name):
        print_status('success', f'Environment "{args.env_name}" deleted successfully!')
    else:
        print_status('error', f'Failed to delete environment "{args.env_name}"')
        sys.exit(1)


def handle_list(args, manager: BasePackageManager):
    """Handle environment listing."""
    print_header('Available Environments')

    envs = manager.list_envs()

    if not envs:
        print_status('warning', 'No environments found')
        return

    print(f"\n{'#':<4} {'Environment Name':<30}")
    print('─' * 50)

    for i, env in enumerate(envs, 1):
        print(f"{i:<4} {env:<30}")

    print(f"\nTotal: {len(envs)} environment(s)\n")


def handle_install(args, manager: BasePackageManager):
    """Handle package installation."""
    print_header(f'Installing Packages in: {args.env_name}')

    for package in args.packages:
        print(f"\nInstalling {package}...")

        if manager.install_package(args.env_name, package):
            print_status('success', f'Package "{package}" installed')
        else:
            print_status('error', f'Failed to install "{package}"')


def handle_update(args, manager: BasePackageManager):
    """Handle package update."""
    print_header(f'Updating Packages in: {args.env_name}')

    if args.package:
        # Update single package
        if manager.update_package(args.env_name, args.package):
            print_status('success', f'Package "{args.package}" updated')
        else:
            print_status('error', f'Failed to update "{args.package}"')
    else:
        # Update all packages
        packages = manager.list_packages(args.env_name)

        if not packages:
            print_status('warning', 'No packages found')
            return

        print(f"Updating {len(packages)} package(s)...")

        for pkg in tqdm(packages, desc="Updating"):
            manager.update_package(args.env_name, pkg['name'])

        print_status('success', 'All packages updated')


def handle_packages(args, manager: BasePackageManager):
    """Handle package listing."""
    print_header(f'Packages in: {args.env_name}')

    packages = manager.list_packages(args.env_name)

    if not packages:
        print_status('warning', 'No packages found')
        return

    if args.json:
        print(json.dumps(packages, indent=2))
    else:
        print(f"\n{'#':<6} {'Package':<35} {'Version':<15}")
        print('─' * 60)

        for i, pkg in enumerate(packages, 1):
            print(f"{i:<6} {pkg['name']:<35} {pkg['version']:<15}")

        print(f"\nTotal: {len(packages)} package(s)\n")


def handle_run(args, manager: BasePackageManager):
    """Handle script execution."""
    print_header(f'Running Script in: {args.env_name}')

    if manager.run_script(args.env_name, args.script, args.args, python=True):
        print_status('success', 'Script completed successfully')
    else:
        print_status('error', 'Script execution failed')
        sys.exit(1)


def handle_exec(args, manager: BasePackageManager):
    """Handle command execution."""
    print_header(f'Executing Command in: {args.env_name}')

    if manager.run_script(args.env_name, args.command, args.args, python=False):
        print_status('success', 'Command completed successfully')
    else:
        print_status('error', 'Command execution failed')
        sys.exit(1)


def handle_registry(args, manager: BasePackageManager):
    """Handle registry creation."""
    print_header(f'Creating Registry for: {args.env_name}')

    packages = manager.list_packages(args.env_name)

    if not packages:
        print_status('warning', 'No packages to register')
        return

    registry_file = Path(f"{args.env_name}_registry.json")

    try:
        with open(registry_file, 'w') as f:
            json.dump(packages, f, indent=2)

        print_status('success', f'Registry created: {registry_file}')
        print_status('info', f'Registered {len(packages)} package(s)')

    except Exception as e:
        print_status('error', f'Failed to create registry: {e}')
        sys.exit(1)


def handle_info(args, manager: BasePackageManager):
    """Handle info display."""
    env_name = args.env_name or 'current'

    print_header(f'Environment Info: {env_name}')

    # Show package count
    packages = manager.list_packages(env_name) if args.env_name else []

    print(f"Package Manager: {manager.runner.pm.value}")
    print(f"Total Packages: {len(packages)}")
    print()


# =================== Main Function ===================

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Determine package manager
    if args.manager:
        pm_type = PackageManager(args.manager)
    else:
        pm_type = detect_package_manager()
        print_status('info', f'Auto-detected package manager: {pm_type.value}')

    # Create manager
    manager = create_manager(pm_type)

    # Handle command
    try:
        if args.command == 'create':
            handle_create(args, manager)
        elif args.command == 'delete':
            handle_delete(args, manager)
        elif args.command == 'list':
            handle_list(args, manager)
        elif args.command == 'install':
            handle_install(args, manager)
        elif args.command == 'update':
            handle_update(args, manager)
        elif args.command == 'packages':
            handle_packages(args, manager)
        elif args.command == 'run':
            handle_run(args, manager)
        elif args.command == 'exec':
            handle_exec(args, manager)
        elif args.command == 'registry':
            handle_registry(args, manager)
        elif args.command == 'update-all':
            handle_update(args, manager)
        elif args.command == 'info':
            handle_info(args, manager)
        elif args.command == 'discover':
            handle_discover(args, manager)

    except KeyboardInterrupt:
        print_status('warning', '\nOperation cancelled by user')
        sys.exit(130)

    except Exception as e:
        print_status('error', f'Unexpected error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
