#!/usr/bin/env python3
"""
ToolBoxV2 Setup Helper
A modern, interactive setup utility with enhanced visual feedback
"""

import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# --- CLI Printing Utilities ---
from toolboxv2.utils.clis.cli_printing import (
    print_box_header,
    print_box_content,
    print_box_footer,
    print_status,
    print_separator
)


# =================== Modern UI Helpers (Spinner fallback) ===================

# Note: print_box_header, print_box_content, print_box_footer, print_status, print_separator
# are now imported from cli_printing




def print_menu_option(number: int, text: str, selected: bool = False):
    """Print a menu option"""
    if selected:
        print(f"  \033[96m▶ {number}. {text}\033[0m")
    else:
        print(f"    {number}. {text}")


def show_spinner_text(message: str):
    """Show a simple progress indicator"""
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    sys.stdout.write(f'\r\033[96m{spinner[0]} {message}...\033[0m')
    sys.stdout.flush()


def clear_spinner():
    """Clear spinner line"""
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()


# =================== Configuration ===================

# Platform groups
PLATFORMS = {
    "Windows": ["winget", "scoop", "choco"],
    "Linux": ["apt", "dnf", "pacman", "asdf"],
    "Darwin": ["brew", "asdf"],
}

# Tool install templates
TEMPLATES = {
    "cargo": {
        "winget": "winget install Rustlang.Rustup",
        "scoop": "scoop install rust",
        "choco": "choco install rust",
        "apt": "sudo apt install -y cargo",
        "dnf": "sudo dnf install -y cargo",
        "pacman": "sudo pacman -S --noconfirm cargo",
        "brew": "brew install rust",
        "asdf": "asdf plugin-add rust || true && asdf install rust latest",
    },
    "node": {
        "winget": "winget install OpenJS.NodeJS",
        "scoop": "scoop install nodejs",
        "choco": "choco install nodejs",
        "apt": "sudo apt install -y nodejs npm",
        "dnf": "sudo dnf install -y nodejs",
        "pacman": "sudo pacman -S --noconfirm nodejs npm",
        "brew": "brew install node",
        "asdf": "asdf plugin-add nodejs || true && asdf install nodejs latest",
    },
    "docker": {
        "winget": "winget install Docker.DockerDesktop",
        "scoop": "scoop install docker",
        "choco": "choco install docker-desktop",
        "apt": "sudo apt install -y docker.io",
        "dnf": "sudo dnf install -y docker",
        "pacman": "sudo pacman -S --noconfirm docker",
        "brew": "brew install --cask docker",
        "asdf": "asdf plugin-add docker || true && asdf install docker latest",
    }
}

# Binaries for detection
BIN_MAP = {
    "winget": "winget",
    "scoop": "scoop",
    "choco": "choco",
    "apt": "apt",
    "dnf": "dnf",
    "pacman": "pacman",
    "brew": "brew",
    "asdf": "asdf"
}


# =================== Helper Functions ===================

def get_tb_root() -> Path:
    """Get ToolBoxV2 root directory"""
    # Try to find it relative to this script
    current = Path(__file__).resolve().parent

    # Look for package.json or src-core as markers
    for parent in [current] + list(current.parents):
        if (parent / "package.json").exists() or (parent / "src-core").exists():
            return parent

    # Fallback to current directory
    return Path.cwd()


def is_installed(tool: str) -> bool:
    """Check if a tool is installed"""
    return shutil.which(tool) is not None


def get_current_managers() -> List[str]:
    """Get available package managers for current platform"""
    system = platform.system()
    return PLATFORMS.get(system, [])


def get_managers_for_tool(tool: str) -> List[dict]:
    """Get available managers that can install the given tool"""
    available_managers = get_current_managers()
    tool_cmds = TEMPLATES.get(tool, {})

    return [
        {
            "name": mgr,
            "bin": BIN_MAP[mgr],
            "install_cmd": tool_cmds[mgr]
        }
        for mgr in available_managers if mgr in tool_cmds
    ]


def run_command(command: str, cwd: Optional[Path] = None, silent: bool = False) -> bool:
    """Run a shell command"""
    if cwd is None:
        cwd = get_tb_root()

    try:
        subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE if silent else None,
            stderr=subprocess.PIPE if silent else None
        )
        return True
    except subprocess.CalledProcessError as e:
        if not silent:
            print_status(f"Command failed: {command}", "error")
            if e.stderr:
                print(f"  Error: {e.stderr.decode()}")
        return False


# =================== User Input Functions ===================

def input_with_validation(prompt: str, valid_options: Optional[List[str]] = None) -> str:
    """Get user input with validation"""
    while True:
        user_input = input(f"\033[96m❯ {prompt}: \033[0m").strip().lower()

        if valid_options is None or user_input in valid_options:
            return user_input

        print_status(f"Invalid input. Valid options: {', '.join(valid_options)}", "error")


def ask_choice(prompt: str, choices: List[str]) -> str:
    """Ask user to choose from a list"""
    print()
    print_separator("═")
    print(f"  {prompt}")
    print_separator("═")
    print()

    for idx, choice in enumerate(choices, 1):
        print_menu_option(idx, choice)

    print()

    while True:
        try:
            selection = int(input("\033[96m❯ Choose option (number): \033[0m").strip())
            if 1 <= selection <= len(choices):
                selected = choices[selection - 1]
                print_status(f"Selected: {selected}", "success")
                return selected
        except (ValueError, KeyboardInterrupt):
            if KeyboardInterrupt:
                print()
                print_status("Setup cancelled by user", "warning")
                sys.exit(0)

        print_status(f"Please enter a number between 1 and {len(choices)}", "error")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask yes/no question"""
    default_str = "Y/n" if default else "y/N"
    response = input(f"\033[96m❯ {prompt} ({default_str}): \033[0m").strip().lower()

    if not response:
        return default

    return response in ['y', 'yes']


# =================== Mode Selection ===================

def select_mode() -> str:
    """Select setup mode"""
    print_box_header("Setup Mode Selection", "🎯")
    print()
    print("  \033[1mDev Mode\033[0m")
    print("    • Build from source")
    print("    • Install Cargo (Rust) and Node.js")
    print("    • Build web distribution and desktop app")
    print("    • Best for development and customization")
    print()
    print("  \033[1mUser Mode\033[0m")
    print("    • Download pre-built binaries")
    print("    • Quick installation")
    print("    • Best for end users")
    print()
    print_box_footer()

    return ask_choice("Select installation mode", ["dev", "user"])


# =================== Python Selection ===================

def select_python_interpreter() -> str:
    """Select Python interpreter"""
    print_box_header("Python Interpreter Selection", "🐍")
    print_box_content("Scanning for available Python installations...", "info")
    print_box_footer()

    # Search for Python interpreters
    candidates = ["python3.11", "python3.10", "python3.9", "python3", "python", "conda", "uv"]
    found = []

    for idx, cmd in enumerate(candidates):
        show_spinner_text(f"Checking {cmd}")
        if is_installed(cmd):
            found.append(cmd)
        time.sleep(0.1)  # Small delay for visual feedback

    clear_spinner()

    if not found:
        print_box_header("No Python Found", "✗")
        print_box_content("No Python installation detected", "error")
        print_box_content("Please install Python 3.9+ first", "info")
        print_box_footer()
        sys.exit(1)

    print_status(f"Found {len(found)} Python installation(s)", "success")
    print()

    return ask_choice("Select Python interpreter/manager", found)


# =================== Tool Installation ===================

def install_with_manager(tool: str) -> Tuple[str, bool]:
    """Try to install a tool using available package managers"""
    managers = get_managers_for_tool(tool)

    for mgr in managers:
        if shutil.which(mgr["bin"]):
            print_status(f"Installing {tool} using {mgr['name']}", "install")

            success = run_command(mgr["install_cmd"], silent=False)

            if success:
                # Verify installation
                time.sleep(1)
                if is_installed(tool):
                    return tool, True
                else:
                    print_status(f"Installation succeeded but {tool} not found in PATH", "warning")

            return tool, success

    print_status(f"No package manager available for {tool}", "error")
    return tool, False


def install_tools_parallel(tools: List[str], max_threads: int = 3) -> List[Tuple[str, bool]]:
    """Install multiple tools in parallel"""
    print_box_header(f"Installing Development Tools", "📦")
    print_box_content(f"Tools to install: {', '.join(tools)}", "info")
    print_box_content(f"Parallel threads: {max_threads}", "info")
    print_box_footer()

    results = []

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(install_with_manager, tool): tool for tool in tools}

        for future in as_completed(futures):
            tool, success = future.result()

            if success:
                print_status(f"Successfully installed: {tool}", "success")
            else:
                print_status(f"Failed to install: {tool}", "error")

            results.append((tool, success))

    print()

    # Summary
    successful = sum(1 for _, success in results if success)
    print_separator()
    print(f"  Installation complete: {successful}/{len(tools)} successful")
    print_separator()
    print()

    return results


def install_dev_tools():
    """Install development tools"""
    print_box_header("Development Tools Setup", "🔧")
    print_box_footer()

    tools = ["cargo", "node"]

    # Ask about Docker
    if ask_yes_no("Install Docker?", default=False):
        tools.append("docker")

    # Filter already installed
    to_install = []
    already_installed = []

    for tool in tools:
        if is_installed(tool):
            already_installed.append(tool)
        else:
            to_install.append(tool)

    if already_installed:
        print()
        print_status("Already installed:", "info")
        for tool in already_installed:
            print(f"  • {tool}")

    if not to_install:
        print()
        print_status("All required tools are already installed", "success")
        return

    print()
    print_status(f"Need to install: {', '.join(to_install)}", "install")
    print()

    # Install missing tools
    install_tools_parallel(to_install, max_threads=3)


# =================== NPM & Build Functions ===================

def install_npm_dependencies(dev_mode: bool = False) -> bool:
    """Install NPM dependencies"""
    print_box_header("Installing NPM Dependencies", "📦")
    print_box_content(f"Mode: {'Development' if dev_mode else 'Production'}", "info")

    tb_root = get_tb_root()
    print_box_content(f"Location: {tb_root}", "info")
    print_box_footer()

    # Check if package.json exists
    package_json = tb_root / "package.json"
    if not package_json.exists():
        print_status(f"package.json not found in {tb_root}", "error")
        return False

    # Run npm install
    command = "npm run init" if dev_mode else "npm run init:prod"

    print_status("Running npm install...", "progress")
    print()

    success = run_command(command, cwd=tb_root)

    print()
    if success:
        print_status("NPM dependencies installed successfully", "success")
    else:
        print_status("Failed to install NPM dependencies", "error")

    return success


def build_web_distribution() -> bool:
    """Build web distribution"""
    print_box_header("Building Web Distribution", "🌐")

    tb_root = get_tb_root()
    print_box_content(f"Location: {tb_root}", "info")
    print_box_footer()

    print_status("Building web assets...", "build")
    print()

    success = run_command("npm run build:web", cwd=tb_root)

    print()
    if success:
        print_status("Web distribution built successfully", "success")
    else:
        print_status("Failed to build web distribution", "error")

    return success


def build_tauri_app() -> bool:
    """Build Tauri desktop application"""
    print_box_header("Building Desktop Application", "🖥️")

    tb_root = get_tb_root()
    print_box_content(f"Location: {tb_root}", "info")
    print_box_content("Platform: " + platform.system(), "info")
    print_box_footer()

    print_status("Building Tauri application...", "build")
    print_status("This may take several minutes", "info")
    print()

    # Build
    success = run_command("npm run tauriB", cwd=tb_root)

    if not success:
        print()
        print_status("Build failed", "error")
        return False

    print()
    print_status("Build completed", "success")

    # Locate and copy binary
    print_status("Locating build artifacts...", "progress")

    tauri_prefix = tb_root / "simple-core"
    target_dir = tauri_prefix / "src-tauri" / "target" / "release" / "bundle"

    # Platform-specific binary extensions
    system = platform.system()
    binary_patterns = {
        "Windows": [".exe"],
        "Darwin": [".app", ".dmg"],
        "Linux": ["", ".AppImage", ".deb"]
    }

    patterns = binary_patterns.get(system, [""])

    # Search for binary
    found_files = []
    if target_dir.exists():
        for root, _, files in os.walk(target_dir):
            for file in files:
                for pattern in patterns:
                    if file.endswith(pattern) and not file.startswith('.'):
                        found_files.append(Path(root) / file)

    if found_files:
        print()
        print_status("Found build artifacts:", "success")
        for file_path in found_files:
            print(f"  • {file_path.name}")

            # Copy to root
            try:
                dest = tb_root / file_path.name
                shutil.copy2(file_path, dest)
                print_status(f"Copied to: {dest}", "success")
            except Exception as e:
                print_status(f"Failed to copy {file_path.name}: {e}", "warning")
    else:
        print()
        print_status("No build artifacts found", "warning")
        print_status(f"Checked: {target_dir}", "info")

    return True


# =================== Download Functions ===================

def detect_os_and_arch() -> Tuple[str, str]:
    """Detect operating system and architecture"""
    current_os = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize architecture names
    arch_map = {
        'x86_64': 'x64',
        'amd64': 'x64',
        'aarch64': 'arm64',
        'arm64': 'arm64'
    }

    machine = arch_map.get(machine, machine)

    return current_os, machine


def query_executable_url(os_name: str, arch: str, binary_type: str) -> Tuple[str, str]:
    """Query URL for executable binary"""
    base_url = "https://downloads.example.com/toolboxv2"  # Replace with actual URL

    # Construct filename
    if os_name == "windows":
        extension = ".exe"
    else:
        extension = ""

    file_name = f"{binary_type}_{os_name}_{arch}{extension}"
    full_url = f"{base_url}/{file_name}"

    return full_url, file_name


def download_executable(url: str, file_name: str) -> bool:
    """Download executable from URL"""
    print_status(f"Downloading: {file_name}", "download")
    print_status(f"From: {url}", "info")

    try:
        import requests
    except ImportError:
        print_status("'requests' library required for download", "error")
        print_status("Install with: pip install requests", "info")
        return False

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(file_name, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Simple progress
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f'\r  Progress: {percent:.1f}%')
                        sys.stdout.flush()

        print()  # New line after progress

        # Make executable on Unix
        if platform.system().lower() != "windows":
            os.chmod(file_name, 0o755)

        print_status(f"Downloaded: {file_name}", "success")
        return True

    except requests.exceptions.RequestException as e:
        print()
        print_status(f"Download failed: {e}", "error")
        return False
    except Exception as e:
        print()
        print_status(f"Error: {e}", "error")
        return False


# =================== Build Pipeline ===================

def full_build_pipeline(dev_mode: bool = False) -> bool:
    """Execute full build pipeline"""
    print()
    print_separator("═")
    print("  🚀 BUILD PIPELINE")
    print_separator("═")
    print()

    # Phase 1: NPM Dependencies
    print_separator()
    print("  PHASE 1: Dependencies")
    print_separator()
    print()

    if not install_npm_dependencies(dev_mode):
        print_status("Build pipeline failed at dependency installation", "error")
        return False

    # Phase 2: Web Distribution
    print()
    print_separator()
    print("  PHASE 2: Web Distribution")
    print_separator()
    print()

    if not build_web_distribution():
        print_status("Build pipeline failed at web distribution", "error")
        return False

    # Phase 3: Desktop App (dev mode only)
    if dev_mode:
        print()
        print_separator()
        print("  PHASE 3: Desktop Application")
        print_separator()
        print()

        if not build_tauri_app():
            print_status("Desktop build failed", "warning")
            print_status("Continuing anyway...", "info")

    print()
    print_separator("═")
    print("  ✅ BUILD PIPELINE COMPLETE")
    print_separator("═")
    print()

    return True


# =================== User Mode Setup ===================

def setup_user_mode():
    """Setup for user mode (download pre-built binaries)"""
    print_box_header("User Mode Setup", "📥")
    print_box_content("Downloading pre-built binaries", "info")
    print_box_footer()

    current_os, machine = detect_os_and_arch()

    print_status(f"Detected OS: {current_os}", "info")
    print_status(f"Architecture: {machine}", "info")
    print()

    # Download server
    print_separator()
    print("  Downloading API Server")
    print_separator()
    print()

    server_url, server_file = query_executable_url(current_os, machine, "server")
    server_success = download_executable(server_url, server_file)

    # Download app
    print()
    print_separator()
    print("  Downloading Desktop App")
    print_separator()
    print()

    app_url, app_file = query_executable_url(current_os, machine, "app")
    app_success = download_executable(app_url, app_file)

    print()

    if server_success and app_success:
        print_status("All binaries downloaded successfully", "success")
        return True
    else:
        print_status("Some downloads failed", "warning")
        return False


# =================== Main Setup Function ===================

def setup_main():
    """Main setup function"""
    # Clear screen
    print('\033[2J\033[H')

    # Welcome
    print_box_header("ToolBoxV2 Setup Wizard", "🚀")
    print_box_content("Interactive installation and configuration", "info")
    print_box_footer()

    try:
        # Step 1: Mode selection
        mode = select_mode()

        # Step 2: Python selection
        python_choice = select_python_interpreter()

        if python_choice == "uv":
            print()
            print_status("UV detected - checking for Python environment", "warning")
            print_status("API requires native Python environment", "info")
            print()

            # Check for uv helper
            uv_helper = Path("uv_api_python_helper.py")
            if uv_helper.exists():
                print_status("Running UV API helper...", "progress")
                run_command(f"{sys.executable} {uv_helper}")
            else:
                print_status("UV helper not found, continuing...", "warning")

        # Step 3: Mode-specific setup
        if mode == "dev":
            # Dev mode: install tools and build
            print()
            install_dev_tools()

            print()
            if not full_build_pipeline(dev_mode=True):
                print_status("Build pipeline encountered errors", "warning")

        else:
            # User mode: download binaries
            print()
            if not setup_user_mode():
                print_status("Binary download encountered errors", "warning")

        # Final message
        print()
        print_box_header("Setup Complete!", "✅")
        print_box_content("ToolBoxV2 is ready to use", "success")
        print()
        print_box_content("Quick Start Commands:", "info")
        print_box_content("  tb login          # Login to ToolBoxV2", "")
        print_box_content("  tb --sm           # Manage as service (auto-start/restart)", "")
        print_box_content("  tb --ipy          # Interactive Python shell", "")
        print_box_content("  tb gui            # Launch GUI", "")
        print_box_content("  tb api start      # Start API server", "")
        print_box_content("  tb --help         # Show all commands", "")
        print_box_footer()

    except KeyboardInterrupt:
        print()
        print()
        print_box_header("Setup Cancelled", "⚠")
        print_box_content("Setup was interrupted by user", "warning")
        print_box_footer()
        sys.exit(0)

    except Exception as e:
        print()
        print_box_header("Setup Failed", "✗")
        print_box_content(f"An error occurred: {str(e)}", "error")
        print_box_footer()

        import traceback
        if ask_yes_no("Show detailed error?", default=False):
            print()
            print_separator()
            traceback.print_exc()
            print_separator()

        sys.exit(1)


# =================== Entry Point ===================

if __name__ == "__main__":
    setup_main()
