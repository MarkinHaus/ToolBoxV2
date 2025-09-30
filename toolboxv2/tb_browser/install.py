#!/usr/bin/env python3
"""
ToolBox Pro Extension - Universal Installer
Supports all major browsers on Windows, macOS, Linux, Android, and iOS
"""

import os
import sys
import platform
import subprocess
import shutil
import json
from pathlib import Path
import zipfile
import webbrowser


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(text, color=Colors.END):
    """Print colored text to terminal"""
    print(f"{color}{text}{Colors.END}")


def print_header(text):
    """Print section header"""
    print_colored(f"\n{'=' * 60}", Colors.CYAN)
    print_colored(f"  {text}", Colors.BOLD + Colors.HEADER)
    print_colored(f"{'=' * 60}", Colors.CYAN)


def print_success(text):
    """Print success message"""
    print_colored(f"✅ {text}", Colors.GREEN)


def print_error(text):
    """Print error message"""
    print_colored(f"❌ {text}", Colors.FAIL)


def print_warning(text):
    """Print warning message"""
    print_colored(f"⚠️  {text}", Colors.WARNING)


def print_info(text):
    """Print info message"""
    print_colored(f"ℹ️  {text}", Colors.BLUE)

def detect_shell() -> tuple[str, str]:
    """
    Detects the best available shell and the argument to execute a command.
    Returns:
        A tuple of (shell_executable, command_argument).
        e.g., ('/bin/bash', '-c') or ('powershell.exe', '-Command')
    """
    if platform.system() == "Windows":
        if shell_path := shutil.which("pwsh"):
            return shell_path, "-Command"
        if shell_path := shutil.which("powershell"):
            return shell_path, "-Command"
        return "cmd.exe", "/c"

    shell_env = os.environ.get("SHELL")
    if shell_env and shutil.which(shell_env):
        return shell_env, "-c"

    for shell in ["bash", "zsh", "sh"]:
        if shell_path := shutil.which(shell):
            return shell_path, "-c"

    return "/bin/sh", "-c"

class ToolBoxInstaller:
    def __init__(self):
        self.project_dir = Path(__file__).parent.absolute()
        self.build_dir = self.project_dir / "build"
        self.package_json = self.project_dir / "package.json"
        self.build_script = self.project_dir / "build.js"
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()

    def check_requirements(self):
        """Check if Node.js is installed"""
        print_header("Checking Requirements")

        try:
            a,b =detect_shell()
            result = subprocess.run([a,b,'node', '--version'],
                                    capture_output=True,
                                    text=True,
                                    check=True)
            node_version = result.stdout.strip()
            print_success(f"Node.js found: {node_version}")

            # Check npm
            npm_result = subprocess.run([a,b,'npm', '--version'],
                                        capture_output=True,
                                        text=True,
                                        check=True)
            npm_version = npm_result.stdout.strip()
            print_success(f"npm found: v{npm_version}")
            return True

        except (subprocess.CalledProcessError, FileNotFoundError):
            print_info("Please install Node.js from: https://nodejs.org/")
            print_info("Minimum required version: 14.0.0")
            # import webbrowser
            # webbrowser.open("https://nodejs.org/")
            return False

    def detect_platform(self):
        """Detect operating system and architecture"""
        print_header("Detecting Platform")

        os_info = {
            'darwin': 'macOS',
            'linux': 'Linux',
            'windows': 'Windows'
        }

        os_name = os_info.get(self.system, self.system)
        print_success(f"Operating System: {os_name}")
        print_success(f"Architecture: {self.machine}")

        # Detect if mobile (simplified detection)
        is_mobile = 'arm' in self.machine or 'aarch64' in self.machine
        if is_mobile:
            print_info("Mobile/ARM architecture detected")

        return os_name

    def run_build(self, mode='build'):
        """Run the Node.js build script"""
        print_header(f"Running {mode.upper()} Build")

        if not self.build_script.exists():
            print_error(f"Build script not found: {self.build_script}")
            return False

        try:
            # Run npm build command

            a,b =detect_shell()
            cmd = [a,b,'node', str(self.build_script), mode]
            print_info(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(cmd,
                                    cwd=str(self.project_dir),
                                    capture_output=False,
                                    text=True,
                                    check=True)

            print_success(f"Build completed successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print_error(f"Build failed with error code {e.returncode}")
            return False

    def create_package(self):
        """Create distributable package"""
        print_header("Creating Distribution Package")

        if not self.build_dir.exists():
            print_error("Build directory not found. Please run build first.")
            return None

        # Read version from package.json
        try:
            with open(self.package_json, 'r') as f:
                package_data = json.load(f)
                version = package_data.get('version', '3.0.0')
        except:
            version = '3.0.0'

        # Create zip file
        zip_name = f"toolbox-pro-extension-v{version}.zip"
        zip_path = self.project_dir / zip_name

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.build_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(self.build_dir)
                        zipf.write(file_path, arcname)

            print_success(f"Package created: {zip_name}")
            return zip_path

        except Exception as e:
            print_error(f"Failed to create package: {e}")
            return None

    def print_browser_instructions(self, browser):
        """Print installation instructions for specific browser"""
        instructions = {
            'chrome': {
                'name': 'Google Chrome / Chromium',
                'desktop': [
                    "1. Open Chrome and navigate to: chrome://extensions/",
                    "2. Enable 'Developer mode' (toggle in top-right corner)",
                    "3. Click 'Load unpacked' button",
                    f"4. Select the folder: {self.build_dir}",
                    "5. The extension should now be installed and active!"
                ],
                'android': [
                    "Chrome on Android doesn't support extensions directly.",
                    "Alternative: Use Kiwi Browser from Google Play Store",
                    "1. Install Kiwi Browser",
                    "2. Open kiwi://extensions/",
                    "3. Enable Developer mode",
                    "4. Load the extension zip file"
                ]
            },
            'firefox': {
                'name': 'Mozilla Firefox',
                'desktop': [
                    "1. Open Firefox and navigate to: about:debugging#/runtime/this-firefox",
                    "2. Click 'Load Temporary Add-on'",
                    f"3. Navigate to: {self.build_dir}",
                    "4. Select the 'manifest.json' file",
                    "5. The extension is now installed (temporary until Firefox restart)"
                ],
                'android': [
                    "Firefox on Android supports limited extensions:",
                    "1. Install Firefox Browser from Google Play",
                    "2. Extensions must be published to AMO (addons.mozilla.org)",
                    "3. For development, use Firefox Nightly with custom collection"
                ]
            },
            'edge': {
                'name': 'Microsoft Edge',
                'desktop': [
                    "1. Open Edge and navigate to: edge://extensions/",
                    "2. Enable 'Developer mode' (toggle in bottom-left)",
                    "3. Click 'Load unpacked'",
                    f"4. Select the folder: {self.build_dir}",
                    "5. The extension should now be installed!"
                ]
            },
            'safari': {
                'name': 'Safari',
                'desktop': [
                    "Safari extensions require Xcode conversion:",
                    "1. Install Xcode from Mac App Store",
                    "2. Use Safari Web Extension Converter:",
                    "   xcrun safari-web-extension-converter <path-to-extension>",
                    "3. Open the generated Xcode project",
                    "4. Build and run the project",
                    "5. Enable extension in Safari Preferences > Extensions"
                ],
                'ios': [
                    "iOS Safari extensions must be converted and published:",
                    "1. Convert extension using Xcode (Mac required)",
                    "2. Test using iOS Simulator or TestFlight",
                    "3. Submit to App Store for distribution"
                ]
            },
            'opera': {
                'name': 'Opera / Opera GX',
                'desktop': [
                    "1. Open Opera and navigate to: opera://extensions/",
                    "2. Enable 'Developer mode'",
                    "3. Click 'Load unpacked'",
                    f"4. Select the folder: {self.build_dir}",
                    "5. Extension is now installed!"
                ]
            },
            'brave': {
                'name': 'Brave Browser',
                'desktop': [
                    "1. Open Brave and navigate to: brave://extensions/",
                    "2. Enable 'Developer mode' (toggle in top-right)",
                    "3. Click 'Load unpacked'",
                    f"4. Select the folder: {self.build_dir}",
                    "5. Extension is now active!"
                ]
            }
        }

        if browser not in instructions:
            print_warning(f"Instructions for {browser} not available")
            return

        info = instructions[browser]
        print_colored(f"\n📱 {info['name']}", Colors.BOLD + Colors.CYAN)

        # Desktop instructions
        if 'desktop' in info and self.system in ['darwin', 'linux', 'windows']:
            print_colored("\n  Desktop Installation:", Colors.BOLD)
            for step in info['desktop']:
                print(f"    {step}")

        # Mobile instructions
        if 'android' in info and self.system == 'linux':
            print_colored("\n  Android Installation:", Colors.BOLD)
            for step in info['android']:
                print(f"    {step}")

        if 'ios' in info and self.system == 'darwin':
            print_colored("\n  iOS Installation:", Colors.BOLD)
            for step in info['ios']:
                print(f"    {step}")

    def show_all_instructions(self):
        """Display installation instructions for all browsers"""
        print_header("Browser Installation Instructions")

        browsers = ['chrome', 'firefox', 'edge', 'safari', 'opera', 'brave']

        for browser in browsers:
            self.print_browser_instructions(browser)
            print()  # Add spacing

    def interactive_menu(self):
        """Display interactive menu"""
        print_colored("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          🧰 ToolBox Pro Extension Installer              ║
║                                                          ║
║          Universal Browser Extension Builder             ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
        """, Colors.CYAN + Colors.BOLD)

        while True:
            print_colored("\nWhat would you like to do?\n", Colors.BOLD)
            print("  1. 🔨 Build Extension (Development)")
            print("  2. 📦 Build Extension (Production)")
            print("  3. 🗜️  Build + Create ZIP Package")
            print("  4. 📖 Show Installation Instructions")
            print("  5. 🌐 Open Extension Folder")
            print("  6. ❌ Exit")

            choice = input(f"\n{Colors.CYAN}Enter your choice (1-6): {Colors.END}").strip()

            if choice == '1':
                if self.run_build('dev'):
                    print_success("Development build ready!")
                    print_info(f"Location: {self.build_dir}")
                    self.show_all_instructions()

            elif choice == '2':
                if self.run_build('build'):
                    print_success("Production build ready!")
                    print_info(f"Location: {self.build_dir}")

            elif choice == '3':
                if self.run_build('build'):
                    zip_path = self.create_package()
                    if zip_path:
                        print_success(f"Package ready: {zip_path}")
                        print_info("You can now distribute this ZIP file")

            elif choice == '4':
                self.show_all_instructions()

            elif choice == '5':
                if self.build_dir.exists():

                    a, b = detect_shell()
                    if self.system == 'darwin':
                        subprocess.run([a,b,'open', str(self.build_dir)])
                    elif self.system == 'windows':
                        os.startfile(str(self.build_dir))
                    elif self.system == 'linux':
                        subprocess.run([a,b,'xdg-open', str(self.build_dir)])
                    print_success(f"Opened: {self.build_dir}")
                else:
                    print_warning("Build folder doesn't exist yet. Build first!")

            elif choice == '6':
                print_colored("\n👋 Thanks for using ToolBox Pro!", Colors.GREEN)
                sys.exit(0)

            else:
                print_warning("Invalid choice. Please select 1-6.")

    def quick_install(self):
        """Quick automated installation"""
        print_header("Quick Install Mode")

        # Check requirements
        if not self.check_requirements():
            return False

        # Detect platform
        self.detect_platform()

        # Build extension
        if not self.run_build('build'):
            return False

        # Show instructions
        self.show_all_instructions()

        print_colored("\n" + "=" * 60, Colors.GREEN)
        print_success("Installation package is ready!")
        print_info(f"Build location: {self.build_dir}")
        print_info("Follow the instructions above to load the extension in your browser")
        print_colored("=" * 60 + "\n", Colors.GREEN)

        return True


def main():
    """Main entry point"""
    installer = ToolBoxInstaller()

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command in ['--help', '-h', 'help']:
            print_colored("""
ToolBox Pro Extension Installer

Usage:
  python install.py [command]

Commands:
  (none)    - Interactive menu mode
  quick     - Quick automated installation
  build     - Build production version
  dev       - Build development version
  package   - Build and create ZIP package
  help      - Show this help message

Examples:
  python install.py           # Interactive mode
  python install.py quick     # Quick install
  python install.py build     # Production build
            """, Colors.CYAN)
            return

        elif command == 'quick':
            if not installer.quick_install():
                sys.exit(1)
            return

        elif command == 'build':
            installer.check_requirements()
            installer.detect_platform()
            if installer.run_build('build'):
                print_success("Build completed!")
            else:
                sys.exit(1)
            return

        elif command == 'dev':
            installer.check_requirements()
            installer.detect_platform()
            if installer.run_build('dev'):
                print_success("Development build completed!")
            else:
                sys.exit(1)
            return

        elif command == 'package':
            installer.check_requirements()
            installer.detect_platform()
            if installer.run_build('build'):
                installer.create_package()
            else:
                sys.exit(1)
            return

    # Default: Interactive menu
    installer.check_requirements()
    installer.detect_platform()
    installer.interactive_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n\n👋 Installation cancelled by user", Colors.WARNING)
        sys.exit(0)
    except Exception as e:
        print_error(f"An error occurred: {e}")
        sys.exit(1)
