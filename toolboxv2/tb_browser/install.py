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
from typing import Tuple, Optional

# =================== ToolBox Style Integration ===================

try:
    from toolboxv2.utils.extras.Style import Style, Spinner

    TOOLBOX_AVAILABLE = True
except ImportError:
    # Fallback wenn ToolBox nicht verfügbar
    TOOLBOX_AVAILABLE = False


    class Style:
        """Fallback Style-Klasse wenn ToolBox nicht verfügbar"""

        @staticmethod
        def RED(text): return f"\033[91m{text}\033[0m"

        @staticmethod
        def GREEN(text): return f"\033[92m{text}\033[0m"

        @staticmethod
        def YELLOW(text): return f"\033[93m{text}\033[0m"

        @staticmethod
        def BLUE(text): return f"\033[94m{text}\033[0m"

        @staticmethod
        def CYAN(text): return f"\033[96m{text}\033[0m"

        @staticmethod
        def VIOLET(text): return f"\033[95m{text}\033[0m"

        @staticmethod
        def WHITE(text): return f"\033[97m{text}\033[0m"

        @staticmethod
        def GREY(text): return f"\033[90m{text}\033[0m"

        @staticmethod
        def Bold(text): return f"\033[1m{text}\033[0m"

        @staticmethod
        def Underline(text): return f"\033[4m{text}\033[0m"

        @staticmethod
        def GREEN2(text): return f"\033[92m{text}\033[0m"

        @staticmethod
        def RED2(text): return f"\033[91m{text}\033[0m"

        @staticmethod
        def VIOLET2(text): return f"\033[95m{text}\033[0m"


    class Spinner:
        """Fallback Spinner wenn ToolBox nicht verfügbar"""

        def __init__(self, message, **kwargs):
            self.message = message

        def __enter__(self):
            print(f"⟳ {self.message}...", flush=True)
            return self

        def __exit__(self, *args):
            pass


# =================== Helper Functions ===================

def print_status(status: str, message: str):
    """Print colored status message."""
    icons = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'running': '⟳'
    }

    colors = {
        'success': Style.GREEN,
        'error': Style.RED,
        'warning': Style.YELLOW,
        'info': Style.BLUE,
        'running': Style.CYAN
    }

    icon = icons.get(status, '•')
    color_fn = colors.get(status, Style.WHITE)

    print(color_fn(f"{icon} {message}"))


def print_header(title: str):
    """Print section header with box."""
    width = 70
    print()
    print(Style.CYAN('╔' + '═' * width + '╗'))
    print(Style.CYAN('║') + Style.Bold(Style.WHITE(title.center(width))) + Style.CYAN('║'))
    print(Style.CYAN('╚' + '═' * width + '╝'))
    print()


def print_box(lines: list, color_fn=Style.WHITE):
    """Print content in a box."""
    if not lines:
        return

    width = max(len(line) for line in lines) + 4

    print(color_fn('┌' + '─' * width + '┐'))
    for line in lines:
        padding = width - len(line) - 2
        print(color_fn('│') + f"  {line}" + ' ' * padding + color_fn('│'))
    print(color_fn('└' + '─' * width + '┘'))


def detect_shell() -> Tuple[str, str]:
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


# =================== Main Installer Class ===================

class ToolBoxInstaller:
    def __init__(self):
        self.project_dir = Path(__file__).parent.absolute()
        self.build_dir = self.project_dir / "build"
        self.package_json = self.project_dir / "package.json"
        self.build_script = self.project_dir / "build.js"
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()

    def check_requirements(self) -> bool:
        """Check if Node.js and npm are installed"""
        print_header("Checking Requirements")

        try:
            shell_exec, shell_arg = detect_shell()

            # Check Node.js
            result = subprocess.run(
                [shell_exec, shell_arg, 'node --version'],
                capture_output=True,
                text=True,
                check=True
            )
            node_version = result.stdout.strip()
            print_status('success', f"Node.js found: {node_version}")

            # Check npm
            npm_result = subprocess.run(
                [shell_exec, shell_arg, 'npm --version'],
                capture_output=True,
                text=True,
                check=True
            )
            npm_version = npm_result.stdout.strip()
            print_status('success', f"npm found: v{npm_version}")

            return True

        except (subprocess.CalledProcessError, FileNotFoundError):
            print_status('error', "Node.js not found")
            print_status('info', "Please install Node.js from: https://nodejs.org/")
            print_status('info', "Minimum required version: 14.0.0")
            return False

    def detect_platform(self) -> str:
        """Detect operating system and architecture"""
        print_header("Detecting Platform")

        os_info = {
            'darwin': 'macOS',
            'linux': 'Linux',
            'windows': 'Windows'
        }

        os_name = os_info.get(self.system, self.system)
        print_status('success', f"Operating System: {os_name}")
        print_status('success', f"Architecture: {self.machine}")

        # Detect if mobile (simplified detection)
        is_mobile = 'arm' in self.machine or 'aarch64' in self.machine
        if is_mobile:
            print_status('info', "Mobile/ARM architecture detected")

        return os_name

    def run_build(self, mode: str = 'build') -> bool:
        """Run the Node.js build script"""
        print_header(f"Running {mode.upper()} Build")

        if not self.build_script.exists():
            print_status('error', f"Build script not found: {self.build_script}")
            return False

        try:
            shell_exec, shell_arg = detect_shell()
            cmd = [shell_exec, shell_arg, f'node {str(self.build_script)} {mode}']

            print_status('info', f"Executing: node {self.build_script} {mode}")

            with Spinner(f"Building extension ({mode} mode)", symbols="d" if TOOLBOX_AVAILABLE else None):
                result = subprocess.run(
                    cmd,
                    cwd=str(self.project_dir),
                    capture_output=False,
                    text=True,
                    check=True
                )

            print_status('success', f"Build completed successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print_status('error', f"Build failed with error code {e.returncode}")
            return False

    def create_package(self) -> Optional[Path]:
        """Create distributable package"""
        print_header("Creating Distribution Package")

        if not self.build_dir.exists():
            print_status('error', "Build directory not found. Please run build first.")
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
            with Spinner("Creating ZIP package", symbols="t" if TOOLBOX_AVAILABLE else None):
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(self.build_dir):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(self.build_dir)
                            zipf.write(file_path, arcname)

            print_status('success', f"Package created: {zip_name}")
            return zip_path

        except Exception as e:
            print_status('error', f"Failed to create package: {e}")
            return None

    def print_browser_instructions(self, browser: str):
        """Print installation instructions for specific browser"""
        instructions = {
            'chrome': {
                'name': 'Google Chrome / Chromium',
                'icon': '🌐',
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
                'icon': '🦊',
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
                'icon': '🌊',
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
                'icon': '🧭',
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
                'icon': '🎭',
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
                'icon': '🦁',
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
            print_status('warning', f"Instructions for {browser} not available")
            return

        info = instructions[browser]

        # Print browser header
        print(f"\n{info['icon']} {Style.Bold(Style.CYAN(info['name']))}")
        print(Style.GREY('─' * 70))

        # Desktop instructions
        if 'desktop' in info and self.system in ['darwin', 'linux', 'windows']:
            print(f"\n  {Style.Bold(Style.WHITE('Desktop Installation:'))}")
            for step in info['desktop']:
                if step.startswith(('1.', '2.', '3.', '4.', '5.')):
                    print(f"    {Style.CYAN(step)}")
                else:
                    print(f"    {Style.WHITE(step)}")

        # Mobile instructions
        if 'android' in info and self.system == 'linux':
            print(f"\n  {Style.Bold(Style.WHITE('Android Installation:'))}")
            for step in info['android']:
                if step.startswith(('1.', '2.', '3.', '4.')):
                    print(f"    {Style.CYAN(step)}")
                else:
                    print(f"    {Style.YELLOW(step)}")

        if 'ios' in info and self.system == 'darwin':
            print(f"\n  {Style.Bold(Style.WHITE('iOS Installation:'))}")
            for step in info['ios']:
                if step.startswith(('1.', '2.', '3.')):
                    print(f"    {Style.CYAN(step)}")
                else:
                    print(f"    {Style.YELLOW(step)}")

    def show_all_instructions(self):
        """Display installation instructions for all browsers"""
        print_header("Browser Installation Instructions")

        browsers = ['chrome', 'firefox', 'edge', 'safari', 'opera', 'brave']

        for browser in browsers:
            self.print_browser_instructions(browser)

    def interactive_menu(self):
        """Display interactive menu"""
        print(Style.CYAN("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          🧰 ToolBox Pro Extension Installer                          ║
║                                                                      ║
║          Universal Browser Extension Builder                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """))

        while True:
            print(
                f"\n{Style.Bold(Style.WHITE('┌─ 🎯 MAIN MENU ───────────────────────────────────────────────────────┐'))}")
            print(Style.WHITE('│') + ' ' * 70 + Style.WHITE('│'))
            print(Style.WHITE(
                '│') + f"  {Style.CYAN('1.')} {Style.WHITE('🔨 Build Extension (Development)')}" + ' ' * 33 + Style.WHITE('│'))
            print(Style.WHITE(
                '│') + f"  {Style.CYAN('2.')} {Style.WHITE('📦 Build Extension (Production)')}" + ' ' * 34 + Style.WHITE('│'))
            print(Style.WHITE(
                '│') + f"  {Style.CYAN('3.')} {Style.WHITE('🗜️  Build + Create ZIP Package')}" + ' ' * 35 + Style.WHITE('│'))
            print(Style.WHITE(
                '│') + f"  {Style.CYAN('4.')} {Style.WHITE('📖 Show Installation Instructions')}" + ' ' * 32 + Style.WHITE('│'))
            print(Style.WHITE(
                '│') + f"  {Style.CYAN('5.')} {Style.WHITE('🌐 Open Extension Folder')}" + ' ' * 41 + Style.WHITE('│'))
            print(Style.WHITE('│') + f"  {Style.CYAN('0.')} {Style.WHITE('❌ Exit')}" + ' ' * 58 + Style.WHITE('│'))
            print(Style.WHITE('│') + ' ' * 70 + Style.WHITE('│'))
            print(Style.Bold(Style.WHITE('└──────────────────────────────────────────────────────────────────────┘')))

            choice = input(f"\n{Style.CYAN('❯')} {Style.WHITE('Enter your choice (0-5):')} ").strip()

            if choice == '1':
                if self.run_build('dev'):
                    print_status('success', "Development build ready!")
                    print_status('info', f"Location: {self.build_dir}")
                    self.show_all_instructions()
                input(f"\n{Style.GREY('Press Enter to continue...')}")

            elif choice == '2':
                if self.run_build('build'):
                    print_status('success', "Production build ready!")
                    print_status('info', f"Location: {self.build_dir}")
                input(f"\n{Style.GREY('Press Enter to continue...')}")

            elif choice == '3':
                if self.run_build('build'):
                    zip_path = self.create_package()
                    if zip_path:
                        print_status('success', f"Package ready: {zip_path}")
                        print_status('info', "You can now distribute this ZIP file")
                input(f"\n{Style.GREY('Press Enter to continue...')}")

            elif choice == '4':
                self.show_all_instructions()
                input(f"\n{Style.GREY('Press Enter to continue...')}")

            elif choice == '5':
                if self.build_dir.exists():
                    shell_exec, shell_arg = detect_shell()
                    try:
                        if self.system == 'darwin':
                            subprocess.run([shell_exec, shell_arg, f'open {str(self.build_dir)}'])
                        elif self.system == 'windows':
                            os.startfile(str(self.build_dir))
                        elif self.system == 'linux':
                            subprocess.run([shell_exec, shell_arg, f'xdg-open {str(self.build_dir)}'])
                        print_status('success', f"Opened: {self.build_dir}")
                    except Exception as e:
                        print_status('error', f"Failed to open folder: {e}")
                else:
                    print_status('warning', "Build folder doesn't exist yet. Build first!")
                input(f"\n{Style.GREY('Press Enter to continue...')}")

            elif choice == '0':
                print(f"\n{Style.GREEN('👋 Thanks for using ToolBox Pro!')}")
                sys.exit(0)

            else:
                print_status('warning', "Invalid choice. Please select 0-5.")

    def quick_install(self) -> bool:
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

        print()
        print(Style.GREEN('═' * 70))
        print_status('success', "Installation package is ready!")
        print_status('info', f"Build location: {self.build_dir}")
        print_status('info', "Follow the instructions above to load the extension in your browser")
        print(Style.GREEN('═' * 70))
        print()

        return True


# =================== Main Entry Point ===================

def main():
    """Main entry point"""
    installer = ToolBoxInstaller()

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command in ['--help', '-h', 'help']:
            print(Style.CYAN("""
╔══════════════════════════════════════════════════════════════════════╗
║          ToolBox Pro Extension Installer - Help                      ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
  tb browser [command]

Commands:
  (none)    - Interactive menu mode
  quick     - Quick automated installation
  build     - Build production version
  dev       - Build development version
  package   - Build and create ZIP package
  help      - Show this help message

Examples:
  tb browser           # Interactive mode
  tb browser quick     # Quick install
  tb browser build     # Production build
            """))
            return

        elif command == 'quick':
            if not installer.quick_install():
                sys.exit(1)
            return

        elif command == 'build':
            installer.check_requirements()
            installer.detect_platform()
            if installer.run_build('build'):
                print_status('success', "Build completed!")
            else:
                sys.exit(1)
            return

        elif command == 'dev':
            installer.check_requirements()
            installer.detect_platform()
            if installer.run_build('dev'):
                print_status('success', "Development build completed!")
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
        print(f"\n\n{Style.YELLOW('👋 Installation cancelled by user')}")
        sys.exit(0)
    except Exception as e:
        print_status('error', f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
