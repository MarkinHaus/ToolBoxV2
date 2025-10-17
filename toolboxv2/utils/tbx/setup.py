# file: toolbox-exec/tb_setup.py
"""
TB Language Setup Utility
- File association (.tbx files)
- Icon registration
- Desktop integration
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path


class TBxSetup:
    """Setup utility for TB Language file associations and icons"""

    def __init__(self):
        self.system = platform.system()
        self.tb_root = self.get_tb_root()
        self.icon_path = Path(os.getenv("FAVI", ".ico"))
        self.executable = self.get_executable()

    def get_tb_root(self) -> Path:
        """Get toolbox root directory"""
        try:
            from toolboxv2 import tb_root_dir
            return Path(tb_root_dir)
        except ImportError:
            return Path(__file__).parent.parent

    def get_executable(self) -> Path:
        """Get TB executable path"""
        if self.system == "Windows":
            exe = self.tb_root / "bin" / "tb.exe"
        else:
            exe = self.tb_root / "bin" / "tb"

        if not exe.exists():
            # Try target/release
            if self.system == "Windows":
                exe = self.tb_root / "tb-exc" / "target" / "release" / "tb.exe"
            else:
                exe = self.tb_root / "tb-exc" / "target" / "release" / "tb"

        return exe

    def setup_all(self):
        """Run complete setup"""
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║         TB Language - System Integration Setup                 ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        print()

        # Check prerequisites
        if not self.executable.exists():
            print("❌ TB executable not found!")
            print(f"   Expected at: {self.executable}")
            print("   Run 'tb x build' first!")
            return False

        print(f"✓ TB executable found: {self.executable}")
        print()

        # Setup icon
        if not self.setup_icon():
            print("⚠️  Icon setup failed (continuing anyway)")

        # Setup file association
        if self.system == "Windows":
            success = self.setup_windows()
        elif self.system == "Linux":
            success = self.setup_linux()
        elif self.system == "Darwin":
            success = self.setup_macos()
        else:
            print(f"❌ Unsupported system: {self.system}")
            return False

        if success:
            print()
            print("╔════════════════════════════════════════════════════════════════╗")
            print("║                    ✓ Setup Complete!                           ║")
            print("╠════════════════════════════════════════════════════════════════╣")
            print("║  .tbx files are now associated with TB Language                ║")
            print("║  Double-click any .tbx file to run it!                         ║")
            print("╚════════════════════════════════════════════════════════════════╝")

        return success

    def setup_icon(self) -> bool:
        """Setup icon file"""
        print("📦 Setting up icon...")

        icon_dir = self.tb_root / "resources"
        icon_dir.mkdir(exist_ok=True)

        # Check if icon exists
        if self.icon_path.exists():
            print(f"   ✓ Icon already exists: {self.icon_path}")
            return True

        # Create placeholder icon info
        print(f"   ⚠️  Icon not found at: {self.icon_path}")
        print(f"   📝 Creating placeholder...")

        # Try to create a simple icon reference
        # User needs to provide actual tb_icon.ico file
        placeholder = icon_dir / "README_ICON.txt"
        placeholder.write_text("""
TB Language Icon
================

Place your icon files here:
- tb_icon.ico   (Windows)
- tb_icon.png   (Linux)
- tb_icon.icns  (macOS)

Recommended size: 256x256 px

You can use the ToolBox V2 logo/icon.
        """)

        print(f"   ℹ️  Place icon file at: {self.icon_path}")
        return False

    def setup_windows(self) -> bool:
        """Setup file association on Windows"""
        print("🪟 Setting up Windows file association...")

        try:
            import winreg

            # Create .tbx extension key
            print("   Creating registry entries...")

            # HKEY_CURRENT_USER\Software\Classes\.tbx
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\.tbx") as key:
                winreg.SetValue(key, "", winreg.REG_SZ, "TBLanguageFile")
                print("   ✓ Registered .tbx extension")

            # HKEY_CURRENT_USER\Software\Classes\TBLanguageFile
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\TBLanguageFile") as key:
                winreg.SetValue(key, "", winreg.REG_SZ, "TB Language Program")

                # Set icon
                if self.icon_path.exists():
                    icon_key = winreg.CreateKey(key, "DefaultIcon")
                    winreg.SetValue(icon_key, "", winreg.REG_SZ, str(self.icon_path))
                    print(f"   ✓ Set icon: {self.icon_path}")

                # Set open command
                command_key = winreg.CreateKey(key, r"shell\open\command")
                cmd = f'"{self.executable}" run "%1"'
                winreg.SetValue(command_key, "", winreg.REG_SZ, cmd)
                print(f"   ✓ Set open command: {cmd}")

                # Add "Run in Terminal" context menu
                terminal_key = winreg.CreateKey(key, r"shell\run_terminal\command")
                terminal_cmd = f'cmd /k "{self.executable}" run "%1" && pause'
                winreg.SetValue(terminal_key, "", winreg.REG_SZ, terminal_cmd)
                winreg.SetValue(winreg.CreateKey(key, r"shell\run_terminal"), "", winreg.REG_SZ, "Run in Terminal")
                print(f"   ✓ Added 'Run in Terminal' context menu")

                # Add "Edit" context menu
                edit_key = winreg.CreateKey(key, r"shell\edit\command")
                winreg.SetValue(edit_key, "", winreg.REG_SZ, 'notepad "%1"')
                winreg.SetValue(winreg.CreateKey(key, r"shell\edit"), "", winreg.REG_SZ, "Edit")
                print(f"   ✓ Added 'Edit' context menu")

            # Refresh shell
            print("   Refreshing Explorer...")
            try:
                import ctypes
                ctypes.windll.shell32.SHChangeNotify(0x08000000, 0x0000, None, None)
            except:
                print("   ⚠️  Could not refresh Explorer (restart may be needed)")

            print("   ✓ Windows setup complete!")
            return True

        except ImportError:
            print("   ❌ winreg module not available")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def setup_linux(self) -> bool:
        """Setup file association on Linux"""
        print("🐧 Setting up Linux file association...")

        try:
            # Create .desktop file
            desktop_dir = Path.home() / ".local" / "share" / "applications"
            desktop_dir.mkdir(parents=True, exist_ok=True)

            desktop_file = desktop_dir / "tb-language.desktop"

            icon_path = self.icon_path.with_suffix('.png')
            if not icon_path.exists():
                icon_path = "text-x-script"  # Fallback icon

            desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=TB Language
Comment=Execute TB Language programs
Exec={self.executable} run %f
Icon={icon_path}
Terminal=false
MimeType=text/x-tb;application/x-tb;
Categories=Development;
"""

            desktop_file.write_text(desktop_content)
            desktop_file.chmod(0o755)
            print(f"   ✓ Created desktop entry: {desktop_file}")

            # Create MIME type
            mime_dir = Path.home() / ".local" / "share" / "mime" / "packages"
            mime_dir.mkdir(parents=True, exist_ok=True)

            mime_file = mime_dir / "tb-language.xml"
            mime_content = """<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
    <mime-type type="text/x-tb">
        <comment>TB Language Program</comment>
        <glob pattern="*.tbx"/>
        <sub-class-of type="text/plain"/>
    </mime-type>
</mime-info>
"""

            mime_file.write_text(mime_content)
            print(f"   ✓ Created MIME type: {mime_file}")

            # Update MIME database
            print("   Updating MIME database...")
            try:
                subprocess.run(["update-mime-database",
                                str(Path.home() / ".local" / "share" / "mime")],
                               check=True, capture_output=True)
                print("   ✓ MIME database updated")
            except:
                print("   ⚠️  Could not update MIME database automatically")
                print("   Run: update-mime-database ~/.local/share/mime")

            # Update desktop database
            print("   Updating desktop database...")
            try:
                subprocess.run(["update-desktop-database", str(desktop_dir)],
                               check=True, capture_output=True)
                print("   ✓ Desktop database updated")
            except:
                print("   ⚠️  Could not update desktop database automatically")

            # Set default application
            try:
                subprocess.run([
                    "xdg-mime", "default", "tb-language.desktop", "text/x-tb"
                ], check=True, capture_output=True)
                print("   ✓ Set as default application for .tbx files")
            except:
                print("   ⚠️  Could not set as default application")

            print("   ✓ Linux setup complete!")
            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def setup_macos(self) -> bool:
        """Setup file association on macOS"""
        print("🍎 Setting up macOS file association...")

        try:
            # Create Info.plist for file association
            app_dir = self.tb_root / "TB Language.app"
            contents_dir = app_dir / "Contents"
            macos_dir = contents_dir / "MacOS"
            resources_dir = contents_dir / "Resources"

            # Create directories
            macos_dir.mkdir(parents=True, exist_ok=True)
            resources_dir.mkdir(parents=True, exist_ok=True)

            # Copy executable
            app_executable = macos_dir / "tb"
            if not app_executable.exists():
                shutil.copy(self.executable, app_executable)
                app_executable.chmod(0o755)

            # Create launcher script
            launcher = macos_dir / "TB Language"
            launcher.write_text(f"""#!/bin/bash
if [ "$#" -gt 0 ]; then
    "{app_executable}" run "$@"
else
    "{app_executable}" repl
fi
""")
            launcher.chmod(0o755)

            # Create Info.plist
            plist_file = contents_dir / "Info.plist"
            plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>TB Language</string>
    <key>CFBundleIconFile</key>
    <string>tb_icon</string>
    <key>CFBundleIdentifier</key>
    <string>dev.tblang.tb</string>
    <key>CFBundleName</key>
    <string>TB Language</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>tbx</string>
            </array>
            <key>CFBundleTypeIconFile</key>
            <string>tb_icon</string>
            <key>CFBundleTypeName</key>
            <string>TB Language Program</string>
            <key>CFBundleTypeRole</key>
            <string>Editor</string>
            <key>LSHandlerRank</key>
            <string>Owner</string>
        </dict>
    </array>
</dict>
</plist>
"""
            plist_file.write_text(plist_content)
            print(f"   ✓ Created app bundle: {app_dir}")

            # Copy icon if exists
            icon_src = self.icon_path.with_suffix('.icns')
            if icon_src.exists():
                shutil.copy(icon_src, resources_dir / "tb_icon.icns")
                print(f"   ✓ Copied icon")

            # Register with Launch Services
            print("   Registering with Launch Services...")
            try:
                subprocess.run([
                    "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister",
                    "-f", str(app_dir)
                ], check=True, capture_output=True)
                print("   ✓ Registered with Launch Services")
            except:
                print("   ⚠️  Could not register automatically")
                print(f"   Run: open '{app_dir}'")

            print("   ✓ macOS setup complete!")
            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def uninstall(self):
        """Remove file associations"""
        print("🗑️  Uninstalling file associations...")

        if self.system == "Windows":
            try:
                import winreg
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\.tbx")
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\TBLanguageFile")
                print("   ✓ Windows registry cleaned")
            except:
                print("   ⚠️  Could not clean registry")

        elif self.system == "Linux":
            desktop_file = Path.home() / ".local" / "share" / "applications" / "tb-language.desktop"
            mime_file = Path.home() / ".local" / "share" / "mime" / "packages" / "tb-language.xml"

            if desktop_file.exists():
                desktop_file.unlink()
                print("   ✓ Removed desktop entry")

            if mime_file.exists():
                mime_file.unlink()
                print("   ✓ Removed MIME type")

        elif self.system == "Darwin":
            app_dir = self.tb_root / "TB Language.app"
            if app_dir.exists():
                shutil.rmtree(app_dir)
                print("   ✓ Removed app bundle")

        print("   ✓ Uninstall complete!")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="TB Language System Integration Setup"
    )
    parser.add_argument('action', choices=['install', 'uninstall'],
                        help='Action to perform')

    args = parser.parse_args()

    setup = TBxSetup()

    if args.action == 'install':
        success = setup.setup_all()
        sys.exit(0 if success else 1)
    elif args.action == 'uninstall':
        setup.uninstall()
        sys.exit(0)

def function_runner(action: str = 'install'):
    setup = TBxSetup()
    if action == 'install':
        success = setup.setup_all()
        return success
    elif action == 'uninstall':
        setup.uninstall()
        return True

if __name__ == "__main__":
    main()
