# file: toolboxv2/utils/tbx/install_support.py
"""
Complete TB Language Setup
- Build executable from Rust source
- Setup file associations (.tbx and .tb)
- Install VS Code extension
- Install PyCharm plugin
- Configure system PATH

Version: 1.0.1
Last Updated: 2025-11-10
"""

import subprocess
import sys
import shutil
import time
import zipfile
import json
from pathlib import Path
import platform


class TBSetup:
    """Complete TB Language setup manager"""

    def __init__(self):
        # Get toolboxv2 root directory
        self.root = Path(__file__).parent.parent.parent
        self.tbx_utils = Path(__file__).parent
        self.system = platform.system()
        self.tb_exc_dir = self.root / "tb-exc" / "src"

        # Verify critical paths
        if not self.tb_exc_dir.exists():
            print(f"⚠️  Warning: tb-exc directory not found at {self.tb_exc_dir}")

        if not (self.tbx_utils / "setup.py").exists():
            print(f"⚠️  Warning: setup.py not found at {self.tbx_utils / 'setup.py'}")

    def setup_all(self):
        """Run complete setup"""
        print("═" * 70)
        print("  TB Language - Complete Setup v1.0.1")
        print("═" * 70)
        print()
        print(f"  Root directory: {self.root}")
        print(f"  TB Compiler:    {self.tb_exc_dir}")
        print(f"  Platform:       {self.system}")
        print()

        success = True

        # Step 1: Build
        if not self.build_executable():
            print("❌ Build failed!")
            return False

        # Step 2: System integration
        if not self.setup_system_integration():
            print("⚠️  System integration failed (optional)")
            success = False

        # Step 3: VS Code extension
        if not self.setup_vscode():
            print("⚠️  VS Code extension setup failed (optional)")
            success = False

        # Step 4: PyCharm plugin
        if not self.setup_pycharm():
            print("⚠️  PyCharm plugin setup failed (optional)")
            success = False

        print()
        print("═" * 70)
        if success:
            print("  ✓ Setup Complete!")
        else:
            print("  ⚠️  Setup completed with warnings")
        print("═" * 70)
        print()
        print("Next steps:")
        print("  1. Restart PyCharm and VS Code (if open)")
        print("  2. Create a test file: test.tbx or test.tb")
        print("  3. Run it: tb run test.tbx")
        print("  4. Or compile it: tb compile test.tbx")
        print("  5. Or double-click test.tbx to run (JIT mode)")
        print("  6. Open .tbx/.tb files in PyCharm/VS Code for syntax highlighting")
        print()

        return success

    def build_executable(self):
        """Step 1: Build TB Language from Rust source"""
        print("Step 1/4: Building TB Language...")
        print("-" * 70)

        if not self.tb_exc_dir.exists():
            print(f"❌ TB compiler source not found at: {self.tb_exc_dir}")
            return False

        # Check if Cargo is available
        try:
            cargo_check = subprocess.run(
                ["cargo", "--version"],
                capture_output=True,
                text=True
            , encoding='utf-8')
            if cargo_check.returncode != 0:
                print("❌ Cargo not found. Please install Rust: https://rustup.rs/")
                return False
            print(f"   Using: {cargo_check.stdout.strip()}")
        except FileNotFoundError:
            print("❌ Cargo not found. Please install Rust: https://rustup.rs/")
            return False

        # Build in release mode
        print(f"   Building from: {self.tb_exc_dir}")
        print("   This may take a few minutes...")

        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=str(self.tb_exc_dir),
            capture_output=False
        , encoding='utf-8')

        if result.returncode != 0:
            print("❌ Build failed!")
            return False

        # Verify executable exists
        if self.system == "Windows":
            exe_path = self.tb_exc_dir / "target" / "release" / "tb.exe"
        else:
            exe_path = self.tb_exc_dir / "target" / "release" / "tb"

        if not exe_path.exists():
            print(f"❌ Executable not found at: {exe_path}")
            return False

        print(f"   ✓ Executable built: {exe_path}")
        print("   ✓ Build successful")
        print()
        return True

    def setup_system_integration(self):
        """Step 2: System integration (file associations)"""
        print("Step 2/4: Setting up system integration...")
        print("-" * 70)

        setup_script = self.tbx_utils / "setup.py"

        if not setup_script.exists():
            print(f"❌ Setup script not found at: {setup_script}")
            print()
            return False

        result = subprocess.run([
            sys.executable,
            str(setup_script),
            "install"
        ], encoding='utf-8')

        print()
        if result.returncode == 0:
            print("   ✓ System integration complete")
        return result.returncode == 0

    def setup_vscode(self):
        """Step 3: VS Code extension"""
        print("Step 3/4: Installing VS Code extension...")
        print("-" * 70)

        # Correct path: utils/tbx/tb-lang-support
        vscode_ext = self.tbx_utils / "tb-lang-support"
        if not vscode_ext.exists():
            print(f"⚠️  VS Code extension directory not found at: {vscode_ext}")
            print()
            return False

        print(f"   Extension directory: {vscode_ext}")

        try:
            # Check if npm is available
            subprocess.run(["npm", "--version"],
                           capture_output=True, check=True, encoding='utf-8')

            # Install dependencies
            print("  Installing npm dependencies...")
            subprocess.run(["npm", "install"],
                           cwd=vscode_ext,
                           capture_output=True,
                           check=True, encoding='utf-8')

            # Compile TypeScript
            print("  Compiling TypeScript...")
            subprocess.run(["npm", "run", "compile"],
                           cwd=vscode_ext,
                           capture_output=True,
                           check=True, encoding='utf-8')

            # Try to install to VS Code
            print("  Installing to VS Code...")
            result = subprocess.run([
                "code", "--install-extension", str(vscode_ext.resolve())
            ], capture_output=True, encoding='utf-8')

            if result.returncode == 0:
                print("✓ VS Code extension installed")
                print()
                return True
            else:
                print("⚠️  Could not auto-install to VS Code")
                print(f"   Manual install: code --install-extension {vscode_ext.resolve()}")
                print()
                return False

        except FileNotFoundError as e:
            print(f"⚠️  Tool not found: {e}")
            print("   npm: https://nodejs.org/")
            print("   VS Code: https://code.visualstudio.com/")
            print()
            return False
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Command failed: {e}")
            print()
            return False

    def setup_pycharm(self):
        """Step 4: PyCharm plugin"""
        print("Step 4/4: Installing PyCharm plugin...")
        print("-" * 70)

        # Correct path: utils/tbx/tb-lang-pycharm
        pycharm_plugin = self.tbx_utils / "tb-lang-pycharm"
        if not pycharm_plugin.exists():
            print(f"⚠️  PyCharm plugin directory not found at: {pycharm_plugin}")
            print()
            return False

        print(f"   Plugin directory: {pycharm_plugin}")

        try:
            # Build plugin JAR
            print("  Building PyCharm plugin...")
            if not self.build_pycharm_plugin():
                print("⚠️  Plugin build failed")
                print()
                return False

            # Install to PyCharm
            print("  Installing to PyCharm...")
            if not self.install_pycharm_plugin():
                print("⚠️  Auto-install failed")
                print()
                return False

            print("✓ PyCharm plugin installed")
            print("  Please restart PyCharm to activate the plugin")
            print()
            return True

        except Exception as e:
            print(f"⚠️  Error: {e}")
            print()
            return False

    def create_pycharm_plugin(self):
        """Create PyCharm plugin structure"""
        plugin_dir = self.root /"utils"/"tbx"/ "tb-lang-pycharm"
        plugin_dir.mkdir(exist_ok=True)

        # Create directory structure
        (plugin_dir / "src" / "main" / "resources" / "fileTypes").mkdir(parents=True, exist_ok=True)
        (plugin_dir / "src" / "main" / "resources" / "META-INF").mkdir(parents=True, exist_ok=True)

        return True

    def build_pycharm_plugin(self):
        """Build PyCharm plugin JAR"""
        plugin_dir = self.root /"utils"/"tbx"/ "tb-lang-pycharm"
        build_script = plugin_dir / "build_plugin.py"

        if not build_script.exists():
            # Create build script
            build_script.write_text('''#!/usr/bin/env python3
import zipfile
from pathlib import Path

plugin_dir = Path(__file__).parent
output_jar = plugin_dir / "tb-language.jar"

with zipfile.ZipFile(output_jar, 'w', zipfile.ZIP_DEFLATED) as jar:
    # Add plugin.xml
    plugin_xml = plugin_dir / "src" / "main" / "resources" / "META-INF" / "plugin.xml"
    if plugin_xml.exists():
        jar.write(plugin_xml, "META-INF/plugin.xml")

    # Add file type definition
    file_type = plugin_dir / "src" / "main" / "resources" / "fileTypes" / "TB.xml"
    if file_type.exists():
        jar.write(file_type, "fileTypes/TB.xml")

print(f"✓ Plugin built: {output_jar}")
''')
            build_script.chmod(0o755)

        # Run build script
        result = subprocess.run([sys.executable, str(build_script)],
                                capture_output=True, text=True, encoding='utf-8')

        if result.returncode == 0:
            print(f"  {result.stdout.strip()}")
            return True
        else:
            print(f"  Build error: {result.stderr}")
            return False

    def install_pycharm_plugin(self):
        """Install plugin to PyCharm"""
        time.sleep(2)
        plugin_jar = self.root  /"utils"/"tbx" / "tb-lang-pycharm" / "tb-language.jar"

        if not plugin_jar.exists():
            print(f"  Plugin JAR not found")
            return False

        # Find PyCharm config directory
        pycharm_dirs = self.find_pycharm_config_dirs()

        if not pycharm_dirs:
            print("  PyCharm installation not found")
            print(f"  Manual install: Copy {plugin_jar} to PyCharm plugins directory")
            return False

        # Install to all found PyCharm installations
        installed = False
        for config_dir in pycharm_dirs:
            plugins_dir = config_dir / "plugins"
            plugins_dir.mkdir(exist_ok=True)

            dest = plugins_dir / "tb-language.jar"
            shutil.copy(plugin_jar, dest)
            print(f"  ✓ Installed to: {dest}")
            installed = True

        return installed

    def find_pycharm_config_dirs(self):
        """Find PyCharm config directories"""
        config_dirs = []
        home = Path.home()

        if self.system == "Windows":
            # Windows: C:\Users\<user>\AppData\Roaming\JetBrains\PyCharm*
            base = home / "AppData" / "Roaming" / "JetBrains"
            if base.exists():
                config_dirs.extend(base.glob("PyCharm*"))

        elif self.system == "Linux":
            # Linux: ~/.config/JetBrains/PyCharm*
            base = home / ".config" / "JetBrains"
            if base.exists():
                config_dirs.extend(base.glob("PyCharm*"))

            # Also check old location
            old_base = home / ".PyCharm*"
            config_dirs.extend(home.glob(".PyCharm*"))

        elif self.system == "Darwin":
            # macOS: ~/Library/Application Support/JetBrains/PyCharm*
            base = home / "Library" / "Application Support" / "JetBrains"
            if base.exists():
                config_dirs.extend(base.glob("PyCharm*"))

        return [d for d in config_dirs if d.is_dir()]


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="TB Language Complete Setup"
    )
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip building the executable')
    parser.add_argument('--skip-system', action='store_true',
                        help='Skip system integration')
    parser.add_argument('--skip-vscode', action='store_true',
                        help='Skip VS Code extension')
    parser.add_argument('--skip-pycharm', action='store_true',
                        help='Skip PyCharm plugin')
    parser.add_argument('--pycharm-only', action='store_true',
                        help='Only setup PyCharm plugin')

    args = parser.parse_args()

    setup = TBSetup()

    if args.pycharm_only:
        success = setup.setup_pycharm()
    else:
        # Full setup with skip options
        success = True

        if not args.skip_build:
            success = setup.build_executable() and success

        if not args.skip_system:
            setup.setup_system_integration()

        if not args.skip_vscode:
            setup.setup_vscode()

        if not args.skip_pycharm:
            setup.setup_pycharm()

    sys.exit(0 if success else 1)

def function_runner(skip_build=False, skip_system=False, skip_vscode=False, skip_pycharm=False):

    sys.argv = [sys.argv[0], '--skip-build' if skip_build else '',
                '--skip-system' if skip_system else '',
                '--skip-vscode' if skip_vscode else '',
                '--skip-pycharm' if skip_pycharm else '']
    sys.argv = [x for x in sys.argv if x != '']
    main()


if __name__ == "__main__":
    main()
