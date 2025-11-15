# file: toolbox-exec/tb_lang_cli.py
# Production-Ready Manager for TB Language

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from toolboxv2.utils.tbx.install_support import function_runner as system_tbx_support
from toolboxv2.utils.tbx.setup import function_runner as language_ide_extension
from toolboxv2.utils.tbx.test.test_tb_lang2 import function_runner as test_tbx_examples

# --- Enhanced UI Imports ---
try:
    from toolboxv2.utils.extras.Style import Spinner, Style
except ImportError:
    try:
        from toolboxv2.extras.Style import Spinner, Style
    except ImportError:
        print("WARNING: UI utilities not found. Using fallback.")


        # Fallback implementations
        class Style:
            @staticmethod
            def RED(s): return f"\033[91m{s}\033[0m"

            @staticmethod
            def GREEN(s): return f"\033[92m{s}\033[0m"

            @staticmethod
            def YELLOW(s): return f"\033[93m{s}\033[0m"

            @staticmethod
            def BLUE(s): return f"\033[94m{s}\033[0m"

            @staticmethod
            def GREY(s): return f"\033[90m{s}\033[0m"


        class Spinner:
            def __init__(self, msg, **kwargs):
                self.msg = msg

            def __enter__(self):
                print(f"⟳ {self.msg}...")
                return self

            def __exit__(self, *args):
                pass

# --- CLI Printing Utilities ---
from toolboxv2.utils.clis.cli_printing import (
    print_box_header,
    print_box_content,
    print_box_footer,
    print_status,
    print_separator
)

# --- Configuration ---
EXECUTABLE_NAME = "tbx"
PROJECT_DIR = "tb-exc/src"

# =================== Helper Functions ===================

def get_tb_root() -> Path:
    """Get the toolbox root directory"""
    try:
        from toolboxv2 import tb_root_dir
        return tb_root_dir
    except ImportError:
        return Path(__file__).parent.parent.parent


def get_project_dir() -> Path:
    """Get the TB language project directory"""
    return get_tb_root() / PROJECT_DIR


def get_executable_path() -> Optional[Path]:
    """Find the compiled TB executable"""
    tb_root = get_tb_root()
    name_with_ext = f"{EXECUTABLE_NAME}.exe" if platform.system() == "Windows" else EXECUTABLE_NAME

    search_paths = [
        tb_root / "bin" / name_with_ext,
        get_project_dir() / "target" / "release" / name_with_ext,
        get_project_dir() / "target" / "debug" / name_with_ext,
    ]

    for path in search_paths:
        if path.is_file():
            return path.resolve()

    return None


def detect_shell():
    """Detect shell for running commands"""
    if platform.system() == "Windows":
        return "powershell", "-Command"
    else:
        return "sh", "-c"


def _build_native(project_dir: Path, release: bool, export_bin: bool) -> bool:
    """Build for the current native platform"""
    shell, shell_flag = detect_shell()

    build_cmd = "cargo build"
    if release:
        build_cmd += " --release"

    with Spinner(f"Compiling TB Language ({'release' if release else 'debug'} mode)", symbols='d'):
        result = subprocess.run(
            [shell, shell_flag, build_cmd],
            cwd=project_dir,
            capture_output=False,
            text=True,
            check=False,
            encoding=sys.stdout.encoding or 'utf-8'
        )

    if result.returncode != 0:
        print_status("Build failed!", "error")
        return False

    print_status("Build successful!", "success")

    # Export to bin directory
    if export_bin:
        return _export_to_bin(project_dir, release, "native")

    return True


def _build_desktop_target(project_dir: Path, release: bool, rust_target: str, export_bin: bool) -> bool:
    """Build for a specific desktop target"""
    shell, shell_flag = detect_shell()

    build_cmd = f"cargo build --target {rust_target}"
    if release:
        build_cmd += " --release"

    with Spinner(f"Compiling for {rust_target}", symbols='d'):
        result = subprocess.run(
            [shell, shell_flag, build_cmd],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
            encoding=sys.stdout.encoding or 'utf-8'
        )

    if result.returncode != 0:
        print_status(f"Build failed for {rust_target}!", "error")
        if result.stderr:
            print(Style.GREY(result.stderr))
        return False

    print_status(f"Build successful for {rust_target}!", "success")

    # Export to bin directory
    if export_bin:
        return _export_to_bin(project_dir, release, rust_target)

    return True


def _build_mobile_platform(project_dir: Path, release: bool, platform_name: str, export_bin: bool) -> bool:
    """Build for mobile platform using build scripts"""
    system = platform.system()

    # Determine which script to use
    if system == "Windows":
        script_path = project_dir / "build-mobile.ps1"
        script_cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path), platform_name]
    else:
        script_path = project_dir / "build-mobile.sh"
        script_cmd = ["bash", str(script_path), platform_name]

    # Add debug flag if needed
    if not release:
        if system == "Windows":
            script_cmd.insert(-1, "-Debug")
        else:
            script_cmd.append("--debug")

    # Check if script exists
    if not script_path.exists():
        print_status(f"Mobile build script not found: {script_path}", "error")
        print_status("Please ensure build-mobile scripts are in the project directory", "info")
        return False

    # Make script executable on Unix
    if system != "Windows":
        os.chmod(script_path, 0o755)

    with Spinner(f"Building for {platform_name} (using {script_path.name})", symbols='d'):
        result = subprocess.run(
            script_cmd,
            cwd=project_dir,
            capture_output=False,
            text=True,
            check=False,
            encoding=sys.stdout.encoding or 'utf-8'
        )

    if result.returncode != 0:
        print_status(f"Mobile build failed for {platform_name}!", "error")
        return False

    print_status(f"Mobile build successful for {platform_name}!", "success")

    # Export to bin directory
    if export_bin:
        return _export_mobile_to_bin(project_dir, release, platform_name)

    return True


def _build_all_platforms(project_dir: Path, release: bool, export_bin: bool) -> bool:
    """Build for all supported platforms"""
    print_status("Building for all platforms...", "info")
    print()

    success = True

    # Build native first
    print_box_header("Building Native", "🖥️")
    if not _build_native(project_dir, release, export_bin):
        success = False
    print()

    # Build Android
    print_box_header("Building Android", "📱")
    if not _build_mobile_platform(project_dir, release, "android", export_bin):
        success = False
    print()

    # Build iOS (only on macOS)
    if platform.system() == "Darwin":
        print_box_header("Building iOS", "🍎")
        if not _build_mobile_platform(project_dir, release, "ios", export_bin):
            success = False
        print()
    else:
        print_status("Skipping iOS build (requires macOS)", "warning")

    return success

def _export_to_bin(project_dir: Path, release: bool, rust_target: str) -> bool:
    """Export compiled binary to bin directory"""
    bin_dir = get_tb_root() / "bin"
    bin_dir.mkdir(exist_ok=True)

    build_type = "release" if release else "debug"

    # Determine source path based on target
    if rust_target == "native":
        target_dir = project_dir / "target" / build_type
    else:
        target_dir = project_dir / "target" / rust_target / build_type

    # Find executable
    exe_name = EXECUTABLE_NAME
    if rust_target == "native" and platform.system() == "Windows":
        exe_name += ".exe"

    source_path = target_dir / exe_name

    if not source_path.exists():
        print_status(f"Warning: Compiled executable not found at {source_path}", "warning")
        return False

    # Create target-specific subdirectory in bin
    if rust_target == "native":
        dest_dir = bin_dir
        dest_name = exe_name
    else:
        dest_dir = bin_dir / rust_target
        dest_dir.mkdir(exist_ok=True)
        dest_name = exe_name

    dest_path = dest_dir / dest_name

    # Copy executable
    if dest_path.exists():
        os.remove(dest_path)
    shutil.copy(source_path, dest_path)

    # Make executable on Unix
    if platform.system() != "Windows" and not dest_name.endswith(".exe"):
        os.chmod(dest_path, 0o755)

    print_status(f"✓ Exported to: {dest_path}", "success")
    return True


def _export_mobile_to_bin(project_dir: Path, release: bool, platform_name: str) -> bool:
    """Export mobile libraries to bin directory"""
    bin_dir = get_tb_root() / "bin" / platform_name
    bin_dir.mkdir(parents=True, exist_ok=True)

    build_type = "release" if release else "debug"
    target_base = project_dir / "target"

    # Define mobile targets
    if platform_name == "android":
        targets = {
            "aarch64-linux-android": "arm64-v8a",
            "armv7-linux-androideabi": "armeabi-v7a",
            "i686-linux-android": "x86",
            "x86_64-linux-android": "x86_64"
        }
        lib_name = "libtb_runtime.so"
    elif platform_name == "ios":
        targets = {
            "aarch64-apple-ios": "device",
            "x86_64-apple-ios": "simulator-intel",
            "aarch64-apple-ios-sim": "simulator-arm64"
        }
        lib_name = "libtb_runtime.a"
    else:
        print_status(f"Unknown mobile platform: {platform_name}", "error")
        return False

    exported_count = 0

    for rust_target, arch_name in targets.items():
        source_path = target_base / rust_target / build_type / lib_name

        if source_path.exists():
            dest_dir = bin_dir / arch_name
            dest_dir.mkdir(exist_ok=True)
            dest_path = dest_dir / lib_name

            shutil.copy(source_path, dest_path)
            print_status(f"✓ Exported {arch_name}: {dest_path}", "success")
            exported_count += 1
        else:
            print_status(f"⚠ Not found: {source_path}", "warning")

    if exported_count > 0:
        print_status(f"Exported {exported_count} {platform_name} libraries to {bin_dir}", "info")
        return True
    else:
        print_status(f"No {platform_name} libraries found to export", "warning")
        return False


def handle_system_support(args):
    """Handle system support operations"""
    return system_tbx_support(*args)

def handle_ide_extension(args):
    """Handle language IDE extension operations"""
    return language_ide_extension(args)

def handle_test_examples(args):
    """Handle TB language testing and examples"""
    return test_tbx_examples(args)
# =================== Command Handlers ===================

def handle_build(release: bool = True, target: str = "native", export_bin: bool = True):
    """
    Build the TB language executable for various targets

    Args:
        release: Build in release mode (default: True)
        target: Build target - native, windows, linux, macos, android, ios, all (default: native)
        export_bin: Export binaries to bin directory (default: True)
    """
    print_box_header("Building TB Language", "🔨")
    print_box_content(f"Mode: {'Release' if release else 'Debug'}", "info")
    print_box_content(f"Target: {target}", "info")
    print_box_footer()

    project_dir = get_project_dir()

    if not project_dir.exists():
        print_status(f"Project directory not found: {project_dir}", "error")
        return False

    # Define target mappings
    desktop_targets = {
        "windows": "x86_64-pc-windows-msvc",
        "linux": "x86_64-unknown-linux-gnu",
        "macos": "x86_64-apple-darwin",
        "macos-arm": "aarch64-apple-darwin",
    }

    mobile_targets = {
        "android": ["aarch64-linux-android", "armv7-linux-androideabi",
                    "i686-linux-android", "x86_64-linux-android"],
        "ios": ["aarch64-apple-ios", "x86_64-apple-ios", "aarch64-apple-ios-sim"],
    }

    try:
        # Handle different target types
        if target == "native":
            # Build for current platform
            return _build_native(project_dir, release, export_bin)

        elif target in ["windows", "linux", "macos", "macos-arm"]:
            # Build for specific desktop platform
            return _build_desktop_target(project_dir, release, desktop_targets[target], export_bin)

        elif target == "android":
            # Build for all Android targets using mobile script
            return _build_mobile_platform(project_dir, release, "android", export_bin)

        elif target == "ios":
            # Build for all iOS targets using mobile script
            return _build_mobile_platform(project_dir, release, "ios", export_bin)

        elif target == "all":
            # Build for all platforms
            return _build_all_platforms(project_dir, release, export_bin)

        else:
            print_status(f"Unknown target: {target}", "error")
            return False

    except FileNotFoundError:
        print_status("Build failed: 'cargo' command not found", "error")
        print_status("Is Rust installed and in your PATH?", "info")
        print_status("Install from: https://rustup.rs", "info")
        return False
    except Exception as e:
        print_status(f"Build failed: {e}", "error")
        return False


def handle_clean():
    """Clean build artifacts"""
    print_box_header("Cleaning Build Artifacts", "🧹")
    print_box_footer()

    project_dir = get_project_dir()

    try:
        shell, shell_flag = detect_shell()

        with Spinner("Running cargo clean", symbols='+'):
            subprocess.run(
                [shell, shell_flag, "cargo clean"],
                cwd=project_dir,
                capture_output=True,
                check=True
            )

        print_status("Clean successful!", "success")
        return True
    except Exception as e:
        print_status(f"Clean failed: {e}", "error")
        return False


def handle_run(file_path: str, mode: str = "jit", watch: bool = False):
    """Run a TB program"""
    exe_path = get_executable_path()

    if not exe_path:
        print_status("TB executable not found!", "error")
        print_status("Build it first with: tb x build", "info")
        return False

    if not Path(file_path).exists():
        print_status(f"File not found: {file_path}", "error")
        return False

    print_box_header(f"Running TB Program", "🚀")
    print_box_content(f"File: {file_path}", "info")
    print_box_content(f"Mode: {mode}", "info")
    print_box_footer()

    try:
        if mode == "compiled":
            # Step 1: Compile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.exe' if os.name == 'nt' else '') as f:
                output_path = f.name

            try:
                print_status("Compiling...", "info")
                compile_start = time.perf_counter()

                compile_result = subprocess.run(
                    [str(exe_path), "compile", file_path, "--output", output_path],
                    capture_output=True, text=True, check=False,
                    encoding='utf-8', errors='replace'
                )

                compile_time = (time.perf_counter() - compile_start) * 1000

                if compile_result.returncode != 0:
                    print()
                    print_status(f"Compilation failed", "error")
                    if compile_result.stderr:
                        print(compile_result.stderr)
                    return False

                print_status(f"Compiled in {compile_time:.2f}ms", "success")

                # Step 2: Execute
                if os.name != 'nt':
                    os.chmod(output_path, 0o755)

                print_status("Executing...", "info")
                exec_start = time.perf_counter()

                exec_result = subprocess.run(
                    [output_path],
                    check=False
                )

                exec_time = (time.perf_counter() - exec_start) * 1000

                print()
                if exec_result.returncode == 0:
                    print_status(f"Execution completed successfully in {exec_time:.2f}ms", "success")
                    return True
                else:
                    print_status(f"Execution failed with code {exec_result.returncode}", "error")
                    return False

            finally:
                try:
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                except:
                    pass

        else:  # JIT mode
            cmd = [str(exe_path), "run", file_path, "--mode", mode]
            result = subprocess.run(cmd, check=False)

            if result.returncode == 0:
                print()
                print_status("Execution completed successfully", "success")
                return True
            else:
                print()
                print_status(f"Execution failed with code {result.returncode}", "error")
                return False

    except KeyboardInterrupt:
        print()
        print_status("Execution interrupted", "warning")
        return False
    except Exception as e:
        print_status(f"Failed to run: {e}", "error")
        return False

def handle_compile(input_file: str, output_file: str, target: str = "native"):
    """Compile a TB program"""
    exe_path = get_executable_path()

    if not exe_path:
        print_status("TB executable not found!", "error")
        return False

    if not Path(input_file).exists():
        print_status(f"Input file not found: {input_file}", "error")
        return False

    print_box_header("Compiling TB Program", "⚙️")
    print_box_content(f"Input: {input_file}", "info")
    print_box_content(f"Output: {output_file}", "info")
    print_box_content(f"Target: {target}", "info")
    print_box_footer()

    try:
        cmd = [str(exe_path), "compile", input_file, output_file, "--target", target]

        result = subprocess.run(cmd, check=True)

        print()
        print_status("Compilation successful!", "success")
        return True

    except subprocess.CalledProcessError:
        print()
        print_status("Compilation failed", "error")
        return False
    except Exception as e:
        print_status(f"Failed to compile: {e}", "error")
        return False


def handle_repl():
    """Start TB REPL"""
    exe_path = get_executable_path()

    if not exe_path:
        print_status("TB executable not found!", "error")
        return False

    try:
        subprocess.run([str(exe_path), "repl"])
        return True
    except KeyboardInterrupt:
        print()
        return True
    except Exception as e:
        print_status(f"Failed to start REPL: {e}", "error")
        return False


def handle_check(file_path: str):
    """Check a TB program without executing"""
    exe_path = get_executable_path()

    if not exe_path:
        print_status("TB executable not found!", "error")
        return False

    if not Path(file_path).exists():
        print_status(f"File not found: {file_path}", "error")
        return False

    try:
        result = subprocess.run([str(exe_path), "check", file_path], check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        print_status(f"Failed to check: {e}", "error")
        return False


def handle_init(project_name: str):
    """Initialize a new TB project"""
    print_box_header(f"Creating TB Project: {project_name}", "📦")
    print_box_footer()

    from toolboxv2 import tb_root_dir, init_cwd

    if init_cwd == tb_root_dir:
        print_status("Cannot create project in TB root directory", "error")
        return False

    project_path = init_cwd / project_name

    if project_path.exists():
        print_status(f"Directory already exists: {project_path}", "error")
        return False

    try:
        # Create directory structure
        project_path.mkdir()
        (project_path / "src").mkdir()
        (project_path / "examples").mkdir()

        # Create main.tb
        main_tb = project_path / "src" / "main.tb"
        main_tb.write_text('''#!tb
@config {
    mode: "jit"
    type_mode: "static"
    optimize: true
}

@shared {
    app_name: "''' + project_name + '''"
}

fn main() {
    echo "Hello from $app_name!"
}

main()
''')

        # Create README
        readme = project_path / "README.md"
        readme.write_text(f'''# {project_name}

A TB Language project.

## Running


```bash
tb run x src/main.tb
Building
bash
tb compile src/main.tb bin/{project_name}
''')
        print_status(f"✓ Created project structure", "success")
        print_status(f"✓ Created src/main.tb", "success")
        print_status(f"✓ Created README.md", "success")
        print()
        print_status(f"Get started with:", "info")
        print(f"  cd {project_name}")
        print(f"  tb run src/main.tb")

        return True

    except Exception as e:
        print_status(f"Failed to create project: {e}", "error")
        return False

def handle_examples():
    """Run example programs"""
    examples_dir = get_project_dir() / "examples"
    if not examples_dir.exists():
        print_status("Examples directory not found", "error")
        return False

    examples = list(examples_dir.glob("*.tb"))

    if not examples:
        print_status("No example files found", "warning")
        return False

    print_box_header("TB Language Examples", "📚")
    print()

    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example.name}")

    print()
    print_box_footer()

    try:
        choice = input("Select example (number) or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            return True

        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            print()
            return handle_run(str(examples[idx]), mode="jit")
        else:
            print_status("Invalid selection", "error")
            return False

    except ValueError:
        print_status("Invalid input", "error")
        return False
    except KeyboardInterrupt:
        print()
        return True

def handle_info():
    """Show system information"""
    print_box_header("TB Language System Information", "ℹ️")
    print()
    # TB Root
    tb_root = get_tb_root()
    print(f"  TB Root:     {tb_root}")

    # Project directory
    project_dir = get_project_dir()
    print(f"  Project Dir: {project_dir}")
    print(f"  Exists:      {project_dir.exists()}")

    # Executable
    exe_path = get_executable_path()
    if exe_path:
        print(f"  Executable:  {exe_path}")
        print(f"  Exists:      {exe_path.exists()}")
    else:
        print(f"  Executable:  Not found (build first)")

    # Rust toolchain
    print()
    print("  Rust Toolchain:")
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"    {result.stdout.strip()}")

        result = subprocess.run(
            ["cargo", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"    {result.stdout.strip()}")
    except FileNotFoundError:
        print(Style.RED("    Rust not found! Install from https://rustup.rs"))
    except subprocess.CalledProcessError:
        print(Style.RED("    Failed to get Rust version"))

    print()
    print_box_footer()

#=================== CLI Entry Point ===================

def cli_tbx_main():
    """Main entry point for TB Language CLI"""
    Copyparser = argparse.ArgumentParser(
        description="🚀 TB Language - Unified Multi-Language Programming Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='tb run',
        epilog="""
╔════════════════════════════════════════════════════════════════════════════╗
║                           Command Examples                                 ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Setup & Build:                                                            ║
║    $ tb run build                    # Build TB Language (native/release)  ║
║    $ tb run build --debug            # Build in debug mode                 ║
║    $ tb run build --target android   # Build for Android (all archs)       ║
║    $ tb run build --target ios       # Build for iOS (all archs)           ║
║    $ tb run build --target windows   # Cross-compile for Windows           ║
║    $ tb run build --target all       # Build for all platforms             ║
║    $ tb run clean                    # Clean build artifacts               ║
║                                                                            ║
║  Running Programs:                                                         ║
║    $ tb run x program.tb           # Run in JIT mode (default)             ║
║    $ tb run x program.tb --mode compiled                                   ║
║    $ tb run x program.tb --mode streaming                                  ║
║                                                                            ║
║  Compilation:                                                              ║
║    $ tb run compile input.tb output  # Compile to native                   ║
║    $ tb run compile app.tb app.wasm --target wasm                          ║
║                                                                            ║
║  Development:                                                              ║
║    $ tb run repl                     # Start interactive REPL              ║
║    $ tb run check program.tb         # Check syntax & types                ║
║    $ tb run examples                 # Browse and run examples             ║
║                                                                            ║
║  Project Management:                                                       ║
║    $ tb run init myproject           # Create new TB project               ║
║    $ tb run info                     # Show system information             ║
║                                                                            ║
║  Nested Tools:                                                             ║
║    $ tb run support [args]           # System support operations           ║
║    $ tb run ide [args]               # Language IDE extension tools        ║
║    $ tb run test [args]              # TB language testing and examples    ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
    )
    Copysubparsers = Copyparser.add_subparsers(dest="command", required=False)

    # Build command
    p_build = Copysubparsers.add_parser('build', help='Build TB Language executable')
    p_build.add_argument('--debug', action='store_true', help='Build in debug mode')
    p_build.add_argument('--target',
                        choices=['native', 'windows', 'linux', 'macos', 'macos-arm',
                                'android', 'ios', 'all'],
                        default='native',
                        help='Build target platform (default: native)')
    p_build.add_argument('--no-export', action='store_true',
                        help='Skip exporting to bin directory')

    # Clean command
    Copysubparsers.add_parser('clean', help='Clean build artifacts')

    # Run command
    p_run = Copysubparsers.add_parser('x', help='Run a TB program')
    p_run.add_argument('file', help='TB program file to run')
    p_run.add_argument('--mode', choices=['compiled', 'jit', 'streaming'],
                       default='jit', help='Execution mode')
    p_run.add_argument('--watch', action='store_true',
                       help='Watch for file changes and re-run')

    # Compile command
    p_compile = Copysubparsers.add_parser('compile', help='Compile TB program')
    p_compile.add_argument('input', help='Input TB file')
    p_compile.add_argument('output', help='Output file')
    p_compile.add_argument('--target', choices=['native', 'wasm', 'library'],
                           default='native', help='Compilation target')

    # REPL command
    Copysubparsers.add_parser('repl', help='Start interactive REPL')

    # Check command
    p_check = Copysubparsers.add_parser('check', help='Check syntax and types')
    p_check.add_argument('file', help='TB file to check')

    # Init command
    p_init = Copysubparsers.add_parser('init', help='Initialize new TB project')
    p_init.add_argument('name', help='Project name')

    # Examples command
    Copysubparsers.add_parser('examples', help='Browse and run examples')

    # Info command
    Copysubparsers.add_parser('info', help='Show system information')

    # System support command
    p_support = Copysubparsers.add_parser('support', help='System support operations')
    p_support.add_argument('support_args', nargs='*', help='Arguments for system support')

    # IDE extension command
    p_ide = Copysubparsers.add_parser('ide', help='Language IDE extension operations')
    p_ide.add_argument('ide_args', nargs='*', help='Arguments for IDE extension')

    # Test examples command
    p_test = Copysubparsers.add_parser('test', help='TB language testing and examples')
    p_test.add_argument('test_args', nargs='*', help='Arguments for testing')
    p_test.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    p_test.add_argument('--filter', help='Filter tests by name')
    p_test.add_argument('--failed', '-f', action='store_true', help='Run only failed tests')
    args = Copyparser.parse_args()

    # Execute command
    if args.command == 'build':
        success = handle_build(
            release=not args.debug,
            target=args.target,
            export_bin=not args.no_export
        )
    elif args.command == 'clean':
        success = handle_clean()
    elif args.command == 'x':
        success = handle_run(args.file, mode=args.mode, watch=args.watch)
    elif args.command == 'compile':
        success = handle_compile(args.input, args.output, target=args.target)
    elif args.command == 'repl':
        success = handle_repl()
    elif args.command == 'check':
        success = handle_check(args.file)
    elif args.command == 'init':
        success = handle_init(args.name)
    elif args.command == 'examples':
        success = handle_examples()
    elif args.command == 'info':
        handle_info()
        success = True
    elif args.command == 'support':
        success = handle_system_support(args.support_args)
    elif args.command == 'ide':
        success = handle_ide_extension(args.ide_args)
    elif args.command == 'test':
        success = handle_test_examples(args.test_args)
    else:
        # No command provided, show help
        Copyparser.print_help()
        success = True

    sys.exit(0 if success else 1)

#=================== Main ===================
if __name__ == "__main__":
    cli_tbx_main()

