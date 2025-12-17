#!/usr/bin/env python3
"""
tauri_cli.py - Tauri Desktop App Build & Management CLI

Commands:
- build-worker: Build tb-worker sidecar with Nuitka (includes toolboxv2 package)
- build-app: Build Tauri app for current platform
- build-all: Build worker + app for all platforms
- dev: Start development server
- clean: Clean build artifacts
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .cli_printing import print_status, print_box_header, print_box_footer, c_print

# Platform detection
SYSTEM = platform.system().lower()
MACHINE = platform.machine().lower()
IS_WINDOWS = SYSTEM == "windows"
IS_MACOS = SYSTEM == "darwin"
IS_LINUX = SYSTEM == "linux"

# Nuitka target triples
TARGET_TRIPLES = {
    ("windows", "amd64"): "x86_64-pc-windows-msvc",
    ("windows", "x86_64"): "x86_64-pc-windows-msvc",
    ("darwin", "arm64"): "aarch64-apple-darwin",
    ("darwin", "x86_64"): "x86_64-apple-darwin",
    ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
}


def get_project_root() -> Path:
    """Get ToolBoxV2 project root."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "toolboxv2").exists():
            return parent
    return Path.cwd()


def get_target_triple() -> str:
    """Get current platform's target triple."""
    key = (SYSTEM, MACHINE)
    return TARGET_TRIPLES.get(key, f"{MACHINE}-unknown-{SYSTEM}")


def get_worker_binary_name(target: str) -> str:
    """Get worker binary name for target."""
    if "windows" in target:
        return f"tb-worker-{target}.exe"
    return f"tb-worker-{target}"


def ensure_pyinstaller() -> bool:
    """Ensure PyInstaller is installed."""
    try:
        subprocess.run([sys.executable, "-m", "PyInstaller", "--version"],
                       capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("Installing PyInstaller...", "install")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"],
                           check=True)
            return True
        except subprocess.CalledProcessError:
            try:
                subprocess.run(
                    ["uv", "pip", "install", "pyinstaller"], check=True
                )
                return True
            except subprocess.CalledProcessError:
                print_status("Failed to install PyInstaller", "error")
                return False


def build_worker(output_dir: Path, target: Optional[str] = None,
                 standalone: bool = True, onefile: bool = True) -> bool:
    """Build tb-worker sidecar with PyInstaller."""
    print_box_header("Building TB-Worker Sidecar", "🔨")

    if not ensure_pyinstaller():
        return False

    target = target or get_target_triple()
    project_root = get_project_root()
    worker_entry = project_root / "toolboxv2" / "utils" / "workers" / "tauri_integration.py"

    if not worker_entry.exists():
        print_status(f"Worker entry not found: {worker_entry}", "error")
        return False

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_name = get_worker_binary_name(target)

    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        f"--distpath={output_dir}",
        f"--workpath={output_dir / 'build'}",
        f"--specpath={output_dir}",
        f"--name={binary_name.replace('.exe', '')}",
        # Collect toolboxv2 packages
        "--collect-all=toolboxv2.utils.workers",
        "--collect-all=toolboxv2.utils.extras",
        "--collect-all=toolboxv2.utils.system",
        # Hidden imports
        "--hidden-import=toolboxv2",
        "--hidden-import=toolboxv2.utils",
        "--hidden-import=toolboxv2.utils.workers",
        "--hidden-import=toolboxv2.utils.extras",
        "--hidden-import=toolboxv2.utils.extras.db",
        "--hidden-import=toolboxv2.utils.system",
        # Exclude problematic/heavy modules
        "--exclude-module=tkinter",
        "--exclude-module=matplotlib",
        "--exclude-module=PIL",
        "--exclude-module=pytest",
        "--exclude-module=sphinx",
        "--exclude-module=numpy",
        "--exclude-module=pandas",
        "--exclude-module=torch",
        "--exclude-module=tensorflow",
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # Platform-specific options
    if IS_WINDOWS:
        cmd.append("--console")  # Keep console for worker logging
    elif IS_MACOS:
        cmd.append("--console")

    cmd.append(str(worker_entry.resolve()))

    print_status(f"Target: {target}", "info")
    print_status(f"Output: {output_dir / binary_name}", "info")
    c_print(f"  Command: pyinstaller {binary_name}...")

    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        if result.returncode != 0:
            print_status("PyInstaller build failed", "error")
            return False

        # Move to correct location for Tauri
        tauri_binaries = project_root / "toolboxv2" / "simple-core" / "src-tauri" / "binaries"
        tauri_binaries.mkdir(parents=True, exist_ok=True)

        # Find built binary
        built = list(output_dir.glob(f"**/{binary_name.replace('.exe', '')}*"))
        if IS_WINDOWS:
            built = [b for b in built if b.suffix == ".exe"] or built
        else:
            built = [b for b in built if b.is_file() and not b.suffix]

        if built:
            dest = tauri_binaries / binary_name
            shutil.copy2(built[0], dest)
            print_status(f"Copied to: {dest}", "success")
        else:
            print_status("Built binary not found!", "warning")

        print_status("Worker build complete!", "success")
        return True
    except Exception as e:
        print_status(f"Build error: {e}", "error")
        return False


def build_frontend(project_root: Path) -> bool:
    """Build frontend with webpack."""
    print_box_header("Building Frontend", "📦")

    web_dir = project_root / "toolboxv2" / "web"
    if not (web_dir / "package.json").exists():
        print_status("No package.json in web directory", "warning")
        return True

    try:
        # Install dependencies
        print_status("Installing npm dependencies...", "install")
        subprocess.run(["npm", "install"], cwd=web_dir, check=True, shell=IS_WINDOWS)

        # Build
        print_status("Running webpack build...", "progress")
        subprocess.run(["npm", "run", "build"], cwd=project_root / "toolboxv2",
                       check=True, shell=IS_WINDOWS)

        print_status("Frontend build complete!", "success")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Frontend build failed: {e}", "error")
        return False
    except FileNotFoundError:
        print_status("npm not found - please install Node.js", "error")
        return False


def build_tauri_app(project_root: Path, target: Optional[str] = None,
                    debug: bool = False) -> bool:
    """Build Tauri desktop app."""
    print_box_header("Building Tauri App", "🚀")

    simple_core = project_root / "toolboxv2" / "simple-core"
    if not (simple_core / "src-tauri" / "Cargo.toml").exists():
        print_status("Tauri project not found", "error")
        return False

    cmd = ["npx", "tauri", "build"]
    if debug:
        cmd.append("--debug")
    if target:
        cmd.extend(["--target", target])

    try:
        print_status(f"Building for: {target or 'current platform'}", "info")
        subprocess.run(cmd, cwd=simple_core, check=True, shell=IS_WINDOWS)
        print_status("Tauri app build complete!", "success")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Tauri build failed: {e}", "error")
        return False
    except FileNotFoundError:
        print_status("npx/tauri not found - run 'npm install' in simple-core", "error")
        return False


def run_worker_debug(project_root: Path, http_port: int = 5000, ws_port: int = 5001) -> subprocess.Popen:
    """Start worker in debug mode (directly, without PyInstaller build)."""
    print_status(f"Starting worker debug mode (HTTP:{http_port}, WS:{ws_port})...", "launch")

    worker_entry = project_root / "toolboxv2" / "utils" / "workers" / "tauri_integration.py"

    env = os.environ.copy()
    env["TB_HTTP_PORT"] = str(http_port)
    env["TB_WS_PORT"] = str(ws_port)
    env["TB_DEBUG"] = "1"
    env["TOOLBOX_LOGGING_LEVEL"] = "DEBUG"
    env["PYTHONPATH"] = str(project_root)

    return subprocess.Popen(
        [sys.executable, str(worker_entry)],
        cwd=project_root,
        env=env,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.STDOUT,
    )


def run_dev_server(project_root: Path, no_worker: bool = False,
                   worker_only: bool = False,
                   http_port: int = 5000, ws_port: int = 5001) -> None:
    """Start Tauri development server with debug options.

    Tauri always uses the pre-built dist folder for UI.
    Worker provides the API (HTTP:5000, WS:5001).
    """
    print_box_header("Starting Development Server", "🔧")

    simple_core = project_root / "toolboxv2" / "simple-core"
    worker_proc = None

    # Check dist folder exists
    dist_folder = project_root / "toolboxv2" / "dist"
    if not dist_folder.exists() or not (dist_folder / "index.html").exists():
        print_status("Warning: dist folder not found or empty!", "warning")
        print_status("Run 'npm run build' in toolboxv2/ first", "info")

    try:
        # Start worker in debug mode if requested
        if not no_worker:
            worker_proc = run_worker_debug(project_root, http_port, ws_port)
            print_status(f"Worker started (PID: {worker_proc.pid})", "success")
            print_status(f"  HTTP API: http://localhost:{http_port}", "info")
            print_status(f"  WebSocket: ws://localhost:{ws_port}", "info")

        if worker_only:
            print_status("Worker-only mode - press Ctrl+C to stop", "info")
            # Stream worker output
            if worker_proc:
                try:
                    for line in iter(worker_proc.stdout.readline, b''):
                        print(line.decode('utf-8', errors='replace'), end='')
                except KeyboardInterrupt:
                    pass
            return

        # Tauri dev always uses dist folder (no devUrl configured)
        cmd = ["npx", "tauri", "dev", "--no-dev-server"]

        print_status("Starting Tauri dev mode (using dist folder)...", "launch")
        subprocess.run(cmd, cwd=simple_core, shell=IS_WINDOWS)

    except KeyboardInterrupt:
        print_status("Dev server stopped", "info")
    except FileNotFoundError:
        print_status("npx/tauri not found", "error")
    finally:
        if worker_proc:
            print_status("Stopping worker...", "progress")
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker_proc.kill()
            print_status("Worker stopped", "success")


def clean_build(project_root: Path) -> None:
    """Clean build artifacts."""
    print_box_header("Cleaning Build Artifacts", "🧹")

    dirs_to_clean = [
        project_root / "toolboxv2" / "simple-core" / "src-tauri" / "target",
        project_root / "toolboxv2" / "simple-core" / "src-tauri" / "binaries",
        project_root / "nuitka-build",
        project_root / "build",
    ]

    for d in dirs_to_clean:
        if d.exists():
            print_status(f"Removing: {d}", "progress")
            shutil.rmtree(d, ignore_errors=True)

    print_status("Clean complete!", "success")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="tb gui",
        description="ToolBoxV2 Tauri Desktop App Build & Management CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # build-worker
    worker_parser = subparsers.add_parser("build-worker", help="Build tb-worker sidecar with Nuitka")
    worker_parser.add_argument("--target", help="Target triple (e.g., x86_64-pc-windows-msvc)")
    worker_parser.add_argument("--output", "-o", type=Path, default=Path("nuitka-build"),
                               help="Output directory")
    worker_parser.add_argument("--no-standalone", action="store_true", help="Don't create standalone")
    worker_parser.add_argument("--no-onefile", action="store_true", help="Don't create single file")

    # build-app
    app_parser = subparsers.add_parser("build-app", help="Build Tauri desktop app")
    app_parser.add_argument("--target", help="Rust target triple")
    app_parser.add_argument("--debug", action="store_true", help="Debug build")
    app_parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend build")
    app_parser.add_argument("--skip-worker", action="store_true", help="Skip worker build")

    # build-all
    all_parser = subparsers.add_parser("build-all", help="Build worker + app for all platforms")
    all_parser.add_argument("--platforms", nargs="+", default=["current"],
                            choices=["current", "windows", "macos", "linux", "all"],
                            help="Platforms to build for")

    # dev
    dev_parser = subparsers.add_parser("dev", help="Start development server")
    dev_parser.add_argument("--no-worker", action="store_true",
                            help="Don't start Python worker (use remote API)")
    dev_parser.add_argument("--worker-only", action="store_true",
                            help="Only start Python worker (no Tauri app)")
    dev_parser.add_argument("--http-port", type=int, default=5000,
                            help="HTTP worker port (default: 5000)")
    dev_parser.add_argument("--ws-port", type=int, default=5001,
                            help="WebSocket worker port (default: 5001)")

    # clean
    subparsers.add_parser("clean", help="Clean build artifacts")

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    project_root = get_project_root()
    print_status(f"Project root: {project_root}", "info")

    if args.command == "build-worker":
        success = build_worker(
            output_dir=args.output,
            target=args.target,
            standalone=not args.no_standalone,
            onefile=not args.no_onefile
        )
        sys.exit(0 if success else 1)

    elif args.command == "build-app":
        if not args.skip_worker:
            if not build_worker(Path("nuitka-build"), args.target):
                sys.exit(1)
        if not args.skip_frontend:
            if not build_frontend(project_root):
                sys.exit(1)
        success = build_tauri_app(project_root, args.target, args.debug)
        sys.exit(0 if success else 1)

    elif args.command == "build-all":
        platforms = args.platforms
        if "all" in platforms:
            platforms = ["windows", "macos", "linux"]
        elif "current" in platforms:
            platforms = [SYSTEM]

        for plat in platforms:
            print_box_header(f"Building for {plat}", "🎯")
            # Map platform to targets
            if plat == "windows":
                targets = ["x86_64-pc-windows-msvc"]
            elif plat == "macos":
                targets = ["aarch64-apple-darwin", "x86_64-apple-darwin"]
            elif plat == "linux":
                targets = ["x86_64-unknown-linux-gnu"]
            else:
                targets = [get_target_triple()]

            for target in targets:
                build_worker(Path("nuitka-build"), target)

        build_frontend(project_root)
        build_tauri_app(project_root)

    elif args.command == "dev":
        run_dev_server(
            project_root,
            no_worker=args.no_worker,
            worker_only=args.worker_only,
            http_port=args.http_port,
            ws_port=args.ws_port
        )

    elif args.command == "clean":
        clean_build(project_root)

    print_box_footer()


if __name__ == "__main__":
    main()
