#!/usr/bin/env python3
"""
tauri_cli.py - Tauri Desktop App Build & Management CLI

Commands:
- run: Run the Tauri app (download if needed)
- download: Download pre-built Tauri app from GitHub/Registry
- build-worker: Build tb-worker sidecar with PyInstaller
- build-app: Build Tauri app for current platform
- build-all: Build worker + app for all platforms
- dev: Start development server
- clean: Clean build artifacts
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from .cli_printing import print_status, print_box_header, print_box_footer, c_print

# Platform detection
SYSTEM = platform.system().lower()
MACHINE = platform.machine().lower()
IS_WINDOWS = SYSTEM == "windows"
IS_MACOS = SYSTEM == "darwin"
IS_LINUX = SYSTEM == "linux"

# Target triples for different platforms
TARGET_TRIPLES = {
    ("windows", "amd64"): "x86_64-pc-windows-msvc",
    ("windows", "x86_64"): "x86_64-pc-windows-msvc",
    ("darwin", "arm64"): "aarch64-apple-darwin",
    ("darwin", "x86_64"): "x86_64-apple-darwin",
    ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
}

# App download configuration
GITHUB_REPO = "MarkinHaus/ToolBoxV2"
REGISTRY_URL = "https://registry.simplecore.app"
APP_NAME = "simple-core"

# Platform-specific executable names
EXECUTABLE_NAMES = {
    "windows": "simple-core.exe",
    "darwin": "simple-core.app",
    "linux": "simple-core",
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


def get_app_install_dir() -> Path:
    """Get the directory where the app should be installed."""
    if IS_WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif IS_MACOS:
        base = Path.home() / "Applications"
    else:
        base = Path.home() / ".local" / "share"
    return base / "ToolBoxV2" / "app"


def get_installed_app_path() -> Optional[Path]:
    """Get path to installed app executable if it exists."""
    install_dir = get_app_install_dir()
    exe_name = EXECUTABLE_NAMES.get(SYSTEM, "simple-core")

    if IS_WINDOWS:
        app_path = install_dir / exe_name
    elif IS_MACOS:
        app_path = install_dir / exe_name / "Contents" / "MacOS" / "simple-core"
    else:
        app_path = install_dir / exe_name

    if app_path.exists():
        return app_path
    return None


def get_installed_version() -> Optional[str]:
    """Get version of installed app."""
    install_dir = get_app_install_dir()
    version_file = install_dir / "version.json"
    if version_file.exists():
        try:
            with open(version_file) as f:
                data = json.load(f)
                return data.get("version")
        except (json.JSONDecodeError, IOError):
            pass
    return None


def fetch_latest_release_info() -> Optional[Dict[str, Any]]:
    """Fetch latest release info from GitHub."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ToolBoxV2-CLI"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print_status(f"Failed to fetch release info: {e}", "warning")
        return None


def fetch_registry_artifacts() -> Optional[Dict[str, Any]]:
    """Fetch artifact info from TB Registry."""
    url = f"{REGISTRY_URL}/api/v1/artifacts/{APP_NAME}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ToolBoxV2-CLI"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print_status(f"Registry not available: {e}", "warning")
        return None


def get_asset_for_platform(release_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find the correct asset for current platform."""
    assets = release_info.get("assets", [])
    target = get_target_triple()

    # Look for matching asset
    patterns = []
    if IS_WINDOWS:
        patterns = [f"{APP_NAME}_{target}.zip", f"{APP_NAME}_x64-setup.exe",
                    f"{APP_NAME}_{target}.msi", "simple-core_x64-setup.exe"]
    elif IS_MACOS:
        if "aarch64" in target:
            patterns = [f"{APP_NAME}_aarch64.dmg", f"{APP_NAME}_universal.dmg",
                        f"{APP_NAME}_{target}.dmg"]
        else:
            patterns = [f"{APP_NAME}_x64.dmg", f"{APP_NAME}_universal.dmg",
                        f"{APP_NAME}_{target}.dmg"]
    else:
        patterns = [f"{APP_NAME}_{target}.AppImage", f"{APP_NAME}_{target}.deb",
                    f"{APP_NAME}_{target}.tar.gz"]

    for pattern in patterns:
        for asset in assets:
            if pattern.lower() in asset.get("name", "").lower():
                return asset

    # Fallback: any matching platform
    platform_keywords = {"windows": ["windows", ".exe", ".msi"],
                         "darwin": ["darwin", "macos", ".dmg"],
                         "linux": ["linux", ".appimage", ".deb"]}

    for keyword in platform_keywords.get(SYSTEM, []):
        for asset in assets:
            if keyword in asset.get("name", "").lower():
                return asset

    return None


def download_file(url: str, dest: Path, show_progress: bool = True) -> bool:
    """Download a file with progress indication."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ToolBoxV2-CLI"})
        with urllib.request.urlopen(req, timeout=300) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if show_progress and total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\r  Downloading: {pct:.1f}% ({downloaded // 1024}KB / {total_size // 1024}KB)", end="")

            if show_progress:
                print()  # New line after progress
        return True
    except Exception as e:
        print_status(f"Download failed: {e}", "error")
        return False


def extract_and_install(archive_path: Path, install_dir: Path) -> bool:
    """Extract downloaded archive and install."""
    install_dir.mkdir(parents=True, exist_ok=True)

    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(install_dir)
        elif archive_path.suffix in (".gz", ".tar"):
            import tarfile
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(install_dir)
        elif archive_path.suffix == ".exe":
            # Windows installer - just copy
            shutil.copy2(archive_path, install_dir / archive_path.name)
        elif archive_path.suffix == ".dmg":
            # macOS - mount and copy
            print_status("Please open the .dmg file and drag the app to Applications", "info")
            subprocess.run(["open", str(archive_path)], check=True)
            return True
        elif archive_path.suffix == ".AppImage":
            # Linux AppImage - make executable and copy
            dest = install_dir / "simple-core"
            shutil.copy2(archive_path, dest)
            os.chmod(dest, 0o755)
        elif archive_path.suffix == ".deb":
            # Debian package - install with dpkg
            subprocess.run(["sudo", "dpkg", "-i", str(archive_path)], check=True)
        else:
            print_status(f"Unknown archive format: {archive_path.suffix}", "error")
            return False

        return True
    except Exception as e:
        print_status(f"Installation failed: {e}", "error")
        return False


def download_app(source: str = "auto", version: str = "latest",
                 force: bool = False) -> bool:
    """Download and install the Tauri app."""
    print_box_header("Downloading SimpleCore Desktop App", "📥")

    install_dir = get_app_install_dir()
    installed_version = get_installed_version()

    # Check if already installed
    if not force and installed_version:
        print_status(f"App already installed: v{installed_version}", "info")
        print_status("Use --force to reinstall", "info")
        return True

    # Try registry first, then GitHub
    release_info = None
    download_url = None
    asset_name = None

    if source in ("auto", "registry"):
        print_status("Checking TB Registry...", "progress")
        registry_info = fetch_registry_artifacts()
        if registry_info:
            # Get download URL from registry
            versions = registry_info.get("versions", [])
            if versions:
                target_version = versions[0] if version == "latest" else next(
                    (v for v in versions if v.get("version") == version), None
                )
                if target_version:
                    download_url = target_version.get("download_url")
                    asset_name = target_version.get("filename", "app.zip")
                    version = target_version.get("version", version)

    if not download_url and source in ("auto", "github"):
        print_status("Checking GitHub Releases...", "progress")
        release_info = fetch_latest_release_info()
        if release_info:
            asset = get_asset_for_platform(release_info)
            if asset:
                download_url = asset.get("browser_download_url")
                asset_name = asset.get("name", "app.zip")
                version = release_info.get("tag_name", version).lstrip("v")

    if not download_url:
        print_status("No download available for your platform", "error")
        print_status(f"Platform: {SYSTEM} ({MACHINE})", "info")
        print_status("Try building from source: tb gui build-app", "info")
        return False

    print_status(f"Found version: {version}", "success")
    print_status(f"Asset: {asset_name}", "info")

    # Download to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        download_path = tmp_path / asset_name

        print_status(f"Downloading from: {download_url[:60]}...", "progress")
        if not download_file(download_url, download_path):
            return False

        print_status("Installing...", "progress")
        if not extract_and_install(download_path, install_dir):
            return False

    # Save version info
    version_file = install_dir / "version.json"
    with open(version_file, "w") as f:
        json.dump({"version": version, "source": source}, f)

    print_status(f"Installed to: {install_dir}", "success")
    print_status(f"Version: {version}", "success")
    return True


def run_app(with_worker: bool = True, http_port: int = 5000,
            ws_port: int = 5001, no_ws: bool = False,
            download_if_missing: bool = True) -> None:
    """Run the Tauri app (download if needed)."""
    print_box_header("Starting SimpleCore Desktop App", "🚀")

    app_path = get_installed_app_path()

    # Check if app is installed
    if not app_path:
        if download_if_missing:
            print_status("App not installed, downloading...", "info")
            if not download_app():
                print_status("Failed to download app", "error")
                return
            app_path = get_installed_app_path()
        else:
            print_status("App not installed. Run: tb gui download", "error")
            return

    if not app_path:
        print_status("App installation failed", "error")
        return

    project_root = get_project_root()
    worker_proc = None

    try:
        # Start worker if requested
        if with_worker:
            print_status("Starting local worker...", "progress")
            worker_proc = run_worker_debug(project_root, http_port, ws_port, no_ws=no_ws)
            print_status(f"Worker started (PID: {worker_proc.pid})", "success")
            print_status(f"  HTTP API: http://localhost:{http_port}", "info")
            if not no_ws:
                print_status(f"  WebSocket: ws://localhost:{ws_port}", "info")

        # Start the app
        print_status(f"Launching: {app_path}", "launch")

        if IS_WINDOWS:
            subprocess.Popen([str(app_path)], creationflags=subprocess.DETACHED_PROCESS)
        elif IS_MACOS:
            subprocess.Popen(["open", "-a", str(app_path.parent.parent.parent)])
        else:
            subprocess.Popen([str(app_path)])

        print_status("App launched successfully!", "success")

        if with_worker:
            print_status("Worker running in background. Press Ctrl+C to stop.", "info")
            try:
                worker_proc.wait()
            except KeyboardInterrupt:
                pass

    except Exception as e:
        print_status(f"Failed to launch app: {e}", "error")
    finally:
        if worker_proc:
            print_status("Stopping worker...", "progress")
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker_proc.kill()
            print_status("Worker stopped", "success")


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


def run_worker_debug(project_root: Path, http_port: int = 5000, ws_port: int = 5001,
                     no_ws: bool = False, verbose: bool = True) -> subprocess.Popen:
    """Start worker in debug mode (directly, without PyInstaller build).

    The worker runs both HTTP and WS servers in a unified process.
    """
    ws_status = "disabled" if no_ws else f"WS:{ws_port}"
    print_status(f"Starting worker debug mode (HTTP:{http_port}, {ws_status})...", "launch")

    worker_entry = project_root / "toolboxv2" / "utils" / "workers" / "tauri_integration.py"

    # Build command with CLI arguments
    cmd = [
        sys.executable, str(worker_entry),
        "--http-port", str(http_port),
        "--ws-port", str(ws_port),
    ]
    if no_ws:
        cmd.append("--no-ws")
    if verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    return subprocess.Popen(
        cmd,
        cwd=project_root,
        env=env,
    )


def run_dev_server(project_root: Path, no_worker: bool = False,
                   worker_only: bool = False,
                   http_port: int = 5000, ws_port: int = 5001,
                   no_ws: bool = False) -> None:
    """Start Tauri development server with debug options.

    Tauri always uses the pre-built dist folder for UI.
    Worker provides the API (HTTP + WS in unified process).
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
            worker_proc = run_worker_debug(project_root, http_port, ws_port, no_ws=no_ws)
            print_status(f"Worker started (PID: {worker_proc.pid})", "success")
            print_status(f"  HTTP API: http://localhost:{http_port}", "info")
            if not no_ws:
                print_status(f"  WebSocket: ws://localhost:{ws_port}", "info")
            else:
                print_status("  WebSocket: disabled", "info")

        if worker_only:
            print_status("Worker-only mode - press Ctrl+C to stop", "info")
            # Just wait for the process
            if worker_proc:
                try:
                    worker_proc.wait()
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
        description="ToolBoxV2 Tauri Desktop App Build & Management CLI",
        epilog="""
Examples:
  tb gui                    # Run app (download if needed)
  tb gui run                # Same as above
  tb gui run --no-worker    # Run app without local worker (use remote API)
  tb gui download           # Download pre-built app
  tb gui download --force   # Force re-download
  tb gui dev                # Start development server
  tb gui build-app          # Build from source
  tb gui status             # Show installation status
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run (default command)
    run_parser = subparsers.add_parser("run", help="Run the desktop app (download if needed)")
    run_parser.add_argument("--no-worker", action="store_true",
                            help="Don't start local worker (use remote API)")
    run_parser.add_argument("--http-port", type=int, default=5000,
                            help="HTTP worker port (default: 5000)")
    run_parser.add_argument("--ws-port", type=int, default=5001,
                            help="WebSocket worker port (default: 5001)")
    run_parser.add_argument("--no-ws", action="store_true",
                            help="Disable WebSocket server")
    run_parser.add_argument("--no-download", action="store_true",
                            help="Don't download if not installed")

    # download
    download_parser = subparsers.add_parser("download", help="Download pre-built app")
    download_parser.add_argument("--source", choices=["auto", "github", "registry"],
                                 default="auto", help="Download source")
    download_parser.add_argument("--version", default="latest",
                                 help="Version to download (default: latest)")
    download_parser.add_argument("--force", action="store_true",
                                 help="Force re-download even if installed")

    # status
    subparsers.add_parser("status", help="Show installation status")

    # build-worker
    worker_parser = subparsers.add_parser("build-worker", help="Build tb-worker sidecar with PyInstaller")
    worker_parser.add_argument("--target", help="Target triple (e.g., x86_64-pc-windows-msvc)")
    worker_parser.add_argument("--output", "-o", type=Path, default=Path("nuitka-build"),
                               help="Output directory")
    worker_parser.add_argument("--no-standalone", action="store_true", help="Don't create standalone")
    worker_parser.add_argument("--no-onefile", action="store_true", help="Don't create single file")

    # build-app
    app_parser = subparsers.add_parser("build-app", help="Build Tauri desktop app from source")
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
    dev_parser.add_argument("--no-ws", action="store_true",
                            help="Disable WebSocket server (HTTP only)")

    # clean
    subparsers.add_parser("clean", help="Clean build artifacts")

    # uninstall
    subparsers.add_parser("uninstall", help="Uninstall the desktop app")

    return parser


def show_status() -> None:
    """Show installation status."""
    print_box_header("SimpleCore Desktop App Status", "📊")

    app_path = get_installed_app_path()
    version = get_installed_version()
    install_dir = get_app_install_dir()

    print_status(f"Platform: {SYSTEM} ({MACHINE})", "info")
    print_status(f"Target: {get_target_triple()}", "info")
    print()

    if app_path:
        print_status(f"Status: Installed ✓", "success")
        print_status(f"Version: {version or 'unknown'}", "info")
        print_status(f"Location: {install_dir}", "info")
        print_status(f"Executable: {app_path}", "info")
    else:
        print_status("Status: Not installed", "warning")
        print_status(f"Install location: {install_dir}", "info")
        print()
        print_status("To install: tb gui download", "info")
        print_status("Or build from source: tb gui build-app", "info")


def uninstall_app() -> bool:
    """Uninstall the desktop app."""
    print_box_header("Uninstalling SimpleCore Desktop App", "🗑️")

    install_dir = get_app_install_dir()

    if not install_dir.exists():
        print_status("App not installed", "info")
        return True

    try:
        shutil.rmtree(install_dir)
        print_status(f"Removed: {install_dir}", "success")
        return True
    except Exception as e:
        print_status(f"Failed to uninstall: {e}", "error")
        return False


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Default to 'run' if no command specified
    if not args.command:
        # Run the app by default (download if needed)
        run_app(
            with_worker=True,
            http_port=5000,
            ws_port=5001,
            no_ws=False,
            download_if_missing=True
        )
        print_box_footer()
        return

    project_root = get_project_root()

    if args.command == "run":
        run_app(
            with_worker=not args.no_worker,
            http_port=args.http_port,
            ws_port=args.ws_port,
            no_ws=args.no_ws,
            download_if_missing=not args.no_download
        )

    elif args.command == "download":
        success = download_app(
            source=args.source,
            version=args.version,
            force=args.force
        )
        sys.exit(0 if success else 1)

    elif args.command == "status":
        show_status()

    elif args.command == "uninstall":
        success = uninstall_app()
        sys.exit(0 if success else 1)

    elif args.command == "build-worker":
        print_status(f"Project root: {project_root}", "info")
        success = build_worker(
            output_dir=args.output,
            target=args.target,
            standalone=not args.no_standalone,
            onefile=not args.no_onefile
        )
        sys.exit(0 if success else 1)

    elif args.command == "build-app":
        print_status(f"Project root: {project_root}", "info")
        if not args.skip_worker:
            if not build_worker(Path("nuitka-build"), args.target):
                sys.exit(1)
        if not args.skip_frontend:
            if not build_frontend(project_root):
                sys.exit(1)
        success = build_tauri_app(project_root, args.target, args.debug)
        sys.exit(0 if success else 1)

    elif args.command == "build-all":
        print_status(f"Project root: {project_root}", "info")
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
        print_status(f"Project root: {project_root}", "info")
        run_dev_server(
            project_root,
            no_worker=args.no_worker,
            worker_only=args.worker_only,
            http_port=args.http_port,
            ws_port=args.ws_port,
            no_ws=args.no_ws
        )

    elif args.command == "clean":
        print_status(f"Project root: {project_root}", "info")
        clean_build(project_root)

    print_box_footer()


if __name__ == "__main__":
    main()
