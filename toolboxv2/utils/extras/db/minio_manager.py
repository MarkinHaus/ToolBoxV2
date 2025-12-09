# file: minio_manager.py
# Production-ready MinIO Installation & Management
# Supports Windows, Linux, macOS - Server & Desktop modes

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
import logging
import threading

try:
    import requests
except ImportError:
    requests = None

# Logger setup
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("minio_manager")
        if not _logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger


class MinIOMode(Enum):
    """Operating mode for MinIO"""
    SERVER = "server"           # Central cloud server
    DESKTOP = "desktop"         # Local desktop with mirroring
    STANDALONE = "standalone"   # Single node, no replication


class MinIOStatus(Enum):
    """Status of MinIO instance"""
    RUNNING = "running"
    STOPPED = "stopped"
    NOT_INSTALLED = "not_installed"
    ERROR = "error"


@dataclass
class MinIOConfig:
    """Configuration for a MinIO instance"""
    mode: MinIOMode = MinIOMode.STANDALONE
    data_dir: str = "./.data/minio"
    port: int = 9000
    console_port: int = 9001
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    host: str = "127.0.0.1"
    use_tls: bool = False

    # Replication settings
    cloud_endpoint: Optional[str] = None
    cloud_access_key: Optional[str] = None
    cloud_secret_key: Optional[str] = None
    sync_bucket: str = "user-data-enc"

    # Process management
    pid_file: Optional[str] = None
    log_file: Optional[str] = None

    def __post_init__(self):
        self.data_dir = str(Path(self.data_dir).expanduser().resolve())
        if self.pid_file is None:
            self.pid_file = str(Path(self.data_dir) / "minio.pid")
        if self.log_file is None:
            self.log_file = str(Path(self.data_dir) / "minio.log")

    @property
    def endpoint(self) -> str:
        scheme = "https" if self.use_tls else "http"
        return f"{scheme}://{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "data_dir": self.data_dir,
            "port": self.port,
            "console_port": self.console_port,
            "access_key": self.access_key,
            "secret_key": self.secret_key,
            "host": self.host,
            "use_tls": self.use_tls,
            "cloud_endpoint": self.cloud_endpoint,
            "cloud_access_key": self.cloud_access_key,
            "cloud_secret_key": self.cloud_secret_key,
            "sync_bucket": self.sync_bucket,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MinIOConfig':
        if "mode" in data and isinstance(data["mode"], str):
            data["mode"] = MinIOMode(data["mode"])
        return cls(**data)


class MinIOInstaller:
    """Cross-platform MinIO installer"""

    DOWNLOAD_URLS = {
        "Linux": {
            "x86_64": "https://dl.min.io/server/minio/release/linux-amd64/minio",
            "aarch64": "https://dl.min.io/server/minio/release/linux-arm64/minio",
            "arm64": "https://dl.min.io/server/minio/release/linux-arm64/minio",
        },
        "Darwin": {
            "x86_64": "https://dl.min.io/server/minio/release/darwin-amd64/minio",
            "arm64": "https://dl.min.io/server/minio/release/darwin-arm64/minio",
        },
        "Windows": {
            "AMD64": "https://dl.min.io/server/minio/release/windows-amd64/minio.exe",
            "x86_64": "https://dl.min.io/server/minio/release/windows-amd64/minio.exe",
        }
    }

    MC_DOWNLOAD_URLS = {
        "Linux": {
            "x86_64": "https://dl.min.io/client/mc/release/linux-amd64/mc",
            "aarch64": "https://dl.min.io/client/mc/release/linux-arm64/mc",
            "arm64": "https://dl.min.io/client/mc/release/linux-arm64/mc",
        },
        "Darwin": {
            "x86_64": "https://dl.min.io/client/mc/release/darwin-amd64/mc",
            "arm64": "https://dl.min.io/client/mc/release/darwin-arm64/mc",
        },
        "Windows": {
            "AMD64": "https://dl.min.io/client/mc/release/windows-amd64/mc.exe",
            "x86_64": "https://dl.min.io/client/mc/release/windows-amd64/mc.exe",
        }
    }

    def __init__(self, install_dir: Optional[str] = None):
        self.system = platform.system()
        self.arch = platform.machine()

        if install_dir is None:
            if self.system == "Windows":
                install_dir = os.path.join(os.environ.get("LOCALAPPDATA", "."), "minio")
            else:
                install_dir = os.path.expanduser("~/.local/bin")

        self.install_dir = Path(install_dir)
        self.install_dir.mkdir(parents=True, exist_ok=True)

    def _get_download_url(self, urls: Dict) -> Optional[str]:
        """Get download URL for current platform"""
        system_urls = urls.get(self.system)
        if not system_urls:
            get_logger().error(f"Unsupported system: {self.system}")
            return None

        url = system_urls.get(self.arch)
        if not url:
            get_logger().error(f"Unsupported architecture: {self.arch} on {self.system}")
            return None

        return url

    def _download_file(self, url: str, dest: Path, progress_callback: Optional[Callable] = None) -> bool:
        """Download file with progress tracking"""
        if requests is None:
            get_logger().error("requests library not available")
            return False

        try:
            get_logger().info(f"Downloading from {url}...")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size:
                            progress_callback(downloaded, total_size)

            # Make executable on Unix
            if self.system != "Windows":
                dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

            get_logger().info(f"Downloaded to {dest}")
            return True

        except Exception as e:
            get_logger().error(f"Download failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def get_minio_path(self) -> Optional[Path]:
        """Get path to MinIO binary"""
        exe_name = "minio.exe" if self.system == "Windows" else "minio"

        # Check install directory
        local_path = self.install_dir / exe_name
        if local_path.exists():
            return local_path

        # Check system PATH
        system_path = shutil.which("minio")
        if system_path:
            return Path(system_path)

        return None

    def get_mc_path(self) -> Optional[Path]:
        """Get path to MinIO Client (mc) binary"""
        exe_name = "mc.exe" if self.system == "Windows" else "mc"

        local_path = self.install_dir / exe_name
        if local_path.exists():
            return local_path

        system_path = shutil.which("mc")
        if system_path:
            return Path(system_path)

        return None

    def is_minio_installed(self) -> bool:
        """Check if MinIO is installed"""
        return self.get_minio_path() is not None

    def is_mc_installed(self) -> bool:
        """Check if MinIO Client is installed"""
        return self.get_mc_path() is not None

    def install_minio(self, progress_callback: Optional[Callable] = None) -> bool:
        """Install MinIO server binary"""
        if self.is_minio_installed():
            get_logger().info("MinIO is already installed")
            return True

        url = self._get_download_url(self.DOWNLOAD_URLS)
        if not url:
            return False

        exe_name = "minio.exe" if self.system == "Windows" else "minio"
        dest = self.install_dir / exe_name

        return self._download_file(url, dest, progress_callback)

    def install_mc(self, progress_callback: Optional[Callable] = None) -> bool:
        """Install MinIO Client (mc) binary"""
        if self.is_mc_installed():
            get_logger().info("MinIO Client (mc) is already installed")
            return True

        url = self._get_download_url(self.MC_DOWNLOAD_URLS)
        if not url:
            return False

        exe_name = "mc.exe" if self.system == "Windows" else "mc"
        dest = self.install_dir / exe_name

        return self._download_file(url, dest, progress_callback)

    def install_all(self, progress_callback: Optional[Callable] = None) -> bool:
        """Install both MinIO server and client"""
        minio_ok = self.install_minio(progress_callback)
        mc_ok = self.install_mc(progress_callback)
        return minio_ok and mc_ok

    def get_version(self) -> Optional[str]:
        """Get installed MinIO version"""
        minio_path = self.get_minio_path()
        if not minio_path:
            return None

        try:
            result = subprocess.run(
                [str(minio_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Parse version from output
                output = result.stdout.strip()
                # Format: "minio version RELEASE.2024-..."
                if "version" in output.lower():
                    parts = output.split()
                    for i, part in enumerate(parts):
                        if part.lower() == "version" and i + 1 < len(parts):
                            return parts[i + 1]
                return output
        except Exception as e:
            get_logger().warning(f"Failed to get MinIO version: {e}")

        return None


class MinIOInstance:
    """Manages a running MinIO instance"""

    def __init__(self, config: MinIOConfig, installer: Optional[MinIOInstaller] = None):
        self.config = config
        self.installer = installer or MinIOInstaller()
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

        # Ensure data directory exists
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)

    def _read_pid(self) -> Optional[int]:
        """Read PID from file"""
        pid_path = Path(self.config.pid_file)
        if pid_path.exists():
            try:
                return int(pid_path.read_text().strip())
            except (ValueError, IOError):
                pass
        return None

    def _write_pid(self, pid: int):
        """Write PID to file"""
        Path(self.config.pid_file).write_text(str(pid))

    def _clear_pid(self):
        """Remove PID file"""
        pid_path = Path(self.config.pid_file)
        if pid_path.exists():
            pid_path.unlink()

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running"""
        try:
            if platform.system() == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x0400, False, pid)  # PROCESS_QUERY_INFORMATION
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            else:
                os.kill(pid, 0)
                return True
        except (OSError, ProcessLookupError, PermissionError):
            return False

    def get_status(self) -> MinIOStatus:
        """Get current status of MinIO instance"""
        if not self.installer.is_minio_installed():
            return MinIOStatus.NOT_INSTALLED

        pid = self._read_pid()
        if pid and self._is_process_running(pid):
            # Verify it's actually responsive
            try:
                if requests:
                    response = requests.get(
                        f"{self.config.endpoint}/minio/health/live",
                        timeout=2
                    )
                    if response.status_code == 200:
                        return MinIOStatus.RUNNING
            except:
                pass
            return MinIOStatus.RUNNING  # Process exists but maybe starting up

        return MinIOStatus.STOPPED

    def start(self, wait_ready: bool = True, timeout: int = 30) -> bool:
        """Start MinIO server"""
        with self._lock:
            status = self.get_status()

            if status == MinIOStatus.NOT_INSTALLED:
                get_logger().info("MinIO not installed, installing now...")
                if not self.installer.install_minio():
                    get_logger().error("Failed to install MinIO")
                    return False

            if status == MinIOStatus.RUNNING:
                get_logger().info("MinIO is already running")
                return True

            minio_path = self.installer.get_minio_path()
            if not minio_path:
                get_logger().error("MinIO binary not found")
                return False

            # Build command
            cmd = [
                str(minio_path),
                "server",
                self.config.data_dir,
                "--address", f"{self.config.host}:{self.config.port}",
                "--console-address", f"{self.config.host}:{self.config.console_port}",
            ]

            # Set environment
            env = os.environ.copy()
            env["MINIO_ROOT_USER"] = self.config.access_key
            env["MINIO_ROOT_PASSWORD"] = self.config.secret_key

            print("Starting MinIO with user:", self.config.access_key, self.config.secret_key)
            # Open log file
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                log_handle = open(log_path, 'a')

                # Start process
                if platform.system() == "Windows":
                    creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                    self._process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=log_handle,
                        stderr=log_handle,
                        creationflags=creationflags
                    )
                else:
                    self._process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=log_handle,
                        stderr=log_handle,
                        start_new_session=True
                    )

                self._write_pid(self._process.pid)
                get_logger().info(f"MinIO started with PID {self._process.pid}")

                # Wait for ready
                if wait_ready:
                    return self._wait_for_ready(timeout)

                return True

            except Exception as e:
                get_logger().error(f"Failed to start MinIO: {e}")
                return False

    def _wait_for_ready(self, timeout: int) -> bool:
        """Wait for MinIO to be ready"""
        if not requests:
            time.sleep(2)  # Fallback if requests not available
            return True

        start_time = time.time()
        health_url = f"{self.config.endpoint}/minio/health/live"

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    get_logger().info("MinIO is ready")
                    return True
            except:
                pass
            time.sleep(0.5)

        get_logger().warning(f"MinIO not ready after {timeout}s")
        return False

    def stop(self, timeout: int = 10) -> bool:
        """Stop MinIO server"""
        with self._lock:
            pid = self._read_pid()
            if not pid:
                get_logger().info("MinIO is not running (no PID file)")
                return True

            if not self._is_process_running(pid):
                get_logger().info("MinIO process not running, cleaning up")
                self._clear_pid()
                return True

            get_logger().info(f"Stopping MinIO (PID {pid})...")

            try:
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                   capture_output=True, timeout=timeout)
                else:
                    os.kill(pid, 15)  # SIGTERM

                    # Wait for graceful shutdown
                    for _ in range(timeout):
                        if not self._is_process_running(pid):
                            break
                        time.sleep(1)
                    else:
                        # Force kill
                        os.kill(pid, 9)  # SIGKILL

                self._clear_pid()
                get_logger().info("MinIO stopped")
                return True

            except Exception as e:
                get_logger().error(f"Failed to stop MinIO: {e}")
                return False

    def restart(self, wait_ready: bool = True) -> bool:
        """Restart MinIO server"""
        self.stop()
        time.sleep(1)
        return self.start(wait_ready=wait_ready)

    def get_health(self) -> Dict[str, Any]:
        """Get health information"""
        result = {
            "status": self.get_status().value,
            "endpoint": self.config.endpoint,
            "console": f"http://{self.config.host}:{self.config.console_port}",
            "data_dir": self.config.data_dir,
            "mode": self.config.mode.value,
        }

        if self.get_status() == MinIOStatus.RUNNING and requests:
            try:
                # Get cluster health
                response = requests.get(
                    f"{self.config.endpoint}/minio/health/cluster",
                    auth=(self.config.access_key, self.config.secret_key),
                    timeout=5
                )
                if response.status_code == 200:
                    result["cluster_health"] = response.json()
            except:
                pass

        return result


class MinIOClientWrapper:
    """Wrapper for MinIO Client (mc) operations"""

    def __init__(self, installer: Optional[MinIOInstaller] = None):
        self.installer = installer or MinIOInstaller()
        self._aliases: Dict[str, MinIOConfig] = {}

    def _get_mc_path(self) -> Optional[str]:
        """Get mc binary path, installing if needed"""
        mc_path = self.installer.get_mc_path()
        if not mc_path:
            if not self.installer.install_mc():
                return None
            mc_path = self.installer.get_mc_path()
        return str(mc_path) if mc_path else None

    def _run_mc(self, args: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
        """Run mc command"""
        mc_path = self._get_mc_path()
        if not mc_path:
            raise RuntimeError("MinIO Client (mc) not available")

        cmd = [mc_path] + args
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def set_alias(self, alias: str, config: MinIOConfig) -> bool:
        """Configure an alias for a MinIO server"""
        try:
            result = self._run_mc([
                "alias", "set", alias,
                config.endpoint,
                config.access_key,
                config.secret_key
            ])

            if result.returncode == 0:
                self._aliases[alias] = config
                get_logger().info(f"Alias '{alias}' configured for {config.endpoint}")
                return True
            else:
                get_logger().error(f"Failed to set alias: {result.stderr}")
                return False

        except Exception as e:
            get_logger().error(f"Failed to set alias: {e}")
            return False

    def remove_alias(self, alias: str) -> bool:
        """Remove an alias"""
        try:
            result = self._run_mc(["alias", "rm", alias])
            if alias in self._aliases:
                del self._aliases[alias]
            return result.returncode == 0
        except Exception as e:
            get_logger().error(f"Failed to remove alias: {e}")
            return False

    def create_bucket(self, alias: str, bucket: str) -> bool:
        """Create a bucket"""
        try:
            result = self._run_mc(["mb", f"{alias}/{bucket}", "--ignore-existing"])
            return result.returncode == 0
        except Exception as e:
            get_logger().error(f"Failed to create bucket: {e}")
            return False

    def setup_replication(self, source_alias: str, target_alias: str, bucket: str) -> bool:
        """Setup bucket replication between two MinIO instances"""
        try:
            # Enable versioning on both sides (required for replication)
            self._run_mc(["version", "enable", f"{source_alias}/{bucket}"])
            self._run_mc(["version", "enable", f"{target_alias}/{bucket}"])

            # Setup replication
            result = self._run_mc([
                "replicate", "add",
                f"{source_alias}/{bucket}",
                f"--remote-bucket={target_alias}/{bucket}",
                "--replicate", "delete,delete-marker,existing-objects"
            ])

            if result.returncode == 0:
                get_logger().info(f"Replication configured: {source_alias}/{bucket} -> {target_alias}/{bucket}")
                return True
            else:
                get_logger().error(f"Replication setup failed: {result.stderr}")
                return False

        except Exception as e:
            get_logger().error(f"Failed to setup replication: {e}")
            return False

    def start_mirror(self, source: str, target: str, watch: bool = True) -> Optional[subprocess.Popen]:
        """Start mirroring between source and target

        Args:
            source: Source path (alias/bucket)
            target: Target path (alias/bucket)
            watch: If True, watch for changes and sync continuously

        Returns:
            Popen object if watch=True, None otherwise
        """
        mc_path = self._get_mc_path()
        if not mc_path:
            return None

        args = [mc_path, "mirror"]
        if watch:
            args.append("--watch")
        args.extend(["--remove", "--overwrite", source, target])

        try:
            if watch:
                # Start as background process
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True
                )
                get_logger().info(f"Mirror started: {source} -> {target}")
                return process
            else:
                # One-shot sync
                result = subprocess.run(args, capture_output=True, text=True, timeout=3600)
                if result.returncode == 0:
                    get_logger().info(f"Mirror complete: {source} -> {target}")
                else:
                    get_logger().error(f"Mirror failed: {result.stderr}")
                return None

        except Exception as e:
            get_logger().error(f"Mirror failed: {e}")
            return None

    def list_objects(self, path: str, recursive: bool = False) -> List[Dict[str, Any]]:
        """List objects in a bucket/path"""
        try:
            args = ["ls", "--json"]
            if recursive:
                args.append("--recursive")
            args.append(path)

            result = self._run_mc(args)

            objects = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        obj = json.loads(line)
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass

            return objects

        except Exception as e:
            get_logger().error(f"Failed to list objects: {e}")
            return []


class MinIOManager:
    """High-level manager for MinIO setup and operations"""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.expanduser("~/.toolboxv2/minio")

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.base_dir / "config.json"
        self.installer = MinIOInstaller(str(self.base_dir / "bin"))
        self.mc_client = MinIOClientWrapper(self.installer)

        self._instances: Dict[str, MinIOInstance] = {}
        self._mirror_processes: List[subprocess.Popen] = []

        self._load_config()

    def _load_config(self):
        """Load saved configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = json.load(f)

                for name, cfg_data in data.get("instances", {}).items():
                    config = MinIOConfig.from_dict(cfg_data)
                    self._instances[name] = MinIOInstance(config, self.installer)

            except Exception as e:
                get_logger().warning(f"Failed to load config: {e}")

    def _save_config(self):
        """Save current configuration"""
        try:
            data = {
                "instances": {
                    name: inst.config.to_dict()
                    for name, inst in self._instances.items()
                }
            }
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            get_logger().error(f"Failed to save config: {e}")

    def install(self, progress_callback: Optional[Callable] = None) -> bool:
        """Install MinIO and mc"""
        return self.installer.install_all(progress_callback)

    def create_instance(self, name: str, config: MinIOConfig) -> MinIOInstance:
        """Create a new MinIO instance configuration"""
        instance = MinIOInstance(config, self.installer)
        self._instances[name] = instance
        self._save_config()
        return instance

    def get_instance(self, name: str) -> Optional[MinIOInstance]:
        """Get instance by name"""
        return self._instances.get(name)

    def remove_instance(self, name: str, delete_data: bool = False) -> bool:
        """Remove an instance"""
        if name not in self._instances:
            return False

        instance = self._instances[name]
        instance.stop()

        if delete_data:
            data_path = Path(instance.config.data_dir)
            if data_path.exists():
                shutil.rmtree(data_path)

        del self._instances[name]
        self._save_config()
        return True

    def setup_server(self, name: str = "cloud",
                     port: int = 9000,
                     data_dir: Optional[str] = None,
                     access_key: str = "admin",
                     secret_key: str = "SecureCloudPass",
                     use_docker: bool = False) -> MinIOInstance:
        """Setup a central cloud server"""

        if data_dir is None:
            data_dir = str(self.base_dir / "data" / name)

        config = MinIOConfig(
            mode=MinIOMode.SERVER,
            data_dir=data_dir,
            port=port,
            console_port=port + 1,
            access_key=access_key,
            secret_key=secret_key,
            host="0.0.0.0"  # Listen on all interfaces for server
        )

        if use_docker:
            return self._setup_docker_server(name, config)

        instance = self.create_instance(name, config)

        # Ensure bucket exists after starting
        if instance.start():
            time.sleep(2)  # Wait for startup
            self.mc_client.set_alias(name, config)
            self.mc_client.create_bucket(name, config.sync_bucket)

        return instance

    def _setup_docker_server(self, name: str, config: MinIOConfig) -> MinIOInstance:
        """Setup MinIO server using Docker"""
        try:
            # Check if Docker is available
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
        except:
            get_logger().warning("Docker not available, falling back to direct installation")
            return self.create_instance(name, config)

        # Create data directory
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)

        # Build docker command
        container_name = f"minio-{name}"
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{config.port}:9000",
            "-p", f"{config.console_port}:9001",
            "-v", f"{config.data_dir}:/data",
            "-e", f"MINIO_ROOT_USER={config.access_key}",
            "-e", f"MINIO_ROOT_PASSWORD={config.secret_key}",
            "quay.io/minio/minio",
            "server", "/data",
            "--console-address", ":9001"
        ]

        try:
            # Remove existing container if any
            subprocess.run(["docker", "rm", "-f", container_name],
                          capture_output=True)

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                get_logger().info(f"Docker container '{container_name}' started")

                # Create instance for tracking
                instance = self.create_instance(name, config)

                # Wait for startup and setup
                time.sleep(3)
                self.mc_client.set_alias(name, config)
                self.mc_client.create_bucket(name, config.sync_bucket)

                return instance
            else:
                get_logger().error(f"Docker start failed: {result.stderr}")
                raise RuntimeError(result.stderr)

        except Exception as e:
            get_logger().error(f"Docker setup failed: {e}")
            raise

    def setup_desktop(self, name: str = "local",
                      cloud_endpoint: Optional[str] = None,
                      cloud_access_key: Optional[str] = None,
                      cloud_secret_key: Optional[str] = None,
                      auto_sync: bool = True) -> MinIOInstance:
        """Setup a desktop client with optional cloud sync"""
        endpoint = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9010")
        host, port = endpoint.split(":")
        config = MinIOConfig(
            mode=MinIOMode.DESKTOP,
            data_dir=str(self.base_dir / "data" / name),
            port=port,  # Different port from server
            console_port=port+1,
            access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "SecurePass123"),
            host=host,
            cloud_endpoint=cloud_endpoint,
            cloud_access_key=cloud_access_key,
            cloud_secret_key=cloud_secret_key,
        )

        instance = self.create_instance(name, config)

        if instance.start():
            time.sleep(2)
            self.mc_client.set_alias("local", config)
            self.mc_client.create_bucket("local", config.sync_bucket)

            # Setup cloud sync if configured
            if cloud_endpoint and cloud_access_key and cloud_secret_key:
                cloud_config = MinIOConfig(
                    endpoint=cloud_endpoint,
                    access_key=cloud_access_key,
                    secret_key=cloud_secret_key,
                )
                self.mc_client.set_alias("cloud", cloud_config)

                if auto_sync:
                    self.start_bidirectional_sync("local", "cloud", config.sync_bucket)

        return instance

    def start_bidirectional_sync(self, local_alias: str, cloud_alias: str, bucket: str):
        """Start bidirectional sync between local and cloud"""
        local_path = f"{local_alias}/{bucket}"
        cloud_path = f"{cloud_alias}/{bucket}"

        # Upload: local -> cloud
        upload_proc = self.mc_client.start_mirror(local_path, cloud_path, watch=True)
        if upload_proc:
            self._mirror_processes.append(upload_proc)

        # Download: cloud -> local
        download_proc = self.mc_client.start_mirror(cloud_path, local_path, watch=True)
        if download_proc:
            self._mirror_processes.append(download_proc)

        get_logger().info("Bidirectional sync started")

    def stop_all_sync(self):
        """Stop all sync processes"""
        for proc in self._mirror_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        self._mirror_processes.clear()
        get_logger().info("All sync processes stopped")

    def setup_replication(self, source_name: str, target_name: str):
        """Setup server-to-server replication"""
        source = self.get_instance(source_name)
        target = self.get_instance(target_name)

        if not source or not target:
            raise ValueError("Both source and target instances must exist")

        # Setup aliases
        self.mc_client.set_alias(source_name, source.config)
        self.mc_client.set_alias(target_name, target.config)

        # Setup bidirectional replication
        bucket = source.config.sync_bucket
        self.mc_client.setup_replication(source_name, target_name, bucket)
        self.mc_client.setup_replication(target_name, source_name, bucket)

        get_logger().info(f"Active-active replication configured between {source_name} and {target_name}")

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all instances"""
        return {
            name: inst.get_health()
            for name, inst in self._instances.items()
        }

    def start_all(self) -> bool:
        """Start all configured instances"""
        success = True
        for name, instance in self._instances.items():
            if not instance.start():
                get_logger().error(f"Failed to start {name}")
                success = False
        return success

    def stop_all(self) -> bool:
        """Stop all instances and sync processes"""
        self.stop_all_sync()

        success = True
        for name, instance in self._instances.items():
            if not instance.stop():
                get_logger().error(f"Failed to stop {name}")
                success = False
        return success


# Convenience functions for CLI usage
def quick_install() -> bool:
    """Quick install MinIO"""
    installer = MinIOInstaller()
    return installer.install_all()


def quick_server_setup(port: int = 9000, access_key: str = "admin",
                       secret_key: str = "SecurePass123") -> MinIOInstance:
    """Quick server setup"""
    manager = MinIOManager()
    return manager.setup_server(port=port, access_key=access_key, secret_key=secret_key)


def quick_desktop_setup(cloud_endpoint: str, cloud_access_key: str,
                        cloud_secret_key: str) -> MinIOInstance:
    """Quick desktop setup with cloud sync"""
    manager = MinIOManager()
    return manager.setup_desktop(
        cloud_endpoint=cloud_endpoint,
        cloud_access_key=cloud_access_key,
        cloud_secret_key=cloud_secret_key,
        auto_sync=True
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    manager = MinIOManager()
    print("Installing MinIO...")
    if manager.install():
        print("Installation successful!")
        print(f"Version: {manager.installer.get_version()}")
    else:
        print("Installation failed!")
