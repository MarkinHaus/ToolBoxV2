# file: db_cli_manager.py
# Production-Ready Manager for MinIO-based Blob Storage
# Features: Cross-platform installation, Server/Desktop/Mobile setup, Replication

import argparse
import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Enhanced UI Imports ---
try:
    from toolboxv2.utils.extras.Style import Spinner, Style
except ImportError:
    try:
        from toolboxv2.extras.Style import Spinner, Style
    except ImportError:
        # Fallback minimal Style class
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
            def CYAN(s): return f"\033[96m{s}\033[0m"
            @staticmethod
            def GREY(s): return f"\033[90m{s}\033[0m"
            @staticmethod
            def BOLD(s): return f"\033[1m{s}\033[0m"

        class Spinner:
            def __init__(self, message="", **kwargs):
                self.message = message
            def __enter__(self):
                print(f"⏳ {self.message}...")
                return self
            def __exit__(self, *args):
                pass

# --- CLI Printing Utilities ---
try:
    from toolboxv2.utils.clis.cli_printing import (
        print_box_content,
        print_box_footer,
        print_box_header,
        print_separator,
        print_status,
        print_table_header,
        print_table_row,
    )
except ImportError:
    # Fallback implementations
    def print_box_header(title, icon=""):
        print(f"\n╔{'═' * 60}╗")
        print(f"║ {icon} {title:<56} ║")
        print(f"╠{'═' * 60}╣")

    def print_box_content(content, style="info"):
        print(f"║ {content:<58} ║")

    def print_box_footer():
        print(f"╚{'═' * 60}╝\n")

    def print_status(message, style="info"):
        icons = {"success": "✓", "error": "✗", "warning": "⚠", "info": "ℹ"}
        colors = {"success": Style.GREEN, "error": Style.RED,
                  "warning": Style.YELLOW, "info": Style.CYAN}
        icon = icons.get(style, "•")
        color = colors.get(style, lambda x: x)
        print(f"  {icon} {color(message)}")

    def print_separator():
        print(f"{'─' * 62}")

    def print_table_header(columns, widths):
        header = " │ ".join(f"{col:<{w}}" for (col, _), w in zip(columns, widths, strict=False))
        print(f"  {header}")
        print(f"  {'─' * sum(widths)}")

    def print_table_row(values, widths, styles=None):
        row = " │ ".join(f"{v:<{w}}" for v, w in zip(values, widths, strict=False))
        print(f"  {row}")

# --- Configuration ---
try:
    import requests
except ImportError:
    requests = None

try:
    import psutil
except ImportError:
    psutil = None

# Import our MinIO manager
try:
    from toolboxv2.utils.extras.db.minio_manager import (
        MinIOClientWrapper,
        MinIOConfig,
        MinIOInstaller,
        MinIOInstance,
        MinIOManager,
        MinIOMode,
        MinIOStatus,
    )
    from toolboxv2.utils.extras.db.mobile_db import MobileDB
except ImportError:
    print(Style.RED("FATAL: minio_manager.py and mobile_db.py must be in the same directory"))
    sys.exit(1)

# Configuration
CONFIG_FILE = "minio_cluster_config.json"
DEFAULT_BASE_DIR = os.getenv('APPDATA') if os.name == 'nt' else os.getenv('XDG_CONFIG_HOME') or os.path.expanduser(
                '~/.config') if os.name == 'posix' else os.path.expanduser("~/")
DEFAULT_BASE_DIR += "/ToolBoxV2/.minio"

@dataclass
class ClusterConfig:
    """Configuration for a MinIO cluster/setup"""
    name: str
    mode: str  # "server", "desktop", "mobile"
    data_dir: str
    port: int = 9000
    console_port: int = 9001
    access_key: str = "admin"
    secret_key: str = "SecurePass123"
    host: str = "127.0.0.1"

    # Cloud sync settings
    cloud_endpoint: str | None = None
    cloud_access_key: str | None = None
    cloud_secret_key: str | None = None
    cloud_bucket: str = "user-data-enc"

    # Replication settings
    replicate_to: str | None = None  # Name of another server to replicate to

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "data_dir": self.data_dir,
            "port": self.port,
            "console_port": self.console_port,
            "access_key": self.access_key,
            "secret_key": self.secret_key,
            "host": self.host,
            "cloud_endpoint": self.cloud_endpoint,
            "cloud_access_key": self.cloud_access_key,
            "cloud_secret_key": self.cloud_secret_key,
            "cloud_bucket": self.cloud_bucket,
            "replicate_to": self.replicate_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterConfig':
        return cls(**data)


class MinIOCLIManager:
    """CLI Manager for MinIO installations and configurations"""

    def __init__(self, config_path: str | None = None):
        self.config_path = Path(config_path or CONFIG_FILE)

        # Find toolbox root if available
        try:
            from toolboxv2 import tb_root_dir
            self.tb_root = tb_root_dir
            if not self.config_path.is_absolute():
                self.config_path = self.tb_root / self.config_path
        except ImportError:
            self.tb_root = Path.cwd()

        self.base_dir = Path(DEFAULT_BASE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.installer = MinIOInstaller(str(self.base_dir / "bin"))
        self.mc_client = MinIOClientWrapper(self.installer)

        self.configs: Dict[str, ClusterConfig] = {}
        self.instances: Dict[str, MinIOInstance] = {}

        self._load_config()

    def _load_config(self):
        """Load cluster configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)

                for name, cfg in data.get("clusters", {}).items():
                    self.configs[name] = ClusterConfig.from_dict(cfg)

            except Exception as e:
                print_status(f"Failed to load config: {e}", "warning")
        else:
            # Create default configuration
            self._create_default_config()

    def _save_config(self):
        """Save configuration to file"""
        try:
            data = {
                "clusters": {
                    name: cfg.to_dict() for name, cfg in self.configs.items()
                }
            }
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print_status(f"Failed to save config: {e}", "error")

    def _create_default_config(self):
        """Create default configuration"""
        default_data_dir = str(self.base_dir / "data" / "default")

        self.configs["default"] = ClusterConfig(
            name="default",
            mode="desktop",
            data_dir=default_data_dir,
            port=9000,
            console_port=9001,
        )
        self._save_config()

    def _get_instance(self, name: str) -> MinIOInstance | None:
        """Get or create MinIO instance"""
        if name not in self.configs:
            return None

        if name not in self.instances:
            config = self.configs[name]
            minio_config = MinIOConfig(
                mode=MinIOMode(config.mode) if config.mode in ["server", "desktop"] else MinIOMode.STANDALONE,
                data_dir=config.data_dir,
                port=config.port,
                console_port=config.console_port,
                access_key=config.access_key,
                secret_key=config.secret_key,
                host=config.host,
                cloud_endpoint=config.cloud_endpoint,
                cloud_access_key=config.cloud_access_key,
                cloud_secret_key=config.cloud_secret_key,
            )
            self.instances[name] = MinIOInstance(minio_config, self.installer)

        return self.instances[name]

    # =================== Installation Commands ===================

    def cmd_install(self, components: List[str] = None):
        """Install MinIO components"""
        print_box_header("Installing MinIO", "📦")

        system = platform.system()
        arch = platform.machine()
        print_box_content(f"System: {system} ({arch})", "info")
        print_box_footer()

        if components is None or "all" in components:
            components = ["server", "client"]

        success = True

        if "server" in components:
            print_status("Installing MinIO Server...", "info")
            with Spinner("Downloading MinIO Server", symbols='d'):
                if self.installer.install_minio(self._progress_callback):
                    print_status("MinIO Server installed successfully", "success")
                else:
                    print_status("Failed to install MinIO Server", "error")
                    success = False

        if "client" in components:
            print_status("Installing MinIO Client (mc)...", "info")
            with Spinner("Downloading MinIO Client", symbols='d'):
                if self.installer.install_mc(self._progress_callback):
                    print_status("MinIO Client installed successfully", "success")
                else:
                    print_status("Failed to install MinIO Client", "error")
                    success = False

        if success:
            version = self.installer.get_version()
            print_status(f"Installation complete. Version: {version or 'unknown'}", "success")

        return success

    def _progress_callback(self, downloaded: int, total: int):
        """Progress callback for downloads"""
        percent = (downloaded / total) * 100 if total > 0 else 0
        bar_width = 30
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  [{bar}] {percent:.1f}%", end="", flush=True)
        if downloaded >= total:
            print()

    def cmd_uninstall(self):
        """Uninstall MinIO"""
        print_box_header("Uninstalling MinIO", "🗑️")
        print_box_footer()

        # Stop all instances first
        self.cmd_stop_all()

        # Remove binaries
        bin_dir = self.base_dir / "bin"
        if bin_dir.exists():
            for item in bin_dir.iterdir():
                if "minio" in item.name.lower() or item.name in ["mc", "mc.exe"]:
                    item.unlink()
                    print_status(f"Removed {item.name}", "info")

        print_status("MinIO uninstalled", "success")

    # =================== Setup Commands ===================

    def cmd_setup_server(self, name: str = "cloud",
                         port: Optional[int] = 9000,
                         access_key: Optional[str] = None,
                         secret_key: Optional[str] = None,
                         host: Optional[str] = "0.0.0.0",
                         use_docker: bool = False):
        """Setup a central cloud server"""
        print_box_header(f"Setting up Server: {name}", "🖥️")
        print_box_content(f"Port: {port}", "info")
        print_box_content(f"Host: {host}", "info")
        print_box_content(f"Docker: {use_docker}", "info")
        print_box_footer()

        entry_point = os.getenv("MINIO_ENDPOINT", f"{host}:{port}")
        access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "admin")
        secret_key = secret_key or os.getenv("MINIO_SECRET_KEY","SecureCloudPass" )
        port = int(entry_point.split(':')[1])
        host = entry_point.split(':')[0]

        # Ensure MinIO is installed
        if not self.installer.is_minio_installed():
            print_status("MinIO not installed, installing now...", "warning")
            if not self.cmd_install(["server"]):
                return False

        data_dir = str(self.base_dir / "data" / name)

        config = ClusterConfig(
            name=name,
            mode="server",
            data_dir=data_dir,
            port=port,
            console_port=port + 1,
            access_key=access_key,
            secret_key=secret_key,
            host=host,
        )

        self.configs[name] = config
        self._save_config()

        if use_docker:
            return self._setup_docker_server(name, config)

        instance = self._get_instance(name)
        if instance and instance.start():
            print_status(f"Server '{name}' started successfully", "success")
            print_status(f"Console: http://{host}:{port + 1}", "info")

            # Setup alias and bucket
            time.sleep(2)
            minio_config = MinIOConfig(
                port=port,
                access_key=access_key,
                secret_key=secret_key,
                host="127.0.0.1" if host == "0.0.0.0" else host,
            )
            self.mc_client.set_alias(name, minio_config)
            self.mc_client.create_bucket(name, config.cloud_bucket)
            print_status(f"Bucket '{config.cloud_bucket}' created", "success")

            return True

        print_status(f"Failed to start server '{name}'", "error")
        return False

    def _setup_docker_server(self, name: str, config: ClusterConfig) -> bool:
        """Setup MinIO server using Docker"""
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
        except:
            print_status("Docker not available, please install Docker first", "error")
            return False

        Path(config.data_dir).mkdir(parents=True, exist_ok=True)

        container_name = f"minio-{name}"
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{config.port}:9000",
            "-p", f"{config.console_port}:9001",
            "-v", f"{config.data_dir}:/data",
            "-e", f"MINIO_ROOT_USER={config.access_key}",
            "-e", f"MINIO_ROOT_PASSWORD={config.secret_key}",
            "--restart", "unless-stopped",
            "quay.io/minio/minio",
            "server", "/data",
            "--console-address", ":9001"
        ]

        # Remove existing container
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

        with Spinner(f"Starting Docker container '{container_name}'"):
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print_status(f"Docker container '{container_name}' started", "success")
            print_status(f"Console: http://localhost:{config.console_port}", "info")

            # Setup alias and bucket
            time.sleep(3)
            minio_config = MinIOConfig(
                port=config.port,
                access_key=config.access_key,
                secret_key=config.secret_key,
                host="127.0.0.1",
            )
            self.mc_client.set_alias(name, minio_config)
            self.mc_client.create_bucket(name, config.cloud_bucket)

            return True

        print_status(f"Docker start failed: {result.stderr}", "error")
        return False

    def cmd_setup_desktop(self, name: str = "local",
                          cloud_endpoint: str | None = None,
                          cloud_access_key: str | None = None,
                          cloud_secret_key: str | None = None,
                          auto_sync: bool = True):
        """Setup a desktop client with local MinIO and optional cloud sync"""
        print_box_header(f"Setting up Desktop: {name}", "💻")

        cloud_endpoint = cloud_endpoint or os.getenv("CLOUD_ENDPOINT")
        cloud_access_key = cloud_access_key or os.getenv("CLOUD_ACCESS_KEY")
        cloud_secret_key = cloud_secret_key or os.getenv("CLOUD_SECRET_KEY")

        if cloud_endpoint:
            print_box_content(f"Cloud: {cloud_endpoint}", "info")
        print_box_content(f"Auto-sync: {auto_sync}", "info")
        print_box_footer()

        # Ensure MinIO is installed
        if not self.installer.is_minio_installed():
            print_status("MinIO not installed, installing now...", "warning")
            if not self.cmd_install():
                return False

        data_dir = str(self.base_dir / "data" / name)

        endpoint = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9010")
        host, port = endpoint.split(":")
        config = ClusterConfig(
            name=name,
            mode="desktop",
            data_dir=data_dir,
            port=port,
            console_port=port+1,
            access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "SecurePass123"),
            host=host,
            cloud_endpoint=cloud_endpoint,
            cloud_access_key=cloud_access_key,
            cloud_secret_key=cloud_secret_key,
        )

        self.configs[name] = config
        self._save_config()

        instance = self._get_instance(name)
        if instance and instance.start():
            print_status(f"Desktop MinIO '{name}' started", "success")
            print_status(f"Console: http://127.0.0.1:{config.console_port}", "info")

            # Setup local alias
            time.sleep(2)
            minio_config = MinIOConfig(
                port=config.port,
                access_key=config.access_key,
                secret_key=config.secret_key,
                host="127.0.0.1",
            )
            self.mc_client.set_alias("local", minio_config)
            self.mc_client.create_bucket("local", config.cloud_bucket)

            # Setup cloud sync if configured
            if cloud_endpoint and cloud_access_key and cloud_secret_key and auto_sync:
                print_status("Setting up cloud sync...", "info")

                cloud_config = MinIOConfig(
                    host=cloud_endpoint.split(":")[0].replace("http://", "").replace("https://", ""),
                    port=int(cloud_endpoint.split(":")[-1]) if ":" in cloud_endpoint else 9000,
                    access_key=cloud_access_key,
                    secret_key=cloud_secret_key,
                    use_tls="https" in cloud_endpoint,
                )
                self.mc_client.set_alias("cloud", cloud_config)

                # Start bidirectional sync
                self._start_sync("local", "cloud", config.cloud_bucket)
                print_status("Cloud sync configured and started", "success")

            return True

        print_status(f"Failed to start desktop MinIO '{name}'", "error")
        return False

    def cmd_setup_mobile(self, name: str = "mobile",
                         cloud_endpoint: str | None = None,
                         cloud_access_key: str | None = None,
                         cloud_secret_key: str | None = None,
                         max_size_mb: int = 500):
        """Setup mobile SQLite database for offline storage"""
        print_box_header(f"Setting up Mobile DB: {name}", "📱")
        print_box_content(f"Max size: {max_size_mb} MB", "info")

        cloud_endpoint = cloud_endpoint or os.getenv("CLOUD_ENDPOINT")
        cloud_access_key = cloud_access_key or os.getenv("CLOUD_ACCESS_KEY")
        cloud_secret_key = cloud_secret_key or os.getenv("CLOUD_SECRET_KEY")

        if cloud_endpoint:
            print_box_content(f"Cloud: {cloud_endpoint}", "info")
        print_box_footer()

        db_dir = self.base_dir / "mobile" / name
        db_dir.mkdir(parents=True, exist_ok=True)

        db_path = db_dir / "data.db"

        # Create MobileDB instance
        mobile_db = MobileDB(
            db_path=str(db_path),
            max_size_mb=max_size_mb
        )

        # Save config
        config = ClusterConfig(
            name=name,
            mode="mobile",
            data_dir=str(db_dir),
            cloud_endpoint=cloud_endpoint,
            cloud_access_key=cloud_access_key,
            cloud_secret_key=cloud_secret_key,
        )
        self.configs[name] = config
        self._save_config()

        print_status(f"Mobile database created at {db_path}", "success")
        print_status("SQLite database ready for offline use", "info")

        if cloud_endpoint:
            print_status("Manual sync required (call sync command when online)", "warning")

        mobile_db.close()
        return True

    # =================== Replication Commands ===================

    def cmd_setup_replication(self, source: str, target: str):
        """Setup server-to-server replication"""
        print_box_header("Setting up Replication", "🔄")
        print_box_content(f"Source: {source}", "info")
        print_box_content(f"Target: {target}", "info")
        print_box_footer()

        if source not in self.configs:
            print_status(f"Source '{source}' not found in config", "error")
            return False

        if target not in self.configs:
            print_status(f"Target '{target}' not found in config", "error")
            return False

        source_config = self.configs[source]
        target_config = self.configs[target]

        # Setup aliases
        source_minio = MinIOConfig(
            port=source_config.port,
            access_key=source_config.access_key,
            secret_key=source_config.secret_key,
            host="127.0.0.1" if source_config.host == "0.0.0.0" else source_config.host,
        )
        target_minio = MinIOConfig(
            port=target_config.port,
            access_key=target_config.access_key,
            secret_key=target_config.secret_key,
            host="127.0.0.1" if target_config.host == "0.0.0.0" else target_config.host,
        )

        self.mc_client.set_alias(source, source_minio)
        self.mc_client.set_alias(target, target_minio)

        # Setup bidirectional replication
        bucket = source_config.cloud_bucket

        print_status("Configuring replication rules...", "info")

        if self.mc_client.setup_replication(source, target, bucket):
            print_status(f"Replication {source} -> {target} configured", "success")
        else:
            print_status(f"Failed to setup {source} -> {target} replication", "error")
            return False

        if self.mc_client.setup_replication(target, source, bucket):
            print_status(f"Replication {target} -> {source} configured", "success")
        else:
            print_status(f"Failed to setup {target} -> {source} replication", "error")
            return False

        # Update config
        source_config.replicate_to = target
        target_config.replicate_to = source
        self._save_config()

        print_status("Active-Active replication configured successfully", "success")
        return True

    def _start_sync(self, local_alias: str, cloud_alias: str, bucket: str):
        """Start background sync processes"""
        local_path = f"{local_alias}/{bucket}"
        cloud_path = f"{cloud_alias}/{bucket}"

        # Create sync script for systemd/launchd
        self._create_sync_service(local_alias, cloud_alias, bucket)

        # Start immediate mirrors
        self.mc_client.start_mirror(local_path, cloud_path, watch=True)
        self.mc_client.start_mirror(cloud_path, local_path, watch=True)

    def _create_sync_service(self, local_alias: str, cloud_alias: str, bucket: str):
        """Create system service for background sync"""
        system = platform.system()

        if system == "Linux":
            self._create_systemd_service(local_alias, cloud_alias, bucket)
        elif system == "Darwin":
            self._create_launchd_service(local_alias, cloud_alias, bucket)
        elif system == "Windows":
            self._create_windows_task(local_alias, cloud_alias, bucket)

    def _create_systemd_service(self, local_alias: str, cloud_alias: str, bucket: str):
        """Create systemd service for Linux"""
        mc_path = self.installer.get_mc_path()
        if not mc_path:
            return

        service_content = f"""[Unit]
Description=MinIO Sync Service ({local_alias} <-> {cloud_alias})
After=network.target

[Service]
Type=simple
User={os.environ.get('USER', 'root')}
ExecStart={mc_path} mirror --watch --remove --overwrite {local_alias}/{bucket} {cloud_alias}/{bucket}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

        service_path = self.base_dir / "minio-sync.service"
        service_path.write_text(service_content)

        print_status(f"Systemd service file created: {service_path}", "info")
        print_status("Install with: sudo cp minio-sync.service /etc/systemd/system/ && sudo systemctl enable --now minio-sync", "info")

    def _create_launchd_service(self, local_alias: str, cloud_alias: str, bucket: str):
        """Create launchd plist for macOS"""
        mc_path = self.installer.get_mc_path()
        if not mc_path:
            return

        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.toolboxv2.minio-sync</string>
    <key>ProgramArguments</key>
    <array>
        <string>{mc_path}</string>
        <string>mirror</string>
        <string>--watch</string>
        <string>--remove</string>
        <string>--overwrite</string>
        <string>{local_alias}/{bucket}</string>
        <string>{cloud_alias}/{bucket}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
"""

        plist_path = self.base_dir / "com.toolboxv2.minio-sync.plist"
        plist_path.write_text(plist_content)

        print_status(f"LaunchAgent plist created: {plist_path}", "info")
        print_status("Install with: cp com.toolboxv2.minio-sync.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.toolboxv2.minio-sync.plist", "info")

    def _create_windows_task(self, local_alias: str, cloud_alias: str, bucket: str):
        """Create Windows Task Scheduler task"""
        mc_path = self.installer.get_mc_path()
        if not mc_path:
            return

        bat_content = f"""@echo off
"{mc_path}" mirror --watch --remove --overwrite {local_alias}/{bucket} {cloud_alias}/{bucket}
"""

        bat_path = self.base_dir / "minio-sync.bat"
        bat_path.write_text(bat_content)

        print_status(f"Batch file created: {bat_path}", "info")
        print_status("Add to Task Scheduler for automatic startup", "info")

    # =================== Instance Management ===================

    def cmd_start(self, name: str | None = None):
        """Start MinIO instance(s)"""
        instances = [name] if name else list(self.configs.keys())

        for inst_name in instances:
            if inst_name not in self.configs:
                print_status(f"Instance '{inst_name}' not found", "error")
                continue

            config = self.configs[inst_name]
            if config.mode == "mobile":
                print_status(f"'{inst_name}' is mobile (SQLite), no server to start", "info")
                continue

            instance = self._get_instance(inst_name)
            if instance:
                print_status(f"Starting '{inst_name}'...", "info")
                if instance.start():
                    print_status(f"'{inst_name}' started successfully", "success")
                else:
                    print_status(f"Failed to start '{inst_name}'", "error")

    def cmd_stop(self, name: str | None = None):
        """Stop MinIO instance(s)"""
        instances = [name] if name else list(self.configs.keys())

        for inst_name in instances:
            if inst_name not in self.configs:
                continue

            instance = self._get_instance(inst_name)
            if instance:
                print_status(f"Stopping '{inst_name}'...", "info")
                if instance.stop():
                    print_status(f"'{inst_name}' stopped", "success")

    def cmd_stop_all(self):
        """Stop all instances"""
        self.cmd_stop(None)

    def cmd_restart(self, name: str):
        """Restart a MinIO instance"""
        instance = self._get_instance(name)
        if instance:
            print_status(f"Restarting '{name}'...", "info")
            if instance.restart():
                print_status(f"'{name}' restarted successfully", "success")
            else:
                print_status(f"Failed to restart '{name}'", "error")

    def cmd_status(self, name: str | None = None):
        """Show status of instance(s)"""
        instances = [name] if name else list(self.configs.keys())

        columns = [
            ("NAME", 15),
            ("MODE", 10),
            ("STATUS", 12),
            ("PORT", 8),
            ("DATA DIR", 30),
        ]
        widths = [w for _, w in columns]

        print("\n🗄️  MinIO Cluster Status\n")
        print_table_header(columns, widths)
        servers = []

        for inst_name in instances:
            if inst_name not in self.configs:
                continue

            config = self.configs[inst_name]

            if config.mode == "mobile":
                status = "READY"
                status_style = "green"
            else:
                instance = self._get_instance(inst_name)
                if instance:
                    inst_status = instance.get_status()
                    status = inst_status.value.upper()
                    status_style = "green" if inst_status == MinIOStatus.RUNNING else "red"
                else:
                    status = "UNKNOWN"
                    status_style = "yellow"

            data_dir = config.data_dir
            if len(data_dir) > 28:
                data_dir = "..." + data_dir[-25:]

            print_table_row(
                [inst_name, config.mode, status, str(config.port), data_dir],
                widths,
                ["white", "cyan", status_style, "yellow", "grey"]
            )

        print()

    def cmd_health(self, name: str | None = None):
        """Health check for instance(s)"""
        instances = [name] if name else list(self.configs.keys())

        print_box_header("Health Check", "🏥")
        print_box_footer()

        for inst_name in instances:
            if inst_name not in self.configs:
                continue

            config = self.configs[inst_name]

            if config.mode == "mobile":
                # Check SQLite database
                db_path = Path(config.data_dir) / "data.db"
                if db_path.exists():
                    size = db_path.stat().st_size / (1024 * 1024)
                    print_status(f"[{inst_name}] Mobile DB: {size:.2f} MB", "success")
                else:
                    print_status(f"[{inst_name}] Mobile DB not found", "warning")
            else:
                instance = self._get_instance(inst_name)
                if instance:
                    health = instance.get_health()
                    status = health.get("status", "unknown")

                    if status == "running":
                        print_status(f"[{inst_name}] Healthy - {health.get('endpoint')}", "success")
                    else:
                        print_status(f"[{inst_name}] {status}", "warning")

    # =================== Sync Commands ===================

    def cmd_sync(self, name: str):
        """Trigger manual sync for mobile/desktop"""
        if name not in self.configs:
            print_status(f"Instance '{name}' not found", "error")
            return False

        config = self.configs[name]

        if not config.cloud_endpoint:
            print_status("No cloud endpoint configured for sync", "error")
            return False

        print_box_header(f"Syncing '{name}'", "🔄")
        print_box_content(f"Cloud: {config.cloud_endpoint}", "info")
        print_box_footer()

        if config.mode == "mobile":
            return self._sync_mobile(name, config)
        else:
            return self._sync_desktop(name, config)

    def _sync_mobile(self, name: str, config: ClusterConfig) -> bool:
        """Sync mobile SQLite with cloud"""
        db_path = Path(config.data_dir) / "data.db"

        if not db_path.exists():
            print_status("Mobile database not found", "error")
            return False

        mobile_db = MobileDB(str(db_path))

        try:
            from minio import Minio

            # Parse endpoint
            endpoint = config.cloud_endpoint.replace("http://", "").replace("https://", "")
            secure = "https" in config.cloud_endpoint

            minio_client = Minio(
                endpoint,
                access_key=config.cloud_access_key,
                secret_key=config.cloud_secret_key,
                secure=secure
            )

            stats = {
                "uploaded": 0,
                "downloaded": 0,
                "errors": []
            }

            # Upload dirty blobs
            dirty_count = len(mobile_db.get_dirty_blobs())
            print_status(f"Uploading {dirty_count} changed blobs...", "info")

            with Spinner("Uploading"):
                for meta in mobile_db.get_dirty_blobs():
                    try:
                        data = mobile_db.get(meta.path)
                        if data:
                            import io
                            minio_client.put_object(
                                config.cloud_bucket,
                                meta.path,
                                io.BytesIO(data),
                                len(data),
                                metadata={"checksum": meta.checksum}
                            )
                            mobile_db.mark_synced(meta.path)
                            stats["uploaded"] += 1
                    except Exception as e:
                        stats["errors"].append(str(e))

            # Download new blobs from cloud
            print_status("Checking for cloud updates...", "info")

            with Spinner("Downloading"):
                try:
                    objects = minio_client.list_objects(config.cloud_bucket, recursive=True)
                    local_paths = set(m.path for m in mobile_db.list())

                    for obj in objects:
                        if obj.object_name not in local_paths:
                            try:
                                response = minio_client.get_object(config.cloud_bucket, obj.object_name)
                                data = response.read()
                                mobile_db.put(obj.object_name, data, skip_sync=True)
                                mobile_db.mark_synced(obj.object_name)
                                stats["downloaded"] += 1
                            except Exception as e:
                                stats["errors"].append(str(e))
                except Exception as e:
                    stats["errors"].append(str(e))

            print_status(f"Uploaded: {stats['uploaded']}, Downloaded: {stats['downloaded']}", "success")
            if stats["errors"]:
                print_status(f"Errors: {len(stats['errors'])}", "warning")

            mobile_db.close()
            return len(stats["errors"]) == 0

        except ImportError:
            print_status("MinIO SDK not installed (pip install minio)", "error")
            return False
        except Exception as e:
            print_status(f"Sync failed: {e}", "error")
            return False

    def _sync_desktop(self, name: str, config: ClusterConfig) -> bool:
        """Sync desktop MinIO with cloud"""
        # Setup aliases if not done
        local_config = MinIOConfig(
            port=config.port,
            access_key=config.access_key,
            secret_key=config.secret_key,
            host="127.0.0.1",
        )

        endpoint = config.cloud_endpoint.replace("http://", "").replace("https://", "")
        cloud_config = MinIOConfig(
            host=endpoint.split(":")[0],
            port=int(endpoint.split(":")[-1]) if ":" in endpoint else 9000,
            access_key=config.cloud_access_key,
            secret_key=config.cloud_secret_key,
            use_tls="https" in config.cloud_endpoint,
        )

        self.mc_client.set_alias("local", local_config)
        self.mc_client.set_alias("cloud", cloud_config)

        print_status("Running bidirectional sync...", "info")

        # One-shot mirror in both directions
        self.mc_client.start_mirror(f"local/{config.cloud_bucket}", f"cloud/{config.cloud_bucket}", watch=False)
        self.mc_client.start_mirror(f"cloud/{config.cloud_bucket}", f"local/{config.cloud_bucket}", watch=False)

        print_status("Sync complete", "success")
        return True

    # =================== Discovery/Info Commands ===================

    def cmd_list_buckets(self, name: str):
        """List buckets in an instance"""
        instance = self._get_instance(name)
        if not instance:
            print_status(f"Instance '{name}' not found", "error")
            return

        config = self.configs[name]

        print_box_header(f"Buckets in '{name}'", "📁")
        print_box_footer()

        try:
            from minio import Minio

            client = Minio(
                f"127.0.0.1:{config.port}",
                access_key=config.access_key,
                secret_key=config.secret_key,
                secure=False
            )

            buckets = client.list_buckets()
            for bucket in buckets:
                print_status(f"{bucket.name} (created: {bucket.creation_date})", "info")

        except Exception as e:
            print_status(f"Failed to list buckets: {e}", "error")

    def cmd_info(self):
        """Show system and installation info"""
        print_box_header("MinIO Manager Info", "ℹ️")
        print_box_content(f"System: {platform.system()} {platform.machine()}", "info")
        print_box_content(f"Python: {platform.python_version()}", "info")
        print_box_content(f"Config: {self.config_path}", "info")
        print_box_content(f"Base Dir: {self.base_dir}", "info")
        print_box_footer()

        minio_path = self.installer.get_minio_path()
        mc_path = self.installer.get_mc_path()

        print_status(f"MinIO Server: {'✓ ' + str(minio_path) if minio_path else '✗ Not installed'}",
                     "success" if minio_path else "warning")
        print_status(f"MinIO Client: {'✓ ' + str(mc_path) if mc_path else '✗ Not installed'}",
                     "success" if mc_path else "warning")

        if minio_path:
            version = self.installer.get_version()
            print_status(f"Version: {version or 'unknown'}", "info")


# =================== CLI Entry Point ===================

async def cli_db_runner():
    """Main CLI entry point"""

    parser = argparse.ArgumentParser(
        description="🗄️  MinIO Blob Storage Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='tb db',
        epilog="""
╔════════════════════════════════════════════════════════════════════════════╗
║                           Command Examples                                 ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Installation:                                                             ║
║    $ tb db install                    # Install MinIO server & client      ║
║    $ tb db info                       # Show installation info             ║
║                                                                            ║
║  Server Setup (Cloud):                                                     ║
║    $ tb db setup-server --name cloud --port 9000                           ║
║    $ tb db setup-server --name cloud --docker   # Use Docker               ║
║                                                                            ║
║  Desktop Setup (Local + Cloud Sync):                                       ║
║    $ tb db setup-desktop --name local                                      ║
║    $ tb db setup-desktop --cloud-endpoint https://cloud.example.com:9000   ║
║                         --cloud-access-key admin                           ║
║                         --cloud-secret-key SecurePass                      ║
║                                                                            ║
║  Mobile Setup (SQLite + Sync):                                             ║
║    $ tb db setup-mobile --name phone --max-size 500                        ║
║                                                                            ║
║  Replication (Server to Server):                                           ║
║    $ tb db setup-replication --source server1 --target server2             ║
║                                                                            ║
║  Instance Management:                                                      ║
║    $ tb db start                      # Start all instances                ║
║    $ tb db stop --name local          # Stop specific instance             ║
║    $ tb db status                     # Show instance status               ║
║    $ tb db health                     # Health check all instances         ║
║                                                                            ║
║  Sync:                                                                     ║
║    $ tb db sync --name mobile         # Manual sync for mobile/desktop     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
        """
    )

    subparsers = parser.add_subparsers(dest="action", help="Available commands")

    # Install command
    p_install = subparsers.add_parser('install', help='Install MinIO binaries')
    p_install.add_argument('--components', nargs='+', choices=['server', 'client', 'all'],
                           default=['all'], help='Components to install')

    # Uninstall command
    subparsers.add_parser('uninstall', help='Uninstall MinIO binaries')

    # Info command
    subparsers.add_parser('info', help='Show system and installation info')

    # Setup server command
    p_server = subparsers.add_parser('setup-server', help='Setup a central MinIO server')
    p_server.add_argument('--name', default='cloud', help='Server name')
    p_server.add_argument('--port', type=int, default=9000, help='Server port')
    p_server.add_argument('--access-key', default='admin', help='Access key')
    p_server.add_argument('--secret-key', default='SecureCloudPass', help='Secret key')
    p_server.add_argument('--host', default='0.0.0.0', help='Bind host')
    p_server.add_argument('--docker', action='store_true', help='Use Docker')

    # Setup desktop command
    p_desktop = subparsers.add_parser('setup-desktop', help='Setup desktop with local MinIO')
    p_desktop.add_argument('--name', default='local', help='Instance name')
    p_desktop.add_argument('--cloud-endpoint', help='Cloud MinIO endpoint for sync')
    p_desktop.add_argument('--cloud-access-key', help='Cloud access key')
    p_desktop.add_argument('--cloud-secret-key', help='Cloud secret key')
    p_desktop.add_argument('--no-sync', action='store_true', help='Disable auto-sync')

    # Setup mobile command
    p_mobile = subparsers.add_parser('setup-mobile', help='Setup mobile SQLite database')
    p_mobile.add_argument('--name', default='mobile', help='Database name')
    p_mobile.add_argument('--max-size', type=int, default=500, help='Max database size in MB')
    p_mobile.add_argument('--cloud-endpoint', help='Cloud MinIO endpoint for sync')
    p_mobile.add_argument('--cloud-access-key', help='Cloud access key')
    p_mobile.add_argument('--cloud-secret-key', help='Cloud secret key')

    # Setup replication command
    p_repl = subparsers.add_parser('setup-replication', help='Setup server-to-server replication')
    p_repl.add_argument('--source', required=True, help='Source server name')
    p_repl.add_argument('--target', required=True, help='Target server name')

    # Instance management commands
    p_start = subparsers.add_parser('start', help='Start instance(s)')
    p_start.add_argument('--name', help='Instance name (all if omitted)')

    p_stop = subparsers.add_parser('stop', help='Stop instance(s)')
    p_stop.add_argument('--name', help='Instance name (all if omitted)')

    p_restart = subparsers.add_parser('restart', help='Restart an instance')
    p_restart.add_argument('--name', required=True, help='Instance name')

    p_status = subparsers.add_parser('status', help='Show instance status')
    p_status.add_argument('--name', help='Instance name (all if omitted)')

    p_health = subparsers.add_parser('health', help='Health check instance(s)')
    p_health.add_argument('--name', help='Instance name (all if omitted)')

    # Sync command
    p_sync = subparsers.add_parser('sync', help='Manual sync with cloud')
    p_sync.add_argument('--name', required=True, help='Instance name')

    # Buckets command
    p_buckets = subparsers.add_parser('buckets', help='List buckets in an instance')
    p_buckets.add_argument('--name', required=True, help='Instance name')

    # Parse arguments
    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        return

    # Create manager
    manager = MinIOCLIManager()

    # Execute command
    if args.action == 'install':
        manager.cmd_install(args.components)

    elif args.action == 'uninstall':
        manager.cmd_uninstall()

    elif args.action == 'info':
        manager.cmd_info()

    elif args.action == 'setup-server':
        manager.cmd_setup_server(
            name=args.name,
            port=args.port,
            access_key=args.access_key,
            secret_key=args.secret_key,
            host=args.host,
            use_docker=args.docker
        )

    elif args.action == 'setup-desktop':
        manager.cmd_setup_desktop(
            name=args.name,
            cloud_endpoint=args.cloud_endpoint,
            cloud_access_key=args.cloud_access_key,
            cloud_secret_key=args.cloud_secret_key,
            auto_sync=not args.no_sync
        )

    elif args.action == 'setup-mobile':
        manager.cmd_setup_mobile(
            name=args.name,
            cloud_endpoint=args.cloud_endpoint,
            cloud_access_key=args.cloud_access_key,
            cloud_secret_key=args.cloud_secret_key,
            max_size_mb=args.max_size
        )

    elif args.action == 'setup-replication':
        manager.cmd_setup_replication(args.source, args.target)

    elif args.action == 'start':
        manager.cmd_start(args.name)

    elif args.action == 'stop':
        manager.cmd_stop(args.name)

    elif args.action == 'restart':
        manager.cmd_restart(args.name)

    elif args.action == 'status':
        manager.cmd_status(args.name)

    elif args.action == 'health':
        manager.cmd_health(args.name)

    elif args.action == 'sync':
        manager.cmd_sync(args.name)

    elif args.action == 'buckets':
        manager.cmd_list_buckets(args.name)


if __name__ == "__main__":
    asyncio.run(cli_db_runner())
