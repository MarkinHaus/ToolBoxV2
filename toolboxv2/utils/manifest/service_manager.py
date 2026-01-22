"""
ManifestServiceManager - Service management based on tb-manifest.yaml

Extends the existing ServiceManager to work with the manifest system.
Provides synchronization between manifest configuration and running services.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .schema import TBManifest, AutostartConfig
from .loader import ManifestLoader


@dataclass
class ServiceSyncResult:
    """Result of synchronizing services with manifest."""
    started: List[str]
    stopped: List[str]
    already_running: List[str]
    failed: Dict[str, str]  # name -> error message
    commands_executed: List[str]


class ManifestServiceManager:
    """
    Service manager that uses tb-manifest.yaml as source of truth.

    Wraps the existing ServiceManager and adds manifest-aware functionality.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize with optional base directory.

        Args:
            base_dir: Directory containing tb-manifest.yaml. Defaults to tb_root_dir.
        """
        from toolboxv2 import tb_root_dir
        self.base_dir = Path(base_dir) if base_dir else tb_root_dir
        self.loader = ManifestLoader(self.base_dir)

        # Import existing ServiceManager
        from toolboxv2.utils.clis.service_manager import ServiceManager
        self.service_manager = ServiceManager()

    def get_manifest(self) -> Optional[TBManifest]:
        """Load and return the manifest, or None if not found."""
        if not self.loader.exists():
            return None
        try:
            return self.loader.load()
        except Exception:
            return None

    def get_autostart_config(self) -> AutostartConfig:
        """Get autostart configuration from manifest or defaults."""
        manifest = self.get_manifest()
        if manifest:
            return manifest.autostart
        return AutostartConfig()

    def get_enabled_services(self) -> List[str]:
        """Get list of services that should be running according to manifest."""
        autostart = self.get_autostart_config()
        if not autostart.enabled:
            return []
        return autostart.services

    def get_autostart_commands(self) -> List[str]:
        """Get list of commands to execute on startup."""
        autostart = self.get_autostart_config()
        if not autostart.enabled:
            return []
        return autostart.commands

    def get_running_services(self) -> Dict[str, int]:
        """Get currently running services with their PIDs."""
        status = self.service_manager.get_all_status(include_registry=False)
        return {
            name: info["pid"]
            for name, info in status.items()
            if info["running"] and info["pid"]
        }

    def sync_services(self, dry_run: bool = False) -> ServiceSyncResult:
        """
        Synchronize running services with manifest configuration.

        Starts services that should be running but aren't.
        Optionally stops services that are running but shouldn't be.

        Args:
            dry_run: If True, don't actually start/stop, just report what would happen.

        Returns:
            ServiceSyncResult with details of what was done.
        """
        result = ServiceSyncResult(
            started=[],
            stopped=[],
            already_running=[],
            failed={},
            commands_executed=[]
        )

        enabled = set(self.get_enabled_services())
        running = self.get_running_services()
        running_names = set(running.keys())

        # Services to start
        to_start = enabled - running_names

        # Services already running
        result.already_running = list(enabled & running_names)

        # Start missing services
        for name in to_start:
            if dry_run:
                result.started.append(name)
            else:
                start_result = self.service_manager.start_service(name)
                if start_result.success:
                    result.started.append(name)
                else:
                    result.failed[name] = start_result.error or "Unknown error"

        # Execute autostart commands
        commands = self.get_autostart_commands()
        for cmd in commands:
            if dry_run:
                result.commands_executed.append(cmd)
            else:
                try:
                    import subprocess
                    subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    result.commands_executed.append(cmd)
                except Exception as e:
                    result.failed[f"cmd:{cmd}"] = str(e)

        return result

    def start_from_manifest(self) -> ServiceSyncResult:
        """Start all services defined in manifest autostart."""
        return self.sync_services(dry_run=False)

    def get_status_report(self) -> Dict[str, dict]:
        """
        Get comprehensive status report comparing manifest vs reality.

        Returns dict with service name -> status info.
        """
        enabled = set(self.get_enabled_services())
        all_status = self.service_manager.get_all_status(include_registry=True)

        report = {}
        for name, info in all_status.items():
            report[name] = {
                **info,
                "in_manifest": name in enabled,
                "should_run": name in enabled,
                "status": self._get_status_string(
                    running=info["running"],
                    should_run=name in enabled
                )
            }

        # Add services in manifest but not in registry
        for name in enabled:
            if name not in report:
                report[name] = {
                    "name": name,
                    "running": False,
                    "pid": None,
                    "in_manifest": True,
                    "should_run": True,
                    "status": "not_started",
                    "registered": False
                }

        return report

    def _get_status_string(self, running: bool, should_run: bool) -> str:
        """Get human-readable status string."""
        if running and should_run:
            return "running"
        elif running and not should_run:
            return "running_extra"  # Running but not in manifest
        elif not running and should_run:
            return "not_started"  # Should be running but isn't
        else:
            return "stopped"  # Not running and shouldn't be

    def stop_all_manifest_services(self, graceful: bool = True) -> List[str]:
        """Stop all services defined in manifest."""
        stopped = []
        for name in self.get_enabled_services():
            if self.service_manager.stop_service(name, graceful=graceful):
                stopped.append(name)
        return stopped

    def restart_manifest_services(self) -> ServiceSyncResult:
        """Restart all manifest-defined services."""
        # Stop all first
        self.stop_all_manifest_services(graceful=True)

        # Wait a moment
        import time
        time.sleep(1)

        # Start again
        return self.start_from_manifest()

    def apply_manifest_to_services_json(self) -> Path:
        """
        Update services.json to match manifest autostart configuration.

        This syncs the manifest autostart.services to the ServiceManager's
        internal configuration.
        """
        manifest = self.get_manifest()
        if not manifest:
            raise ValueError("No manifest found")

        config = self.service_manager.load_config()
        services = config.get("services", {})

        # Update auto_start flags based on manifest
        enabled = set(manifest.autostart.services)

        for name in enabled:
            if name not in services:
                services[name] = {}
            services[name]["auto_start"] = manifest.autostart.enabled

        # Disable auto_start for services not in manifest
        for name in services:
            if name not in enabled:
                services[name]["auto_start"] = False

        config["services"] = services

        # Add autostart commands to config
        config["autostart"] = {
            "enabled": manifest.autostart.enabled,
            "services": manifest.autostart.services,
            "commands": manifest.autostart.commands
        }

        # Add database mode
        config["database"] = {
            "mode": manifest.database.mode.value
        }

        self.service_manager.save_config(config)
        return self.service_manager.config_path


def run_manifest_startup() -> int:
    """
    Entry point for manifest-based service startup.

    Can be called from `tb --sm` when manifest exists.

    Returns:
        Exit code: 0 = success, 1 = partial failure, 2 = complete failure
    """
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header, print_box_footer, print_status
    )

    manager = ManifestServiceManager()
    manifest = manager.get_manifest()

    if not manifest:
        print_status("No tb-manifest.yaml found, using legacy services.json", "info")
        # Fall back to legacy behavior
        from toolboxv2.utils.clis.service_manager import run_service_manager_startup
        return run_service_manager_startup()

    print_box_header("ToolBoxV2 Manifest Service Manager", "📋")
    print_status(f"Environment: {manifest.app.environment.value}", "info")
    print_status(f"DB Mode: {manifest.database.mode.value}", "info")
    print()

    # Sync services
    result = manager.sync_services()

    # Report results
    if result.already_running:
        for name in result.already_running:
            print_status(f"{name}: already running", "info")

    if result.started:
        for name in result.started:
            print_status(f"{name}: started", "success")

    if result.commands_executed:
        for cmd in result.commands_executed:
            print_status(f"Executed: {cmd}", "success")

    if result.failed:
        for name, error in result.failed.items():
            print_status(f"{name}: {error}", "error")

    print()

    # Summary
    total = len(result.started) + len(result.already_running)
    failed = len(result.failed)

    if failed == 0:
        print_status(f"All {total} service(s) ready", "success")
        print_box_footer()
        return 0
    elif total > 0:
        print_status(f"{total} ready, {failed} failed", "warning")
        print_box_footer()
        return 1
    else:
        print_status(f"All {failed} service(s) failed", "error")
        print_box_footer()
        return 2

