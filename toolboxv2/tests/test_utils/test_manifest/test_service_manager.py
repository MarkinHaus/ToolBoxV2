"""Tests for ManifestServiceManager."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from toolboxv2.utils.manifest.schema import TBManifest, AutostartConfig
from toolboxv2.utils.manifest.service_manager import (
    ManifestServiceManager,
    ServiceSyncResult,
)


class TestManifestServiceManager:
    """Tests for ManifestServiceManager class."""

    def test_get_enabled_services_from_manifest(self):
        """Test getting enabled services from manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest with autostart services
            manifest_content = """
manifest_version: "1.0"
autostart:
  enabled: true
  services:
    - workers
    - db
  commands:
    - echo "test"
"""
            manifest_path = Path(tmpdir) / "tb-manifest.yaml"
            manifest_path.write_text(manifest_content, encoding="utf-8")

            # Create manager with explicit root_dir
            manager = ManifestServiceManager(Path(tmpdir))
            # Replace service_manager with mock
            manager.service_manager = MagicMock()

            enabled = manager.get_enabled_services()
            assert "workers" in enabled
            assert "db" in enabled
            assert len(enabled) == 2

    def test_get_enabled_services_when_disabled(self):
        """Test that no services returned when autostart disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_content = """
manifest_version: "1.0"
autostart:
  enabled: false
  services:
    - workers
"""
            manifest_path = Path(tmpdir) / "tb-manifest.yaml"
            manifest_path.write_text(manifest_content, encoding="utf-8")

            manager = ManifestServiceManager(Path(tmpdir))
            manager.service_manager = MagicMock()

            enabled = manager.get_enabled_services()
            assert enabled == []

    def test_get_autostart_commands(self):
        """Test getting autostart commands from manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_content = """
manifest_version: "1.0"
autostart:
  enabled: true
  services: []
  commands:
    - "tb -v"
    - "echo hello"
"""
            manifest_path = Path(tmpdir) / "tb-manifest.yaml"
            manifest_path.write_text(manifest_content, encoding="utf-8")

            manager = ManifestServiceManager(Path(tmpdir))
            manager.service_manager = MagicMock()

            commands = manager.get_autostart_commands()
            assert "tb -v" in commands
            assert "echo hello" in commands

    def test_sync_services_dry_run(self):
        """Test sync_services in dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_content = """
manifest_version: "1.0"
autostart:
  enabled: true
  services:
    - workers
  commands:
    - "echo test"
"""
            manifest_path = Path(tmpdir) / "tb-manifest.yaml"
            manifest_path.write_text(manifest_content, encoding="utf-8")

            # Create mock ServiceManager
            mock_sm = MagicMock()
            mock_sm.get_all_status.return_value = {}  # No services running

            manager = ManifestServiceManager(Path(tmpdir))
            manager.service_manager = mock_sm

            result = manager.sync_services(dry_run=True)

            assert "workers" in result.started
            assert "echo test" in result.commands_executed
            # In dry run, start_service should NOT be called
            mock_sm.start_service.assert_not_called()

    def test_get_status_report(self):
        """Test status report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_content = """
manifest_version: "1.0"
autostart:
  enabled: true
  services:
    - workers
"""
            manifest_path = Path(tmpdir) / "tb-manifest.yaml"
            manifest_path.write_text(manifest_content, encoding="utf-8")

            # Create mock ServiceManager
            mock_sm = MagicMock()
            mock_sm.get_all_status.return_value = {
                "workers": {"running": True, "pid": 1234, "auto_start": True},
                "db": {"running": False, "pid": None, "auto_start": False},
            }

            manager = ManifestServiceManager(Path(tmpdir))
            manager.service_manager = mock_sm

            report = manager.get_status_report()

            assert "workers" in report
            assert report["workers"]["in_manifest"] is True
            assert report["workers"]["status"] == "running"

            assert "db" in report
            assert report["db"]["in_manifest"] is False

    def test_no_manifest_returns_defaults(self):
        """Test behavior when no manifest exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ManifestServiceManager(Path(tmpdir))
            manager.service_manager = MagicMock()

            # Should return empty list when no manifest
            enabled = manager.get_enabled_services()
            assert enabled == []

            commands = manager.get_autostart_commands()
            assert commands == []

