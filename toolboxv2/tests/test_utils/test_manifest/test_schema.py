"""
Tests for toolboxv2.utils.manifest.schema
==========================================
"""

import pytest
from pydantic import ValidationError

from toolboxv2.utils.manifest.schema import (
    TBManifest,
    AppConfig,
    AutostartConfig,
    ModsConfig,
    DatabaseConfig,
    DatabaseMode,
    ServicesConfig,
    WorkersConfig,
    HTTPWorkerInstance,
    WSWorkerInstance,
    NginxConfig,
    AuthConfig,
    AuthProvider,
    PathsConfig,
    RegistryConfig,
    ToolboxConfig,
    IsaaConfig,
    Environment,
    LogLevel,
    _apply_nested_override,
)


class TestAppConfig:
    """Tests for AppConfig model."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = AppConfig()
        assert config.name == "ToolBoxV2"
        assert config.version == "0.1.0"
        assert config.instance_id == "tbv2_main"
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is False
        assert config.log_level == LogLevel.INFO

    def test_custom_values(self):
        """Test custom values are accepted."""
        config = AppConfig(
            name="MyApp",
            version="1.0.0",
            environment=Environment.PRODUCTION,
            debug=True,
            log_level=LogLevel.DEBUG,
        )
        assert config.name == "MyApp"
        assert config.environment == Environment.PRODUCTION
        assert config.debug is True


class TestAutostartConfig:
    """Tests for AutostartConfig model."""

    def test_default_values(self):
        """Test default values."""
        config = AutostartConfig()
        assert config.enabled is False
        assert config.services == []
        assert config.commands == []

    def test_with_services_and_commands(self):
        """Test with services and commands."""
        config = AutostartConfig(
            enabled=True,
            services=["workers", "db"],
            commands=["tb -v", "tb status"],
        )
        assert config.enabled is True
        assert "workers" in config.services
        assert "tb -v" in config.commands


class TestDatabaseConfig:
    """Tests for DatabaseConfig model."""

    def test_default_mode_is_lc(self):
        """Test default mode is LOCAL_DICT."""
        config = DatabaseConfig()
        assert config.mode == DatabaseMode.LC

    def test_all_modes_valid(self):
        """Test all database modes are valid."""
        for mode in DatabaseMode:
            config = DatabaseConfig(mode=mode)
            assert config.mode == mode


class TestWorkersConfig:
    """Tests for WorkersConfig model."""

    def test_default_workers(self):
        """Test default worker configuration."""
        config = WorkersConfig()
        assert len(config.http) == 1
        assert len(config.websocket) == 1
        assert config.http[0].port == 8000
        assert config.websocket[0].port == 8100

    def test_multiple_workers(self):
        """Test multiple worker instances."""
        config = WorkersConfig(
            http=[
                HTTPWorkerInstance(name="http1", port=8000),
                HTTPWorkerInstance(name="http2", port=8001),
            ],
            websocket=[
                WSWorkerInstance(name="ws1", port=8100),
            ],
        )
        assert len(config.http) == 2
        assert config.http[1].port == 8001


class TestTBManifest:
    """Tests for main TBManifest model."""

    def test_default_manifest(self):
        """Test default manifest creation."""
        manifest = TBManifest()
        assert manifest.manifest_version == "1.0.0"
        assert manifest.app.name == "ToolBoxV2"
        assert manifest.database.mode == DatabaseMode.LC
        assert manifest.isaa is None

    def test_isaa_auto_enabled_when_installed(self):
        """Test ISAA config is auto-created when isaa is in installed mods."""
        manifest = TBManifest(
            mods={"installed": {"isaa": "^0.1.0"}}
        )
        assert manifest.isaa is not None
        assert manifest.isaa.enabled is True

    def test_environment_overrides(self):
        """Test environment-specific overrides."""
        manifest = TBManifest(
            app={"environment": "production"}
        )
        effective = manifest.get_effective_config()
        # Production should have debug=False
        assert effective.app.debug is False


class TestApplyNestedOverride:
    """Tests for _apply_nested_override helper."""

    def test_simple_override(self):
        """Test simple nested override."""
        data = {"app": {"debug": False}}
        _apply_nested_override(data, "app.debug", True)
        assert data["app"]["debug"] is True

    def test_deep_override(self):
        """Test deeply nested override."""
        data = {"auth": {"session": {"cookie_secure": False}}}
        _apply_nested_override(data, "auth.session.cookie_secure", True)
        assert data["auth"]["session"]["cookie_secure"] is True

