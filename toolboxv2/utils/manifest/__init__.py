"""
ToolBoxV2 Manifest System
=========================

Unified configuration management through tb-manifest.yaml.

Components:
- schema: Pydantic models for manifest validation
- loader: Load, validate, and apply environment overrides
- converter: Generate sub-configs (.config.yaml, config.toml, etc.)
- service_manager: Start/stop services based on manifest
"""

from .schema import (
    TBManifest,
    AppConfig,
    AutostartConfig,
    ModsConfig,
    DatabaseConfig,
    ServicesConfig,
    WorkersConfig,
    NginxConfig,
    AuthConfig,
    PathsConfig,
    RegistryConfig,
    ToolboxConfig,
    UtilitiesConfig,
    IsaaConfig,
)
from .loader import ManifestLoader
from .converter import ConfigConverter
from .service_manager import ManifestServiceManager, ServiceSyncResult, run_manifest_startup

__all__ = [
    "TBManifest",
    "AppConfig",
    "AutostartConfig",
    "ModsConfig",
    "DatabaseConfig",
    "ServicesConfig",
    "WorkersConfig",
    "NginxConfig",
    "AuthConfig",
    "PathsConfig",
    "RegistryConfig",
    "ToolboxConfig",
    "UtilitiesConfig",
    "IsaaConfig",
    "ManifestLoader",
    "ConfigConverter",
    "ManifestServiceManager",
    "ServiceSyncResult",
    "run_manifest_startup",
]

