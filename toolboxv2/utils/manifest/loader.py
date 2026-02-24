"""
ManifestLoader - Load, validate, and manage tb-manifest.yaml
=============================================================

Responsibilities:
- Load manifest from file
- Validate against schema
- Apply environment overrides
- Generate default manifest
- Resolve environment variables in values
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .schema import TBManifest, Environment


# =============================================================================
# Environment Variable Resolution
# =============================================================================


ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def resolve_env_vars(value: Any) -> Any:
    """
    Resolve environment variables in a value.

    Supports:
    - ${VAR_NAME} - Required variable
    - ${VAR_NAME:default} - Variable with default
    """
    if isinstance(value, str):
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            return match.group(0)  # Keep original if no value and no default

        return ENV_VAR_PATTERN.sub(replacer, value)

    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]

    return value


# =============================================================================
# ManifestLoader
# =============================================================================


class ManifestLoader:
    """
    Load and manage tb-manifest.yaml configuration.

    Usage:
        loader = ManifestLoader()
        manifest = loader.load()  # Load from default path
        manifest = loader.load("custom/path/tb-manifest.yaml")

        # Get effective config with env overrides applied
        effective = manifest.get_effective_config()
    """

    DEFAULT_FILENAME = "tb-manifest.yaml"

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize loader.

        Args:
            base_dir: Base directory for manifest. Defaults to current working directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._manifest: Optional[TBManifest] = None
        self._manifest_path: Optional[Path] = None

    @property
    def manifest_path(self) -> Path:
        """Get the manifest file path."""
        if self._manifest_path:
            return self._manifest_path
        if (self.base_dir / self.DEFAULT_FILENAME).exists():
            return self.base_dir / self.DEFAULT_FILENAME
        if (self.base_dir / ".config.yaml").exists():
            return self.base_dir / ".config.yaml"
        return self.base_dir / self.DEFAULT_FILENAME

    @property
    def manifest(self) -> Optional[TBManifest]:
        """Get the loaded manifest."""
        return self._manifest

    def exists(self) -> bool:
        """Check if manifest file exists."""
        return self.manifest_path.exists()

    def load(self, path: Optional[str] = None, resolve_env: bool = True) -> TBManifest:
        """
        Load manifest from file.

        Args:
            path: Optional custom path to manifest file
            resolve_env: Whether to resolve environment variables

        Returns:
            Loaded and validated TBManifest

        Raises:
            FileNotFoundError: If manifest file doesn't exist
            ValidationError: If manifest is invalid
        """
        if path:
            self._manifest_path = Path(path)

        manifest_file = self.manifest_path

        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_file}\n"
                f"Run 'tb init config' to create one."
            )

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._manifest = TBManifest.model_validate(data)
        if resolve_env:
            resolved_data = resolve_env_vars(self._manifest.model_dump())
            self._manifest = TBManifest.model_validate(resolved_data)

        return self._manifest

    def load_or_create_default(self, resolve_env: bool = True) -> TBManifest:
        """Load manifest or create default if not exists."""
        if self.exists():
            return self.load(resolve_env=resolve_env)
        return self.create_default()

    def create_default(self, save: bool = True) -> TBManifest:
        """
        Create a minimal default manifest.

        Args:
            save: Whether to save the manifest to file

        Returns:
            Default TBManifest
        """
        manifest = TBManifest(
            manifest_version="1.0.0",
            app={"name": "ToolBoxV2", "environment": "development"},
            mods={
                "installed": {"CloudM": "^0.1.0", "DB": "^0.0.3"},
                "init_modules": ["CloudM", "DB"],
                "open_modules": ["CloudM.AuthHelper", "CloudM.Auth"],
            },
            database={"mode": "LC"},
            services={"enabled": ["workers", "db"]},
        )

        self._manifest = manifest

        if save:
            self.save(manifest)

        return manifest

    def save(self, manifest: Optional[TBManifest] = None) -> Path:
        """
        Save manifest to file.

        Args:
            manifest: Manifest to save. Uses loaded manifest if not provided.

        Returns:
            Path to saved file
        """
        if manifest is None:
            manifest = self._manifest

        if manifest is None:
            raise ValueError("No manifest to save. Load or create one first.")

        # Convert to dict, excluding None values
        data = manifest.model_dump(exclude_none=True, exclude_unset=False)

        # Add header comment
        yaml_content = self._generate_yaml_with_comments(data)

        manifest_file = self.manifest_path
        manifest_file.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        self._manifest = manifest
        return manifest_file

    def _generate_yaml_with_comments(self, data: dict) -> str:
        """Generate YAML with helpful comments."""
        header = '''# ═══════════════════════════════════════════════════════════════════════════════
# tb-manifest.yaml - ToolBoxV2 Unified Configuration Manifest
# ═══════════════════════════════════════════════════════════════════════════════
# This is the single source of truth for your ToolBox installation.
# Changes here are applied to all sub-configs when saved.
#
# Generate with: tb init config
# Validate with: tb manifest validate
# Apply with:    tb manifest apply
#
# Database Modes:
#   LC = LOCAL_DICT    - Local JSON file (no config needed)
#   LR = LOCAL_REDIS   - Local Redis (needs redis.url)
#   RR = REMOTE_REDIS  - Remote Redis (needs redis.url or credentials)
#   CB = CLUSTER_BLOB  - MinIO storage (needs minio.* config)
# ═══════════════════════════════════════════════════════════════════════════════

'''
        yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return header + yaml_str

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the current manifest.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        if self._manifest is None:
            return False, ["No manifest loaded"]

        manifest = self._manifest

        # Check mod dependencies
        for mod, deps in manifest.mods.dependencies.items():
            if mod not in manifest.mods.installed:
                errors.append(f"Dependency defined for non-installed mod: {mod}")
            for dep in deps:
                # Parse dependency spec (e.g., "CloudM>=0.1.0")
                dep_name = re.split(r"[<>=!]", dep)[0]
                if dep_name not in manifest.mods.installed:
                    errors.append(f"Mod '{mod}' depends on '{dep_name}' which is not installed")

        # Check init_modules are installed
        for mod in manifest.mods.init_modules:
            base_mod = mod.split(".")[0]
            if base_mod not in manifest.mods.installed:
                errors.append(f"init_module '{mod}' is not installed")

        # Check open_modules are installed
        for mod in manifest.mods.open_modules:
            base_mod = mod.split(".")[0]
            if base_mod not in manifest.mods.installed:
                errors.append(f"open_module '{mod}' is not installed")

        # Check database mode requirements
        mode = manifest.database.mode
        if mode.value in ("LR", "RR"):
            redis = manifest.database.redis
            if "${" in redis.url and not os.environ.get("DB_CONNECTION_URI"):
                errors.append(f"Database mode {mode.value} requires DB_CONNECTION_URI env var or redis.url")

        # Check SSL config if enabled
        if manifest.nginx.ssl_enabled:
            if not manifest.nginx.ssl_certificate:
                errors.append("nginx.ssl_enabled is true but ssl_certificate is not set")
            if not manifest.nginx.ssl_certificate_key:
                errors.append("nginx.ssl_enabled is true but ssl_certificate_key is not set")

        # Check worker ports don't conflict
        all_ports = []
        for http in manifest.workers.http:
            if http.port in all_ports:
                errors.append(f"Duplicate port {http.port} in HTTP workers")
            all_ports.append(http.port)
        for ws in manifest.workers.websocket:
            if ws.port in all_ports:
                errors.append(f"Duplicate port {ws.port} (conflicts with HTTP worker)")
            all_ports.append(ws.port)

        return len(errors) == 0, errors

    def get_effective_manifest(self) -> TBManifest:
        """Get manifest with environment overrides applied."""
        if self._manifest is None:
            raise ValueError("No manifest loaded")
        return self._manifest.get_effective_config()

