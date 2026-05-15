"""
Tests for toolboxv2.utils.manifest.loader
==========================================
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from toolboxv2.utils.manifest.loader import (
    ManifestLoader,
    resolve_env_vars,
    ENV_VAR_PATTERN,
)
from toolboxv2.utils.manifest.schema import TBManifest, DatabaseMode


class TestResolveEnvVars:
    """Tests for environment variable resolution."""

    def test_simple_env_var(self, monkeypatch):
        """Test simple environment variable resolution."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = resolve_env_vars("${TEST_VAR}")
        assert result == "test_value"

    def test_env_var_with_default(self):
        """Test environment variable with default value."""
        # Ensure var doesn't exist
        os.environ.pop("NONEXISTENT_VAR", None)
        result = resolve_env_vars("${NONEXISTENT_VAR:default_value}")
        assert result == "default_value"

    def test_env_var_override_default(self, monkeypatch):
        """Test that env var overrides default."""
        monkeypatch.setenv("MY_VAR", "from_env")
        result = resolve_env_vars("${MY_VAR:default}")
        assert result == "from_env"

    def test_nested_dict_resolution(self, monkeypatch):
        """Test resolution in nested dictionaries."""
        monkeypatch.setenv("DB_HOST", "localhost")
        data = {
            "database": {
                "host": "${DB_HOST:127.0.0.1}",
                "port": 5432,
            }
        }
        result = resolve_env_vars(data)
        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == 5432

    def test_list_resolution(self, monkeypatch):
        """Test resolution in lists."""
        monkeypatch.setenv("ITEM1", "value1")
        data = ["${ITEM1}", "${ITEM2:default2}"]
        result = resolve_env_vars(data)
        assert result == ["value1", "default2"]


class TestManifestLoader:
    """Tests for ManifestLoader class."""

    def test_exists_false_when_no_file(self):
        """Test exists() returns False when no manifest file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ManifestLoader(tmpdir)
            assert loader.exists() is False

    def test_create_default_manifest(self):
        """Test creating default manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ManifestLoader(tmpdir)
            manifest = loader.create_default(save=True)

            assert manifest.manifest_version == "1.0.0"
            assert manifest.app.name == "ToolBoxV2"
            assert loader.exists() is True

    def test_load_manifest(self):
        """Test loading manifest from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a manifest file
            manifest_path = Path(tmpdir) / "tb-manifest.yaml"
            manifest_data = {
                "manifest_version": "1.0.0",
                "app": {
                    "name": "TestApp",
                    "environment": "production",
                },
                "database": {
                    "mode": "CB",
                },
            }
            with open(manifest_path, "w") as f:
                yaml.dump(manifest_data, f)

            loader = ManifestLoader(tmpdir)
            manifest = loader.load()

            assert manifest.app.name == "TestApp"
            assert manifest.database.mode == DatabaseMode.CB

    def test_load_nonexistent_raises(self):
        """Test loading nonexistent manifest raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ManifestLoader(tmpdir)
            with pytest.raises(FileNotFoundError):
                loader.load()

    def test_load_or_create_default(self):
        """Test load_or_create_default creates when missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ManifestLoader(tmpdir)
            manifest = loader.load_or_create_default()

            assert manifest is not None
            assert loader.exists() is True

    def test_save_manifest(self):
        """Test saving manifest to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ManifestLoader(tmpdir)
            manifest = TBManifest(
                app={"name": "SavedApp"}
            )

            path = loader.save(manifest)

            assert path.exists()
            with open(path, encoding="utf-8") as f:
                content = f.read()
                assert "SavedApp" in content

    def test_validate_valid_manifest(self):
        """Test validation of valid manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ManifestLoader(tmpdir)
            loader.create_default(save=False)

            is_valid, errors = loader.validate()
            assert is_valid is True
            assert errors == []

    def test_validate_missing_dependency(self):
        """Test validation catches missing dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ManifestLoader(tmpdir)
            manifest = TBManifest(
                mods={
                    "installed": {"ModA": "^1.0.0"},
                    "dependencies": {"ModA": ["ModB>=1.0.0"]},  # ModB not installed
                }
            )
            loader._manifest = manifest

            is_valid, errors = loader.validate()
            assert is_valid is False
            assert any("ModB" in e for e in errors)

    def test_validate_duplicate_ports(self):
        """Test validation catches duplicate ports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ManifestLoader(tmpdir)
            manifest = TBManifest(
                workers={
                    "http": [
                        {"name": "http1", "port": 8000},
                        {"name": "http2", "port": 8000},  # Duplicate!
                    ]
                }
            )
            loader._manifest = manifest

            is_valid, errors = loader.validate()
            assert is_valid is False
            assert any("Duplicate port" in e for e in errors)

