"""
Tests for toolboxv2.utils.manifest.converter
=============================================
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from toolboxv2.utils.manifest.converter import ConfigConverter
from toolboxv2.utils.manifest.schema import TBManifest, DatabaseMode


class TestConfigConverter:
    """Tests for ConfigConverter class."""

    @pytest.fixture
    def manifest(self):
        """Create a test manifest."""
        return TBManifest(
            app={"name": "TestApp", "instance_id": "test_instance"},
            mods={
                "installed": {"CloudM": "^0.1.0"},
                "init_modules": ["CloudM"],
                "open_modules": ["CloudM.AuthHelper"],
            },
            database={"mode": "LC"},
            workers={
                "http": [{"name": "http_main", "port": 8000, "workers": 4}],
                "websocket": [{"name": "ws_main", "port": 8100}],
            },
            autostart={
                "enabled": True,
                "services": ["workers", "db"],
                "commands": ["tb -v"],
            },
        )

    def test_apply_all_generates_files(self, manifest):
        """Test apply_all generates all config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = ConfigConverter(manifest, Path(tmpdir))
            generated = converter.apply_all()

            assert len(generated) >= 3

            # Check .config.yaml exists
            config_yaml = Path(tmpdir) / ".config.yaml"
            assert config_yaml.exists()

            # Check bin/config.toml exists
            config_toml = Path(tmpdir) / "bin" / "config.toml"
            assert config_toml.exists()

            # Check services.json exists
            services_json = Path(tmpdir) / "services.json"
            assert services_json.exists()

    def test_worker_config_content(self, manifest):
        """Test .config.yaml content is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = ConfigConverter(manifest, Path(tmpdir))
            converter.apply_all()

            config_path = Path(tmpdir) / ".config.yaml"
            with open(config_path) as f:
                content = f.read()
                # Skip header comments
                yaml_content = "\n".join(
                    line for line in content.split("\n")
                    if not line.startswith("#")
                )
                config = yaml.safe_load(yaml_content)

            assert config["toolbox"]["instance_id"] == "test_instance"
            assert config["toolbox"]["modules_preload"] == ["CloudM"]
            assert config["http_worker"]["port"] == 8000
            assert config["ws_worker"]["port"] == 8100

    def test_rust_config_content(self, manifest):
        """Test bin/config.toml content is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = ConfigConverter(manifest, Path(tmpdir))
            converter.apply_all()

            config_path = Path(tmpdir) / "bin" / "config.toml"
            with open(config_path) as f:
                content = f.read()

            assert 'open_modules = ["CloudM.AuthHelper"]' in content
            assert 'init_modules = ["CloudM"]' in content
            assert 'client_prefix = "api-client"' in content

    def test_services_json_content(self, manifest):
        """Test services.json content is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = ConfigConverter(manifest, Path(tmpdir))
            converter.apply_all()

            services_path = Path(tmpdir) / "services.json"
            with open(services_path) as f:
                services = json.load(f)

            assert services["autostart"]["enabled"] is True
            assert "workers" in services["autostart"]["services"]
            assert "tb -v" in services["autostart"]["commands"]
            assert services["database"]["mode"] == "LC"

    def test_suggest_env_vars_for_redis(self, monkeypatch):
        """Test env var suggestions for Redis mode."""
        # Clear env vars that would prevent suggestions
        monkeypatch.delenv("DB_CONNECTION_URI", raising=False)

        manifest = TBManifest(database={"mode": "LR"})

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = ConfigConverter(manifest, Path(tmpdir))
            suggestions = converter._suggest_env_vars()

            assert "DB_CONNECTION_URI" in suggestions

    def test_suggest_env_vars_for_minio(self, monkeypatch):
        """Test env var suggestions for MinIO mode."""
        # Clear env vars that would prevent suggestions
        monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
        monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
        monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)

        manifest = TBManifest(database={"mode": "CB"})

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = ConfigConverter(manifest, Path(tmpdir))
            suggestions = converter._suggest_env_vars()

            assert "MINIO_ENDPOINT" in suggestions
            assert "MINIO_ACCESS_KEY" in suggestions
            assert "MINIO_SECRET_KEY" in suggestions

    def test_append_missing_env_vars_creates_file(self, monkeypatch):
        """Test append_missing_env_vars creates .env if missing."""
        # Clear env vars that would prevent suggestions
        monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
        monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
        monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)

        manifest = TBManifest(database={"mode": "CB"})

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = ConfigConverter(manifest, Path(tmpdir))
            added = converter.append_missing_env_vars()

            env_path = Path(tmpdir) / ".env"
            assert env_path.exists()
            assert len(added) > 0

    def test_append_missing_env_vars_preserves_existing(self):
        """Test append_missing_env_vars doesn't overwrite existing values."""
        manifest = TBManifest(database={"mode": "CB"})

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing .env with a value
            env_path = Path(tmpdir) / ".env"
            with open(env_path, "w") as f:
                f.write("MINIO_ENDPOINT=my-custom-endpoint\n")

            converter = ConfigConverter(manifest, Path(tmpdir))
            added = converter.append_missing_env_vars()

            # MINIO_ENDPOINT should NOT be in added (it already exists)
            assert "MINIO_ENDPOINT" not in added

            # Original value should be preserved
            with open(env_path) as f:
                content = f.read()
            assert "my-custom-endpoint" in content

    def test_never_overwrites_env_file(self):
        """Test that .env is NEVER completely overwritten."""
        manifest = TBManifest()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .env with custom content
            env_path = Path(tmpdir) / ".env"
            original_content = "MY_CUSTOM_VAR=my_value\nANOTHER_VAR=another\n"
            with open(env_path, "w") as f:
                f.write(original_content)

            converter = ConfigConverter(manifest, Path(tmpdir))
            converter.apply_all()
            converter.append_missing_env_vars()

            # Original content should still be there
            with open(env_path) as f:
                content = f.read()
            assert "MY_CUSTOM_VAR=my_value" in content
            assert "ANOTHER_VAR=another" in content

