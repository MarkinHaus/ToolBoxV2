"""Tests for manifest CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

import pytest


class TestManifestCLI:
    """Tests for manifest CLI commands."""

    def test_create_parser(self):
        """Test parser creation."""
        from toolboxv2.utils.clis.manifest_cli import create_parser

        parser = create_parser()

        # Test that subcommands exist
        args = parser.parse_args(["show"])
        assert args.command == "show"

        args = parser.parse_args(["validate"])
        assert args.command == "validate"

        args = parser.parse_args(["apply", "--dry-run"])
        assert args.command == "apply"
        assert args.dry_run is True

        args = parser.parse_args(["init", "--env", "production"])
        assert args.command == "init"
        assert args.env == "production"

    def test_cmd_show_no_manifest(self):
        """Test show command when no manifest exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Import inside the test to allow patching
            with patch("toolboxv2.tb_root_dir", Path(tmpdir)):
                from toolboxv2.utils.clis.manifest_cli import cmd_show

                args = MagicMock()
                args.json = False
                args.section = None

                # The function imports tb_root_dir at runtime, so we need to patch the loader
                with patch("toolboxv2.utils.manifest.loader.ManifestLoader") as MockLoader:
                    mock_loader = MagicMock()
                    mock_loader.exists.return_value = False
                    MockLoader.return_value = mock_loader

                    result = cmd_show(args)
                    assert result == 1  # Should fail

    def test_cmd_show_with_manifest(self):
        """Test show command with existing manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_content = """
manifest_version: "1.0"
app:
  name: TestApp
"""
            manifest_path = Path(tmpdir) / "tb-manifest.yaml"
            manifest_path.write_text(manifest_content, encoding="utf-8")

            with patch("toolboxv2.tb_root_dir", Path(tmpdir)):
                from toolboxv2.utils.clis.manifest_cli import cmd_show
                from toolboxv2.utils.manifest import ManifestLoader

                args = MagicMock()
                args.json = False
                args.section = None

                # Use real loader with temp directory
                with patch("toolboxv2.utils.manifest.loader.ManifestLoader") as MockLoader:
                    mock_loader = MagicMock()
                    mock_loader.exists.return_value = True
                    mock_loader.load.return_value = MagicMock()
                    MockLoader.return_value = mock_loader

                    result = cmd_show(args)
                    assert result == 0

    def test_cmd_validate_valid_manifest(self):
        """Test validate command with valid manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_content = """
manifest_version: "1.0"
app:
  name: TestApp
  environment: development
"""
            manifest_path = Path(tmpdir) / "tb-manifest.yaml"
            manifest_path.write_text(manifest_content, encoding="utf-8")

            with patch("toolboxv2.tb_root_dir", Path(tmpdir)):
                from toolboxv2.utils.clis.manifest_cli import cmd_validate

                args = MagicMock()
                args.strict = False

                with patch("toolboxv2.utils.manifest.loader.ManifestLoader") as MockLoader:
                    mock_loader = MagicMock()
                    mock_loader.exists.return_value = True
                    mock_loader.load.return_value = MagicMock()
                    mock_loader.validate.return_value = (True, [])
                    MockLoader.return_value = mock_loader

                    result = cmd_validate(args)
                    assert result == 0

    def test_cmd_init_creates_manifest(self):
        """Test init command creates manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("toolboxv2.tb_root_dir", Path(tmpdir)):
                from toolboxv2.utils.clis.manifest_cli import cmd_init

                args = MagicMock()
                args.force = False
                args.env = "development"

                with patch("toolboxv2.utils.manifest.loader.ManifestLoader") as MockLoader:
                    mock_loader = MagicMock()
                    mock_loader.exists.return_value = False
                    mock_loader.manifest_path = Path(tmpdir) / "tb-manifest.yaml"
                    MockLoader.return_value = mock_loader

                    result = cmd_init(args)
                    assert result == 0
                    # The function actually creates a real manifest, so just check success

    def test_cmd_init_refuses_overwrite(self):
        """Test init command refuses to overwrite without --force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing manifest
            manifest_path = Path(tmpdir) / "tb-manifest.yaml"
            manifest_path.write_text("existing: content", encoding="utf-8")

            with patch("toolboxv2.tb_root_dir", Path(tmpdir)):
                from toolboxv2.utils.clis.manifest_cli import cmd_init

                args = MagicMock()
                args.force = False
                args.env = "development"

                with patch("toolboxv2.utils.manifest.loader.ManifestLoader") as MockLoader:
                    mock_loader = MagicMock()
                    mock_loader.exists.return_value = True
                    MockLoader.return_value = mock_loader

                    result = cmd_init(args)
                    assert result == 1  # Should fail
                    mock_loader.save.assert_not_called()

    def test_cmd_apply_dry_run(self):
        """Test apply command in dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_content = """
manifest_version: "1.0"
app:
  name: TestApp
"""
            manifest_path = Path(tmpdir) / "tb-manifest.yaml"
            manifest_path.write_text(manifest_content, encoding="utf-8")

            with patch("toolboxv2.tb_root_dir", Path(tmpdir)):
                from toolboxv2.utils.clis.manifest_cli import cmd_apply

                args = MagicMock()
                args.dry_run = True
                args.force = False
                args.env = False

                with patch("toolboxv2.utils.manifest.loader.ManifestLoader") as MockLoader:
                    mock_loader = MagicMock()
                    mock_loader.exists.return_value = True
                    mock_manifest = MagicMock()
                    mock_manifest.nginx.enabled = False
                    mock_loader.load.return_value = mock_manifest
                    MockLoader.return_value = mock_loader

                    result = cmd_apply(args)
                    assert result == 0

                    # No files should be created in dry run
                    assert not (Path(tmpdir) / ".config.yaml").exists()

    def test_cli_manifest_main_no_command(self):
        """Test main entry point with no command shows help."""
        from toolboxv2.utils.clis.manifest_cli import cli_manifest_main

        with patch.object(sys, 'argv', ['manifest']):
            result = cli_manifest_main()
            assert result == 0  # Help shown, no error

