"""Tests for Installation Validator."""

import yaml
from pathlib import Path

import pytest

from toolboxv2.utils.extras.install.install_validator import (
    ValidationResult,
    validate_tb_config,
    validate_config_structure,
    validate_dependencies,
    validate_module_import,
    validate_module_files,
    validate_installation,
    validate_package_integrity,
    get_validation_summary,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for tests."""
    test_dir = tmp_path / "test_validator"
    test_dir.mkdir()
    yield test_dir


@pytest.fixture
def sample_module(temp_dir):
    """Create a sample module directory."""
    module_dir = temp_dir / "test_module"
    module_dir.mkdir()

    # Create __init__.py
    (module_dir / "__init__.py").write_text("# Test module")

    # Create module file
    (module_dir / "test_module.py").write_text("def test(): pass")

    # Create config
    config = {
        "version": "1.0.0",
        "config_version": "2.0",
        "module_name": "test_module",
        "module_type": "package",
        "description": "Test module",
        "author": "Test Author",
        "license": "MIT",
        "homepage": "https://example.com",
        "dependencies_file": "requirements.txt",
        "platforms": {
            "server": {"files": ["*.py"], "required": True},
            "client": {"files": [], "required": False},
        },
        "metadata": {}
    }

    with open(module_dir / "test_module.yaml", "w") as f:
        yaml.dump(config, f)

    return module_dir


# =================== ValidationResult Tests ===================


def test_validation_result_default():
    """Test default ValidationResult is valid."""
    result = ValidationResult()
    assert result.is_valid is True
    assert result.errors == []
    assert result.warnings == []


def test_validation_result_add_error():
    """Test adding error invalidates result."""
    result = ValidationResult()
    result.add_error("Test error")

    assert result.is_valid is False
    assert len(result.errors) == 1
    assert result.errors[0] == "Test error"


def test_validation_result_add_warning():
    """Test adding warning doesn't invalidate result."""
    result = ValidationResult()
    result.add_warning("Test warning")

    assert result.is_valid is True
    assert len(result.warnings) == 1
    assert result.warnings[0] == "Test warning"


def test_validation_result_merge():
    """Test merging results."""
    result1 = ValidationResult()
    result1.add_warning("Warning 1")

    result2 = ValidationResult()
    result2.add_error("Error 1")
    result2.add_warning("Warning 2")

    result1.merge(result2)

    assert result1.is_valid is False
    assert len(result1.errors) == 1
    assert len(result1.warnings) == 2


# =================== Config Validation Tests ===================


def test_validate_tb_config_valid(sample_module):
    """Test validation of valid config."""
    config_path = sample_module / "test_module.yaml"
    result = validate_tb_config(config_path)

    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validate_tb_config_missing_file(temp_dir):
    """Test validation with missing config file."""
    result = validate_tb_config(temp_dir / "nonexistent.yaml")

    assert result.is_valid is False
    assert "not found" in result.errors[0]


def test_validate_tb_config_invalid_yaml(temp_dir):
    """Test validation with invalid YAML."""
    config_file = temp_dir / "invalid.yaml"
    config_file.write_text("invalid: yaml: content: [")

    result = validate_tb_config(config_file)

    assert result.is_valid is False


def test_validate_tb_config_missing_required_fields(temp_dir):
    """Test validation with missing required fields."""
    config_file = temp_dir / "incomplete.yaml"
    config_file.write_text("version: 1.0.0")

    result = validate_tb_config(config_file)

    assert result.is_valid is False
    # Should have errors about missing fields


def test_validate_config_structure_valid():
    """Test config structure validation with valid config."""
    config = {
        "module_name": "test",
        "module_type": "package",
    }

    result = validate_config_structure(config)

    assert result.is_valid is True


def test_validate_config_structure_missing_required():
    """Test config structure validation with missing required field."""
    config = {"description": "test"}

    result = validate_config_structure(config)

    assert result.is_valid is False
    assert "module_name" in result.errors[0]


def test_validate_config_structure_unknown_type():
    """Test config structure validation with unknown module_type."""
    config = {
        "module_name": "test",
        "module_type": "unknown_type",
    }

    result = validate_config_structure(config)

    assert result.is_valid is True  # Unknown type is a warning
    assert len(result.warnings) > 0


# =================== Dependency Validation Tests ===================


def test_validate_dependencies_no_deps():
    """Test dependency validation with no dependencies."""
    result = validate_dependencies([], Path("/tmp"))
    assert result.is_valid is True


def test_validate_dependencies_satisfied(temp_dir, sample_module):
    """Test dependency validation when deps are satisfied."""
    # Create a dependency module
    dep_dir = temp_dir / "dependency"
    dep_dir.mkdir()
    (dep_dir / "__init__.py").write_text("# Dep")

    result = validate_dependencies(["dependency"], temp_dir)

    assert result.is_valid is True


def test_validate_dependencies_missing(temp_dir):
    """Test dependency validation with missing deps."""
    result = validate_dependencies(["nonexistent_dep"], temp_dir)

    assert result.is_valid is False
    assert "Missing dependencies" in result.errors[0]


def test_validate_dependencies_single_file(temp_dir):
    """Test dependency validation with single file module."""
    # Create a .py file module
    (temp_dir / "single_mod.py").write_text("# Single file module")

    result = validate_dependencies(["single_mod"], temp_dir)

    assert result.is_valid is True


# =================== Module Import Validation Tests ===================


def test_validate_module_import_valid(sample_module):
    """Test module import validation with valid module."""
    result = validate_module_import("test_module", sample_module)

    assert result.is_valid is True


def test_validate_module_import_no_files(temp_dir):
    """Test module import validation with no Python files."""
    empty_dir = temp_dir / "empty"
    empty_dir.mkdir()

    result = validate_module_import("empty", empty_dir)

    assert result.is_valid is False
    assert len(result.errors) > 0


def test_validate_module_import_main_py(temp_dir):
    """Test module import validation with main .py file."""
    (temp_dir / "main_mod.py").write_text("# Main module")

    result = validate_module_import("main_mod", temp_dir)

    # Should be valid - has main .py file
    assert result.is_valid is True


def test_validate_module_files_valid(sample_module):
    """Test module files validation with valid structure."""
    result = validate_module_files(sample_module)

    assert result.is_valid is True


def test_validate_module_files_no_files(temp_dir):
    """Test module files validation with no files."""
    empty_dir = temp_dir / "empty"
    empty_dir.mkdir()

    result = validate_module_files(empty_dir)

    assert result.is_valid is False


def test_validate_module_files_executable_warning(temp_dir):
    """Test module files validation detects executables."""
    mod_dir = temp_dir / "test_mod"
    mod_dir.mkdir()

    # Create normal file
    (mod_dir / "__init__.py").write_text("# Test")

    # Create executable file (suspicious)
    (mod_dir / "suspicious.exe").write_bytes(b"fake exe")

    result = validate_module_files(mod_dir)

    assert result.is_valid is True  # Still valid, just a warning
    assert len(result.warnings) > 0
    assert "suspicious" in result.warnings[0].lower()


# =================== Full Installation Validation Tests ===================


def test_validate_installation_valid(sample_module):
    """Test full installation validation with valid installation."""
    result = validate_installation(
        package_name="test_module",
        install_path=sample_module,
        mods_dir=sample_module.parent,
    )

    assert result.is_valid is True
    assert result.details["package_name"] == "test_module"


def test_validate_installation_missing_path(temp_dir):
    """Test installation validation with missing path."""
    result = validate_installation(
        package_name="nonexistent",
        install_path=temp_dir / "does_not_exist",
    )

    assert result.is_valid is False
    assert "not found" in result.errors[0]


# =================== Package Integrity Tests ===================


def test_validate_package_integrity_valid(sample_module):
    """Test package integrity validation."""
    result = validate_package_integrity(sample_module)

    assert result.is_valid is True
    assert "checksum" in result.details


def test_validate_package_integrity_checksum_mismatch(temp_dir):
    """Test package integrity with checksum mismatch."""
    (temp_dir / "test.txt").write_text("test content")

    result = validate_package_integrity(
        temp_dir / "test.txt",
        expected_checksum="wrongchecksum",
    )

    assert result.is_valid is False
    assert "Checksum mismatch" in result.errors[0]


def test_validate_package_integrity_checksum_match(temp_dir):
    """Test package integrity with matching checksum."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")

    # Get the actual checksum first
    from toolboxv2.utils.extras.install.rollback_manager import calculate_checksum
    actual_checksum = calculate_checksum(test_file)

    result = validate_package_integrity(
        test_file,
        expected_checksum=actual_checksum,
    )

    assert result.is_valid is True
    assert result.details.get("checksum_verified") is True


# =================== Summary Tests ===================


def test_get_validation_summary_success():
    """Test summary for successful validation."""
    result = ValidationResult()
    summary = get_validation_summary(result)

    assert "Validation passed" in summary
    assert "✓" in summary


def test_get_validation_summary_with_errors():
    """Test summary with errors."""
    result = ValidationResult()
    result.add_error("Error 1")
    result.add_error("Error 2")

    summary = get_validation_summary(result)

    assert "Validation failed" in summary
    assert "Error 1" in summary
    assert "Error 2" in summary


def test_get_validation_summary_with_warnings():
    """Test summary with warnings."""
    result = ValidationResult()
    result.add_warning("Warning 1")

    summary = get_validation_summary(result)

    assert "Validation passed" in summary  # Still valid with warnings
    assert "Warning 1" in summary
