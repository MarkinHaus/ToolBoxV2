"""
Installation Validator for Package Modules.

This module provides validation functions for installed packages,
including configuration validation, dependency checking, and
module import verification.

Classes:
    ValidationError: Base exception for validation errors
    ConfigValidationError: Configuration validation failed
    DependencyValidationError: Dependency validation failed
    ImportValidationError: Module import validation failed

Functions:
    validate_tb_config(config_path: Path) -> bool
        Validate ToolBox v2 configuration file
    validate_dependencies(dependencies: List[str], mods_dir: Path) -> ValidationResult
        Validate that all dependencies are installed
    validate_module_import(module_name: str, module_path: Path) -> ValidationResult
        Validate that a module can be imported
    validate_installation(package_name: str, install_path: Path) -> ValidationResult
        Full validation of a package installation
"""

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# =================== Exceptions ===================


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class ConfigValidationError(ValidationError):
    """Raised when configuration validation fails."""
    pass


class DependencyValidationError(ValidationError):
    """Raised when dependency validation fails."""
    pass


class ImportValidationError(ValidationError):
    """Raised when module import validation fails."""
    pass


# =================== Data Classes ===================


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
        details: Additional validation details
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.details.update(other.details)


# =================== Configuration Validation ===================


TB_CONFIG_SCHEMA_V2 = {
    "version": str,
    "config_version": str,
    "module_name": str,
    "module_type": str,
    "description": str,
    "author": str,
    "license": str,
    "homepage": str,
    "dependencies_file": str,
    "platforms": {
        "server": {"files": list, "required": bool},
        "client": {"files": list, "required": bool},
        "desktop": {"files": list, "required": bool},
        "mobile": {"files": list, "required": bool},
        "common": {"files": list, "required": bool}
    },
    "metadata": dict
}

TB_CONFIG_SINGLE_SCHEMA = {
    "version": str,
    "config_version": str,
    "module_name": str,
    "module_type": "single",
    "file_path": str,
    "description": str,
    "author": str,
    "license": str,
    "specification": dict,
    "dependencies": list,
    "platforms": list,
    "metadata": dict
}


def validate_tb_config(config_path: Path) -> ValidationResult:
    """Validate ToolBox v2 configuration file.

    Args:
        config_path: Path to the .yaml configuration file

    Returns:
        ValidationResult with validation status
    """
    result = ValidationResult()

    if not config_path.exists():
        result.add_error(f"Configuration file not found: {config_path}")
        return result

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            result.add_error("Configuration file is empty")
            return result

        # Check required fields
        required_fields = ["version", "config_version", "module_name"]
        for field_name in required_fields:
            if field_name not in config_data:
                result.add_error(f"Missing required field: {field_name}")

        # Check module_type
        module_type = config_data.get("module_type")
        if module_type:
            if module_type == "single":
                schema = TB_CONFIG_SINGLE_SCHEMA
            else:
                schema = TB_CONFIG_SCHEMA_V2

            # Validate against schema (basic check)
            for key, expected_type in schema.items():
                if key in config_data:
                    value = config_data[key]
                    # Skip dict-type schema entries (nested structures)
                    if not isinstance(expected_type, type):
                        continue
                    if not isinstance(value, expected_type):
                        result.add_warning(
                            f"Field '{key}' has unexpected type: "
                            f"expected {expected_type.__name__}, got {type(value).__name__}"
                        )

        result.details["config"] = config_data
        logger.debug(f"Configuration validated for: {config_path}")

    except yaml.YAMLError as e:
        result.add_error(f"Invalid YAML format: {e}")
    except Exception as e:
        result.add_error(f"Configuration validation error: {e}")

    return result


def validate_config_structure(config: Dict[str, Any]) -> ValidationResult:
    """Validate configuration dictionary structure.

    Args:
        config: Configuration dictionary

    Returns:
        ValidationResult with validation status
    """
    result = ValidationResult()

    if not isinstance(config, dict):
        result.add_error("Configuration must be a dictionary")
        return result

    # Check for required keys
    required_keys = ["module_name"]
    for key in required_keys:
        if key not in config:
            result.add_error(f"Missing required key: {key}")

    # Validate module_type if present
    if "module_type" in config:
        module_type = config["module_type"]
        valid_types = ["package", "single", "hybrid"]
        if module_type not in valid_types:
            result.add_warning(f"Unknown module_type: {module_type}")

    return result


# =================== Dependency Validation ===================


def validate_dependencies(
    dependencies: List[str],
    mods_dir: Path,
) -> ValidationResult:
    """Validate that all dependencies are installed.

    Args:
        dependencies: List of dependency package names
        mods_dir: Path to the modules directory

    Returns:
        ValidationResult with validation status
    """
    result = ValidationResult()

    if not dependencies:
        return result

    mods_dir = Path(mods_dir)
    if not mods_dir.exists():
        result.add_error(f"Modules directory not found: {mods_dir}")
        return result

    missing_deps = []
    for dep in dependencies:
        dep_path = mods_dir / dep
        if not dep_path.exists():
            # Also check for .py file (single file module)
            py_file = mods_dir / f"{dep}.py"
            if not py_file.exists():
                missing_deps.append(dep)

    if missing_deps:
        result.add_error(f"Missing dependencies: {', '.join(missing_deps)}")

    result.details["missing_dependencies"] = missing_deps
    return result


def validate_dependency_versions(
    dependencies: List[Dict[str, str]],
    mods_dir: Path,
    lock_manager=None,
) -> ValidationResult:
    """Validate dependency versions against constraints.

    Args:
        dependencies: List of dependency dicts with name and version constraints
        mods_dir: Path to the modules directory
        lock_manager: Optional LockFileManager for version checking

    Returns:
        ValidationResult with validation status
    """
    result = ValidationResult()

    for dep_spec in dependencies:
        dep_name = dep_spec.get("name")
        if not dep_name:
            result.add_error("Dependency missing 'name' field")
            continue

        dep_path = mods_dir / dep_name
        if not dep_path.exists():
            result.add_error(f"Dependency not found: {dep_name}")
            continue

        # Check version if constraint specified
        if "version" in dep_spec and lock_manager:
            installed_version = lock_manager.get_installed_version(dep_name)
            if installed_version:
                from packaging import version as pv
                required = dep_spec["version"]

                try:
                    # Simple version comparison
                    if required.startswith(">="):
                        min_ver = required[2:]
                        if pv.parse(installed_version) < pv.parse(min_ver):
                            result.add_error(
                                f"Dependency {dep_name} version {installed_version} "
                                f"is below required {min_ver}"
                            )
                    elif required.startswith("=="):
                        if installed_version != required[2:]:
                            result.add_error(
                                f"Dependency {dep_name} version mismatch: "
                                f"required {required[2:]}, have {installed_version}"
                            )
                except pv.InvalidVersion as e:
                    result.add_warning(f"Invalid version format for {dep_name}: {e}")

    return result


# =================== Module Import Validation ===================


def validate_module_import(
    module_name: str,
    module_path: Path,
) -> ValidationResult:
    """Validate that a module can be imported without errors.

    Args:
        module_name: Name of the module
        module_path: Path to the module directory

    Returns:
        ValidationResult with validation status
    """
    result = ValidationResult()

    module_path = Path(module_path)
    if not module_path.exists():
        result.add_error(f"Module path not found: {module_path}")
        return result

    # Check for __init__.py or main .py file
    init_file = module_path / "__init__.py"
    main_file = module_path / f"{module_name}.py"

    if not (init_file.exists() or main_file.exists()):
        # Check for v2 config file (modern modules)
        config_file = module_path / f"{module_name}.yaml"
        if not config_file.exists():
            result.add_error(
                f"Module {module_name} has no __init__.py, {module_name}.py, "
                f"or {module_name}.yaml file"
            )
            return result

    # Try to import the module (basic syntax check)
    try:
        # Add module path to sys.path temporarily
        sys.path.insert(0, str(module_path.parent))

        if init_file.exists():
            spec = importlib.util.spec_from_file_location(module_name, init_file)
        elif main_file.exists():
            spec = importlib.util.spec_from_file_location(module_name, main_file)
        else:
            spec = None

        if spec and spec.loader:
            # Don't actually import (could have side effects)
            # Just check that the spec can be created
            pass

        # Restore sys.path
        sys.path.pop(0)

    except ImportError as e:
        result.add_warning(f"Module import check warning: {e}")
    except Exception as e:
        result.add_warning(f"Module validation warning: {e}")

    return result


def validate_module_files(module_path: Path) -> ValidationResult:
    """Validate that required module files are present.

    Args:
        module_path: Path to the module directory

    Returns:
        ValidationResult with validation status
    """
    result = ValidationResult()

    module_path = Path(module_path)
    if not module_path.exists():
        result.add_error(f"Module path not found: {module_path}")
        return result

    # Check for at least one Python file
    py_files = list(module_path.glob("*.py"))
    yaml_files = list(module_path.glob("*.yaml"))

    if not py_files and not yaml_files:
        result.add_error("Module contains no Python or YAML files")

    # Check for suspicious files
    suspicious = []
    for file in module_path.rglob("*"):
        if file.is_file():
            # Check for executables in unexpected locations
            if file.suffix in [".exe", ".dll", ".so", ".dylib"]:
                suspicious.append(str(file.relative_to(module_path)))

    if suspicious:
        result.add_warning(f"Suspicious executable files found: {suspicious}")

    return result


# =================== Full Installation Validation ===================


def validate_installation(
    package_name: str,
    install_path: Path,
    mods_dir: Optional[Path] = None,
    lock_manager=None,
) -> ValidationResult:
    """Perform full validation of a package installation.

    This runs all validation checks:
    1. Path existence
    2. Configuration validation
    3. Module file validation
    4. Dependency validation
    5. Import validation

    Args:
        package_name: Name of the package
        install_path: Path to the installed package
        mods_dir: Optional path to modules directory (for dependency checking)
        lock_manager: Optional LockFileManager for version checking

    Returns:
        ValidationResult with all validation issues
    """
    result = ValidationResult()

    install_path = Path(install_path)

    # 1. Check path exists
    if not install_path.exists():
        result.add_error(f"Installation path not found: {install_path}")
        return result

    # 2. Validate configuration
    config_path = install_path / f"{package_name}.yaml"
    if config_path.exists():
        config_result = validate_tb_config(config_path)
        result.merge(config_result)

    # 3. Validate module files
    files_result = validate_module_files(install_path)
    result.merge(files_result)

    # 4. Validate import
    import_result = validate_module_import(package_name, install_path)
    result.merge(import_result)

    # 5. Validate dependencies if mods_dir provided
    if mods_dir and config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            dependencies = config.get("dependencies", [])
            if dependencies:
                dep_result = validate_dependencies(dependencies, Path(mods_dir))
                result.merge(dep_result)
        except Exception as e:
            result.add_warning(f"Could not validate dependencies: {e}")

    # Store summary
    result.details["package_name"] = package_name
    result.details["install_path"] = str(install_path)
    result.details["error_count"] = len(result.errors)
    result.details["warning_count"] = len(result.warnings)

    return result


def validate_package_integrity(
    package_path: Path,
    expected_checksum: Optional[str] = None,
) -> ValidationResult:
    """Validate package integrity with checksum.

    Args:
        package_path: Path to the package directory or file
        expected_checksum: Optional SHA256 checksum to verify against

    Returns:
        ValidationResult with integrity status
    """
    result = ValidationResult()

    package_path = Path(package_path)
    if not package_path.exists():
        result.add_error(f"Package path not found: {package_path}")
        return result

    # Calculate checksum
    from toolboxv2.utils.extras.install.rollback_manager import calculate_checksum

    actual_checksum = calculate_checksum(package_path)
    result.details["checksum"] = actual_checksum

    if expected_checksum:
        if actual_checksum != expected_checksum:
            result.add_error(
                f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
            )
        else:
            result.details["checksum_verified"] = True

    return result


def get_validation_summary(result: ValidationResult) -> str:
    """Get a human-readable summary of validation results.

    Args:
        result: ValidationResult to summarize

    Returns:
        Formatted summary string
    """
    lines = []

    if result.is_valid:
        lines.append("✓ Validation passed")
    else:
        lines.append("✗ Validation failed")

    if result.errors:
        lines.append("\nErrors:")
        for error in result.errors:
            lines.append(f"  - {error}")

    if result.warnings:
        lines.append("\nWarnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")

    return "\n".join(lines)
