"""Semantic versioning utilities."""

import re
from typing import Optional

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def parse_version(version_str: str) -> Version:
    """Parse a version string into a Version object.

    Args:
        version_str: Version string (e.g., "1.0.0", "2.1.0-beta.1").

    Returns:
        Parsed Version object.

    Raises:
        ValueError: If version string is invalid.
    """
    return Version(version_str)


def parse_specifier(spec_str: str) -> SpecifierSet:
    """Parse a version specifier string.

    Supports:
    - Exact: "1.0.0" or "==1.0.0"
    - Range: ">=1.0.0,<2.0.0"
    - Caret: "^1.0.0" (>=1.0.0,<2.0.0)
    - Tilde: "~1.0.0" (>=1.0.0,<1.1.0)
    - Wildcard: "*"

    Args:
        spec_str: Version specifier string.

    Returns:
        SpecifierSet for matching versions.
    """
    spec_str = spec_str.strip()

    # Handle wildcard
    if spec_str == "*":
        return SpecifierSet("")

    # Handle caret (^) - compatible with major version
    if spec_str.startswith("^"):
        version = spec_str[1:]
        parsed = parse_version(version)
        if parsed.major == 0:
            # For 0.x.y, only compatible with same minor
            return SpecifierSet(f">={version},<0.{parsed.minor + 1}.0")
        return SpecifierSet(f">={version},<{parsed.major + 1}.0.0")

    # Handle tilde (~) - compatible with minor version
    if spec_str.startswith("~"):
        version = spec_str[1:]
        parsed = parse_version(version)
        return SpecifierSet(f">={version},<{parsed.major}.{parsed.minor + 1}.0")

    # Handle plain version (treat as exact match)
    if re.match(r"^\d+\.\d+\.\d+", spec_str) and not spec_str.startswith(("=", ">", "<", "!")):
        return SpecifierSet(f"=={spec_str}")

    # Standard specifier
    return SpecifierSet(spec_str)


def matches(version: Version, specifier: SpecifierSet) -> bool:
    """Check if a version matches a specifier.

    Args:
        version: Version to check.
        specifier: Specifier to match against.

    Returns:
        True if version matches specifier.
    """
    return version in specifier


def find_best_match(
    versions: list[Version],
    specifier: SpecifierSet,
) -> Optional[Version]:
    """Find the best (newest) matching version.

    Args:
        versions: List of available versions.
        specifier: Specifier to match against.

    Returns:
        Best matching version or None if no match.
    """
    matching = [v for v in versions if v in specifier]
    if not matching:
        return None
    return max(matching)


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings.

    Args:
        v1: First version string.
        v2: Second version string.

    Returns:
        -1 if v1 < v2, 0 if equal, 1 if v1 > v2.
    """
    parsed_v1 = parse_version(v1)
    parsed_v2 = parse_version(v2)

    if parsed_v1 < parsed_v2:
        return -1
    elif parsed_v1 > parsed_v2:
        return 1
    return 0


def is_prerelease(version_str: str) -> bool:
    """Check if a version is a prerelease.

    Args:
        version_str: Version string.

    Returns:
        True if version is a prerelease.
    """
    return parse_version(version_str).is_prerelease


def get_major(version_str: str) -> int:
    """Get the major version number.

    Args:
        version_str: Version string.

    Returns:
        Major version number.
    """
    return parse_version(version_str).major


def get_minor(version_str: str) -> int:
    """Get the minor version number.

    Args:
        version_str: Version string.

    Returns:
        Minor version number.
    """
    return parse_version(version_str).minor


def get_patch(version_str: str) -> int:
    """Get the patch version number.

    Args:
        version_str: Version string.

    Returns:
        Patch version number.
    """
    return parse_version(version_str).micro

