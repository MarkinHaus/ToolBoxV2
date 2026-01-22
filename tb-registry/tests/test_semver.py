"""Tests for semver utilities."""

import pytest
from packaging.version import Version

from registry.resolver.semver import (
    compare_versions,
    find_best_match,
    get_major,
    get_minor,
    get_patch,
    is_prerelease,
    matches,
    parse_specifier,
    parse_version,
)


def test_parse_version() -> None:
    """Test version parsing."""
    v = parse_version("1.2.3")
    assert v.major == 1
    assert v.minor == 2
    assert v.micro == 3


def test_parse_version_prerelease() -> None:
    """Test prerelease version parsing."""
    v = parse_version("1.0.0a1")
    assert v.is_prerelease


def test_parse_specifier_exact() -> None:
    """Test exact version specifier."""
    spec = parse_specifier("1.0.0")
    assert Version("1.0.0") in spec
    assert Version("1.0.1") not in spec


def test_parse_specifier_caret() -> None:
    """Test caret specifier."""
    spec = parse_specifier("^1.2.0")
    assert Version("1.2.0") in spec
    assert Version("1.9.9") in spec
    assert Version("2.0.0") not in spec


def test_parse_specifier_tilde() -> None:
    """Test tilde specifier."""
    spec = parse_specifier("~1.2.0")
    assert Version("1.2.0") in spec
    assert Version("1.2.9") in spec
    assert Version("1.3.0") not in spec


def test_parse_specifier_wildcard() -> None:
    """Test wildcard specifier."""
    spec = parse_specifier("*")
    assert Version("1.0.0") in spec
    assert Version("99.99.99") in spec


def test_parse_specifier_range() -> None:
    """Test range specifier."""
    spec = parse_specifier(">=1.0.0,<2.0.0")
    assert Version("1.0.0") in spec
    assert Version("1.5.0") in spec
    assert Version("2.0.0") not in spec


def test_matches() -> None:
    """Test version matching."""
    spec = parse_specifier(">=1.0.0")
    assert matches(Version("1.0.0"), spec)
    assert matches(Version("2.0.0"), spec)
    assert not matches(Version("0.9.0"), spec)


def test_find_best_match() -> None:
    """Test finding best matching version."""
    versions = [Version("1.0.0"), Version("1.1.0"), Version("1.2.0"), Version("2.0.0")]
    spec = parse_specifier("^1.0.0")
    best = find_best_match(versions, spec)
    assert best == Version("1.2.0")


def test_find_best_match_no_match() -> None:
    """Test finding best match with no matches."""
    versions = [Version("0.1.0"), Version("0.2.0")]
    spec = parse_specifier(">=1.0.0")
    best = find_best_match(versions, spec)
    assert best is None


def test_compare_versions() -> None:
    """Test version comparison."""
    assert compare_versions("1.0.0", "2.0.0") == -1
    assert compare_versions("2.0.0", "1.0.0") == 1
    assert compare_versions("1.0.0", "1.0.0") == 0


def test_is_prerelease() -> None:
    """Test prerelease detection."""
    assert is_prerelease("1.0.0a1")
    assert is_prerelease("1.0.0b1")
    assert is_prerelease("1.0.0rc1")
    assert not is_prerelease("1.0.0")


def test_get_major() -> None:
    """Test getting major version."""
    assert get_major("1.2.3") == 1


def test_get_minor() -> None:
    """Test getting minor version."""
    assert get_minor("1.2.3") == 2


def test_get_patch() -> None:
    """Test getting patch version."""
    assert get_patch("1.2.3") == 3

