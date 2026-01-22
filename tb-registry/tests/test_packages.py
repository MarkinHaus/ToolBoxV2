"""Tests for package endpoints and services.

Note: Tests that require storage (list_packages, get_package, search)
are skipped when running without MinIO. These tests require integration
testing with a running MinIO instance.
"""

import pytest
from fastapi.testclient import TestClient

from registry.models.package import PackageType, Visibility


def test_create_package_unauthorized(client: TestClient) -> None:
    """Test creating package without auth.

    Args:
        client: Test client.
    """
    response = client.post(
        "/api/v1/packages",
        json={
            "name": "test-package",
            "display_name": "Test Package",
            "package_type": "mod",
        },
    )
    assert response.status_code == 401


def test_resolve_empty_requirements(client: TestClient) -> None:
    """Test resolving empty requirements.

    Args:
        client: Test client.
    """
    response = client.post(
        "/api/v1/resolve",
        json={"requirements": []},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["resolved"] == {}


def test_resolve_nonexistent_package(client: TestClient) -> None:
    """Test resolving non-existent package.

    Args:
        client: Test client.
    """
    response = client.post(
        "/api/v1/resolve",
        json={"requirements": ["nonexistent>=1.0.0"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert len(data["conflicts"]) > 0




