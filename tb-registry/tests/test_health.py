"""Tests for health check endpoints."""

from fastapi.testclient import TestClient


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint.

    Args:
        client: Test client.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_readiness_check(client: TestClient) -> None:
    """Test readiness check endpoint returns ready status with details.

    Args:
        client: Test client.
    """
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "details" in data
    assert "database" in data["details"]
    assert "storage" in data["details"]
    # Database should be healthy
    assert data["details"]["database"] is True
    # Storage may be not configured in test mode, but should report healthy
    assert data["details"]["storage"]["healthy"] is True


def test_health_check_head(client: TestClient) -> None:
    """Test health check HEAD endpoint.

    Args:
        client: Test client.
    """
    response = client.head("/api/v1/health")
    assert response.status_code == 200

