"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from registry.app import create_app
from registry.config import Settings


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Create test settings with temporary database.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Test settings.
    """
    db_path = tmp_path / "test_registry.db"
    return Settings(
        host="127.0.0.1",
        port=8000,
        debug=True,
        database_url=f"sqlite:///{db_path}",
        cors_origins=["http://localhost:3000"],
        # CloudM.Auth configuration
        cloudm_jwt_secret="test_jwt_secret_for_cloudm_auth",
        cloudm_auth_url="http://localhost:4025",
        # Clerk (deprecated, for backward compatibility)
        clerk_secret_key="test_secret",
        clerk_publishable_key="test_publishable",
        # MinIO
        minio_primary_endpoint="localhost:9000",
        minio_primary_access_key="minioadmin",
        minio_primary_secret_key="minioadmin",
        minio_primary_bucket="test-bucket",
        minio_primary_secure=False,
    )


@pytest.fixture
def app(test_settings: Settings, monkeypatch):
    """Create test application.

    Args:
        test_settings: Test settings.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        FastAPI application.
    """
    # Monkeypatch get_settings to return test settings
    monkeypatch.setattr("registry.app.get_settings", lambda: test_settings)
    monkeypatch.setattr("registry.config.get_settings", lambda: test_settings)
    monkeypatch.setattr("registry.db.database.get_settings", lambda: test_settings)
    # Skip storage initialization for tests (no MinIO required)
    return create_app(test_settings, skip_storage=True)


@pytest.fixture
def client(app) -> TestClient:
    """Create test client with lifespan.

    Args:
        app: FastAPI application.

    Returns:
        Test client.
    """
    # Use context manager to trigger lifespan events
    with TestClient(app) as client:
        yield client

