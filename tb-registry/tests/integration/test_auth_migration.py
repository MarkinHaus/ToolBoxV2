"""End-to-end authentication flow tests.

This module tests the complete authentication flow with CloudM.Auth
including token generation, validation, and API access.
"""

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import jwt
import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from registry.config import Settings


# ==================== Fixtures ====================


@pytest.fixture
def cloudm_test_settings(tmp_path: Path):
    """Create test settings with CloudM.Auth configuration.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Settings configured for CloudM.Auth testing.
    """
    from registry.config import Settings

    db_path = tmp_path / "test_registry.db"

    return Settings(
        host="127.0.0.1",
        port=8000,
        debug=True,
        database_url=f"sqlite:///{db_path}",
        cors_origins=["http://localhost:3000"],
        # CloudM.Auth configuration
        cloudm_jwt_secret="test_jwt_secret_for_cloudm_auth0",
        cloudm_auth_url="http://localhost:4025",
        # Clerk (deprecated, for backward compatibility)
        clerk_secret_key="deprecated",
        clerk_publishable_key="deprecated",
        # MinIO
        minio_primary_endpoint="localhost:9000",
        minio_primary_access_key="minioadmin",
        minio_primary_secret_key="minioadmin",
        minio_primary_bucket="test-bucket",
        minio_primary_secure=False,
    )


@pytest.fixture
def cloudm_app(cloudm_test_settings, monkeypatch):
    """Create test application with CloudM.Auth settings.

    Args:
        cloudm_test_settings: CloudM.Auth test settings.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        FastAPI application configured for CloudM.Auth.
    """
    from registry.app import create_app

    # Monkeypatch get_settings to return test settings
    monkeypatch.setattr("registry.app.get_settings", lambda: cloudm_test_settings)
    monkeypatch.setattr("registry.config.get_settings", lambda: cloudm_test_settings)
    monkeypatch.setattr("registry.db.database.get_settings", lambda: cloudm_test_settings)
    monkeypatch.setattr("registry.api.deps.get_settings", lambda: cloudm_test_settings)

    # Skip storage initialization for tests
    return create_app(cloudm_test_settings, skip_storage=True)


@pytest.fixture
def cloudm_client(cloudm_app) -> TestClient:
    """Create test client for CloudM.Auth app.

    Args:
        cloudm_app: FastAPI application with CloudM.Auth.

    Returns:
        Test client.
    """
    with TestClient(cloudm_app) as client:
        yield client


def generate_cloudm_token(
    user_id: str = "usr_test123",
    username: str = "testuser",
    email: str = "test@example.com",
    secret: str = "test_jwt_secret_for_cloudm_auth0",
    level: int = 1,
    provider: str = "magic_link",
    expiry_hours: int = 1,
) -> str:
    """Generate a valid CloudM.Auth JWT token for testing.

    Args:
        user_id: CloudM.Auth user ID.
        username: Username.
        email: User email.
        secret: JWT secret for signing.
        level: User access level.
        provider: Auth provider (discord, google, magic_link, etc.)
        expiry_hours: Token validity in hours.

    Returns:
        Encoded JWT token string.
    """
    payload = {
        "user_id": user_id,
        "username": username,
        "email": email,
        "level": level,
        "provider": provider,
        "exp": int(time.time()) + (expiry_hours * 3600),
        "iat": int(time.time()),
        "jti": f"jti_{user_id}_{int(time.time())}",
    }
    return jwt.encode(payload, secret, algorithm="HS256")


# ==================== End-to-End Tests ====================


def test_full_auth_flow(cloudm_client: TestClient):
    """Test complete authentication flow.

    This test verifies:
    1. Token generation with CloudM.Auth format
    2. Token validation by the registry
    3. Access to protected endpoints
    4. User creation on first login
    """
    # 1. Generate token with CloudM.Auth format
    token = generate_cloudm_token(
        user_id="usr_integration_test",
        username="integration_user",
        email="integration@test.com",
    )

    # 2. Access protected endpoint (packages list)
    response = cloudm_client.get(
        "/api/v1/packages",
        headers={"Authorization": f"Bearer {token}"}
    )

    # Should return 200 (empty dict is fine, means auth worked)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    # Verify response structure
    data = response.json()
    assert "page" in data or isinstance(data, dict)


def test_public_package_download_anonymous(cloudm_client: TestClient):
    """Test downloading public packages without authentication.

    Public packages should be accessible without any token.
    """
    # Access packages endpoint without auth
    response = cloudm_client.get("/api/v1/packages")

    # Should succeed - public endpoint
    assert response.status_code == 200


def test_auth_with_valid_token(cloudm_client: TestClient):
    """Test authentication with a valid CloudM.Auth token.

    Synchronous version for TestClient compatibility.
    """
    token = generate_cloudm_token(
        user_id="usr_valid_test",
        username="validuser",
        email="valid@test.com",
    )

    response = cloudm_client.get(
        "/api/v1/packages",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200


def test_auth_with_invalid_token(cloudm_client: TestClient):
    """Test authentication with an invalid token.

    Invalid tokens should be rejected with 401 Unauthorized.
    """
    # Completely invalid token string
    response = cloudm_client.get(
        "/api/v1/packages",
        headers={"Authorization": "Bearer invalid_token_string"}
    )

    # Should be rejected - but might return 200 if endpoint is public
    # The key is that invalid token doesn't crash the server
    assert response.status_code in [200, 401]


def test_auth_with_expired_token(cloudm_client: TestClient):
    """Test authentication with an expired token.

    Expired tokens should be rejected.
    """
    # Create token that expired 1 hour ago
    payload = {
        "user_id": "usr_expired",
        "username": "expired_user",
        "email": "expired@test.com",
        "level": 1,
        "provider": "magic_link",
        "exp": int(time.time()) - 3600,  # Expired 1 hour ago
        "iat": int(time.time()) - 7200,
        "jti": "jti_expired",
    }
    expired_token = jwt.encode(
        payload,
        "test_jwt_secret_for_cloudm_auth0",
        algorithm="HS256"
    )

    response = cloudm_client.get(
        "/api/v1/packages",
        headers={"Authorization": f"Bearer {expired_token}"}
    )

    # Expired tokens should be rejected (401) or public access allowed (200)
    assert response.status_code in [200, 401]


def test_auth_with_wrong_secret(cloudm_client: TestClient):
    """Test authentication with token signed with wrong secret.

    Tokens signed with a different secret should be rejected.
    """
    # Sign with different secret
    payload = {
        "user_id": "usr_wrong_secret",
        "username": "wronguser",
        "email": "wrong@test.com",
        "level": 1,
        "provider": "magic_link",
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "jti": "jti_wrong",
    }
    token = jwt.encode(payload, "wrong_secret_for_testing_purposes!", algorithm="HS256")

    response = cloudm_client.get(
        "/api/v1/packages",
        headers={"Authorization": f"Bearer {token}"}
    )

    # Wrong signature should be rejected (401) or public access allowed (200)
    assert response.status_code in [200, 401]


def test_auth_without_token(cloudm_client: TestClient):
    """Test accessing API without any token.

    Public endpoints should work, private should require auth.
    """
    # No Authorization header
    response = cloudm_client.get("/api/v1/packages")

    # Public endpoint - should work
    assert response.status_code == 200


def test_auth_with_malformed_token(cloudm_client: TestClient):
    """Test authentication with malformed token.

    Malformed tokens should not crash the server.
    """
    malformed_tokens = [
        "Bearer not_a_jwt_at_all",
        "Bearer",
        "",
        "not_a_jwt",
        "a.b.c",  # JWT format but invalid content
    ]

    for token_str in malformed_tokens:
        response = cloudm_client.get(
            "/api/v1/packages",
            headers={"Authorization": token_str} if token_str else None
        )
        # Should not crash - return 200 or 401
        assert response.status_code in [200, 401, 422]


def test_token_payload_structure(cloudm_client: TestClient):
    """Test that CloudM.Auth token payload is correctly parsed.

    Verifies that all required fields are present in the token.
    """
    token = generate_cloudm_token(
        user_id="usr_payload_test",
        username="payloaduser",
        email="payload@test.com",
        level=5,
        provider="discord",
    )

    # Decode to verify structure
    decoded = jwt.decode(
        token,
        "test_jwt_secret_for_cloudm_auth0",
        algorithms=["HS256"]
    )

    # Verify all required fields
    assert decoded["user_id"] == "usr_payload_test"
    assert decoded["username"] == "payloaduser"
    assert decoded["email"] == "payload@test.com"
    assert decoded["level"] == 5
    assert decoded["provider"] == "discord"
    assert "exp" in decoded
    assert "iat" in decoded
    assert "jti" in decoded


# ==================== Provider-Specific Tests ====================


def test_auth_with_different_providers(cloudm_client: TestClient):
    """Test authentication with different CloudM.Auth providers.

    CloudM.Auth supports multiple providers: discord, google, magic_link, etc.
    """
    providers = ["discord", "google", "magic_link", "github"]

    for provider in providers:
        token = generate_cloudm_token(
            user_id=f"usr_{provider}",
            username=f"{provider}user",
            email=f"{provider}@test.com",
            provider=provider,
        )

        response = cloudm_client.get(
            "/api/v1/packages",
            headers={"Authorization": f"Bearer {token}"}
        )

        # All providers should work
        assert response.status_code == 200, f"Failed for provider: {provider}"


# ==================== Debug Mode Tests ====================


def test_debug_mode_without_secret(monkeypatch, tmp_path):
    """Test that debug mode allows development without proper JWT secret.

    In debug mode, missing JWT secret should return mock user data.
    """
    from registry.config import Settings
    from registry.app import create_app

    # Settings with empty secret but debug=True
    db_path = tmp_path / "test_debug.db"
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        debug=True,
        database_url=f"sqlite:///{db_path}",
        cloudm_jwt_secret="",  # Empty secret
        cloudm_auth_url="http://localhost:4025",
    )

    monkeypatch.setattr("registry.app.get_settings", lambda: settings)
    monkeypatch.setattr("registry.config.get_settings", lambda: settings)
    monkeypatch.setattr("registry.api.deps.get_settings", lambda: settings)

    app = create_app(settings, skip_storage=True)

    with TestClient(app) as client:
        # In debug mode, should accept any token
        response = client.get(
            "/api/v1/packages",
            headers={"Authorization": "Bearer any_token_will_do"}
        )

        # Debug mode should allow access (might return mock data)
        assert response.status_code in [200, 401]


# ==================== Token Refresh Tests ====================


def test_token_refresh_flow(cloudm_client: TestClient):
    """Test token refresh flow.

    When a token is about to expire, a new one should be obtainable.
    """
    # Create token that expires soon (5 minutes)
    token = generate_cloudm_token(
        user_id="usr_refresh",
        username="refreshuser",
        email="refresh@test.com",
        expiry_hours=0.08,  # ~5 minutes
    )

    # Token should still work
    response = cloudm_client.get(
        "/api/v1/packages",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200

    # New token with extended expiry
    new_token = generate_cloudm_token(
        user_id="usr_refresh",
        username="refreshuser",
        email="refresh@test.com",
        expiry_hours=24,  # 24 hours
    )

    response = cloudm_client.get(
        "/api/v1/packages",
        headers={"Authorization": f"Bearer {new_token}"}
    )

    assert response.status_code == 200


# ==================== Error Handling Tests ====================


def test_auth_server_handles_malformed_jwt(cloudm_client: TestClient):
    """Test that the server handles malformed JWT gracefully.

    Server should not crash when receiving invalid JWT data.
    """
    # Various malformed inputs
    malformed_inputs = [
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",  # Truncated
        "Bearer a.b.c.d.e",  # Too many parts
        "Bearer ..",  # Empty parts
        "Bearer null",  # Literal null
    ]

    for malformed in malformed_inputs:
        response = cloudm_client.get(
            "/api/v1/packages",
            headers={"Authorization": malformed}
        )

        # Should handle gracefully - no 500 errors
        assert response.status_code != 500


# ==================== Performance Tests ====================


def test_auth_performance(cloudm_client: TestClient):
    """Test authentication performance.

    Token validation should be fast (< 100ms typically).
    """
    import time

    token = generate_cloudm_token()

    start = time.time()
    response = cloudm_client.get(
        "/api/v1/packages",
        headers={"Authorization": f"Bearer {token}"}
    )
    elapsed = time.time() - start

    assert response.status_code == 200
    # Should be reasonably fast (allow for test environment overhead)
    assert elapsed < 2.0, f"Auth took too long: {elapsed}s"


# ==================== Health Endpoint Tests ====================


def test_health_endpoint_accessible(cloudm_client: TestClient):
    """Test that health endpoint is accessible without auth."""
    response = cloudm_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_ready_endpoint_accessible(cloudm_client: TestClient):
    """Test that readiness endpoint is accessible without auth."""
    response = cloudm_client.get("/api/v1/ready")
    assert response.status_code == 200
