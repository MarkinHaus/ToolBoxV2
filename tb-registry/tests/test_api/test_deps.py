"""Tests for FastAPI dependencies."""

import jwt
import time

import pytest
from fastapi import HTTPException
from registry.api.deps import verify_cloudm_token, _get_cloudm_client
from registry.config import Settings


class MockCredentials:
    """Mock HTTPAuthorizationCredentials."""

    def __init__(self, token: str):
        self.credentials = token


@pytest.mark.asyncio
async def test_verify_cloudm_token_with_valid_token():
    """Test dependency with valid token."""
    settings = Settings(cloudm_jwt_secret="test_jwt_secret_for_cloudm_auth0", debug=False)

    # Create valid token
    payload = {
        "user_id": "usr_test",
        "username": "testuser",
        "email": "test@example.com",
        "level": 1,
        "provider": "magic_link",
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "jti": "test_jti",
    }
    token = jwt.encode(payload, "test_jwt_secret_for_cloudm_auth0", algorithm="HS256")

    # Mock credentials
    result = await verify_cloudm_token(MockCredentials(token), settings)
    assert result is not None
    assert result.user_id == "usr_test"


@pytest.mark.asyncio
async def test_verify_cloudm_token_with_no_token():
    """Test dependency with no token."""
    settings = Settings(cloudm_jwt_secret="test_secret", debug=False)

    result = await verify_cloudm_token(None, settings)
    assert result is None


@pytest.mark.asyncio
async def test_verify_cloudm_token_debug_mode():
    """Test dependency in debug mode."""
    settings = Settings(cloudm_jwt_secret="", debug=True)

    result = await verify_cloudm_token(MockCredentials("any_token"), settings)
    assert result is not None
    assert result.user_id == "user_debug"


@pytest.mark.asyncio
async def test_verify_cloudm_token_invalid_token():
    """Test dependency with invalid token."""
    settings = Settings(cloudm_jwt_secret="test_secret", debug=False)

    result = await verify_cloudm_token(MockCredentials("invalid_token"), settings)
    assert result is None


@pytest.mark.asyncio
async def test_verify_cloudm_token_not_configured():
    """Test dependency when JWT secret not configured and not in debug mode."""
    settings = Settings(cloudm_jwt_secret="", debug=False)

    with pytest.raises(HTTPException) as exc_info:
        await verify_cloudm_token(MockCredentials("any_token"), settings)

    assert exc_info.value.status_code == 500
    assert "not configured" in exc_info.value.detail


def test_get_cloudm_client_caching():
    """Test that client is cached."""
    client1 = _get_cloudm_client("secret1")
    client2 = _get_cloudm_client("secret1")
    client3 = _get_cloudm_client("secret2")

    # Same secret should return cached instance
    assert client1 is client2
    # Different secret should return different instance
    assert client1 is not client3
