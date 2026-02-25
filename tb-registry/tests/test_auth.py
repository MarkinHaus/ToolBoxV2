"""
Tests for CloudM.Auth JWT authentication in tb-registry.

Tests cover:
- verify_cloudm_token function
- Token verification with HS256 JWT
- Error handling for missing configuration
- Debug mode fallback
"""

import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from registry.config import Settings
from registry.api.deps import verify_cloudm_token, get_current_user
from registry.auth.cloudm_client import TokenPayload


class TestVerifyCloudmToken(unittest.IsolatedAsyncioTestCase):
    """Tests for verify_cloudm_token dependency."""

    def _get_test_settings(
        self,
        cloudm_jwt_secret: str = "test_jwt_secret",
        debug: bool = False
    ) -> Settings:
        """Create test settings."""
        return Settings(
            host="127.0.0.1",
            port=8000,
            debug=debug,
            database_url="sqlite:///test.db",
            cors_origins=["http://localhost:3000"],
            cloudm_jwt_secret=cloudm_jwt_secret,
            cloudm_auth_url="http://localhost:4025",
            minio_primary_endpoint="localhost:9000",
            minio_primary_access_key="minioadmin",
            minio_primary_secret_key="minioadmin",
            minio_primary_bucket="test-bucket",
            minio_primary_secure=False,
        )

    def _make_credentials(self, token: str) -> HTTPAuthorizationCredentials:
        """Create HTTPAuthorizationCredentials for testing."""
        creds = MagicMock(spec=HTTPAuthorizationCredentials)
        creds.credentials = token
        return creds

    def _make_valid_token(
        self,
        secret: str = "test_jwt_secret",
        user_id: str = "usr_test123",
        exp_hours: int = 1
    ) -> str:
        """Create a valid HS256 JWT token."""
        payload = {
            "user_id": user_id,
            "username": "testuser",
            "email": "test@example.com",
            "level": 1,
            "provider": "magic_link",
            "exp": int(time.time()) + (exp_hours * 3600),
            "iat": int(time.time()),
            "jti": f"jti_{user_id}_{int(time.time())}",
        }
        return jwt.encode(payload, secret, algorithm="HS256")

    async def test_no_credentials_returns_none(self):
        """Test that missing credentials returns None."""
        settings = self._get_test_settings()
        result = await verify_cloudm_token(credentials=None, settings=settings)
        self.assertIsNone(result)

    async def test_missing_secret_debug_mode(self):
        """Test debug mode returns mock data when CloudM secret not configured."""
        settings = self._get_test_settings(cloudm_jwt_secret="", debug=True)
        creds = self._make_credentials("any_token")

        result = await verify_cloudm_token(credentials=creds, settings=settings)

        self.assertIsNotNone(result)
        self.assertEqual(result.user_id, "user_debug")
        self.assertEqual(result.email, "debug@example.com")

    async def test_missing_secret_production_raises(self):
        """Test production mode raises error when secret not configured."""
        settings = self._get_test_settings(cloudm_jwt_secret="", debug=False)
        creds = self._make_credentials("any_token")

        with self.assertRaises(HTTPException) as ctx:
            await verify_cloudm_token(credentials=creds, settings=settings)

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("not configured", ctx.exception.detail)

    async def test_valid_token_returns_payload(self):
        """Test that valid token returns decoded payload."""
        settings = self._get_test_settings()
        token = self._make_valid_token()
        creds = self._make_credentials(token)

        result = await verify_cloudm_token(credentials=creds, settings=settings)

        self.assertIsNotNone(result)
        self.assertEqual(result.user_id, "usr_test123")
        self.assertEqual(result.username, "testuser")
        self.assertEqual(result.email, "test@example.com")
        self.assertEqual(result.level, 1)
        self.assertEqual(result.provider, "magic_link")

    async def test_invalid_token_returns_none(self):
        """Test that invalid token returns None."""
        settings = self._get_test_settings()
        creds = self._make_credentials("not_a_valid_jwt")

        result = await verify_cloudm_token(credentials=creds, settings=settings)
        self.assertIsNone(result)

    async def test_expired_token_returns_none(self):
        """Test that expired token is rejected."""
        settings = self._get_test_settings()

        # Create token that expired 1 hour ago
        payload = {
            "user_id": "usr_expired",
            "username": "expired_user",
            "email": "expired@test.com",
            "level": 1,
            "provider": "magic_link",
            "exp": int(time.time()) - 3600,
            "iat": int(time.time()) - 7200,
            "jti": "jti_expired",
        }
        expired_token = jwt.encode(payload, "test_jwt_secret", algorithm="HS256")
        creds = self._make_credentials(expired_token)

        result = await verify_cloudm_token(credentials=creds, settings=settings)
        self.assertIsNone(result)

    async def test_wrong_signature_returns_none(self):
        """Test that token with wrong signature is rejected."""
        settings = self._get_test_settings()

        # Sign with different secret
        payload = {
            "user_id": "usr_wrong",
            "username": "wrong_user",
            "email": "wrong@test.com",
            "level": 1,
            "provider": "magic_link",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "jti": "jti_wrong",
        }
        token = jwt.encode(payload, "wrong_secret", algorithm="HS256")
        creds = self._make_credentials(token)

        result = await verify_cloudm_token(credentials=creds, settings=settings)
        self.assertIsNone(result)


class TestTokenPayload(unittest.TestCase):
    """Tests for TokenPayload dataclass."""

    def test_token_payload_creation(self):
        """Test creating TokenPayload."""
        payload = TokenPayload(
            user_id="usr_123",
            username="testuser",
            email="test@example.com",
            level=1,
            provider="discord",
            exp=1740451200,
        )

        self.assertEqual(payload.user_id, "usr_123")
        self.assertEqual(payload.username, "testuser")
        self.assertEqual(payload.email, "test@example.com")
        self.assertEqual(payload.level, 1)
        self.assertEqual(payload.provider, "discord")
        self.assertEqual(payload.exp, 1740451200)


class TestGetUserFromToken(unittest.TestCase):
    """Tests for extracting user info from token payload."""

    def test_extract_user_id(self):
        """Test extracting user_id from token."""
        payload = TokenPayload(
            user_id="usr_123",
            username="testuser",
            email="test@example.com",
            level=1,
            provider="magic_link",
            exp=1740451200,
        )

        self.assertEqual(payload.user_id, "usr_123")

    def test_extract_username(self):
        """Test extracting username from token."""
        payload = TokenPayload(
            user_id="usr_123",
            username="testuser",
            email="test@example.com",
            level=1,
            provider="magic_link",
            exp=1740451200,
        )

        self.assertEqual(payload.username, "testuser")

    def test_extract_level(self):
        """Test extracting level from token."""
        payload = TokenPayload(
            user_id="usr_123",
            username="admin",
            email="admin@example.com",
            level=5,
            provider="discord",
            exp=1740451200,
        )

        self.assertEqual(payload.level, 5)
