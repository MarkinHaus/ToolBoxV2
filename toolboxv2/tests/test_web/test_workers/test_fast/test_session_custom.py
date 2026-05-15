"""
Tests for session_custom.py – CustomSessionVerifier & SessionData.

Tests cover:
- SessionData dataclass (serialization, validity, frontend format)
- CustomSessionVerifier singleton & test-mode reset
- Token verification with cache
- Session refresh
- Cache invalidation
"""

import time
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from toolboxv2.utils.workers.session_custom import (
    SessionData,
    CustomSessionVerifier,
    get_session_verifier,
)


class TestSessionData(unittest.TestCase):
    """Tests for the SessionData dataclass."""

    def test_defaults(self):
        sd = SessionData()
        self.assertEqual(sd.user_id, "")
        self.assertEqual(sd.level, 0)
        self.assertFalse(sd.is_valid())  # expires_at == 0 is in the past

    def test_is_valid_when_not_expired(self):
        sd = SessionData(expires_at=time.time() + 3600)
        self.assertTrue(sd.is_valid())

    def test_is_valid_when_expired(self):
        sd = SessionData(expires_at=time.time() - 1)
        self.assertFalse(sd.is_valid())

    def test_to_dict_round_trip(self):
        sd = SessionData(
            user_id="u_1",
            username="alice",
            email="a@e.com",
            level=1,
            provider="discord",
            provider_user_id="d_123",
            token="tok",
            refresh_token="rtok",
            expires_at=9999999999.0,
            created_at=1700000000.0,
            last_validated=1700000001.0,
            provider_data={"avatar": "http://img.example.com/1.png"},
        )
        d = sd.to_dict()
        restored = SessionData.from_dict(d)
        self.assertEqual(restored.user_id, "u_1")
        self.assertEqual(restored.provider, "discord")
        self.assertEqual(restored.provider_data["avatar"], "http://img.example.com/1.png")

    def test_from_dict_with_missing_keys(self):
        """from_dict should gracefully handle missing keys."""
        sd = SessionData.from_dict({"user_id": "u_x"})
        self.assertEqual(sd.user_id, "u_x")
        self.assertEqual(sd.username, "")
        self.assertEqual(sd.level, 0)

    def test_to_frontend_format(self):
        sd = SessionData(
            user_id="u_1",
            username="Alice Smith",
            email="a@e.com",
            level=1,
            token="tok",
            refresh_token="rtok",
            provider="google",
            expires_at=time.time() + 3600,
            provider_data={"avatar": "http://img.example.com/1.png"},
        )
        ff = sd.to_frontend_format()
        self.assertTrue(ff["isAuthenticated"])
        self.assertEqual(ff["userId"], "u_1")
        self.assertEqual(ff["username"], "Alice Smith")
        self.assertEqual(ff["userData"]["firstName"], "Alice")
        self.assertEqual(ff["userData"]["lastName"], "Smith")
        self.assertEqual(ff["provider"], "google")

    def test_to_frontend_format_single_name(self):
        sd = SessionData(username="Alice", expires_at=time.time() + 3600)
        ff = sd.to_frontend_format()
        self.assertEqual(ff["userData"]["firstName"], "Alice")
        self.assertEqual(ff["userData"]["lastName"], "")


class TestCustomSessionVerifier(unittest.IsolatedAsyncioTestCase):
    """Tests for CustomSessionVerifier."""

    def setUp(self):
        """Enable test mode before each test."""
        CustomSessionVerifier.reset_instance()

    def tearDown(self):
        """Clean up singleton state."""
        CustomSessionVerifier._test_mode = False
        CustomSessionVerifier._instance = None

    def test_singleton_in_normal_mode(self):
        CustomSessionVerifier._test_mode = False
        CustomSessionVerifier._instance = None
        app = MagicMock()
        v1 = CustomSessionVerifier(app, "CloudM.Auth")
        v2 = CustomSessionVerifier(app, "CloudM.Auth")
        self.assertIs(v1, v2)
        # Cleanup
        CustomSessionVerifier._instance = None

    def test_test_mode_creates_new_instances(self):
        app = MagicMock()
        v1 = CustomSessionVerifier(app, "CloudM.Auth")
        v2 = CustomSessionVerifier(app, "CloudM.Auth")
        # In test mode __new__ creates fresh object, but __init__ is gated on _initialized
        # Both should exist (test mode allows it)
        self.assertIsNotNone(v1)
        self.assertIsNotNone(v2)

    async def test_verify_empty_token_returns_empty_session(self):
        app = MagicMock()
        verifier = CustomSessionVerifier(app, "CloudM.Auth")
        result = await verifier.verify_session("")
        self.assertEqual(result.user_id, "")
        self.assertFalse(result.is_valid())

    async def test_verify_whitespace_token_returns_empty_session(self):
        app = MagicMock()
        verifier = CustomSessionVerifier(app, "CloudM.Auth")
        result = await verifier.verify_session("   ")
        self.assertEqual(result.user_id, "")

    async def test_verify_session_success(self):
        app = MagicMock()
        mock_result = MagicMock()
        mock_result.ok.return_value = True
        mock_result.unwrap_or.return_value = {
            "user_id": "u_123",
            "user_name": "alice",
            "email": "a@e.com",
            "level": 1,
            "provider": "discord",
            "exp": time.time() + 3600,
        }
        app.a_run_any = AsyncMock(return_value=mock_result)

        verifier = CustomSessionVerifier(app, "CloudM.Auth")
        session = await verifier.verify_session("valid.jwt.token")

        self.assertEqual(session.user_id, "u_123")
        self.assertEqual(session.username, "alice")
        self.assertEqual(session.provider, "discord")
        self.assertTrue(session.is_valid())

        # Verify it was called with the right module/function
        app.a_run_any.assert_called_once_with(
            ("CloudM.Auth", "validate_session"),
            token="valid.jwt.token"
        )

    async def test_verify_session_caches_result(self):
        app = MagicMock()
        mock_result = MagicMock()
        mock_result.ok.return_value = True
        mock_result.unwrap_or.return_value = {
            "user_id": "u_123",
            "user_name": "alice",
            "exp": time.time() + 3600,
        }
        app.a_run_any = AsyncMock(return_value=mock_result)

        verifier = CustomSessionVerifier(app, "CloudM.Auth")
        s1 = await verifier.verify_session("tok1")
        s2 = await verifier.verify_session("tok1")

        # Should only call the auth module once (second time from cache)
        self.assertEqual(app.a_run_any.call_count, 1)
        self.assertEqual(s1.user_id, s2.user_id)

    async def test_verify_session_auth_failure(self):
        app = MagicMock()
        mock_result = MagicMock()
        mock_result.ok.return_value = False
        app.a_run_any = AsyncMock(return_value=mock_result)

        verifier = CustomSessionVerifier(app, "CloudM.Auth")
        session = await verifier.verify_session("bad.token")
        self.assertEqual(session.user_id, "")
        self.assertFalse(session.is_valid())

    async def test_verify_session_exception_returns_empty(self):
        app = MagicMock()
        app.a_run_any = AsyncMock(side_effect=RuntimeError("network error"))
        app.logger = MagicMock()

        verifier = CustomSessionVerifier(app, "CloudM.Auth")
        session = await verifier.verify_session("broken.token")
        self.assertEqual(session.user_id, "")

    async def test_refresh_session_success(self):
        app = MagicMock()
        mock_result = MagicMock()
        mock_result.ok.return_value = True
        mock_result.unwrap_or.return_value = {
            "user_id": "u_123",
            "token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "exp": time.time() + 3600,
        }
        app.a_run_any = AsyncMock(return_value=mock_result)

        verifier = CustomSessionVerifier(app, "CloudM.Auth")
        session = await verifier.refresh_session("old_refresh_token")

        self.assertEqual(session.user_id, "u_123")
        self.assertEqual(session.token, "new_access_token")
        self.assertEqual(session.refresh_token, "new_refresh_token")

    async def test_refresh_empty_token_returns_empty(self):
        app = MagicMock()
        verifier = CustomSessionVerifier(app, "CloudM.Auth")
        session = await verifier.refresh_session("")
        self.assertEqual(session.user_id, "")

    def test_invalidate_session(self):
        app = MagicMock()
        verifier = CustomSessionVerifier(app, "CloudM.Auth")
        # Manually populate cache
        verifier._cache["tok_a"] = SessionData(user_id="u_1")
        self.assertTrue(verifier.invalidate_session("tok_a"))
        self.assertFalse(verifier.invalidate_session("tok_nonexistent"))

    def test_clear_cache(self):
        app = MagicMock()
        verifier = CustomSessionVerifier(app, "CloudM.Auth")
        verifier._cache["tok_a"] = SessionData(user_id="u_1")
        verifier._cache["tok_b"] = SessionData(user_id="u_2")
        verifier.clear_cache()
        self.assertEqual(len(verifier._cache), 0)


class TestGetSessionVerifier(unittest.TestCase):
    """Tests for the get_session_verifier convenience function."""

    def setUp(self):
        CustomSessionVerifier.reset_instance()

    def tearDown(self):
        CustomSessionVerifier._test_mode = False
        CustomSessionVerifier._instance = None

    def test_returns_verifier_instance(self):
        app = MagicMock()
        v = get_session_verifier(app, "CloudM.Auth")
        self.assertIsInstance(v, CustomSessionVerifier)


if __name__ == "__main__":
    unittest.main()
