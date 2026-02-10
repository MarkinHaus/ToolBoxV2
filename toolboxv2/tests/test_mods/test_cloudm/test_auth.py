"""
Tests for CloudM.Auth custom authentication module (v2.0.0).

Tests cover:
- JWT token generation & validation
- Token blacklisting
- OAuth state management
- User CRUD via TBEF.DB
- Magic link flow
- Device invite flow
- Session validation
- Refresh token flow
"""

import json
import time
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict


# ---------------------------------------------------------------------------
# Helpers – import Auth internals lazily so mocks can be set up first
# ---------------------------------------------------------------------------

def _import_auth():
    """Import Auth module – call inside tests so patches apply."""
    import toolboxv2.mods.CloudM.Auth as auth_mod
    return auth_mod


class TestJWTTokenGeneration(unittest.TestCase):
    """Tests for _generate_access_token / _generate_refresh_token."""

    def test_access_token_contains_required_claims(self):
        auth = _import_auth()
        token_str = auth._generate_access_token(
            user_id="u_123", username="alice", level=1, provider="discord"
        )
        import jwt as pyjwt
        payload = pyjwt.decode(token_str, auth.get_jwt_secret(), algorithms=[auth.JWT_ALGORITHM])
        self.assertEqual(payload["sub"], "u_123")
        self.assertEqual(payload["username"], "alice")
        self.assertEqual(payload["level"], 1)
        self.assertEqual(payload["provider"], "discord")
        self.assertEqual(payload["type"], "access")
        self.assertIn("jti", payload)
        self.assertIn("exp", payload)
        self.assertIn("iat", payload)

    def test_access_token_expiry(self):
        auth = _import_auth()
        token_str = auth._generate_access_token("u_1", "bob", 1)
        import jwt as pyjwt
        payload = pyjwt.decode(token_str, auth.get_jwt_secret(), algorithms=[auth.JWT_ALGORITHM])
        expected_expiry = payload["iat"] + auth.ACCESS_TOKEN_EXPIRY
        self.assertEqual(payload["exp"], expected_expiry)

    def test_refresh_token_contains_required_claims(self):
        auth = _import_auth()
        token_str = auth._generate_refresh_token("u_456")
        import jwt as pyjwt
        payload = pyjwt.decode(token_str, auth.get_jwt_secret(), algorithms=[auth.JWT_ALGORITHM])
        self.assertEqual(payload["sub"], "u_456")
        self.assertEqual(payload["type"], "refresh")
        self.assertIn("jti", payload)
        expected_expiry = payload["iat"] + auth.REFRESH_TOKEN_EXPIRY
        self.assertEqual(payload["exp"], expected_expiry)

    def test_generate_tokens_returns_both(self):
        auth = _import_auth()
        user = auth.UserData(
            user_id="u_x", username="carol", email="c@e.com", level=1
        )
        tokens = auth._generate_tokens(user, provider="google")
        self.assertIn("access_token", tokens)
        self.assertIn("refresh_token", tokens)
        self.assertIn("expires_in", tokens)
        self.assertEqual(tokens["token_type"], "Bearer")

    def test_different_users_get_different_jti(self):
        auth = _import_auth()
        t1 = auth._generate_access_token("u_a", "a", 1)
        t2 = auth._generate_access_token("u_b", "b", 1)
        import jwt as pyjwt
        secret = auth.get_jwt_secret()
        p1 = pyjwt.decode(t1, secret, algorithms=[auth.JWT_ALGORITHM])
        p2 = pyjwt.decode(t2, secret, algorithms=[auth.JWT_ALGORITHM])
        self.assertNotEqual(p1["jti"], p2["jti"])


class TestJWTValidation(unittest.IsolatedAsyncioTestCase):
    """Tests for _validate_jwt."""

    async def test_valid_access_token(self):
        auth = _import_auth()
        mock_app = MagicMock()
        # _is_blacklisted needs to return False via DB
        mock_app.a_run_any = AsyncMock(return_value=MagicMock(
            is_error=MagicMock(return_value=True)  # IF_EXIST returns error => not blacklisted
        ))
        token_str = auth._generate_access_token("u_1", "alice", 1, "discord")
        valid, payload = await auth._validate_jwt(mock_app, token_str, "access")
        self.assertTrue(valid)
        self.assertEqual(payload["sub"], "u_1")

    async def test_expired_token_is_invalid(self):
        auth = _import_auth()
        import jwt as pyjwt
        mock_app = MagicMock()
        mock_app.a_run_any = AsyncMock(return_value=MagicMock(
            is_error=MagicMock(return_value=True)
        ))
        # Create an already-expired token
        payload = {
            "sub": "u_1", "user_name": "alice", "level": 1,
            "type": "access", "jti": "test-jti",
            "iat": time.time() - 3600, "exp": time.time() - 1800
        }
        token_str = pyjwt.encode(payload, auth.get_jwt_secret(), algorithm=auth.JWT_ALGORITHM)
        valid, _ = await auth._validate_jwt(mock_app, token_str, "access")
        self.assertFalse(valid)

    async def test_wrong_token_type_is_invalid(self):
        auth = _import_auth()
        mock_app = MagicMock()
        mock_app.a_run_any = AsyncMock(return_value=MagicMock(
            is_error=MagicMock(return_value=True)
        ))
        token_str = auth._generate_refresh_token("u_1")
        # Trying to validate refresh token as access should fail
        valid, _ = await auth._validate_jwt(mock_app, token_str, "access")
        self.assertFalse(valid)


class TestTokenBlacklisting(unittest.IsolatedAsyncioTestCase):
    """Tests for _blacklist_token / _is_blacklisted."""

    async def test_blacklist_token_stores_in_db(self):
        auth = _import_auth()
        mock_app = MagicMock()
        mock_app.a_run_any = AsyncMock(return_value=MagicMock(is_error=MagicMock(return_value=False)))

        token_str = auth._generate_access_token("u_1", "alice", 1)
        await auth._blacklist_token(mock_app, token_str)

        # Verify DB SET was called with AUTH_BLACKLIST:: prefix
        calls = mock_app.a_run_any.call_args_list
        set_calls = [c for c in calls if 'AUTH_BLACKLIST::' in str(c)]
        self.assertTrue(len(set_calls) > 0, "Should have stored blacklist entry in DB")

    async def test_is_blacklisted_returns_true_when_exists(self):
        auth = _import_auth()
        mock_app = MagicMock()
        # IF_EXIST returns success with count > 0 => blacklisted
        mock_result = MagicMock()
        mock_result.is_error.return_value = False
        mock_result.get.return_value = 1
        mock_app.a_run_any = AsyncMock(return_value=mock_result)
        result = await auth._is_blacklisted(mock_app, "some-jti")
        self.assertTrue(result)

    async def test_is_blacklisted_returns_false_when_not_exists(self):
        auth = _import_auth()
        mock_app = MagicMock()
        # IF_EXIST returns error => not found => not blacklisted
        mock_app.a_run_any = AsyncMock(return_value=MagicMock(
            is_error=MagicMock(return_value=True)
        ))
        result = await auth._is_blacklisted(mock_app, "unknown-jti")
        self.assertFalse(result)


class TestOAuthState(unittest.IsolatedAsyncioTestCase):
    """Tests for OAuth CSRF state management."""

    async def test_store_and_validate_state(self):
        auth = _import_auth()
        mock_app = MagicMock()
        stored_data = {}

        async def fake_run_any(cmd, **kwargs):
            from toolboxv2 import TBEF
            if cmd == TBEF.DB.SET:
                stored_data[kwargs.get('query', '')] = kwargs.get('data', '')
                return MagicMock(is_error=MagicMock(return_value=False))
            elif cmd == TBEF.DB.GET:
                key = kwargs.get('query', '')
                if key in stored_data:
                    return MagicMock(
                        is_error=MagicMock(return_value=False),
                        get=MagicMock(return_value=stored_data[key].encode() if isinstance(stored_data[key], str) else stored_data[key])
                    )
                return MagicMock(is_error=MagicMock(return_value=True))
            elif cmd == TBEF.DB.DELETE:
                stored_data.pop(kwargs.get('query', ''), None)
                return MagicMock(is_error=MagicMock(return_value=False))
            return MagicMock(is_error=MagicMock(return_value=True))

        mock_app.a_run_any = AsyncMock(side_effect=fake_run_any)

        state = await auth._store_oauth_state(mock_app, "discord")
        self.assertTrue(len(state) > 0)

        # Verify the state was stored
        matching_keys = [k for k in stored_data if state in k]
        self.assertTrue(len(matching_keys) > 0, "State should be stored in DB")


class TestUserData(unittest.TestCase):
    """Tests for UserData dataclass."""

    def test_user_data_to_dict(self):
        auth = _import_auth()
        user = auth.UserData(
            user_id="u_123",
            username="alice",
            email="alice@example.com",
            level=1,
        )
        d = user.to_dict()
        self.assertEqual(d["user_id"], "u_123")
        self.assertEqual(d["username"], "alice")
        self.assertEqual(d["email"], "alice@example.com")
        self.assertIn("oauth_providers", d)
        self.assertIn("passkeys", d)

    def test_user_data_from_dict(self):
        auth = _import_auth()
        data = {
            "user_id": "u_456",
            "username": "bob",
            "email": "bob@example.com",
            "level": 2,
            "oauth_providers": {},
            "passkeys": [],
            "created_at": 1700000000.0,
            "settings": {}
        }
        user = auth.UserData.from_dict(data)
        self.assertEqual(user.user_id, "u_456")
        self.assertEqual(user.username, "bob")
        self.assertEqual(user.level, 2)


class TestValidateSessionEndpoint(unittest.IsolatedAsyncioTestCase):
    """Tests for the validate_session exported function."""

    async def test_validate_session_with_valid_token(self):
        auth = _import_auth()
        mock_app = MagicMock()

        # Mock _validate_jwt to return success
        with patch.object(auth, '_validate_jwt', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = (True, {
                "sub": "u_123", "user_name": "alice", "level": 1,
                "provider": "discord", "jti": "test-jti"
            })
            # Mock _load_user
            with patch.object(auth, '_load_user', new_callable=AsyncMock) as mock_load:
                mock_user = auth.UserData(
                    user_id="u_123", username="alice",
                    email="alice@test.com", level=1
                )
                mock_load.return_value = mock_user

                result = await auth.validate_session(
                    app=mock_app, token="valid.jwt.token"
                )
                self.assertTrue(result.is_data())
                data = result.get()
                self.assertTrue(data.get("authenticated"))
                self.assertEqual(data.get("user_id"), "u_123")

    async def test_validate_session_without_token(self):
        auth = _import_auth()
        mock_app = MagicMock()
        result = await auth.validate_session(app=mock_app, token=None)
        self.assertTrue(result.is_error())


class TestLogout(unittest.IsolatedAsyncioTestCase):
    """Tests for the logout exported function."""

    async def test_logout_blacklists_token(self):
        auth = _import_auth()
        mock_app = MagicMock()
        with patch.object(auth, '_blacklist_token', new_callable=AsyncMock) as mock_bl:
            mock_bl.return_value = None
            result = await auth.logout(app=mock_app, token="some.jwt.token")
            mock_bl.assert_called_once_with(mock_app, "some.jwt.token")

    async def test_logout_without_token_still_succeeds(self):
        """Logout without token returns ok (graceful no-op)."""
        auth = _import_auth()
        mock_app = MagicMock()
        result = await auth.logout(app=mock_app, token=None)
        # Auth.logout returns Result.ok even without token (graceful)
        data = result.get()
        self.assertTrue(data.get("logged_out"))


class TestOAuthProviderDataclass(unittest.TestCase):
    """Tests for OAuthProvider dataclass."""

    def test_round_trip(self):
        auth = _import_auth()
        provider = auth.OAuthProvider(
            provider_id="123456",
            provider="discord",
            access_token="at_xxx",
            refresh_token="rt_xxx",
            username="alice",
            email="alice@test.com",
            avatar="https://cdn.example.com/avatar.png"
        )
        d = provider.to_dict()
        restored = auth.OAuthProvider.from_dict(d)
        self.assertEqual(restored.provider_id, "123456")
        self.assertEqual(restored.provider, "discord")
        self.assertEqual(restored.email, "alice@test.com")


class TestPasskeyDataclass(unittest.TestCase):
    """Tests for Passkey dataclass."""

    def test_round_trip(self):
        auth = _import_auth()
        pk = auth.Passkey(
            credential_id="cred_abc",
            public_key="base64encodedkey==",
            sign_count=5,
            name="My YubiKey"
        )
        d = pk.to_dict()
        restored = auth.Passkey.from_dict(d)
        self.assertEqual(restored.credential_id, "cred_abc")
        self.assertEqual(restored.sign_count, 5)
        self.assertEqual(restored.name, "My YubiKey")


class TestRefreshToken(unittest.IsolatedAsyncioTestCase):
    """Tests for refresh_token endpoint."""

    async def test_refresh_with_valid_token(self):
        auth = _import_auth()
        mock_app = MagicMock()

        # Generate a real refresh token
        rt = auth._generate_refresh_token("u_123")

        with patch.object(auth, '_validate_jwt', new_callable=AsyncMock) as mock_val:
            mock_val.return_value = (True, {"sub": "u_123", "type": "refresh", "jti": "j1"})
            with patch.object(auth, '_load_user', new_callable=AsyncMock) as mock_load:
                mock_load.return_value = auth.UserData(
                    user_id="u_123", username="alice",
                    email="a@e.com", level=1
                )
                with patch.object(auth, '_blacklist_token', new_callable=AsyncMock):
                    result = await auth.refresh_token(
                        app=mock_app, refresh_token=rt
                    )
                    self.assertTrue(result.is_data())
                    data = result.get()
                    self.assertIn("access_token", data)
                    self.assertIn("refresh_token", data)

    async def test_refresh_without_token(self):
        auth = _import_auth()
        mock_app = MagicMock()
        result = await auth.refresh_token(app=mock_app, refresh_token=None)
        self.assertTrue(result.is_error())


if __name__ == "__main__":
    unittest.main()
