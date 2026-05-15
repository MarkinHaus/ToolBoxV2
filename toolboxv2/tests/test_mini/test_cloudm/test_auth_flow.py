"""
Tests for CloudM.Auth sub-modules and end-to-end auth flows.

Covers:
- Models (OAuthProvider, Passkey, UserData) serialization
- Config (constants, environment helpers)
- JWT token generation and validation
- OAuth state round-trip (store -> retrieve -> delete)
- WebAuthn challenge round-trip
- Token blacklisting
- User storage CRUD
- Token bridge page rendering (web + Tauri contexts)
- Base64url encoding/decoding for WebAuthn
"""

import os
import time
import json
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# =================== Model Tests ===================


class TestModels(unittest.TestCase):
    """Test data model serialization."""

    def test_user_data_roundtrip(self):
        from toolboxv2.mods.CloudM.auth.models import UserData
        user = UserData(
            user_id="usr_abc123",
            username="testuser",
            email="test@example.com",
            level=2,
        )
        d = user.to_dict()
        self.assertEqual(d["user_id"], "usr_abc123")
        self.assertEqual(d["username"], "testuser")
        self.assertEqual(d["email"], "test@example.com")
        self.assertEqual(d["level"], 2)
        self.assertIsInstance(d["oauth_providers"], dict)
        self.assertIsInstance(d["passkeys"], list)

        user2 = UserData.from_dict(d)
        self.assertEqual(user2.user_id, user.user_id)
        self.assertEqual(user2.username, user.username)

    def test_user_data_from_dict_ignores_extra_keys(self):
        from toolboxv2.mods.CloudM.auth.models import UserData
        d = {"user_id": "usr_x", "username": "x", "email": "x@x.com", "extra_field": "ignored"}
        user = UserData.from_dict(d)
        self.assertEqual(user.user_id, "usr_x")
        self.assertFalse(hasattr(user, "extra_field"))

    def test_oauth_provider_roundtrip(self):
        from toolboxv2.mods.CloudM.auth.models import OAuthProvider
        p = OAuthProvider(provider_id="123", provider="discord", username="testuser")
        d = p.to_dict()
        p2 = OAuthProvider.from_dict(d)
        self.assertEqual(p2.provider_id, "123")
        self.assertEqual(p2.provider, "discord")

    def test_passkey_roundtrip(self):
        from toolboxv2.mods.CloudM.auth.models import Passkey
        pk = Passkey(credential_id="abc", public_key="def", sign_count=5)
        d = pk.to_dict()
        pk2 = Passkey.from_dict(d)
        self.assertEqual(pk2.credential_id, "abc")
        self.assertEqual(pk2.sign_count, 5)


# =================== Config Tests ===================


class TestConfig(unittest.TestCase):
    """Test config constants and env helpers."""

    def test_constants_positive(self):
        from toolboxv2.mods.CloudM.auth.config import (
            ACCESS_TOKEN_EXPIRY, REFRESH_TOKEN_EXPIRY,
            STATE_EXPIRY, CHALLENGE_EXPIRY, MAGIC_LINK_EXPIRY, DEVICE_INVITE_EXPIRY,
        )
        self.assertGreater(ACCESS_TOKEN_EXPIRY, 0)
        self.assertGreater(REFRESH_TOKEN_EXPIRY, ACCESS_TOKEN_EXPIRY)
        self.assertGreater(STATE_EXPIRY, 0)
        self.assertGreater(CHALLENGE_EXPIRY, 0)
        self.assertGreater(MAGIC_LINK_EXPIRY, 0)
        self.assertGreater(DEVICE_INVITE_EXPIRY, 0)

    @patch.dict(os.environ, {"TB_ENV": "production"})
    def test_is_production(self):
        from toolboxv2.mods.CloudM.auth.config import is_production
        self.assertTrue(is_production())

    @patch.dict(os.environ, {"TB_ENV": "development"})
    def test_is_not_production(self):
        from toolboxv2.mods.CloudM.auth.config import is_production
        self.assertFalse(is_production())

    @patch.dict(os.environ, {"TB_JWT_SECRET": "test_secret_key"})
    def test_get_jwt_secret(self):
        from toolboxv2.mods.CloudM.auth.config import get_jwt_secret
        self.assertEqual(get_jwt_secret(), "test_secret_key")

    @patch.dict(os.environ, {}, clear=True)
    def test_get_jwt_secret_missing_raises(self):
        # Clear all potentially relevant env vars
        for key in ["TB_JWT_SECRET", "TB_COOKIE_SECRET"]:
            os.environ.pop(key, None)
        from toolboxv2.mods.CloudM.auth.config import get_jwt_secret
        with self.assertRaises(ValueError):
            get_jwt_secret()

    @patch.dict(os.environ, {"APP_BASE_URL": "http://localhost:9999"})
    def test_get_base_url_dev(self):
        os.environ.pop("TB_ENV", None)
        from toolboxv2.mods.CloudM.auth.config import get_base_url
        self.assertIn("9999", get_base_url())

    def test_discord_config_structure(self):
        from toolboxv2.mods.CloudM.auth.config import get_discord_config
        config = get_discord_config()
        self.assertIn("client_id", config)
        self.assertIn("client_secret", config)
        self.assertIn("redirect_uri", config)
        self.assertIn("authorize_url", config)
        self.assertIn("token_url", config)

    def test_google_config_structure(self):
        from toolboxv2.mods.CloudM.auth.config import get_google_config
        config = get_google_config()
        self.assertIn("client_id", config)
        self.assertIn("scopes", config)
        self.assertIn("openid", config["scopes"])

    def test_passkey_config_structure(self):
        from toolboxv2.mods.CloudM.auth.config import get_passkey_config
        config = get_passkey_config()
        self.assertIn("rp_id", config)
        self.assertIn("rp_name", config)
        self.assertIn("origin", config)


# =================== JWT Token Tests ===================


class TestJWTTokens(unittest.TestCase):
    """Test JWT generation and validation."""

    @patch.dict(os.environ, {"TB_JWT_SECRET": "test_jwt_secret_for_testing_1234567890"})
    def test_generate_access_token(self):
        from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_access_token
        import jwt
        token = _generate_access_token("usr_123", "testuser", 2, "discord")
        payload = jwt.decode(token, "test_jwt_secret_for_testing_1234567890", algorithms=["HS256"])
        self.assertEqual(payload["sub"], "usr_123")
        self.assertEqual(payload["username"], "testuser")
        self.assertEqual(payload["level"], 2)
        self.assertEqual(payload["provider"], "discord")
        self.assertEqual(payload["type"], "access")
        self.assertIn("jti", payload)

    @patch.dict(os.environ, {"TB_JWT_SECRET": "test_jwt_secret_for_testing_1234567890"})
    def test_generate_refresh_token(self):
        from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_refresh_token
        import jwt
        token = _generate_refresh_token("usr_123")
        payload = jwt.decode(token, "test_jwt_secret_for_testing_1234567890", algorithms=["HS256"])
        self.assertEqual(payload["sub"], "usr_123")
        self.assertEqual(payload["type"], "refresh")

    @patch.dict(os.environ, {"TB_JWT_SECRET": "test_jwt_secret_for_testing_1234567890"})
    def test_generate_tokens_pair(self):
        from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_tokens
        from toolboxv2.mods.CloudM.auth.models import UserData
        user = UserData(user_id="usr_x", username="x", email="x@x.com", level=1)
        tokens = _generate_tokens(user, "google")
        self.assertIn("access_token", tokens)
        self.assertIn("refresh_token", tokens)
        self.assertIn("expires_in", tokens)
        self.assertEqual(tokens["token_type"], "Bearer")

    @patch.dict(os.environ, {"TB_JWT_SECRET": "test_jwt_secret_for_testing_1234567890"})
    def test_validate_jwt_valid(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_access_token, _validate_jwt
        token = _generate_access_token("usr_x", "x", 1)
        mock_app = MagicMock()
        # Mock _is_blacklisted to return False
        mock_result = MagicMock()
        mock_result.is_error.return_value = True  # key doesn't exist = not blacklisted
        mock_app.a_run_any = AsyncMock(return_value=mock_result)
        valid, payload = asyncio.get_event_loop().run_until_complete(
            _validate_jwt(mock_app, token, "access")
        )
        self.assertTrue(valid)
        self.assertEqual(payload["sub"], "usr_x")

    @patch.dict(os.environ, {"TB_JWT_SECRET": "test_jwt_secret_for_testing_1234567890"})
    def test_validate_jwt_wrong_type(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_access_token, _validate_jwt
        token = _generate_access_token("usr_x", "x", 1)
        mock_app = MagicMock()
        valid, payload = asyncio.get_event_loop().run_until_complete(
            _validate_jwt(mock_app, token, "refresh")  # Wrong type
        )
        self.assertFalse(valid)
        self.assertIn("Expected refresh token", payload["error"])

    def test_validate_jwt_no_token(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.jwt_tokens import _validate_jwt
        mock_app = MagicMock()
        valid, payload = asyncio.get_event_loop().run_until_complete(
            _validate_jwt(mock_app, "", "access")
        )
        self.assertFalse(valid)

    @patch.dict(os.environ, {"TB_JWT_SECRET": "test_jwt_secret_for_testing_1234567890"})
    def test_validate_jwt_expired(self):
        import asyncio
        import jwt as pyjwt
        from toolboxv2.mods.CloudM.auth.jwt_tokens import _validate_jwt
        # Create an expired token
        token = pyjwt.encode(
            {"sub": "usr_x", "type": "access", "exp": time.time() - 100, "jti": "test"},
            "test_jwt_secret_for_testing_1234567890",
            algorithm="HS256",
        )
        mock_app = MagicMock()
        valid, payload = asyncio.get_event_loop().run_until_complete(
            _validate_jwt(mock_app, token, "access")
        )
        self.assertFalse(valid)
        self.assertIn("expired", payload["error"].lower())

    @patch.dict(os.environ, {"TB_JWT_SECRET": "test_jwt_secret_for_testing_1234567890"})
    def test_validate_jwt_blacklisted(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_access_token, _validate_jwt
        token = _generate_access_token("usr_x", "x", 1)

        mock_app = MagicMock()
        # Mock _db_exists to return True (token is blacklisted)
        mock_result = MagicMock()
        mock_result.is_error.return_value = False
        mock_result.get.return_value = 1
        mock_app.a_run_any = AsyncMock(return_value=mock_result)

        valid, payload = asyncio.get_event_loop().run_until_complete(
            _validate_jwt(mock_app, token, "access")
        )
        self.assertFalse(valid)
        self.assertIn("revoked", payload["error"].lower())


# =================== DB Helpers Tests ===================


class TestDBHelpers(unittest.TestCase):
    """Test DB helper parsing."""

    def test_parse_db_result_none(self):
        from toolboxv2.mods.CloudM.auth.db_helpers import _parse_db_result
        self.assertIsNone(_parse_db_result(None))

    def test_parse_db_result_json_string(self):
        from toolboxv2.mods.CloudM.auth.db_helpers import _parse_db_result
        result = _parse_db_result('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})

    def test_parse_db_result_bytes(self):
        from toolboxv2.mods.CloudM.auth.db_helpers import _parse_db_result
        result = _parse_db_result(b'{"key": "value"}')
        self.assertEqual(result, {"key": "value"})

    def test_parse_db_result_list(self):
        from toolboxv2.mods.CloudM.auth.db_helpers import _parse_db_result
        result = _parse_db_result(['{"key": "value"}'])
        self.assertEqual(result, {"key": "value"})

    def test_parse_db_result_empty_list(self):
        from toolboxv2.mods.CloudM.auth.db_helpers import _parse_db_result
        self.assertIsNone(_parse_db_result([]))

    def test_parse_db_result_invalid_json(self):
        from toolboxv2.mods.CloudM.auth.db_helpers import _parse_db_result
        self.assertIsNone(_parse_db_result("not json"))

    def test_parse_db_result_dict_passthrough(self):
        from toolboxv2.mods.CloudM.auth.db_helpers import _parse_db_result
        d = {"already": "parsed"}
        self.assertEqual(_parse_db_result(d), d)


# =================== State Management Tests ===================


class TestStateManagement(unittest.TestCase):
    """Test OAuth state, challenges, blacklist (all DB-backed)."""

    def test_store_oauth_state(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.state import _store_oauth_state
        mock_app = MagicMock()
        mock_result = MagicMock()
        mock_result.is_error.return_value = False
        mock_app.a_run_any = AsyncMock(return_value=mock_result)

        state = asyncio.get_event_loop().run_until_complete(
            _store_oauth_state(mock_app, "discord", {"redirect_after": "http://localhost"})
        )
        self.assertIsInstance(state, str)
        self.assertGreater(len(state), 10)
        # Verify DB.SET was called
        mock_app.a_run_any.assert_called()

    def test_validate_and_consume_state(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.state import _validate_and_consume_state
        mock_app = MagicMock()
        # Mock _db_get returning valid state
        mock_result = MagicMock()
        mock_result.is_error.return_value = False
        mock_result.get.return_value = json.dumps({
            "provider": "discord",
            "created_at": time.time(),
            "extra": {"redirect_after": "http://tauri.localhost"},
        })
        # Mock _db_delete
        mock_delete = MagicMock()
        mock_delete.is_error.return_value = False
        mock_app.a_run_any = AsyncMock(side_effect=[mock_result, mock_delete])

        valid, extra = asyncio.get_event_loop().run_until_complete(
            _validate_and_consume_state(mock_app, "test_state", "discord")
        )
        self.assertTrue(valid)
        self.assertEqual(extra.get("redirect_after"), "http://tauri.localhost")

    def test_validate_state_wrong_provider(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.state import _validate_and_consume_state
        mock_app = MagicMock()
        mock_result = MagicMock()
        mock_result.is_error.return_value = False
        mock_result.get.return_value = json.dumps({
            "provider": "google",  # Stored as google
            "created_at": time.time(),
            "extra": {},
        })
        mock_delete = MagicMock()
        mock_delete.is_error.return_value = False
        mock_app.a_run_any = AsyncMock(side_effect=[mock_result, mock_delete])

        valid, _ = asyncio.get_event_loop().run_until_complete(
            _validate_and_consume_state(mock_app, "test_state", "discord")  # Validate as discord
        )
        self.assertFalse(valid)

    def test_validate_state_expired(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.state import _validate_and_consume_state
        mock_app = MagicMock()
        mock_result = MagicMock()
        mock_result.is_error.return_value = False
        mock_result.get.return_value = json.dumps({
            "provider": "discord",
            "created_at": time.time() - 700,  # > 600s STATE_EXPIRY
            "extra": {},
        })
        mock_delete = MagicMock()
        mock_delete.is_error.return_value = False
        mock_app.a_run_any = AsyncMock(side_effect=[mock_result, mock_delete])

        valid, _ = asyncio.get_event_loop().run_until_complete(
            _validate_and_consume_state(mock_app, "test_state", "discord")
        )
        self.assertFalse(valid)


# =================== Token Bridge Tests (Web vs Tauri) ===================


class TestTokenBridge(unittest.TestCase):
    """Test the token bridge HTML that server_worker returns after OAuth callback."""

    def _get_bridge_html(self, **kwargs):
        from toolboxv2.utils.workers.server_worker import AuthHandler
        return AuthHandler._build_token_bridge_html(**kwargs)

    def test_bridge_html_contains_token(self):
        html = self._get_bridge_html(
            token="test_token_123",
            refresh_token="refresh_456",
            user_id="usr_abc",
            username="testuser",
        )
        self.assertIn("test_token_123", html)
        self.assertIn("refresh_456", html)
        self.assertIn("usr_abc", html)
        self.assertIn("testuser", html)

    def test_bridge_html_with_redirect_after(self):
        """Tauri context: redirect_after is the Tauri origin."""
        html = self._get_bridge_html(
            token="tk",
            refresh_token="rtk",
            user_id="uid",
            username="user",
            redirect_after="tauri://localhost",
        )
        self.assertIn("tauri://localhost", html)

    def test_bridge_html_stores_in_localstorage(self):
        """Bridge must write to tbjs_user_session localStorage."""
        html = self._get_bridge_html(
            token="tk",
            refresh_token="rtk",
            user_id="uid",
            username="user",
        )
        self.assertIn("tbjs_user_session", html)
        self.assertIn("localStorage.setItem", html)

    def test_bridge_html_with_error(self):
        html = self._get_bridge_html(error="OAuth failed")
        self.assertIn("OAuth failed", html)

    def test_bridge_html_no_sensitive_data_in_error(self):
        """Error page should not contain tokens."""
        html = self._get_bridge_html(error="Something went wrong")
        self.assertNotIn("access_token", html)


# =================== Base64url Encoding Tests ===================


class TestBase64urlEncoding(unittest.TestCase):
    """Test base64url <-> ArrayBuffer encoding used for WebAuthn.

    Both user.js and custom_auth.js must handle base64url from py_webauthn.
    """

    def test_base64url_to_standard_base64(self):
        """Simulate what JS _base64ToArrayBuffer does: base64url -> standard base64 -> atob."""
        import base64
        # py_webauthn outputs base64url (- instead of +, _ instead of /, no padding)
        test_bytes = b"\x00\x01\x02\xff\xfe\xfd"
        b64url = base64.urlsafe_b64encode(test_bytes).decode().rstrip("=")

        # Simulate JS conversion: base64url -> standard base64
        standard = b64url.replace("-", "+").replace("_", "/")
        pad = (4 - len(standard) % 4) % 4
        standard += "=" * pad

        # Decode
        decoded = base64.b64decode(standard)
        self.assertEqual(decoded, test_bytes)

    def test_standard_base64_to_base64url(self):
        """Simulate what JS _arrayBufferToBase64 does: btoa -> base64url."""
        import base64
        test_bytes = b"\x00\x01\x02\xff\xfe\xfd"
        standard = base64.b64encode(test_bytes).decode()

        # Simulate JS conversion: standard base64 -> base64url
        b64url = standard.replace("+", "-").replace("/", "_").rstrip("=")

        # Verify it matches urlsafe encoding
        expected = base64.urlsafe_b64encode(test_bytes).decode().rstrip("=")
        self.assertEqual(b64url, expected)

    def test_challenge_roundtrip(self):
        """Simulate a full challenge round-trip: server generates -> JS receives -> JS sends back."""
        import base64
        import secrets

        # Server generates challenge (py_webauthn)
        challenge_bytes = secrets.token_bytes(32)
        challenge_b64url = base64.urlsafe_b64encode(challenge_bytes).decode().rstrip("=")

        # JS receives base64url, converts to ArrayBuffer (simulated)
        standard = challenge_b64url.replace("-", "+").replace("_", "/")
        pad = (4 - len(standard) % 4) % 4
        standard += "=" * pad
        js_decoded = base64.b64decode(standard)
        self.assertEqual(js_decoded, challenge_bytes)

        # JS sends back as base64url
        js_encoded = base64.b64encode(js_decoded).decode()
        js_b64url = js_encoded.replace("+", "-").replace("/", "_").rstrip("=")
        self.assertEqual(js_b64url, challenge_b64url)


# =================== User Store Tests ===================


class TestUserStore(unittest.TestCase):
    """Test user storage CRUD with mocked DB."""

    def test_save_user_stores_indexes(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.user_store import _save_user
        from toolboxv2.mods.CloudM.auth.models import UserData

        mock_app = MagicMock()
        mock_result = MagicMock()
        mock_result.is_error.return_value = False
        mock_app.a_run_any = AsyncMock(return_value=mock_result)

        user = UserData(
            user_id="usr_test",
            username="testuser",
            email="test@example.com",
            oauth_providers={"discord": {"provider_id": "disc_123"}},
        )

        asyncio.get_event_loop().run_until_complete(_save_user(mock_app, user))

        # Should have been called for: user profile, email index, provider index
        calls = mock_app.a_run_any.call_args_list
        call_queries = [str(c) for c in calls]
        call_str = " ".join(call_queries)
        self.assertIn("AUTH_USER::usr_test", call_str)
        self.assertIn("AUTH_USER_EMAIL::test@example.com", call_str)
        self.assertIn("AUTH_USER_PROVIDER::discord::disc_123", call_str)

    def test_load_user(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.user_store import _load_user

        mock_app = MagicMock()
        mock_result = MagicMock()
        mock_result.is_error.return_value = False
        mock_result.get.return_value = json.dumps({
            "user_id": "usr_test",
            "username": "testuser",
            "email": "test@example.com",
            "level": 2,
            "created_at": 1000,
            "last_login": 2000,
            "settings": {},
            "mod_data": {},
            "oauth_providers": {},
            "passkeys": [],
        })
        mock_app.a_run_any = AsyncMock(return_value=mock_result)

        user = asyncio.get_event_loop().run_until_complete(_load_user(mock_app, "usr_test"))
        self.assertIsNotNone(user)
        self.assertEqual(user.user_id, "usr_test")
        self.assertEqual(user.level, 2)

    def test_load_user_not_found(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.user_store import _load_user

        mock_app = MagicMock()
        mock_result = MagicMock()
        mock_result.is_error.return_value = True
        mock_app.a_run_any = AsyncMock(return_value=mock_result)

        user = asyncio.get_event_loop().run_until_complete(_load_user(mock_app, "nonexistent"))
        self.assertIsNone(user)

    def test_create_or_update_user_new(self):
        import asyncio
        from toolboxv2.mods.CloudM.auth.user_store import _create_or_update_user

        mock_app = MagicMock()
        # First call: _find_user_by_provider -> not found
        # Second call: _find_user_by_email -> not found
        # Third+ calls: _save_user writes
        not_found = MagicMock()
        not_found.is_error.return_value = True
        save_ok = MagicMock()
        save_ok.is_error.return_value = False
        mock_app.a_run_any = AsyncMock(side_effect=[not_found, not_found, save_ok, save_ok, save_ok])

        user, is_new = asyncio.get_event_loop().run_until_complete(
            _create_or_update_user(
                mock_app,
                "discord",
                {"provider_id": "disc_999", "username": "newuser", "email": "new@test.com"},
                {"access_token": "at", "refresh_token": "rt", "expires_in": 3600},
            )
        )
        self.assertTrue(is_new)
        self.assertEqual(user.username, "newuser")
        self.assertIn("discord", user.oauth_providers)


# =================== Integration: Auth Module Import ===================


class TestAuthModuleImports(unittest.TestCase):
    """Verify all symbols are accessible through the facade Auth.py."""

    def test_models_accessible(self):
        from toolboxv2.mods.CloudM.Auth import OAuthProvider, Passkey, UserData
        self.assertTrue(callable(UserData))

    def test_config_accessible(self):
        from toolboxv2.mods.CloudM.Auth import (
            JWT_ALGORITHM, ACCESS_TOKEN_EXPIRY, get_discord_config, get_passkey_config,
        )
        self.assertEqual(JWT_ALGORITHM, "HS256")
        self.assertGreater(ACCESS_TOKEN_EXPIRY, 0)

    def test_internal_functions_accessible(self):
        from toolboxv2.mods.CloudM.Auth import (
            _load_user, _save_user, _find_user_by_provider, _find_user_by_email,
            _generate_tokens, _validate_jwt,
            _store_oauth_state, _validate_and_consume_state,
        )
        self.assertTrue(callable(_load_user))
        self.assertTrue(callable(_generate_tokens))

    def test_api_functions_accessible(self):
        from toolboxv2.mods.CloudM.Auth import (
            get_auth_config, get_discord_auth_url, get_google_auth_url,
            login_discord, login_google,
            validate_session, refresh_token, logout,
            get_user_data, update_user_data, list_users,
            passkey_register_start, passkey_register_finish,
            passkey_login_start, passkey_login_finish,
            request_magic_link, verify_magic_link,
            create_device_invite, verify_device_invite,
        )
        self.assertTrue(callable(validate_session))
        self.assertTrue(callable(login_discord))

    def test_name_and_version(self):
        from toolboxv2.mods.CloudM.Auth import Name, version
        self.assertEqual(Name, "CloudM.Auth")
        self.assertEqual(version, "2.0.0")

    def test_widget_import_still_works(self):
        """UI/widget.py imports UserData as LocalUserData — must still work."""
        from toolboxv2.mods.CloudM.Auth import UserData as LocalUserData
        self.assertTrue(callable(LocalUserData))


if __name__ == "__main__":
    unittest.main()
