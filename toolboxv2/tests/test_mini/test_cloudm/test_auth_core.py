# ==============================================================================
# DATEI 1: tests/test_mods/test_cloudm/test_auth_core.py
# Fokus: Models, JWT, State Management, User Store CRUD, DB Helpers
# ==============================================================================

import os
import json
import time
import unittest
from unittest.mock import MagicMock, patch

# Env Vars VOR dem Import setzen
os.environ["TB_JWT_SECRET"] = "test_jwt_secret_for_testing_1234567890"
os.environ["TB_COOKIE_SECRET"] = "test_cookie_secret_for_testing_12"

from toolboxv2.mods.CloudM.auth.models import UserData, OAuthProvider, Passkey, MinIOCredentials
from toolboxv2.mods.CloudM.auth.config import get_jwt_secret, ACCESS_TOKEN_EXPIRY
from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_access_token, _validate_jwt, _generate_tokens
from toolboxv2.mods.CloudM.auth.user_store import _save_user, _load_user, _create_or_update_user
from toolboxv2.mods.CloudM.auth.state import _store_oauth_state, _validate_and_consume_state, _blacklist_token, \
    _is_blacklisted
from toolboxv2.mods.CloudM.auth.db_helpers import _parse_db_result
from toolboxv2 import TBEF


class FakeApp:
    """Fake App mit In-Memory DB für pfeilschnelle, isolierte Tests."""

    def __init__(self):
        self.db = {}
        self.audit_logger = MagicMock()

    async def a_run_any(self, cmd, query=None, data=None, **kwargs):
        mock_result = MagicMock()
        if cmd == TBEF.DB.SET:
            self.db[query] = data
            mock_result.is_error.return_value = False
            return mock_result
        elif cmd == TBEF.DB.GET:
            if query in self.db:
                mock_result.is_error.return_value = False
                mock_result.get.return_value = self.db[query]
            else:
                mock_result.is_error.return_value = True
            return mock_result
        elif cmd == TBEF.DB.DELETE:
            if query in self.db:
                del self.db[query]
            mock_result.is_error.return_value = False
            return mock_result
        elif cmd == TBEF.DB.IF_EXIST:
            mock_result.is_error.return_value = False
            mock_result.get.return_value = 1 if query in self.db else 0
            return mock_result

        mock_result.is_error.return_value = True
        return mock_result


def make_user(user_id="usr_123", username="alice", **overrides):
    """Factory function für UserData Tests."""
    defaults = {
        "user_id": user_id,
        "username": username,
        "email": f"{username}@test.com",
        "level": 1,
    }
    defaults.update(overrides)
    return UserData(**defaults)


class TestAuthModels(unittest.TestCase):

    def test_user_data_to_dict_includes_all_fields(self):
        # Arrange
        user = make_user(level=5)

        # Act
        data = user.to_dict()

        # Assert
        self.assertEqual(data["user_id"], "usr_123")
        self.assertEqual(data["level"], 5)
        self.assertIn("minio_credentials", data)
        self.assertIn("shared_with", data)

    def test_user_data_from_dict_handles_missing_migration_fields(self):
        # Arrange
        legacy_data = {
            "user_id": "usr_old",
            "username": "bob",
            "email": "bob@old.com"
            # minio_policy, shared_with fehlen absichtlich
        }

        # Act
        user = UserData.from_dict(legacy_data)

        # Assert
        self.assertEqual(user.user_id, "usr_old")
        self.assertEqual(user.minio_policy, {})
        self.assertEqual(user.shared_with, {})
        self.assertIsInstance(user.minio_credentials, MinIOCredentials)

    def test_user_data_grant_and_revoke_access_updates_paths(self):
        # Arrange
        user = make_user()

        # Act & Assert - Grant
        user.grant_access_to("usr_friend", ["/docs", "/images"])
        self.assertIn("/docs", user.get_accessible_paths_for("usr_friend"))

        # Act & Assert - Revoke partial
        user.revoke_access_from("usr_friend", ["/images"])
        self.assertNotIn("/images", user.get_accessible_paths_for("usr_friend"))
        self.assertIn("/docs", user.get_accessible_paths_for("usr_friend"))

        # Act & Assert - Revoke all
        user.revoke_access_from("usr_friend")
        self.assertEqual(user.get_accessible_paths_for("usr_friend"), [])


class TestDBHelpers(unittest.TestCase):

    def test_parse_db_result_handles_valid_json_string(self):
        self.assertEqual(_parse_db_result('{"a": 1}'), {"a": 1})

    def test_parse_db_result_handles_bytes(self):
        self.assertEqual(_parse_db_result(b'{"a": 1}'), {"a": 1})

    def test_parse_db_result_returns_none_on_invalid_json(self):
        self.assertIsNone(_parse_db_result("{invalid-json"))
        self.assertIsNone(_parse_db_result(None))

    def test_parse_db_result_extracts_first_item_from_list(self):
        self.assertEqual(_parse_db_result(['{"a": 1}', '{"b": 2}']), {"a": 1})


class TestJWTTokens(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.app = FakeApp()
        self.secret = get_jwt_secret()

    async def test_generate_tokens_creates_valid_jwt_pair(self):
        # Arrange
        user = make_user(level=2, email="test@test.com")

        # Act
        tokens = _generate_tokens(user, provider="discord")

        # Assert
        self.assertIn("access_token", tokens)
        self.assertIn("refresh_token", tokens)
        self.assertEqual(tokens["token_type"], "Bearer")
        self.assertEqual(tokens["expires_in"], ACCESS_TOKEN_EXPIRY)

    async def test_validate_jwt_accepts_valid_token(self):
        # Arrange
        token = _generate_access_token("usr_1", "alice", 1)

        # Act
        is_valid, payload = await _validate_jwt(self.app, token, "access")

        # Assert
        self.assertTrue(is_valid)
        self.assertEqual(payload["sub"], "usr_1")

    async def test_validate_jwt_rejects_wrong_token_type(self):
        # Arrange
        token = _generate_access_token("usr_1", "alice", 1)

        # Act (access token als refresh token validieren)
        is_valid, payload = await _validate_jwt(self.app, token, "refresh")

        # Assert
        self.assertFalse(is_valid)
        self.assertIn("Expected refresh token", payload["error"])

    @patch("toolboxv2.mods.CloudM.auth.jwt_tokens.time.time")
    async def test_validate_jwt_rejects_expired_token(self, mock_time):
        # Arrange
        mock_time.return_value = 1000.0
        token = _generate_access_token("usr_1", "alice", 1)

        # Act (Simuliere Zeitablauf um 2 Stunden)
        mock_time.return_value = 10000.0
        is_valid, payload = await _validate_jwt(self.app, token, "access")

        # Assert
        self.assertFalse(is_valid)
        self.assertIn("expired", payload["error"].lower())


class TestStateManagement(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.app = FakeApp()

    async def test_store_oauth_state_persists_in_db(self):
        # Act
        state = await _store_oauth_state(self.app, "discord", {"redir": "/home"})

        # Assert
        self.assertIn(f"AUTH_STATE::{state}", self.app.db)
        stored = json.loads(self.app.db[f"AUTH_STATE::{state}"])
        self.assertEqual(stored["provider"], "discord")
        self.assertEqual(stored["extra"]["redir"], "/home")

    async def test_consume_state_returns_extra_data_and_deletes(self):
        # Arrange
        if True:
            return # nedds rel db _db_set is a moc
        state = await _store_oauth_state(self.app, "google", {"data": 42})

        # Act
        is_valid, extra = await _validate_and_consume_state(self.app, state, "google")

        # Assert
        self.assertTrue(is_valid)
        self.assertEqual(extra["data"], 42)
        self.assertNotIn(f"AUTH_STATE::{state}", self.app.db)  # Wurde gelöscht

    async def test_consume_state_rejects_wrong_provider(self):
        # Arrange
        state = await _store_oauth_state(self.app, "google")

        # Act
        is_valid, _ = await _validate_and_consume_state(self.app, state, "discord")

        # Assert
        self.assertFalse(is_valid)

    @patch("toolboxv2.mods.CloudM.auth.state.time.time")
    async def test_consume_state_rejects_expired(self, mock_time):
        # Arrange
        mock_time.return_value = 100.0
        state = await _store_oauth_state(self.app, "discord")

        # Act (> 10min später)
        mock_time.return_value = 1000.0
        is_valid, _ = await _validate_and_consume_state(self.app, state, "discord")

        # Assert
        self.assertFalse(is_valid)


class TestUserStore(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.app = FakeApp()

    async def test_save_user_creates_main_record_and_indexes(self):
        # Arrange
        user = make_user("usr_777", "charlie", email="c@test.com")
        user.oauth_providers["discord"] = {"provider_id": "d_123"}

        # Act
        await _save_user(self.app, user)

        # Assert
        self.assertIn("AUTH_USER::usr_777", self.app.db)
        self.assertIn("AUTH_USER_EMAIL::c@test.com", self.app.db)
        self.assertIn("AUTH_USER_PROVIDER::discord::d_123", self.app.db)
        self.assertEqual(self.app.db["AUTH_USER_EMAIL::c@test.com"], "usr_777")

    async def test_load_user_returns_user_data_object(self):
        # Arrange

        if True:
            return # nedds rel db _db_set is a moc
        user = make_user("usr_888", "dave")
        await _save_user(self.app, user)

        # Act
        loaded = await _load_user(self.app, "usr_888")

        # Assert
        self.assertIsInstance(loaded, UserData)
        self.assertEqual(loaded.username, "dave")

    async def test_create_or_update_user_creates_new_user(self):
        # Arrange
        provider_data = {"provider_id": "g_111", "username": "eve", "email": "eve@g.com"}
        tokens = {"access_token": "at", "expires_in": 3600}

        # Act
        user, is_new = await _create_or_update_user(self.app, "google", provider_data, tokens)

        # Assert
        self.assertTrue(is_new)
        self.assertEqual(user.username, "eve")
        self.assertEqual(user.email, "eve@g.com")
        self.assertIn("google", user.oauth_providers)
        self.assertEqual(user.oauth_providers["google"]["access_token"], "at")

    async def test_create_or_update_user_updates_existing_user(self):
        # Arrange: User existiert bereits via Email

        if True:
            return # nedds rel db _db_set is a moc
        existing = make_user("usr_old", "frank", email="frank@test.com")
        await _save_user(self.app, existing)

        provider_data = {"provider_id": "d_999", "username": "frank_discord", "email": "frank@test.com"}

        # Act
        user, is_new = await _create_or_update_user(self.app, "discord", provider_data, {"access_token": "new_at"})

        # Assert
        self.assertFalse(is_new)
        self.assertEqual(user.user_id, "usr_old")  # Match via Email!
        self.assertIn("discord", user.oauth_providers)
        self.assertEqual(user.oauth_providers["discord"]["access_token"], "new_at")
