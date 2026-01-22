"""
Tests for CloudM AuthClerk module.

Tests cover:
- JWT token generation
- Token refresh functionality
- Session token storage
- CLI verification flow
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from toolboxv2 import Result


class TestSessionTokenStorage(unittest.TestCase):
    """Tests for session token storage functions."""

    @patch('toolboxv2.mods.CloudM.AuthClerk.BlobFile')
    @patch('toolboxv2.mods.CloudM.AuthClerk.Code')
    def test_save_session_token_success(self, mock_code, mock_blobfile):
        """Test saving session token successfully."""
        from toolboxv2.mods.CloudM.AuthClerk import save_session_token

        mock_code.DK.return_value = lambda: "test_key"
        mock_blob_instance = MagicMock()
        mock_blobfile.return_value.__enter__ = MagicMock(return_value=mock_blob_instance)
        mock_blobfile.return_value.__exit__ = MagicMock(return_value=None)

        result = save_session_token(
            identifier="user_123",
            token="jwt_token_abc",
            username="testuser",
            session_id="sess_456"
        )

        self.assertTrue(result)
        mock_blob_instance.clear.assert_called_once()
        mock_blob_instance.write.assert_called_once()

        # Verify the written data contains session_id
        written_data = mock_blob_instance.write.call_args[0][0]
        parsed = json.loads(written_data.decode())
        self.assertEqual(parsed["token"], "jwt_token_abc")
        self.assertEqual(parsed["username"], "testuser")
        self.assertEqual(parsed["session_id"], "sess_456")
        self.assertIn("created_at", parsed)

    @patch('toolboxv2.mods.CloudM.AuthClerk.BlobFile')
    @patch('toolboxv2.mods.CloudM.AuthClerk.Code')
    def test_save_session_token_without_session_id(self, mock_code, mock_blobfile):
        """Test saving session token without session_id."""
        from toolboxv2.mods.CloudM.AuthClerk import save_session_token

        mock_code.DK.return_value = lambda: "test_key"
        mock_blob_instance = MagicMock()
        mock_blobfile.return_value.__enter__ = MagicMock(return_value=mock_blob_instance)
        mock_blobfile.return_value.__exit__ = MagicMock(return_value=None)

        result = save_session_token(
            identifier="user_123",
            token="jwt_token_abc",
            username="testuser"
        )

        self.assertTrue(result)
        written_data = mock_blob_instance.write.call_args[0][0]
        parsed = json.loads(written_data.decode())
        self.assertIsNone(parsed["session_id"])

    @patch('toolboxv2.mods.CloudM.AuthClerk.BlobFile')
    @patch('toolboxv2.mods.CloudM.AuthClerk.Code')
    def test_load_session_token_success(self, mock_code, mock_blobfile):
        """Test loading session token successfully."""
        from toolboxv2.mods.CloudM.AuthClerk import load_session_token

        mock_code.DK.return_value = lambda: "test_key"
        session_data = {
            "token": "jwt_token",
            "username": "testuser",
            "session_id": "sess_123",
            "created_at": time.time()
        }
        mock_blob_instance = MagicMock()
        mock_blob_instance.read.return_value = json.dumps(session_data).encode()
        mock_blobfile.return_value.__enter__ = MagicMock(return_value=mock_blob_instance)
        mock_blobfile.return_value.__exit__ = MagicMock(return_value=None)

        result = load_session_token("user_123")

        self.assertIsNotNone(result)
        self.assertEqual(result["token"], "jwt_token")
        self.assertEqual(result["session_id"], "sess_123")

    @patch('toolboxv2.mods.CloudM.AuthClerk.BlobFile')
    @patch('toolboxv2.mods.CloudM.AuthClerk.Code')
    def test_load_session_token_not_found(self, mock_code, mock_blobfile):
        """Test loading non-existent session token."""
        from toolboxv2.mods.CloudM.AuthClerk import load_session_token

        mock_code.DK.return_value = lambda: "test_key"
        mock_blobfile.side_effect = FileNotFoundError("Not found")

        result = load_session_token("nonexistent_user")

        self.assertIsNone(result)


class TestRefreshJwtToken(unittest.IsolatedAsyncioTestCase):
    """Tests for JWT token refresh functionality."""

    @patch('toolboxv2.mods.CloudM.AuthClerk.get_clerk_client')
    @patch('toolboxv2.mods.CloudM.AuthClerk.load_local_user_data')
    @patch('toolboxv2.mods.CloudM.AuthClerk.save_local_user_data')
    @patch('toolboxv2.mods.CloudM.AuthClerk.save_session_token')
    @patch('toolboxv2.mods.CloudM.AuthClerk.get_app')
    @patch('toolboxv2.mods.CloudM.AuthClerk.get_logger')
    async def test_refresh_jwt_token_success(
        self, mock_get_logger, mock_get_app, mock_save_token, mock_save_local, mock_load_local, mock_get_clerk
    ):
        """Test successful JWT token refresh."""
        from toolboxv2.mods.CloudM.AuthClerk import refresh_jwt_token

        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Mock Clerk client
        mock_clerk = MagicMock()
        mock_session = MagicMock()
        mock_session.user_id = "user_123"
        mock_session.status = "active"  # Session must be active
        mock_clerk.sessions.get.return_value = mock_session

        mock_token_response = MagicMock()
        mock_token_response.jwt = "new_jwt_token"
        mock_clerk.sessions.create_token_from_template.return_value = mock_token_response
        mock_get_clerk.return_value = mock_clerk

        # Mock local user data
        mock_local_data = MagicMock()
        mock_local_data.username = "testuser"
        mock_load_local.return_value = mock_local_data

        mock_app = MagicMock()

        result = await refresh_jwt_token(
            mock_app,
            session_id="sess_123",
            clerk_user_id="user_123"
        )

        self.assertTrue(result.is_ok())
        data = result.get()
        self.assertEqual(data["session_token"], "new_jwt_token")
        self.assertTrue(data["success"])

    async def test_refresh_jwt_token_no_session_id(self):
        """Test refresh fails without session_id or user_id."""
        from toolboxv2.mods.CloudM.AuthClerk import refresh_jwt_token

        mock_app = MagicMock()

        result = await refresh_jwt_token(
            mock_app,
            session_id=None,
            clerk_user_id=None
        ).print()

        self.assertTrue(result.is_error())


if __name__ == "__main__":
    unittest.main()

