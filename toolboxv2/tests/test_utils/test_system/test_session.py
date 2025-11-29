import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import json

from toolboxv2.tests.a_util import async_test
from toolboxv2.utils.system.session import Session


class TestSession(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

        # Mock the get_app() to prevent actual application initialization
        self.app_patcher = patch('toolboxv2.utils.system.session.get_app')
        self.mock_get_app = self.app_patcher.start()

        # Create a mock app instance
        self.mock_app = MagicMock()
        self.mock_app.info_dir = self.test_dir
        self.mock_app.id = 'test_app'
        self.mock_app.get_username.return_value = 'test_user'
        self.mock_get_app.return_value = self.mock_app

        # Mock get_logger to prevent logging issues
        self.logger_patcher = patch('toolboxv2.utils.system.session.get_logger')
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

        # Reset singleton instances before each test
        Session._instances = {}

    def tearDown(self):
        # Stop all patchers
        self.app_patcher.stop()
        self.logger_patcher.stop()

        # Clean up singleton instances
        Session._instances = {}

    @async_test
    async def test_session_initialization(self):
        """Test basic session initialization"""
        session = Session('test_user')

        # Check basic attributes
        self.assertEqual(session.username, 'test_user')
        self.assertIsNone(session._session)
        self.assertFalse(session.valid)
        self.assertIsNone(session.clerk_user_id)
        self.assertIsNone(session.clerk_session_token)

        # Check base URL
        expected_base = os.environ.get("TOOLBOXV2_REMOTE_BASE", "https://simplecore.app")
        self.assertEqual(session.base, expected_base)

    @async_test
    async def test_session_initialization_with_custom_base(self):
        """Test session initialization with custom base URL"""
        custom_base = "https://custom.example.com"
        session = Session('test_user', base=custom_base)

        self.assertEqual(session.base, custom_base)

    @async_test
    async def test_session_initialization_strips_api_suffix(self):
        """Test that /api/ suffix is properly stripped from base URL"""
        session = Session('test_user', base="https://example.com/api/")

        self.assertEqual(session.base, "https://example.com")

    @async_test
    async def test_invalid_login_no_token(self):
        """Test login failure when no session token exists"""
        session = Session('test_user')

        # Mock BlobFile to simulate no stored token
        with patch('toolboxv2.utils.system.session.BlobFile') as mock_blob:
            mock_blob.return_value.__enter__.return_value.read.return_value = b''

            # Attempt to log in
            result = await session.login(verbose=False)

            # Assert login failed
            self.assertFalse(result)
            self.assertFalse(session.valid)

    @async_test
    async def test_invalid_login_with_invalid_token(self):
        """Test login failure with invalid session token"""
        session = Session('test_user')

        # Mock stored session token
        mock_session_data = {
            "token": "invalid_token",
            "user_id": "test_user_id",
            "username": "test_user"
        }

        with patch('toolboxv2.utils.system.session.BlobFile') as mock_blob:
            mock_blob.return_value.__enter__.return_value.read.return_value = \
                json.dumps(mock_session_data).encode()

            # Mock the HTTP response to return authentication failure
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.json = AsyncMock(return_value={
                "result": {"authenticated": False}
            })
            mock_response.__aenter__.return_value = mock_response

            with patch.object(session, '_ensure_session'):
                session._session = AsyncMock()
                session._session.request.return_value = mock_response

                # Attempt to log in
                result = await session.login(verbose=False)

                # Assert login failed
                self.assertFalse(result)
                self.assertFalse(session.valid)

    @async_test
    async def test_logout_with_active_session(self):
        """Test logout functionality with active session"""
        session = Session('test_user')

        # Setup mock session data
        session.clerk_user_id = "test_user_id"
        session.clerk_session_token = "test_token"
        session.valid = True
        session.username = "test_user"

        # Mock the HTTP response for sign out
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response

        # Mock the session object
        mock_session = AsyncMock()
        mock_session.post.return_value = mock_response
        mock_session.closed = False
        mock_session.close = AsyncMock()

        # Mock BlobFile for clearing token
        with patch('toolboxv2.utils.system.session.BlobFile') as mock_blob:
            mock_blob.return_value.__enter__.return_value.clear = MagicMock()

            with patch.object(session, '_ensure_session'):
                session._session = mock_session

                # Perform logout
                result = await session.logout()

                # Verify results
                self.assertTrue(result)
                self.assertFalse(session.valid)
                self.assertIsNone(session.username)
                self.assertIsNone(session.clerk_session_token)
                self.assertIsNone(session.clerk_user_id)

                # Verify session was closed
                mock_session.close.assert_awaited_once()

    @async_test
    async def test_logout_without_session(self):
        """Test logout when no active session exists"""
        session = Session('test_user')
        session._session = None

        with patch('toolboxv2.utils.system.session.BlobFile') as mock_blob:
            mock_blob.return_value.__enter__.return_value.clear = MagicMock()

            result = await session.logout()

            self.assertTrue(result)
            self.assertFalse(session.valid)

    @async_test
    async def test_save_and_load_session_token(self):
        """Test saving and loading session tokens"""
        session = Session('test_user')

        test_token = "test_session_token"
        test_user_id = "test_user_123"

        # Mock BlobFile for saving
        with patch('toolboxv2.utils.system.session.BlobFile') as mock_blob:
            mock_file = MagicMock()
            mock_blob.return_value.__enter__.return_value = mock_file

            # Save token
            result = session._save_session_token(test_token, test_user_id)

            self.assertTrue(result)
            self.assertEqual(session.clerk_session_token, test_token)
            self.assertEqual(session.clerk_user_id, test_user_id)

            # Verify write was called
            mock_file.write.assert_called_once()

    @async_test
    async def test_get_auth_headers(self):
        """Test authentication header generation"""
        session = Session('test_user')

        # Without token
        headers = session._get_auth_headers()
        self.assertEqual(headers, {})

        # With token
        session.clerk_session_token = "test_token"
        headers = session._get_auth_headers()
        self.assertEqual(headers, {"Authorization": "Bearer test_token"})

    @async_test
    async def test_fetch_with_relative_url(self):
        """Test fetch with relative URL"""
        session = Session('test_user', base="https://example.com")

        mock_response = AsyncMock()
        mock_response.status = 200

        with patch.object(session, '_ensure_session'):
            session._session = AsyncMock()
            session._session.get.return_value = mock_response

            response = await session.fetch('/api/test', method='GET')

            # Verify the full URL was constructed
            session._session.get.assert_called_once()
            call_args = session._session.get.call_args
            self.assertEqual(call_args[0][0], 'https://example.com/api/test')

    @async_test
    async def test_singleton_behavior(self):
        """Test that Session behaves as a singleton"""
        session1 = Session('user1')
        session2 = Session('user1')

        # Should return the same instance
        self.assertIs(session1, session2)

        # Different username should create different instance
        Session._instances = {}
        session3 = Session('user2')
        self.assertIsNot(session1, session3)


if __name__ == '__main__':
    unittest.main()
