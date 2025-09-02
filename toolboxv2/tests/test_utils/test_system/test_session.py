import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

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

    def tearDown(self):
        # Stop the app patcher
        self.app_patcher.stop()

    @async_test
    async def test_session_initialization(self):
        """Test basic session initialization"""
        session = Session('test_user')

        # Check basic attributes
        self.assertEqual(session.username, 'test_user')
        self.assertIsNone(session.session)
        self.assertFalse(session.valid)

        # Check base URL
        self.assertEqual(session.base, os.environ.get("TOOLBOXV2_REMOTE_BASE", "https://simplecore.app"))

    @async_test
    async def test_invalid_login_with_mock(self):
        """Test login failure scenarios"""
        # Create a session
        session = Session('test_user')

        # Mock the authentication methods
        with patch.object(session, 'auth_with_prv_key', new_callable=AsyncMock) as mock_auth:
            # Simulate authentication failure
            mock_auth.return_value = False

            # Attempt to log in
            result = await session.login()

            # Assert login failed
            self.assertFalse(result)

            # Verify auth_with_prv_key was called
            mock_auth.assert_called_once()

    @async_test
    async def test_download_file_without_session(self):
        """Test file download without an active session"""
        session = Session('test_user')
        session.session = None

        # Try to download file without initializing session
        with self.assertRaises(Exception) as context:
            await session.download_file('http://example.com/test.txt')

        self.assertIn('Session not initialized', str(context.exception))

    @async_test
    async def test_logout(self):
        """Test logout functionality with correct async mocking."""
        session_instance = Session('test_user')

        # 1. Erstellen Sie einen Mock für das Response-Objekt
        mock_response = MagicMock()
        mock_response.status = 200
        # Da die Response im `async with`-Block verwendet wird, müssen wir das magische `__aenter__` mocken.
        mock_response.__aenter__.return_value = mock_response

        # 2. Konfigurieren Sie den `AsyncMock` für die `post`-Methode
        # Er soll eine Coroutine zurückgeben, die beim `await` den `mock_response` liefert.
        mock_session = AsyncMock()
        mock_session.post.return_value = mock_response
        mock_session.closed = False  # Die Session ist anfangs offen

        # 3. Weisen Sie den Mock der Session-Instanz zu
        session_instance.session = mock_session

        # Führen Sie den Logout durch
        result = await session_instance.logout()

        # Überprüfen Sie die Ergebnisse
        self.assertTrue(result, "Logout should return True on success")
        self.assertIsNone(session_instance.session, "Session object should be set to None after logout")

        # Überprüfen Sie, ob die Methoden des Mocks aufgerufen wurden
        mock_session.post.assert_awaited_once_with(f'{session_instance.base}/web/logoutS')
        mock_session.close.assert_awaited_once()

    @async_test
    async def test_init_log_in_mk_link_invalid(self):
        """Test initialization with an invalid login link"""
        session = Session('test_user')

        # Test with an invalid or empty invitation link
        result = await session.init_log_in_mk_link('/')

        # Should return False or an error result
        self.assertFalse(result)

    def test_get_prv_key(self):
        """Test retrieval of private key"""
        # Ensure the test directory exists
        os.makedirs(os.path.join(self.test_dir), exist_ok=True)

        # Mock the key generation and saving
        with patch('toolboxv2.utils.system.session.Code.load_keys_from_files') as mock_load_keys:
            mock_load_keys.return_value = ('mock_pub_key', 'mock_prv_key')

            session = Session('test_user')
            prv_key = session.get_prv_key()

            # Assert private key is returned
            self.assertEqual(prv_key, 'mock_prv_key')

if __name__ == '__main__':
    unittest.main()
