import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

from toolboxv2.utils.singelton_class import Singleton
from toolboxv2.utils.system.session import Session


def _reset_session_singleton():
    """Reset Session singleton for test isolation."""
    try:
        from toolboxv2.utils.system.session import Session
        if Session in Singleton._instances:
            del Singleton._instances[Session]
            Singleton._args.pop(Session, None)
            Singleton._kwargs.pop(Session, None)
    except (ImportError, KeyError):
        pass


class TestSessionStorage(unittest.TestCase):
    """Unit tests for encrypted BlobStorage interactions."""

    def setUp(self):
        _reset_session_singleton()
        self.env_patcher = patch.dict(os.environ, {"TB_DATA_DIR": "/fake/dir"})
        self.env_patcher.start()

        # Mock the storage subsystem to prevent any real I/O
        self.mock_storage_patcher = patch('toolboxv2.utils.system.session.BlobStorage')
        self.mock_storage_cls = self.mock_storage_patcher.start()

        from toolboxv2.utils.system.session import Session
        self.sess = Session(username="testuser", base="http://localhost:8000")

    def tearDown(self):
        self.env_patcher.stop()
        self.mock_storage_patcher.stop()
        _reset_session_singleton()

    def test_get_blob_name_sanitizes_special_characters(self):
        # Arrange
        self.sess.username = "user@evil/../../name!"

        # Act
        blob_name = self.sess._get_blob_name()

        # Assert
        self.assertEqual(blob_name, "user_evil_______name__session.json")

    def test_get_blob_name_empty_username_uses_default(self):
        # Arrange
        self.sess.username = None

        # Act
        blob_name = self.sess._get_blob_name()

        # Assert
        self.assertEqual(blob_name, "default_session.json")

    @patch('toolboxv2.utils.system.session.BlobFile')
    @patch('toolboxv2.utils.system.session.Code')
    def test_save_session_token_writes_encrypted_json(self, mock_code, mock_blob_file):
        # Arrange
        mock_code.DK.return_value.return_value = "fake-device-key"
        mock_blob_instance = mock_blob_file.return_value.__enter__.return_value

        # Act
        result = self.sess._save_session_token("access_123", "refresh_456", "user_789")

        # Assert
        self.assertTrue(result)
        self.assertEqual(self.sess.access_token, "access_123")

        mock_blob_file.assert_called_with(
            "testuser_session.json",
            mode="w",
            key="fake-device-key",
            storage=self.sess._storage
        )
        mock_blob_instance.write_json.assert_called_once()
        written_data = mock_blob_instance.write_json.call_args[0][0]
        self.assertEqual(written_data["access_token"], "access_123")
        self.assertEqual(written_data["username"], "testuser")

    @patch('toolboxv2.utils.system.session.BlobFile')
    def test_load_session_token_restores_state(self, mock_blob_file):
        # Arrange
        self.sess._storage.list_blobs.return_value = [{"blob_id": "testuser_session.json"}]
        mock_blob_instance = mock_blob_file.return_value.__enter__.return_value
        mock_blob_instance.read_json.return_value = {
            "access_token": "acc_loaded",
            "refresh_token": "ref_loaded",
            "user_id": "user_loaded",
            "username": "testuser",
            "base_url": "https://remote.app"
        }

        # Temporarily clear base to test restoration
        del os.environ["TOOLBOXV2_REMOTE_BASE"]

        # Act
        data = self.sess._load_session_token()

        # Assert
        self.assertIsNotNone(data)
        self.assertEqual(self.sess.access_token, "acc_loaded")
        self.assertEqual(self.sess.base, "https://remote.app")

    def test_clear_session_token_deletes_blob_and_clears_attributes(self):
        # Arrange
        self.sess.access_token = "some_token"
        self.sess._storage.list_blobs.return_value = [{"blob_id": "testuser_session.json"}]

        # Act
        result = self.sess._clear_session_token()

        # Assert
        self.assertTrue(result)
        self.assertIsNone(self.sess.access_token)
        self.sess._storage.delete_blob.assert_called_with("testuser_session.json")


class TestSessionAuth(unittest.IsolatedAsyncioTestCase):
    """Unit tests for asynchronous authentication flows."""

    async def asyncSetUp(self):
        _reset_session_singleton()
        self.env_patcher = patch.dict(os.environ, {"TB_DATA_DIR": "/fake/dir"})
        self.env_patcher.start()

        from toolboxv2.utils.system.session import Session
        self.sess = Session(username="testuser", base="http://test.local")
        self.sess._storage = MagicMock()

        # Initialize loop/session
        self.sess._ensure_session()

        # Intercept aiohttp requests
        self.mock_request = AsyncMock()
        self.sess._session.request = self.mock_request

    async def asyncTearDown(self):
        await self.sess.cleanup()
        self.env_patcher.stop()
        _reset_session_singleton()

    @patch.object(Session, "_load_session_token")
    async def test_login_without_token_returns_false(self, mock_load):
        # Arrange
        mock_load.return_value = None

        # Act
        success = await self.sess.login()

        # Assert
        self.assertFalse(success)
        self.assertFalse(self.sess.valid)

    @patch.object(Session, "_load_session_token")
    async def test_login_valid_token_sets_session_valid(self, mock_load):
        # Arrange
        mock_load.return_value = {"access_token": "valid_token"}

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {"result": {"authenticated": True, "username": "server_user"}}
        self.mock_request.return_value.__aenter__.return_value = mock_resp

        # Act
        success = await self.sess.login()

        # Assert
        self.assertTrue(success)
        self.assertTrue(self.sess.valid)
        self.assertEqual(self.sess.username, "server_user")

        self.mock_request.assert_called_with(
            "POST",
            url="http://test.local/api/CloudM.Auth/validate_session",
            json={"token": "valid_token"}
        )

    @patch.object(Session, "_load_session_token")
    @patch.object(Session, "_save_session_token")
    async def test_login_invalid_token_triggers_successful_refresh(self, mock_save, mock_load):
        # Arrange
        mock_load.return_value = {"access_token": "expired", "refresh_token": "refresh_val"}

        # Validation response (fails)
        mock_val_resp = AsyncMock()
        mock_val_resp.status = 401

        # Refresh response (succeeds)
        mock_ref_resp = AsyncMock()
        mock_ref_resp.status = 200
        mock_ref_resp.json.return_value = {
            "result": {"access_token": "new_access", "refresh_token": "new_refresh"}
        }

        # Sequence of mocked returns for context managers
        self.mock_request.return_value.__aenter__.side_effect = [mock_val_resp, mock_ref_resp]

        # Act
        success = await self.sess.login()

        # Assert
        self.assertTrue(success)
        self.assertTrue(self.sess.valid)
        mock_save.assert_called_with("new_access", "new_refresh", None)

    async def test_login_with_magic_link_sends_correct_payload(self):
        # Arrange
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {"error": 0, "result": {"sent": True}}
        self.mock_request.return_value.__aenter__.return_value = mock_resp

        # Act
        res = await self.sess.login_with_magic_link("test@example.com")

        # Assert
        self.assertFalse(res.is_error())
        self.mock_request.assert_called_with(
            "POST",
            url="http://test.local/api/CloudM.Auth/request_magic_link",
            json={"email": "test@example.com"}
        )

    @patch.object(Session, "_save_session_token")
    async def test_verify_magic_link_saves_token_on_success(self, mock_save):
        # Arrange
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {
            "error": 0,
            "result": {"access_token": "token1", "refresh_token": "token2", "user_id": "u1", "username": "bob"}
        }
        self.mock_request.return_value.__aenter__.return_value = mock_resp

        # Act
        res = await self.sess.verify_magic_link("magic-code")

        # Assert
        self.assertFalse(res.is_error())
        self.assertTrue(self.sess.valid)
        self.assertEqual(self.sess.username, "bob")
        mock_save.assert_called_with("token1", "token2", "u1")


class TestSessionNetwork(unittest.IsolatedAsyncioTestCase):
    """Unit tests for file and network wrapper functionalities."""

    async def asyncSetUp(self):
        _reset_session_singleton()
        from toolboxv2.utils.system.session import Session
        self.sess = Session(username="testuser", base="http://test.local")
        self.sess.access_token = "auth_token_123"
        self.sess._storage = MagicMock()
        self.sess._ensure_session()

    async def asyncTearDown(self):
        await self.sess.cleanup()
        _reset_session_singleton()

    async def test_fetch_get_adds_auth_headers_and_uses_session(self):
        # Arrange
        mock_resp = AsyncMock()
        self.sess._session.get = AsyncMock(return_value=mock_resp)

        # Act
        await self.sess.fetch("/api/data", method="GET", headers={"Custom": "1"})

        # Assert
        self.sess._session.get.assert_called_with(
            "http://test.local/api/data",
            headers={"Custom": "1", "Authorization": "Bearer auth_token_123"}
        )

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("os.makedirs")
    async def test_download_file_writes_chunks_to_disk(self, mock_makedirs, mock_file):
        # Arrange
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.content.read = AsyncMock(side_effect=[b"chunk1", b"chunk2", b""])

        mock_get = AsyncMock()
        mock_get.return_value.__aenter__.return_value = mock_resp
        self.sess._session.get = mock_get

        # Act
        success = await self.sess.download_file("http://test.local/file.zip", "test_out")

        # Assert
        self.assertTrue(success)
        mock_makedirs.assert_called_with("test_out", exist_ok=True)

        # Verify writing logic
        handle = mock_file()
        handle.write.assert_any_call(b"chunk1")
        handle.write.assert_any_call(b"chunk2")

        expected_path = os.path.join("test_out", "file.zip")
        mock_file.assert_called_with(expected_path, 'wb')
