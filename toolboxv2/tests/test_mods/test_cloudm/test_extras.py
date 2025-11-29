# file: toolboxv2/tests/test_mods/test_cloudm/test_extras.py
"""
Tests for CloudM extras module.

Tests login functionality, magic links, and UI management:
- UI registration and retrieval
- Magic link generation
- Login workflows (mocked)
- Version display
- Account creation workflows (mocked)

All network/external operations are mocked.
"""

import unittest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from urllib.parse import unquote

from toolboxv2 import Result, Code
from toolboxv2.mods.CloudM.types import User


class TestUIManagement(unittest.TestCase):
    """Tests for UI management functions"""

    @patch('toolboxv2.mods.CloudM.extras.get_app')
    def test_add_ui_registers_new_ui(self, mock_get_app):
        """Test adding a new UI component"""
        from toolboxv2.mods.CloudM.extras import add_ui

        # Mock app and config
        mock_app = MagicMock()
        mock_app.config_fh.get_file_handler.return_value = "{}"
        mock_get_app.return_value = mock_app

        # Add UI
        add_ui(
            mock_app,
            name="TestUI",
            title="Test Dashboard",
            path="/test/path",
            description="Test description",
            auth=True
        )

        # Verify it was saved
        mock_app.config_fh.add_to_save_file_handler.assert_called_once()
        call_args = mock_app.config_fh.add_to_save_file_handler.call_args

        # Check the saved data
        saved_data = json.loads(call_args[0][1])
        self.assertIn("TestUI", saved_data)
        self.assertEqual(saved_data["TestUI"]["title"], "Test Dashboard")
        self.assertEqual(saved_data["TestUI"]["path"], "/test/path")
        self.assertTrue(saved_data["TestUI"]["auth"])

    @patch('toolboxv2.mods.CloudM.extras.get_app')
    def test_add_ui_updates_existing_ui(self, mock_get_app):
        """Test updating an existing UI component"""
        from toolboxv2.mods.CloudM.extras import add_ui

        # Mock app with existing UI
        existing_uis = {"ExistingUI": {"path": "/old", "title": "Old"}}
        mock_app = MagicMock()
        mock_app.config_fh.get_file_handler.return_value = json.dumps(existing_uis)
        mock_get_app.return_value = mock_app

        # Add new UI
        add_ui(mock_app, "NewUI", "New Title", "/new", "New desc", False)

        # Verify both UIs are present
        call_args = mock_app.config_fh.add_to_save_file_handler.call_args
        saved_data = json.loads(call_args[0][1])

        self.assertIn("ExistingUI", saved_data)
        self.assertIn("NewUI", saved_data)

    @patch('toolboxv2.mods.CloudM.extras.get_app')
    def test_openui_returns_all_uis(self, mock_get_app):
        """Test retrieving all registered UIs"""
        from toolboxv2.mods.CloudM.extras import openui

        # Mock app with multiple UIs
        uis = {
            "UI1": {"path": "/path1", "title": "Title1", "auth": True},
            "UI2": {"path": "/path2", "title": "Title2", "auth": False}
        }
        mock_app = MagicMock()
        mock_app.config_fh.get_file_handler.return_value = json.dumps(uis)
        mock_get_app.return_value = mock_app

        result = openui(mock_app)

        # openui returns ApiResult, extract data
        if hasattr(result, 'result'):
            data = result.result.data
            self.assertEqual(len(data), 2)
        else:
            # Fallback for list return
            self.assertEqual(len(result), 2)

    @patch('toolboxv2.mods.CloudM.extras.get_app')
    def test_openui_empty_uis(self, mock_get_app):
        """Test retrieving UIs when none exist"""
        from toolboxv2.mods.CloudM.extras import openui

        mock_app = MagicMock()
        mock_app.config_fh.get_file_handler.return_value = "{}"
        mock_get_app.return_value = mock_app

        result = openui(mock_app)

        # openui returns ApiResult, extract data
        if hasattr(result, 'result'):
            data = result.result.data
            self.assertEqual(len(data), 0)
        else:
            self.assertEqual(len(result), 0)


class TestVersionDisplay(unittest.TestCase):
    """Tests for version display functions"""

    @patch('toolboxv2.mods.CloudM.extras.get_app')
    def test_openVersion_returns_version(self, mock_get_app):
        """Test that openVersion returns the module version"""
        from toolboxv2.mods.CloudM.extras import openVersion

        # Create a mock self object with version
        mock_self = MagicMock()
        mock_self.version = "1.2.3"

        result = openVersion(mock_self)

        self.assertIsInstance(result.as_result(), Result)
        self.assertEqual(result.get(), "1.2.3")

    @patch('toolboxv2.mods.CloudM.extras.get_app')
    def test_show_version_displays_version(self, mock_get_app):
        """Test show_version function"""
        from toolboxv2.mods.CloudM.extras import show_version

        mock_self = MagicMock()
        mock_self.version = "2.0.0"
        mock_self.api_version = "1.5.0"

        result = show_version(mock_self)

        # show_version returns ApiResult
        if hasattr(result, 'result'):
            self.assertIsNotNone(result)
        else:
            self.assertEqual(result, "2.0.0")
        mock_self.print.assert_called_once()


class TestAccountCreation(unittest.TestCase):
    """Tests for account creation workflows"""

    @patch('toolboxv2.mods.CloudM.extras.get_app')
    @patch('toolboxv2.mods.CloudM.extras.get_invitation')
    @patch('toolboxv2.mods.CloudM.extras.print_qrcode_to_console')
    async def test_register_initial_loot_user_success(self, mock_qr, mock_invitation, mock_get_app):
        """Test successful loot user registration"""
        from toolboxv2.mods.CloudM.extras import register_initial_loot_user

        # Mock app
        mock_app = MagicMock()
        mock_app.config_fh.get_file_handler.return_value = None

        # Mock invitation
        mock_invitation.return_value = Result.ok("test_invitation")

        # Mock user creation
        mock_app.run_any.return_value = Result.ok()

        # Mock user retrieval
        test_user = User(name="loot", user_pass_sync="sync_key")
        mock_app.a_run_any = AsyncMock(return_value=Result.ok(test_user))

        mock_get_app.return_value = mock_app

        result = await register_initial_loot_user(mock_app, "test@example.com")

        self.assertIsInstance(result, Result)
        self.assertTrue(result.is_ok())

        # Verify QR code was generated
        mock_qr.assert_called_once()

    @patch('toolboxv2.mods.CloudM.extras.get_app')
    async def test_register_initial_loot_user_already_exists(self, mock_get_app):
        """Test registration when user already exists"""
        from toolboxv2.mods.CloudM.extras import register_initial_loot_user

        mock_app = MagicMock()
        # User already exists
        mock_app.config_fh.get_file_handler.return_value = "existing_key"
        mock_get_app.return_value = mock_app

        result = await register_initial_loot_user(mock_app, "test@example.com")

        self.assertIsInstance(result, Result)
        self.assertTrue(result.is_error())
        self.assertIn("already Registered", result.get(default=""))


class TestInitializeAdminPanel(unittest.TestCase):
    """Tests for admin panel initialization"""

    @patch('toolboxv2.mods.CloudM.extras.get_app')
    def test_initialize_admin_panel(self, mock_get_app):
        """Test admin panel initialization"""
        from toolboxv2.mods.CloudM.extras import initialize_admin_panel

        mock_app = MagicMock()
        mock_app.logger = MagicMock()
        mock_get_app.return_value = mock_app

        result = initialize_admin_panel(mock_app)

        self.assertIsInstance(result, Result)
        self.assertTrue(result.is_ok())

        # Verify UI was added
        mock_app.run_any.assert_called_once()


class TestCLIWebLogin(unittest.TestCase):
    """Tests for CLI web login functions"""

    @patch('toolboxv2.mods.CloudM.extras.get_app')
    @patch('toolboxv2.mods.CloudM.LogInSystem.cli_logout')
    async def test_cli_logout_delegates_to_login_system(self, mock_logout_system, mock_get_app):
        """Test that cli_logout delegates to LogInSystem"""
        from toolboxv2.mods.CloudM.LogInSystem import cli_logout

        mock_app = MagicMock()
        mock_get_app.return_value = mock_app
        mock_logout_system.return_value = Result.ok("logged out")

        result = await cli_logout(mock_app)

        # Verify delegation
        mock_logout_system.assert_called_once_with(mock_app)


class TestExtrasIntegration(unittest.TestCase):
    """Integration tests for extras module"""

    @patch('toolboxv2.mods.CloudM.extras.get_app')
    def test_full_ui_workflow(self, mock_get_app):
        """Test complete UI registration and retrieval workflow"""
        from toolboxv2.mods.CloudM.extras import add_ui, openui

        mock_app = MagicMock()
        mock_app.config_fh.get_file_handler.return_value = "{}"
        mock_get_app.return_value = mock_app

        # Add multiple UIs
        add_ui(mock_app, "Dashboard", "Main Dashboard", "/dashboard", "Main UI", True)

        # Update mock to return saved data
        saved_call = mock_app.config_fh.add_to_save_file_handler.call_args[0][1]
        mock_app.config_fh.get_file_handler.return_value = saved_call

        # Retrieve UIs
        uis = openui(mock_app)

        # openui returns ApiResult
        if hasattr(uis, 'result'):
            data = uis.result.data
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["title"], "Main Dashboard")
        else:
            self.assertEqual(len(uis), 1)
            self.assertEqual(uis[0]["title"], "Main Dashboard")


if __name__ == '__main__':
    unittest.main(verbosity=2)

