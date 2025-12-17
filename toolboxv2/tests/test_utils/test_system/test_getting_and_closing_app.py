import time
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from toolboxv2 import Style, get_app
from toolboxv2.utils.system import AppType, override_main_app
from toolboxv2.utils.system.getting_and_closing_app import (
    save_closing_app,
)


class TestGetApp(unittest.TestCase):

    @patch('toolboxv2.utils.system.getting_and_closing_app.registered_apps',
           [None])  # Patching registered_apps to simulate empty list
    def test_get_app_singleton(self):
        # Test whether get_app returns the same instance when called multiple times
        app1 = get_app(from_="Test1", name="test")
        app2 = get_app(from_="Test2", name="test")

        self.assertIs(app1, app2)  # Both instances should be the same

    @patch('toolboxv2.utils.system.getting_and_closing_app.get_logger', MagicMock())
    @patch('toolboxv2.utils.system.getting_and_closing_app.asyncio.get_event_loop',
           MagicMock())  # Mocking asyncio functions
    def test_get_app_initial_startup(self):
        # Test get_app behavior when called from InitialStartUp
        app = get_app(from_="InitialStartUp", name="test")
        self.assertIsNotNone(app)  # Ensure app instance is created
        self.assertIsInstance(app, AppType)  # Ensure app is of type AppType
        # You can add more assertions as needed

    # Add more test cases as needed


class TestOverrideMainApp(unittest.TestCase):

    def setUp(self):
        # Initialize registered_apps to None before each test
        self.registered_apps = [None]

    @patch('toolboxv2.utils.system.getting_and_closing_app.registered_apps', [None])
    def test_override_main_app(self):
        # Create a mock application instance
        mock_app = MagicMock(spec=AppType)

        # Test the function with a mock app instance
        returned_app = override_main_app(mock_app)

        # Check if the function returns the same app instance that was passed
        self.assertIs(returned_app, mock_app)

        # Check if the registered_apps contains the overridden app instance
        self.assertIs(get_app(name="test"), mock_app)

    @patch('toolboxv2.utils.system.getting_and_closing_app.registered_apps',
           [MagicMock(spec=AppType, called_exit=[False, 0])])  # Mocking existing app instance
    @patch('toolboxv2.utils.system.getting_and_closing_app.time.time',
           MagicMock(return_value=time.time()))  # Mocking time.time() function
    def test_override_main_app_overtime(self):
        # Test the function behavior when called more than once within 30 seconds
        with self.assertRaises(PermissionError):
            # Call the function with a mock app instance
            override_main_app(MagicMock(spec=AppType))

    # Add more test cases as needed
