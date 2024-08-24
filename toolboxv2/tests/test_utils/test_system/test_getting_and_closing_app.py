import asyncio
import atexit
import time
from typing import List, Optional

import unittest
from unittest.mock import MagicMock, patch, Mock, AsyncMock

from toolboxv2 import get_app, Style
from toolboxv2.utils.system import AppType, override_main_app
from toolboxv2.utils.system.getting_and_closing_app import save_closing_app, a_save_closing_app


class TestGetApp(unittest.TestCase):

    @patch('toolboxv2.utils.system.getting_and_closing_app.registered_apps',
           [None])  # Patching registered_apps to simulate empty list
    def test_get_app_singleton(self):
        # Test whether get_app returns the same instance when called multiple times
        app1 = get_app(from_="Test1")
        app2 = get_app(from_="Test2")

        self.assertIs(app1, app2)  # Both instances should be the same

    @patch('toolboxv2.utils.system.getting_and_closing_app.get_logger', MagicMock())
    @patch('toolboxv2.utils.system.getting_and_closing_app.asyncio.get_event_loop',
           MagicMock())  # Mocking asyncio functions
    def test_get_app_initial_startup(self):
        # Test get_app behavior when called from InitialStartUp
        app = get_app(from_="InitialStartUp")
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
        self.assertIs(get_app(), mock_app)

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


class TestSaveClosingApp(unittest.TestCase):

    @staticmethod
    async def helper_exit(*x):
        return x

    @patch('toolboxv2.utils.system.getting_and_closing_app.registered_apps',
           [MagicMock(called_exit=[False, 0], a_exit=AsyncMock())])
    @patch('asyncio.get_event_loop')
    def test_a_save_closing_app_with_registered_app(self, mock_get_event_loop):
        # Mock the event loop
        loop_mock = MagicMock()
        mock_get_event_loop.return_value = loop_mock

        # Set up a registered app
        app_mock = get_app()
        app_mock.start_dir = "test"
        app_mock.print = Mock()

        # Call the function
        # Call the coroutine
        async def test_coroutine():
            await a_save_closing_app()

        # Run the coroutine
        asyncio.run(test_coroutine())

        # Assert that the appropriate methods were called on the app
        app_mock.a_exit.assert_called_once()
        app_mock.print.assert_any_call(Style.Bold(Style.ITALIC("- completed -")))

    # @patch('asyncio.get_event_loop')
    @patch('toolboxv2.utils.system.getting_and_closing_app.registered_apps',
           [MagicMock(called_exit=[False, 0], exit=Mock(), a_exit=AsyncMock())])
    def test_save_closing_app_with_registered_app(self):
        # Mock the event loop
        # Set up a registered app
        app_mock = get_app()

        app_mock.start_dir = "test"
        app_mock.print = Mock()

        save_closing_app()

        time.sleep(1)

        # Assert that the appropriate methods were called on the app

        app_mock.a_exit.assert_called_once()
        app_mock.print.assert_any_call(Style.Bold(Style.ITALIC("- completed -")))

