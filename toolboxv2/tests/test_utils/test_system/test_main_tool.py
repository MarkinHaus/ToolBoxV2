import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from toolboxv2.utils.system.getting_and_closing_app import get_app
from toolboxv2.utils.system.tb_logger import get_logger
from toolboxv2.utils.system.types import Result, ToolBoxError, ToolBoxInterfaces


class TestMainTool(unittest.TestCase):
    def setUp(self):
        # Create a mock for get_app to prevent actual app initialization
        self.app_patcher = patch('toolboxv2.utils.system.getting_and_closing_app.get_app')
        self.mock_get_app = self.app_patcher.start()

        # Create a mock app instance
        self.mock_app = MagicMock()
        self.mock_app.print = MagicMock()
        self.mock_app.interface_type = ToolBoxInterfaces.cli
        self.mock_get_app.return_value = self.mock_app

        # Create a mock logger
        self.logger_patcher = patch('toolboxv2.utils.system.tb_logger.get_logger')
        self.mock_get_logger = self.logger_patcher.start()
        self.mock_logger = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger

    def tearDown(self):
        # Stop the patchers
        self.app_patcher.stop()
        self.logger_patcher.stop()

    @patch('toolboxv2.utils.system.MainTool.MainTool')
    async def test_maintool_initialization(self, MockMainTool):
        """Test MainTool initialization with basic parameters"""
        # Prepare initialization parameters
        init_params = {
            "v": "1.0.0",
            "name": "TestTool",
            "tool": {},
            "color": "BLUE",
            "description": "Test tool description"
        }

        # Create an instance of MainTool
        tool = MockMainTool()
        tool.__ainit__ = AsyncMock()

        # Call __ainit__ method
        await tool.__ainit__(**init_params)

        # Assert initialization attributes
        self.assertEqual(tool.version, "1.0.0")
        self.assertEqual(tool.name, "TestTool")
        self.assertEqual(tool.color, "BLUE")
        self.assertEqual(tool.description, "Test tool description")
        self.assertIsNotNone(tool.logger)

    def test_return_result_default(self):
        """Test MainTool.return_result with default parameters"""
        # Use the actual MainTool class for this test
        from toolboxv2.utils.system.main_tool import MainTool

        # Set interface to ensure consistent testing
        MainTool.interface = ToolBoxInterfaces.cli

        result = MainTool.return_result()

        # Verify result components
        self.assertEqual(result.error, ToolBoxError.none)
        self.assertEqual(result.result.data_to, ToolBoxInterfaces.cli)
        self.assertEqual(result.result.data, {})
        self.assertEqual(result.result.data_info, {})
        self.assertEqual(result.info.exec_code, 0)
        self.assertEqual(result.info.help_text, "")

    def test_return_result_custom(self):
        """Test MainTool.return_result with custom parameters"""
        from toolboxv2.utils.system.main_tool import MainTool

        # Set interface to ensure consistent testing
        MainTool.interface = ToolBoxInterfaces.cli

        custom_result = MainTool.return_result(
            error=ToolBoxError.none,
            exec_code=500,
            help_text="Authentication failed",
            data_info={"key": "value"},
            data={"user": "test"},
            data_to=ToolBoxInterfaces.cli
        )
        custom_result.print()
        # Verify custom result components
        self.assertEqual(custom_result.error, ToolBoxError.none)
        self.assertEqual(custom_result.result.data_to, ToolBoxInterfaces.cli)
        self.assertEqual(custom_result.result.data, {"user": "test"})
        self.assertEqual(custom_result.result.data_info, {"key": "value"})
        self.assertEqual(custom_result.info.exec_code, 500)
        self.assertEqual(custom_result.info.help_text, "Authentication failed")


    @patch('toolboxv2.utils.system.main_tool.MainTool.CLOUDM_AUTHMANAGER')
    async def test_get_user(self, mock_authmanager):
        """Test get_user method"""
        from toolboxv2.utils.system.main_tool import MainTool

        # Create a mock MainTool instance
        tool = MainTool()
        tool.app = self.mock_app

        # Configure the mock to return a predefined result
        expected_result = Result(ToolBoxError.none, MagicMock(), MagicMock())
        self.mock_app.a_run_any.return_value = expected_result

        # Call get_user
        result = await tool.get_user("testuser")

        # Verify method calls and result
        self.mock_app.a_run_any.assert_called_once()
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
