import unittest
from unittest.mock import MagicMock, patch

from toolboxv2.tests.a_util import async_test
from toolboxv2.utils.system.types import ToolBoxError, ToolBoxInterfaces


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


    def tearDown(self):
        # Stop the patchers
        self.app_patcher.stop()

    @async_test
    async def test_maintool_initialization(self):
        """Test MainTool initialization with basic parameters"""
        # Prepare initialization parameters
        init_params = {
            "v": "1.0.0",
            "name": "TestTool",
            "tool": {},
            "color": "BLUE",
            "description": "Test tool description"
        }
        from toolboxv2.utils.system.main_tool import MainTool
        # Create an instance of MainTool
        tool = MainTool()

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



if __name__ == '__main__':
    unittest.main()
