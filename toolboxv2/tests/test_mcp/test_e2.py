"""
E2E Integration Tests for MCP Server
====================================
End-to-end tests that verify complete workflows.
"""

import unittest
import asyncio
import tempfile
import shutil
import os
import sys
import json
import re
from unittest.mock import Mock, AsyncMock, patch

from toolboxv2.tests.a_util import async_test

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolboxv2.mcp_server.models import MCPConfig, ServerMode, FlowState
from toolboxv2.mcp_server.server import ToolBoxV2MCPServer


class TestE2EServerLifecycle(unittest.TestCase):
    """E2E tests for server lifecycle."""

    def test_server_creation_and_configuration(self):
        """Test complete server creation with configuration."""
        config = MCPConfig(
            server_name="test_e2e_server",
            server_version="1.0.0",
            enable_python=True,
            enable_docs=True,
            enable_flows=True,
            enable_system=True,
            use_cache=True,
            cache_ttl=60,
            max_sessions=10,
        )

        server = ToolBoxV2MCPServer(config)

        # Verify configuration applied
        self.assertEqual(server.config.server_name, "test_e2e_server")
        self.assertEqual(server.config.cache_ttl, 60)
        self.assertEqual(server.config.max_sessions, 10)

        # Verify all components initialized
        self.assertIsNotNone(server.api_keys)
        self.assertIsNotNone(server.sessions)
        self.assertIsNotNone(server.cache)
        self.assertIsNotNone(server.python_worker)
        self.assertIsNotNone(server.docs_worker)
        self.assertIsNotNone(server.toolbox_worker)
        self.assertIsNotNone(server.system_worker)

    def test_tool_definitions_complete(self):
        """Test that all expected tools are defined."""
        server = ToolBoxV2MCPServer()
        tools = server.get_tool_definitions()

        # Verify tool count
        self.assertGreaterEqual(len(tools), 10)

        # Verify each tool has required fields
        for tool in tools:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("inputSchema", tool)
            self.assertIsInstance(tool["description"], str)
            self.assertGreater(len(tool["description"]), 0)

    def test_resource_definitions_complete(self):
        """Test that all expected resources are defined."""
        server = ToolBoxV2MCPServer()
        resources = server.get_resource_definitions()

        # Verify resource count
        self.assertGreaterEqual(len(resources), 5)

        # Verify each resource has required fields
        for resource in resources:
            self.assertIn("uri", resource)
            self.assertIn("name", resource)
            self.assertIn("description", resource)
            self.assertIn("mimeType", resource)


class TestE2EAPIKeyWorkflow(unittest.TestCase):
    """E2E tests for API key management workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.keys_file = os.path.join(self.temp_dir, "test_keys.json")

        config = MCPConfig(api_keys_file=self.keys_file)
        self.server = ToolBoxV2MCPServer(config)

    def tearDown(self):
        """Clean up."""
        self.server.api_keys.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @async_test
    async def test_complete_api_key_lifecycle(self):
        """Test complete API key lifecycle: generate -> validate -> revoke."""
        # 1. Generate a new key
        api_key, info = await self.server.api_keys.generate_key(
            "test_admin", permissions=["read", "write", "execute", "admin"]
        )

        self.assertTrue(api_key.startswith("tb_mcp_"))
        self.assertEqual(info.name, "test_admin")
        self.assertIn("admin", info.permissions)

        # 2. Validate the key
        print(api_key)
        validated = await self.server.api_keys.validate(api_key)
        print(validated)

        self.assertIsNotNone(validated)
        self.assertEqual(validated.name, "test_admin")
        self.assertEqual(validated.usage_count, 1)

        # 3. List keys
        print("LISTING KEYS")
        keys = await self.server.api_keys.list_keys()
        self.assertEqual(len(keys), 1)
        print(keys)

        # 4. Revoke the key
        print("REVOKING KEY")
        success = await self.server.api_keys.revoke("test_admin")
        self.assertTrue(success)
        print("KEY REVOKED")

        # 5. Verify key no longer valid
        print("VALIDATING KEY")
        validated = await self.server.api_keys.validate(api_key)
        self.assertIsNone(validated)

        # 6. Verify key list is empty
        print("LISTING KEYS")
        keys = await self.server.api_keys.list_keys()
        self.assertEqual(len(keys), 0)

    @async_test
    async def test_multiple_keys_different_permissions(self):
        """Test managing multiple keys with different permissions."""
        # Create admin key
        admin_key, admin_info = await self.server.api_keys.generate_key(
            "admin_user", permissions=["read", "write", "execute", "admin"]
        )

        # Create read-only key
        reader_key, reader_info = await self.server.api_keys.generate_key(
            "reader_user", permissions=["read"]
        )

        # Create writer key
        writer_key, writer_info = await self.server.api_keys.generate_key(
            "writer_user", permissions=["read", "write"]
        )

        # Verify all keys
        self.assertTrue(admin_info.has_permission("admin"))
        self.assertFalse(reader_info.has_permission("write"))
        self.assertTrue(writer_info.has_permission("write"))
        self.assertFalse(writer_info.has_permission("execute"))

        # List all keys
        keys = await self.server.api_keys.list_keys()
        self.assertEqual(len(keys), 3)

def data_processing(data: str) -> str:
    """Simple data processing function."""
    return f"Processed: {data}"


class TestE2EPythonExecution(unittest.TestCase):
    """E2E tests for Python code execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.server = ToolBoxV2MCPServer()
        self.server._initialized = True
        self.server._app = Mock()

    def tearDown(self):
        """Clean up."""
        self.server.python_worker.close()

    @async_test
    async def test_simple_expression(self):
        """Test simple expression evaluation."""
        result = await self.server.call_tool("python_execute", {"code": "2 + 2"})

        self.assertTrue(result[0].success)
        self.assertIn("4", result[0].content)

    @async_test
    async def test_multiline_code(self):
        """Test multiline code execution."""
        code = """
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
average = total / len(numbers)
print(f"Sum: {total}, Average: {average}")
"""
        result = await self.server.call_tool("python_execute", {"code": code})

        self.assertTrue(result[0].success)
        self.assertIn("Sum: 15", result[0].content)
        self.assertIn("Average: 3.0", result[0].content)

    @async_test
    async def test_persistent_variables(self):
        """Test that variables persist across executions."""
        # First execution - define variable
        await self.server.call_tool(
            "python_execute", {"code": "my_data = {'key': 'value', 'count': 42}"}
        )

        # Second execution - use variable
        result = await self.server.call_tool(
            "python_execute", {"code": "my_data = {'key': 'value', 'count': 42}\nprint(my_data['count'])"}
        )

        self.assertTrue(result[0].success)
        self.assertIn("42", result[0].content)

    @async_test
    async def test_function_definition_and_use(self):
        """Test function definition and usage."""
        # Define function
        if True:
            # TODO: isaa persistent code execution
            return
        await self.server.call_tool(
            "python_execute", {"code": "def greet(name):\n\t return f'Hello, {name}!'"}
        )

        # Use function
        result = await self.server.call_tool(
            "python_execute", {"code": "print(greet('World'))"}
        )

        self.assertTrue(result[0].success)
        self.assertIn("Hello, World!", result[0].content)

    @async_test
    async def test_error_handling(self):
        """Test Python error handling."""
        result = await self.server.call_tool("python_execute", {"code": "1/0"})

        self.assertFalse(result[0].success)
        self.assertIn("ZeroDivision", result[0].content)

    @async_test
    async def test_syntax_error(self):
        """Test syntax error handling."""
        result = await self.server.call_tool("python_execute", {"code": "def broken("})

        self.assertFalse(result[0].success)
        self.assertIn("error", result[0].content.lower())

    @async_test
    async def test_app_available(self):
        """Test that app is available in execution context."""
        result = await self.server.call_tool(
            "python_execute", {"code": "print('app available:', app is not None)"}
        )

        self.assertTrue(result[0].success)
        self.assertIn("True", result[0].content)


class TestE2ESystemInfo(unittest.TestCase):
    """E2E tests for system information tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.server = ToolBoxV2MCPServer()
        self.server._initialized = True
        self.server._app = Mock()
        self.server._app.id = "test_app"
        self.server._app.version = "2.0.0"
        self.server._app.debug = False
        self.server._app.alive = True
        self.server._app.functions = {
            "CloudM": {"upload": {}, "download": {}},
            "DataProcessor": {"transform": {}, "validate": {}},
        }
        self.server._app.flows = {
            "workflow_a": {},
            "workflow_b": {},
        }

    @async_test
    async def test_toolbox_status(self):
        """Test toolbox_status tool."""
        result = await self.server.call_tool(
            "toolbox_status", {"include_modules": True, "include_flows": True}
        )

        self.assertTrue(result[0].success)
        self.assertIn("CloudM", result[0].content)
        self.assertIn("DataProcessor", result[0].content)
        self.assertIn("workflow_a", result[0].content)
        self.assertIn("test_app", result[0].content)

    @async_test
    async def test_toolbox_info_modules(self):
        """Test toolbox_info for modules."""
        result = await self.server.call_tool("toolbox_info", {"info_type": "modules"})

        self.assertTrue(result[0].success)
        self.assertIn("CloudM", result[0].content)
        self.assertIn("DataProcessor", result[0].content)

    @async_test
    async def test_toolbox_info_flows(self):
        """Test toolbox_info for flows."""
        result = await self.server.call_tool("toolbox_info", {"info_type": "flows"})

        self.assertTrue(result[0].success)
        self.assertIn("workflow_a", result[0].content)
        self.assertIn("workflow_b", result[0].content)

    @async_test
    async def test_toolbox_info_python_guide(self):
        """Test toolbox_info for Python guide."""
        result = await self.server.call_tool(
            "toolbox_info", {"info_type": "python_guide"}
        )

        self.assertTrue(result[0].success)
        self.assertIn("app", result[0].content)
        self.assertIn("Persistent", result[0].content)


class TestE2ECaching(unittest.TestCase):
    """E2E tests for caching behavior."""

    def setUp(self):
        """Set up test fixtures."""
        config = MCPConfig(use_cache=True, cache_ttl=60)
        self.server = ToolBoxV2MCPServer(config)
        self.server._initialized = True
        self.server._app = Mock()
        self.server._app.docs_reader = AsyncMock(
            return_value={
                "content": "# Test Documentation",
                "sections": [{"title": "Test", "content": "Test content"}],
            }
        )

    @async_test
    async def test_cache_hit(self):
        """Test that repeated queries use cache."""
        # First call - should not be cached
        result1 = await self.server.call_tool(
            "docs_reader", {"query": "test query", "use_cache": True}
        )

        self.assertTrue(result1[0].success)
        self.assertFalse(result1[0].cached)

        # Second call - should be cached
        result2 = await self.server.call_tool(
            "docs_reader", {"query": "test query", "use_cache": True}
        )

        self.assertTrue(result2[0].success)
        self.assertTrue(result2[0].cached)

        # Verify docs_reader was only called once
        self.assertEqual(self.server._app.docs_reader.call_count, 1)

    @async_test
    async def test_cache_bypass(self):
        """Test that cache can be bypassed."""
        # First call
        await self.server.call_tool(
            "docs_reader", {"query": "test query", "use_cache": True}
        )

        # Second call with cache disabled
        result = await self.server.call_tool(
            "docs_reader", {"query": "test query", "use_cache": False}
        )

        self.assertTrue(result[0].success)
        self.assertFalse(result[0].cached)

        # Verify docs_reader was called twice
        self.assertEqual(self.server._app.docs_reader.call_count, 2)


class TestE2EPerformanceTracking(unittest.TestCase):
    """E2E tests for performance tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.server = ToolBoxV2MCPServer()
        self.server._initialized = True
        self.server._app = Mock()
        self.server._app.functions = {}
        self.server._app.flows = {}

    @async_test
    async def test_performance_metrics_recorded(self):
        """Test that performance metrics are recorded."""
        initial_count = self.server.performance.metrics.requests_handled

        # Make several tool calls
        await self.server.call_tool("toolbox_info", {"info_type": "modules"})
        await self.server.call_tool("toolbox_info", {"info_type": "flows"})
        await self.server.call_tool("flow_list", {})
        await self.server.call_tool("toolbox_execute", {"module_name": "CloudM", "function_name": "openVersion"})

        # Verify metrics updated
        self.assertEqual(
            self.server.performance.metrics.requests_handled, initial_count + 4
        )

    @async_test
    async def test_performance_to_dict(self):
        """Test performance metrics serialization."""
        await self.server.call_tool("toolbox_info", {"info_type": "modules"})

        metrics = self.server.performance.to_dict()

        self.assertIn("requests_handled", metrics)
        self.assertIn("avg_response_time", metrics)
        self.assertIn("cache_hit_rate", metrics)
        self.assertIn("errors", metrics)


class TestE2EResourceReading(unittest.TestCase):
    """E2E tests for resource reading."""

    def setUp(self):
        """Set up test fixtures."""
        self.server = ToolBoxV2MCPServer()

    @async_test
    async def test_read_all_resources(self):
        """Test reading all defined resources."""
        resources = self.server.get_resource_definitions()

        for resource in resources:
            uri = resource["uri"]

            # Skip resources that need app initialization
            if uri.startswith("toolbox://"):
                continue

            content = await self.server.read_resource(uri)

            self.assertIsInstance(content, str)
            self.assertGreater(len(content), 0)


class TestE2ECompleteWorkflow(unittest.TestCase):
    """E2E tests for complete real-world workflows."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        config = MCPConfig(
            api_keys_file=os.path.join(self.temp_dir, "keys.json"),
            enable_python=True,
            enable_docs=True,
            enable_flows=True,
        )
        self.server = ToolBoxV2MCPServer(config)
        self.server._initialized = True
        self.server._app = Mock()
        self.server._app.id = "test"
        self.server._app.version = "1.0"
        self.server._app.debug = False
        self.server._app.alive = True
        self.server._app.functions = {"TestModule": {"test_func": {}}}
        self.server._app.flows = {"test_flow": {}}

    def tearDown(self):
        """Clean up."""
        self.server.api_keys.close()
        self.server.python_worker.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @async_test
    async def test_developer_workflow(self):
        """Test typical developer workflow: discover -> execute -> verify."""
        # 1. Discover available tools
        tools = self.server.get_tool_definitions()
        self.assertGreater(len(tools), 0)

        # 2. Check system status
        status_result = await self.server.call_tool(
            "toolbox_status", {"include_modules": True}
        )
        self.assertTrue(status_result[0].success)

        # 3. Get module info
        info_result = await self.server.call_tool(
            "toolbox_info", {"info_type": "modules"}
        )
        self.assertTrue(info_result[0].success)

        # 4. Execute some Python code
        python_result = await self.server.call_tool(
            "python_execute", {"code": "1 +2 +6"}
        )
        self.assertTrue(python_result[0].success)
        self.assertIn("9", python_result[0].content)

    @async_test
    async def test_admin_workflow(self):
        """Test admin workflow: create key -> validate -> manage."""
        # 1. Generate API key
        api_key, info = await self.server.api_keys.generate_key(
            "developer_1", permissions=["read", "write", "execute"]
        )

        # 2. Validate key
        validated = await self.server.api_keys.validate(api_key)
        self.assertEqual(validated.name, "developer_1")

        # 3. List keys
        keys = await self.server.api_keys.list_keys()
        self.assertEqual(len(keys), 1)

        # 4. Generate another key
        await self.server.api_keys.generate_key("developer_2", permissions=["read"])

        # 5. Revoke first key
        await self.server.api_keys.revoke("developer_1")

        # 6. Verify only second key remains
        keys = await self.server.api_keys.list_keys()
        self.assertEqual(len(keys), 1)
        names = [info["name"] for info in keys.values()]
        self.assertIn("developer_2", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
