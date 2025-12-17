"""
Unit Tests for server.py
========================
Tests for the main MCP server facade.
"""

import unittest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from toolboxv2.tests.a_util import async_test

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolboxv2.mcp_server.models import MCPConfig, ServerMode, FlowState, ToolResult
from toolboxv2.mcp_server.server import ToolBoxV2MCPServer


class TestToolBoxV2MCPServerInit(unittest.TestCase):
    """Test server initialization."""

    def test_default_config(self):
        """Test initialization with default config."""
        server = ToolBoxV2MCPServer()

        self.assertEqual(server.config.server_name, "toolboxv2_mcp")
        self.assertFalse(server._initialized)
        self.assertIsNone(server._app)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = MCPConfig(
            server_name="custom_server", enable_python=False, enable_docs=False
        )
        server = ToolBoxV2MCPServer(config)

        self.assertEqual(server.config.server_name, "custom_server")
        self.assertFalse(server.config.enable_python)
        self.assertFalse(server.config.enable_docs)

    def test_managers_initialized(self):
        """Test that all managers are initialized."""
        server = ToolBoxV2MCPServer()

        self.assertIsNotNone(server.api_keys)
        self.assertIsNotNone(server.sessions)
        self.assertIsNotNone(server.cache)
        self.assertIsNotNone(server.py_context)
        self.assertIsNotNone(server.performance)

    def test_workers_initialized(self):
        """Test that all workers are initialized."""
        server = ToolBoxV2MCPServer()

        self.assertIsNotNone(server.python_worker)
        self.assertIsNotNone(server.docs_worker)
        self.assertIsNotNone(server.toolbox_worker)
        self.assertIsNotNone(server.system_worker)


class TestToolDefinitions(unittest.TestCase):
    """Test tool definitions."""

    def setUp(self):
        """Set up test fixtures."""
        self.server = ToolBoxV2MCPServer()

    def test_get_tool_definitions_all_enabled(self):
        """Test tool definitions with all features enabled."""
        tools = self.server.get_tool_definitions()

        tool_names = [t["name"] for t in tools]

        # Core tools
        self.assertIn("toolbox_execute", tool_names)
        self.assertIn("toolbox_status", tool_names)
        self.assertIn("toolbox_info", tool_names)

        # Python tools
        self.assertIn("python_execute", tool_names)

        # Docs tools
        self.assertIn("docs_reader", tool_names)
        self.assertIn("docs_writer", tool_names)
        self.assertIn("source_code_lookup", tool_names)
        self.assertIn("get_task_context", tool_names)

        # Flow tools
        self.assertIn("flow_start", tool_names)
        self.assertIn("flow_input", tool_names)
        self.assertIn("flow_status", tool_names)
        self.assertIn("flow_list_available", tool_names)

    def test_get_tool_definitions_python_disabled(self):
        """Test tool definitions with Python disabled."""
        config = MCPConfig(enable_python=False)
        server = ToolBoxV2MCPServer(config)

        tools = server.get_tool_definitions()
        tool_names = [t["name"] for t in tools]

        self.assertNotIn("python_execute", tool_names)

    def test_get_tool_definitions_docs_disabled(self):
        """Test tool definitions with docs disabled."""
        config = MCPConfig(enable_docs=False)
        server = ToolBoxV2MCPServer(config)

        tools = server.get_tool_definitions()
        tool_names = [t["name"] for t in tools]

        self.assertNotIn("docs_reader", tool_names)
        self.assertNotIn("docs_writer", tool_names)

    def test_get_tool_definitions_flows_disabled(self):
        """Test tool definitions with flows disabled."""
        config = MCPConfig(enable_flows=False)
        server = ToolBoxV2MCPServer(config)

        tools = server.get_tool_definitions()
        tool_names = [t["name"] for t in tools]

        self.assertNotIn("flow_start", tool_names)
        self.assertNotIn("flow_continue", tool_names)

    def test_tool_schema_structure(self):
        """Test that tool schemas have correct structure."""
        tools = self.server.get_tool_definitions()

        for tool in tools:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("inputSchema", tool)
            self.assertIn("type", tool["inputSchema"])
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertIn("properties", tool["inputSchema"])


class TestResourceDefinitions(unittest.TestCase):
    """Test resource definitions."""

    def setUp(self):
        """Set up test fixtures."""
        self.server = ToolBoxV2MCPServer()

    def test_get_resource_definitions(self):
        """Test getting resource definitions."""
        resources = self.server.get_resource_definitions()

        self.assertIsInstance(resources, list)
        self.assertGreater(len(resources), 0)

        uris = [r["uri"] for r in resources]
        self.assertIn("flowagents://discovery", uris)
        self.assertIn("flowagents://python_guide", uris)
        self.assertIn("flowagents://performance", uris)
        self.assertIn("toolbox://status", uris)
        self.assertIn("toolbox://performance", uris)

    def test_resource_structure(self):
        """Test resource structure."""
        resources = self.server.get_resource_definitions()

        for resource in resources:
            self.assertIn("uri", resource)
            self.assertIn("name", resource)
            self.assertIn("description", resource)
            self.assertIn("mimeType", resource)

    @async_test
    async def test_read_resource_discovery(self):
        """Test reading discovery resource."""
        content = await self.server.read_resource("flowagents://discovery")

        self.assertIsInstance(content, str)
        self.assertIn("Server Capabilities", content)

    @async_test
    async def test_read_resource_python_guide(self):
        """Test reading Python guide resource."""
        content = await self.server.read_resource("flowagents://python_guide")

        self.assertIsInstance(content, str)
        self.assertIn("app", content)

    @async_test
    async def test_read_resource_performance(self):
        """Test reading performance resource."""
        content = await self.server.read_resource("flowagents://performance")

        self.assertIsInstance(content, str)
        self.assertIn("Cache", content)

    @async_test
    async def test_read_resource_unknown(self):
        """Test reading unknown resource."""
        with self.assertRaises(ValueError):
            await self.server.read_resource("unknown://resource")


class TestToolExecution(unittest.TestCase):
    """Test tool execution routing."""

    def setUp(self):
        """Set up test fixtures."""
        self.server = ToolBoxV2MCPServer()

    @async_test
    async def test_call_tool_unknown(self):
        """Test calling unknown tool."""
        results = await self.server.call_tool("unknown_tool", {})

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIn("Unknown", results[0].content)

    @async_test
    async def test_call_tool_toolbox_status(self):
        """Test calling toolbox_status without app."""
        # Mock the _ensure_app to avoid loading real toolbox
        self.server._initialized = True
        self.server._app = Mock()
        self.server._app.id = "test"
        self.server._app.version = "1.0"
        self.server._app.debug = False
        self.server._app.alive = True
        self.server._app.functions = {"test_mod": {}}
        self.server._app.flows = {}

        results = await self.server.call_tool("toolbox_status", {})

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertIn("test_mod", results[0].content)

    @async_test
    async def test_call_tool_toolbox_info(self):
        """Test calling toolbox_info."""
        self.server._initialized = True
        self.server._app = Mock()
        self.server._app.functions = {"ModuleA": {}, "ModuleB": {}}

        results = await self.server.call_tool("toolbox_info", {"info_type": "modules"})

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertIn("ModuleA", results[0].content)


class TestPerformanceTracking(unittest.TestCase):
    """Test performance tracking integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.server = ToolBoxV2MCPServer()
        self.server._initialized = True
        self.server._app = Mock()
        self.server._app.functions = {}
        self.server._app.flows = {}

    @async_test
    async def test_performance_recorded(self):
        """Test that performance is recorded for tool calls."""
        initial_count = self.server.performance.metrics.requests_handled

        await self.server.call_tool("toolbox_info", {"info_type": "modules"})

        self.assertEqual(
            self.server.performance.metrics.requests_handled, initial_count + 1
        )


class TestLazyLoading(unittest.TestCase):
    """Test lazy loading behavior."""

    def test_not_initialized_on_creation(self):
        """Test that app is not loaded on creation."""
        server = ToolBoxV2MCPServer()

        self.assertFalse(server._initialized)
        self.assertIsNone(server._app)

    def test_tool_definitions_without_loading(self):
        """Test that tool definitions work without loading app."""
        server = ToolBoxV2MCPServer()

        # This should NOT load the app
        tools = server.get_tool_definitions()

        self.assertFalse(server._initialized)
        self.assertIsNotNone(tools)
        self.assertGreater(len(tools), 0)

    def test_resource_definitions_without_loading(self):
        """Test that resource definitions work without loading app."""
        server = ToolBoxV2MCPServer()

        # This should NOT load the app
        resources = server.get_resource_definitions()

        self.assertFalse(server._initialized)
        self.assertIsNotNone(resources)


if __name__ == "__main__":
    unittest.main(verbosity=2)
