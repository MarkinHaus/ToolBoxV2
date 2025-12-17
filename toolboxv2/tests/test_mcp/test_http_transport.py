"""
Unit Tests for http_transport.py
================================
Tests for HTTP/REST transport layer.
"""

import unittest
import asyncio
import json
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from toolboxv2.tests.a_util import async_test

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolboxv2.mcp_server.models import MCPConfig, ServerMode, ToolResult

# Check if aiohttp is available
try:
    from aiohttp import web
    from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
    from toolboxv2.mcp_server.http_transport import HTTPTransport, AIOHTTP_AVAILABLE

    SKIP_HTTP_TESTS = not AIOHTTP_AVAILABLE
except ImportError:
    SKIP_HTTP_TESTS = True
    AIOHTTP_AVAILABLE = False


@unittest.skipIf(SKIP_HTTP_TESTS, "aiohttp not available")
class TestHTTPTransportInit(unittest.TestCase):
    """Test HTTP transport initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_server = Mock()
        self.mock_server.config = MCPConfig(server_mode=ServerMode.HTTP)
        self.mock_server.api_keys = Mock()
        self.mock_server.sessions = Mock()
        self.mock_server.cache = Mock()
        self.mock_server.performance = Mock()
        self.mock_server.performance.metrics = Mock()
        self.mock_server.performance.metrics.init_time = 0
        self.mock_server._initialized = True

    def test_initialization(self):
        """Test basic initialization."""
        transport = HTTPTransport(self.mock_server)

        self.assertIsNotNone(transport.app)
        self.assertEqual(transport.sessions, {})
        self.assertEqual(transport.config, self.mock_server.config)

    def test_routes_setup(self):
        """Test that routes are set up."""
        transport = HTTPTransport(self.mock_server)

        # Get all route paths
        routes = [r.resource.canonical for r in transport.app.router.routes()]

        self.assertIn("/mcp/initialize", routes)
        self.assertIn("/mcp/tools/list", routes)
        self.assertIn("/mcp/tools/call", routes)
        self.assertIn("/mcp/resources/list", routes)
        self.assertIn("/mcp/resources/read", routes)
        self.assertIn("/health", routes)
        self.assertIn("/status", routes)
        self.assertIn("/api/keys", routes)


@unittest.skipIf(SKIP_HTTP_TESTS, "aiohttp not available")
class TestHTTPTransportAuthentication(unittest.TestCase):
    """Test HTTP transport authentication."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_server = Mock()
        self.mock_server.config = MCPConfig(
            server_mode=ServerMode.HTTP, require_auth=True
        )
        self.mock_server.api_keys = Mock()
        self.mock_server.api_keys.validate = AsyncMock()
        self.mock_server.sessions = Mock()
        self.mock_server.cache = Mock()
        self.mock_server.performance = Mock()
        self.mock_server.performance.metrics = Mock()
        self.mock_server.performance.metrics.init_time = 0
        self.mock_server._initialized = True

        self.transport = HTTPTransport(self.mock_server)

    @async_test
    async def test_authenticate_bearer_token(self):
        """Test authentication with Bearer token."""
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer tb_mcp_test_key_12345"}

        mock_key_info = Mock()
        mock_key_info.permissions = ["read", "write"]
        mock_key_info.name = "test_user"
        self.mock_server.api_keys.validate.return_value = mock_key_info

        result = await self.transport._authenticate(mock_request)

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "test_user")
        self.assertIn("read", result["permissions"])

    @async_test
    async def test_authenticate_api_key_header(self):
        """Test authentication with X-API-Key header."""
        mock_request = Mock()
        mock_request.headers = {"X-API-Key": "tb_mcp_test_key_12345"}

        mock_key_info = Mock()
        mock_key_info.permissions = ["read"]
        mock_key_info.name = "api_user"
        self.mock_server.api_keys.validate.return_value = mock_key_info

        result = await self.transport._authenticate(mock_request)

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "api_user")

    @async_test
    async def test_authenticate_no_key(self):
        """Test authentication without key."""
        mock_request = Mock()
        mock_request.headers = {}

        result = await self.transport._authenticate(mock_request)

        self.assertIsNone(result)

    @async_test
    async def test_authenticate_invalid_key(self):
        """Test authentication with invalid key."""
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer invalid_key"}

        self.mock_server.api_keys.validate.return_value = None

        result = await self.transport._authenticate(mock_request)

        self.assertIsNone(result)

    @async_test
    async def test_authenticate_auth_disabled(self):
        """Test authentication when disabled."""
        self.mock_server.config.require_auth = False

        mock_request = Mock()
        mock_request.headers = {}

        result = await self.transport._authenticate(mock_request)

        self.assertIsNotNone(result)
        self.assertIn("admin", result["permissions"])


@unittest.skipIf(SKIP_HTTP_TESTS, "aiohttp not available")
class TestHTTPTransportPermissions(unittest.TestCase):
    """Test HTTP transport permission checking."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_server = Mock()
        self.mock_server.config = MCPConfig(server_mode=ServerMode.HTTP)
        self.mock_server.api_keys = Mock()
        self.mock_server.sessions = Mock()
        self.mock_server.cache = Mock()
        self.mock_server.performance = Mock()
        self.mock_server.performance.metrics = Mock()
        self.mock_server.performance.metrics.init_time = 0
        self.mock_server._initialized = True

        self.transport = HTTPTransport(self.mock_server)

    def test_check_permission_has_permission(self):
        """Test permission check when permission exists."""
        key_info = {"permissions": ["read", "write", "execute"]}

        self.assertTrue(self.transport._check_permission(key_info, "read"))
        self.assertTrue(self.transport._check_permission(key_info, "write"))
        self.assertTrue(self.transport._check_permission(key_info, "execute"))

    def test_check_permission_missing_permission(self):
        """Test permission check when permission missing."""
        key_info = {"permissions": ["read"]}

        self.assertFalse(self.transport._check_permission(key_info, "write"))
        self.assertFalse(self.transport._check_permission(key_info, "admin"))

    def test_check_permission_no_key_info(self):
        """Test permission check with no key info."""
        self.assertFalse(self.transport._check_permission(None, "read"))


@unittest.skipIf(SKIP_HTTP_TESTS, "aiohttp not available")
class TestHTTPTransportResponses(unittest.TestCase):
    """Test HTTP transport response helpers."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_server = Mock()
        self.mock_server.config = MCPConfig(server_mode=ServerMode.HTTP)
        self.mock_server.api_keys = Mock()
        self.mock_server.sessions = Mock()
        self.mock_server.cache = Mock()
        self.mock_server.performance = Mock()
        self.mock_server.performance.metrics = Mock()
        self.mock_server.performance.metrics.init_time = 0
        self.mock_server._initialized = True

        self.transport = HTTPTransport(self.mock_server)

    def test_json_response(self):
        """Test JSON response creation."""
        response = self.transport._json_response({"key": "value"})

        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "application/json")

    def test_json_response_custom_status(self):
        """Test JSON response with custom status."""
        response = self.transport._json_response({"created": True}, status=201)

        self.assertEqual(response.status, 201)

    def test_error_response(self):
        """Test error response creation."""
        response = self.transport._error_response("Something went wrong")

        self.assertEqual(response.status, 400)
        body = json.loads(response.body)
        self.assertEqual(body["error"], "Something went wrong")

    def test_error_response_custom_status(self):
        """Test error response with custom status."""
        response = self.transport._error_response("Unauthorized", status=401)

        self.assertEqual(response.status, 401)


@unittest.skipIf(SKIP_HTTP_TESTS, "aiohttp not available")
class TestHTTPTransportToolFiltering(unittest.TestCase):
    """Test HTTP transport tool filtering based on permissions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_server = Mock()
        self.mock_server.config = MCPConfig(server_mode=ServerMode.HTTP)
        self.mock_server.api_keys = Mock()
        self.mock_server.api_keys.validate = AsyncMock()
        self.mock_server.sessions = Mock()
        self.mock_server.cache = Mock()
        self.mock_server.performance = Mock()
        self.mock_server.performance.metrics = Mock()
        self.mock_server.performance.metrics.init_time = 0
        self.mock_server._initialized = True

        # Mock tool definitions
        self.mock_server.get_tool_definitions = Mock(
            return_value=[
                {"name": "toolbox_execute", "description": "Execute"},
                {"name": "python_execute", "description": "Python"},
                {"name": "docs_reader", "description": "Read docs"},
                {"name": "docs_writer", "description": "Write docs"},
                {"name": "admin_tool", "description": "Admin only"},
            ]
        )

        self.transport = HTTPTransport(self.mock_server)

    @async_test
    async def test_tools_filtered_by_permission(self):
        """Test that tools are filtered by permission."""
        # Mock request with read-only permissions
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer test_key"}

        mock_key_info = Mock()
        mock_key_info.permissions = ["read"]  # No execute, write, or admin
        mock_key_info.name = "limited_user"
        self.mock_server.api_keys.validate.return_value = mock_key_info

        response = await self.transport._handle_list_tools(mock_request)

        self.assertEqual(response.status, 200)
        body = json.loads(response.body)

        tool_names = [t["name"] for t in body["tools"]]

        # Should have read-allowed tools
        self.assertIn("toolbox_execute", tool_names)
        self.assertIn("docs_reader", tool_names)

        # Should NOT have execute/write/admin tools
        self.assertNotIn("python_execute", tool_names)
        self.assertNotIn("docs_writer", tool_names)
        self.assertNotIn("admin_tool", tool_names)

    @async_test
    async def test_tools_all_with_admin(self):
        """Test that admin gets all tools."""
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer admin_key"}

        mock_key_info = Mock()
        mock_key_info.permissions = ["read", "write", "execute", "admin"]
        mock_key_info.name = "admin_user"
        self.mock_server.api_keys.validate.return_value = mock_key_info

        response = await self.transport._handle_list_tools(mock_request)

        self.assertEqual(response.status, 200)
        body = json.loads(response.body)

        # Admin should get all 5 tools
        self.assertEqual(len(body["tools"]), 5)


@unittest.skipIf(SKIP_HTTP_TESTS, "aiohttp not available")
class TestHTTPTransportHealthEndpoint(unittest.TestCase):
    """Test HTTP transport health endpoint."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_server = Mock()
        self.mock_server.config = MCPConfig(
            server_name="test_server", server_version="1.0.0", server_mode=ServerMode.HTTP
        )
        self.mock_server.api_keys = Mock()
        self.mock_server.sessions = Mock()
        self.mock_server.cache = Mock()
        self.mock_server.performance = Mock()
        self.mock_server.performance.metrics = Mock()
        self.mock_server.performance.metrics.init_time = 100
        self.mock_server._initialized = True

        self.transport = HTTPTransport(self.mock_server)

    @async_test
    async def test_health_endpoint(self):
        """Test health endpoint response."""
        mock_request = Mock()

        response = await self.transport._handle_health(mock_request)

        self.assertEqual(response.status, 200)
        body = json.loads(response.body)

        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["server"], "test_server")
        self.assertEqual(body["version"], "1.0.0")
        self.assertEqual(body["mode"], "http")
        self.assertTrue(body["initialized"])


@unittest.skipIf(SKIP_HTTP_TESTS, "aiohttp not available")
class TestHTTPTransportToolExecution(unittest.TestCase):
    """Test HTTP transport tool execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_server = Mock()
        self.mock_server.config = MCPConfig(server_mode=ServerMode.HTTP)
        self.mock_server.api_keys = Mock()
        self.mock_server.api_keys.validate = AsyncMock()
        self.mock_server.sessions = Mock()
        self.mock_server.cache = Mock()
        self.mock_server.performance = Mock()
        self.mock_server.performance.metrics = Mock()
        self.mock_server.performance.metrics.init_time = 0
        self.mock_server._initialized = True

        # Mock call_tool
        self.mock_server.call_tool = AsyncMock()

        self.transport = HTTPTransport(self.mock_server)

    @async_test
    async def test_call_tool_success(self):
        """Test successful tool call."""
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer test_key"}
        mock_request.json = AsyncMock(
            return_value={"name": "toolbox_status", "arguments": {}}
        )

        mock_key_info = Mock()
        mock_key_info.permissions = ["read", "write", "execute", "admin"]
        mock_key_info.name = "test_user"
        self.mock_server.api_keys.validate.return_value = mock_key_info

        # Mock tool result
        mock_result = ToolResult(success=True, content="Status OK", execution_time=0.1)
        self.mock_server.call_tool.return_value = [mock_result]

        response = await self.transport._handle_call_tool(mock_request)

        self.assertEqual(response.status, 200)
        body = json.loads(response.body)

        self.assertIn("content", body)
        self.assertEqual(body["content"][0]["text"], "Status OK")
        self.assertFalse(body["isError"])

    @async_test
    async def test_call_tool_unauthorized(self):
        """Test tool call without authorization."""
        mock_request = Mock()
        mock_request.headers = {}

        response = await self.transport._handle_call_tool(mock_request)

        self.assertEqual(response.status, 401)

    @async_test
    async def test_call_tool_permission_denied(self):
        """Test tool call with insufficient permissions."""
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer test_key"}
        mock_request.json = AsyncMock(
            return_value={
                "name": "python_execute",
                "arguments": {"code": 'print("test")'},
            }
        )

        mock_key_info = Mock()
        mock_key_info.permissions = ["read"]  # No execute permission
        mock_key_info.name = "limited_user"
        self.mock_server.api_keys.validate.return_value = mock_key_info

        response = await self.transport._handle_call_tool(mock_request)

        self.assertEqual(response.status, 403)


if __name__ == "__main__":
    unittest.main(verbosity=2)
