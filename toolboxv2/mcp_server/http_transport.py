"""
ToolBoxV2 MCP Server - HTTP Transport
=====================================
REST/HTTP transport with API key authentication
Following MCP Best Practices
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .server import ToolBoxV2MCPServer

logger = logging.getLogger("mcp.http")

# Check for aiohttp availability
try:
    from aiohttp import web
    try:
        import aiohttp_cors
    except ImportError:
        aiohttp_cors = None
        # print("aiohttp_cors not available, CORS support disabled")
        print("Install with: pip install aiohttp-cors")

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None
    aiohttp_cors = None


class HTTPTransport:
    """
    HTTP/REST Transport for MCP Server.

    Features:
    - RESTful API endpoints
    - API key authentication
    - CORS support for web clients
    - Session management
    - Health monitoring

    Endpoints:
    - POST /mcp/initialize - Initialize session
    - POST /mcp/tools/list - List available tools
    - POST /mcp/tools/call - Execute a tool
    - POST /mcp/resources/list - List resources
    - POST /mcp/resources/read - Read a resource
    - GET /health - Health check
    - GET /api/keys - List API keys (admin)
    """

    def __init__(self, server: "ToolBoxV2MCPServer"):
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "HTTP transport requires aiohttp. Install with:\n"
                "  pip install aiohttp aiohttp-cors --break-system-packages"
            )

        self.server = server
        self.config = server.config
        self.app = web.Application()
        self.sessions: Dict[str, Dict] = {}
        self._runner: Optional[web.AppRunner] = None

        self._setup_routes()
        self._setup_cors()

    def _setup_routes(self):
        """Setup HTTP routes."""
        # MCP Protocol endpoints
        self.app.router.add_post("/mcp/initialize", self._handle_initialize)
        self.app.router.add_post("/mcp/tools/list", self._handle_list_tools)
        self.app.router.add_post("/mcp/tools/call", self._handle_call_tool)
        self.app.router.add_post("/mcp/resources/list", self._handle_list_resources)
        self.app.router.add_post("/mcp/resources/read", self._handle_read_resource)

        # Management endpoints
        self.app.router.add_get("/health", self._handle_health)
        self.app.router.add_get("/api/keys", self._handle_list_keys)
        self.app.router.add_post("/api/keys", self._handle_create_key)
        self.app.router.add_delete("/api/keys/{name}", self._handle_revoke_key)

        # Status endpoint
        self.app.router.add_get("/status", self._handle_status)

    def _setup_cors(self):
        """Setup CORS for web clients."""
        cors = aiohttp_cors.setup(
            self.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                )
            },
        )

        for route in list(self.app.router.routes()):
            try:
                cors.add(route)
            except ValueError:
                pass  # Route already added

    async def _authenticate(self, request: web.Request) -> Optional[Dict]:
        """Authenticate request via API key."""
        if not self.config.require_auth:
            return {
                "permissions": ["read", "write", "execute", "admin"],
                "name": "anonymous",
            }

        # Check Authorization header (Bearer token)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
        else:
            # Check X-API-Key header
            api_key = request.headers.get("X-API-Key", "")

        if not api_key:
            return None

        # Validate with API key manager
        key_info = await self.server.api_keys.validate(api_key)
        if key_info:
            return {"permissions": key_info.permissions, "name": key_info.name}
        return None

    def _check_permission(self, key_info: Optional[Dict], required: str) -> bool:
        """Check if key has required permission."""
        if not key_info:
            return False
        return required in key_info.get("permissions", [])

    def _json_response(self, data: Any, status: int = 200) -> web.Response:
        """Create JSON response."""
        return web.json_response(data, status=status)

    def _error_response(self, message: str, status: int = 400) -> web.Response:
        """Create error response."""
        return web.json_response({"error": message}, status=status)

    # =========================================================================
    # Health & Status Endpoints
    # =========================================================================

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint (no auth required)."""
        uptime = time.time() - self.server.performance.metrics.init_time

        return self._json_response(
            {
                "status": "healthy",
                "server": self.config.server_name,
                "version": self.config.server_version,
                "mode": "http",
                "uptime_seconds": uptime,
                "initialized": self.server._initialized,
            }
        )

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Detailed status endpoint."""
        key_info = await self._authenticate(request)
        if not key_info:
            return self._error_response("Unauthorized", 401)

        return self._json_response(
            {
                "server": {
                    "name": self.config.server_name,
                    "version": self.config.server_version,
                    "mode": "http",
                },
                "initialized": self.server._initialized,
                "performance": self.server.performance.to_dict(),
                "sessions": {
                    "http": len(self.sessions),
                    "flows": self.server.sessions.count,
                },
                "cache": self.server.cache.stats,
            }
        )

    # =========================================================================
    # MCP Protocol Endpoints
    # =========================================================================

    async def _handle_initialize(self, request: web.Request) -> web.Response:
        """MCP initialize endpoint."""
        key_info = await self._authenticate(request)
        if not key_info:
            return self._error_response("Unauthorized", 401)

        # Create HTTP session
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {"key_info": key_info, "created": time.time()}

        return self._json_response(
            {
                "session_id": session_id,
                "protocol_version": "2024-11-05",
                "server_info": {
                    "name": self.config.server_name,
                    "version": self.config.server_version,
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "permissions": key_info.get("permissions", []),
            }
        )

    async def _handle_list_tools(self, request: web.Request) -> web.Response:
        """List available tools."""
        key_info = await self._authenticate(request)
        if not key_info:
            return self._error_response("Unauthorized", 401)

        # Get tools from server
        tools = self.server.get_tool_definitions()

        # Filter tools based on permissions
        permissions = key_info.get("permissions", [])
        filtered_tools = []

        for tool in tools:
            tool_name = tool.get("name", "")

            # Permission checks
            if "python" in tool_name and "execute" not in permissions:
                continue
            if "docs_writer" in tool_name and "write" not in permissions:
                continue
            if "admin" in tool_name and "admin" not in permissions:
                continue

            filtered_tools.append(tool)

        return self._json_response({"tools": filtered_tools})

    async def _handle_call_tool(self, request: web.Request) -> web.Response:
        """Call a tool."""
        key_info = await self._authenticate(request)
        if not key_info:
            return self._error_response("Unauthorized", 401)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._error_response("Invalid JSON", 400)

        tool_name = data.get("name")
        arguments = data.get("arguments", {})

        if not tool_name:
            return self._error_response("Missing tool name", 400)

        # Permission checks
        permissions = key_info.get("permissions", [])

        if "python" in tool_name and "execute" not in permissions:
            return self._error_response("Permission denied: execute required", 403)
        if "docs_writer" in tool_name and "write" not in permissions:
            return self._error_response("Permission denied: write required", 403)
        if "admin" in tool_name and "admin" not in permissions:
            return self._error_response("Permission denied: admin required", 403)

        # Execute tool
        try:
            result = await self.server.call_tool(tool_name, arguments)

            return self._json_response(
                {
                    "content": [{"type": "text", "text": r.content} for r in result],
                    "isError": any(not r.success for r in result)
                    if hasattr(result[0], "success")
                    else False,
                }
            )

        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return self._json_response(
                {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True},
                status=500,
            )

    async def _handle_list_resources(self, request: web.Request) -> web.Response:
        """List available resources."""
        key_info = await self._authenticate(request)
        if not key_info:
            return self._error_response("Unauthorized", 401)

        if not self._check_permission(key_info, "read"):
            return self._error_response("Permission denied", 403)

        resources = self.server.get_resource_definitions()

        return self._json_response({"resources": resources})

    async def _handle_read_resource(self, request: web.Request) -> web.Response:
        """Read a resource."""
        key_info = await self._authenticate(request)
        if not key_info:
            return self._error_response("Unauthorized", 401)

        if not self._check_permission(key_info, "read"):
            return self._error_response("Permission denied", 403)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._error_response("Invalid JSON", 400)

        uri = data.get("uri")
        if not uri:
            return self._error_response("Missing URI", 400)

        try:
            content = await self.server.read_resource(uri)
            return self._json_response(
                {"contents": [{"uri": uri, "text": content, "mimeType": "text/plain"}]}
            )
        except Exception as e:
            return self._error_response(f"Resource error: {e}", 500)

    # =========================================================================
    # API Key Management Endpoints
    # =========================================================================

    async def _handle_list_keys(self, request: web.Request) -> web.Response:
        """List API keys (admin only)."""
        key_info = await self._authenticate(request)
        if not key_info:
            return self._error_response("Unauthorized", 401)

        if not self._check_permission(key_info, "admin"):
            return self._error_response("Permission denied: admin required", 403)

        keys = await self.server.api_keys.list_keys()

        return self._json_response({"keys": list(keys.values())})

    async def _handle_create_key(self, request: web.Request) -> web.Response:
        """Create new API key (admin only)."""
        key_info = await self._authenticate(request)
        if not key_info:
            return self._error_response("Unauthorized", 401)

        if not self._check_permission(key_info, "admin"):
            return self._error_response("Permission denied: admin required", 403)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._error_response("Invalid JSON", 400)

        name = data.get("name")
        permissions = data.get("permissions")

        if not name:
            return self._error_response("Missing key name", 400)

        api_key, info = await self.server.api_keys.generate_key(name, permissions)

        return self._json_response(
            {
                "api_key": api_key,
                "name": info.name,
                "permissions": info.permissions,
                "warning": "Store this key securely - it won't be shown again!",
            },
            status=201,
        )

    async def _handle_revoke_key(self, request: web.Request) -> web.Response:
        """Revoke API key (admin only)."""
        key_info = await self._authenticate(request)
        if not key_info:
            return self._error_response("Unauthorized", 401)

        if not self._check_permission(key_info, "admin"):
            return self._error_response("Permission denied: admin required", 403)

        name = request.match_info.get("name")
        if not name:
            return self._error_response("Missing key name", 400)

        success = await self.server.api_keys.revoke(name)

        if success:
            return self._json_response({"message": f"Key '{name}' revoked"})
        else:
            return self._error_response(f"Key '{name}' not found", 404)

    # =========================================================================
    # Server Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start HTTP server."""
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self.config.http_host, self.config.http_port)
        await site.start()

        logger.info(
            f"HTTP server started on http://{self.config.http_host}:{self.config.http_port}"
        )
        logger.info("Endpoints: /mcp/*, /health, /status, /api/keys")

    async def stop(self) -> None:
        """Stop HTTP server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            logger.info("HTTP server stopped")

    async def run_forever(self) -> None:
        """Run HTTP server until interrupted."""
        await self.start()

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()
