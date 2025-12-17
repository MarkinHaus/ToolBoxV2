"""
ToolBoxV2 MCP Server
====================
Production-ready MCP server with STDIO and HTTP transport.

Architecture: Facade & Workers Pattern
- types.py: Memory-efficient data containers (__slots__)
- managers.py: Stateful management with thread safety
- workers.py: Stateless logic handlers
- server.py: Main facade that orchestrates everything
- http_transport.py: HTTP/REST transport
- __main__.py: CLI entry point

Usage:
    # STDIO mode (Claude Desktop, Cursor)
    python -m mcp_server

    # HTTP mode
    python -m mcp_server --mode http --port 8765

    # Setup & API keys
    python -m mcp_server --setup
    python -m mcp_server --generate-key admin

For ToolBoxV2 integration:
    from mcp_server import ToolBoxV2MCPServer, MCPConfig

    config = MCPConfig(server_mode=ServerMode.HTTP)
    server = ToolBoxV2MCPServer(config)
    await server.run_http()
"""

__version__ = "3.0.0"
__author__ = "ToolBoxV2"

# Core exports
from .models import (
    MCPConfig,
    ServerMode,
    ResponseFormat,
    PermissionLevel,
    FlowState,
    APIKeyInfo,
    FlowSession,
    ToolResult,
    CacheEntry,
    PerformanceMetrics,
)

from .managers import (
    APIKeyManager,
    FlowSessionManager,
    CacheManager,
    PythonContextManager,
    PerformanceTracker,
)

from .workers import (
    MCPSafeIO,
    PythonWorker,
    DocsWorker,
    ToolboxWorker,
    SystemWorker,
)

from .server import ToolBoxV2MCPServer

# HTTP transport (optional - requires aiohttp)
try:
    from .http_transport import HTTPTransport, AIOHTTP_AVAILABLE
except ImportError:
    HTTPTransport = None
    AIOHTTP_AVAILABLE = False

__all__ = [
    # Version
    "__version__",

    # Config & Types
    "MCPConfig",
    "ServerMode",
    "ResponseFormat",
    "PermissionLevel",
    "FlowState",
    "APIKeyInfo",
    "FlowSession",
    "ToolResult",
    "CacheEntry",
    "PerformanceMetrics",

    # Managers
    "APIKeyManager",
    "FlowSessionManager",
    "CacheManager",
    "PythonContextManager",
    "PerformanceTracker",

    # Workers
    "MCPSafeIO",
    "PythonWorker",
    "DocsWorker",
    "ToolboxWorker",
    "SystemWorker",

    # Server
    "ToolBoxV2MCPServer",

    # HTTP (optional)
    "HTTPTransport",
    "AIOHTTP_AVAILABLE",
]
