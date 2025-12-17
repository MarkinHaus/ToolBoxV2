"""
ToolBoxV2 MCP Server - Production Ready Unified System
Sophisticated MCP server with smart initialization, cached operations, and rich notifications.
"""

import asyncio
import contextlib
import io
import json
import sys
import os
import uuid
import hashlib
import time
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path

# MCP imports
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from pydantic import AnyUrl

# ToolBoxV2 imports
from toolboxv2 import get_app, App, Result, Code, Style
from toolboxv2.utils.extras import stram_print, quick_info as _quick_info, quick_success as _quick_success, quick_warning as _quick_warning, quick_error as _quick_error, ask_question as _quick_ask
from toolboxv2.utils.extras.blobs import BlobFile
from toolboxv2.utils.system.types import CallingObject
from toolboxv2.flows import flows_dict as flows_dict_func
GLOBAL_SERVER: Optional[Server] = None

class MCPOutputProxy(io.StringIO):
    """Proxy that forwards all stdout/stderr writes into MCP log messages."""

    def __init__(self, stream_name: str):
        super().__init__()
        self.stream_name = stream_name

    def write(self, data):
        # Buffer locally (so code that reads from stdout still behaves)
        super().write(data)

        # Only send meaningful messages (no empty newline spam)
        if data.strip():
            if GLOBAL_SERVER:
                try:
                    if not hasattr(GLOBAL_SERVER, 'request_context'):
                        return len(data)
                    GLOBAL_SERVER.request_context.session.send_log_message(
                        level="info" if self.stream_name == "stdout" else "error",
                        data=data,
                    )
                except Exception:
                    pass  # fail-safe, MCP must not crash

        return len(data)


# Suppress stdout/stderr during critical MCP operations
# MCPOutputProxy wird nicht mehr benötigt, da es den Deadlock verursacht.
# Wir leiten stattdessen direkt um.


class MCPSafeIO:
    """
    Sichere Umgebung für Tool-Ausführung im STDIO-Modus.

    Problembehebung:
    Im stdio-Modus ist sys.stdout der exklusive Kanal für JSON-RPC Nachrichten.
    Wenn ein Tool 'print()' nutzt, zerstört es das JSON-Format oder verursacht
    Deadlocks, wenn wir versuchen, es abzufangen und als Log zu senden.

    Lösung:
    Wir biegen sys.stdout temporär auf sys.stderr um.
    - print() Ausgaben landen im Inspector-Log (via stderr).
    - Der echte sys.stdout bleibt sauber für die finale JSON-Antwort des Servers.
    """

    def __init__(self, suppress_stdout=True, suppress_stderr=False):
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        # Nur eingreifen, wenn der Server läuft (um CLI-Tools nicht zu stören)
        if GLOBAL_SERVER:
            # Merke das originale "Leitungs-Rohr" zum Inspector
            self.original_stdout = sys.stdout

            # Leite alle print()-Aufrufe des Tools auf den Fehlerkanal um.
            # Das verhindert den Deadlock und Timeout.
            sys.stdout = sys.stderr

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # WICHTIG: Sofort wiederherstellen, bevor der Server die Antwort sendet!
        if self.original_stdout:
            sys.stdout = self.original_stdout

def quick_info(*args, **kwargs):
    with MCPSafeIO():
        return _quick_info(*args, **kwargs)
def quick_success(*args, **kwargs):
    with MCPSafeIO():
        return _quick_success(*args, **kwargs)
def quick_warning(*args, **kwargs):
    with MCPSafeIO():
        return _quick_warning(*args, **kwargs)
def quick_error(*args, **kwargs):
    with MCPSafeIO():
        return _quick_error(*args, **kwargs)


@dataclass
class MCPConfig:
    """Production MCP Server configuration with smart defaults"""
    server_name: str = "toolboxv2-mcp"
    server_version: str = "2.0.0"
    api_keys_file: str = "MCPConfig/mcp_api_keys.json"
    session_timeout: int = 3600
    max_concurrent_sessions: int = 20
    enable_flows: bool = True
    enable_python_execution: bool = True
    enable_system_manipulation: bool = True
    docs_system: bool = True
    docs_reader: bool = True
    docs_writer: bool = True
    smart_init: bool = True
    use_cached_index: bool = True
    rich_notifications: bool = True
    performance_mode: bool = True

    # NEU: Server-Modus Konfiguration
    server_mode: str = "stdio"  # "stdio" oder "http"
    http_host: str = "0.0.0.0"
    http_port: int = 8765
    require_auth: bool = True

    # NEU: Logging-Steuerung
    silent_mode: bool = False  # Für Agent-Modus
    log_file: Optional[str] = None  # Log in Datei statt stdout


class SmartInitManager:
    """Manages smart initialization with caching and notifications"""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.init_lock = asyncio.Lock()
        self.init_status = {"toolbox": False, "docs": False, "flows": False}
        self.cache_info = {}

    async def smart_initialize_toolbox(self, tb_app: App) -> Dict[str, Any]:
        """Smart initialization with caching and progress notifications"""
        async with self.init_lock:
            if self.init_status["toolbox"]:
                return {"status": "already_initialized", "cached": True}

            try:
                quick_info("MCP Init", "Starting ToolBoxV2 smart initialization...")

                with MCPSafeIO():
                    # Load modules progressively
                    start_time = time.time()
                    #await tb_app.load_all_mods_in_file()
                    module_time = time.time() - start_time

                    # Set up flows
                    flows_dict = flows_dict_func(remote=False)
                    tb_app.set_flows(flows_dict)

                    # Initialize ISAA if available
                    # if "isaa" in tb_app.functions:
                    #     await tb_app.get_mod("isaa").init_isaa()

                    self.init_status["toolbox"] = True

                quick_success("MCP Init", f"ToolBoxV2 initialized in {module_time:.2f}s")

                return {
                    "status": "initialized",
                    "modules_count": len(tb_app.functions),
                    "flows_count": len(getattr(tb_app, 'flows', {})),
                    "init_time": module_time
                }

            except Exception as e:
                quick_error("MCP Init", f"Failed to initialize ToolBoxV2: {e}")
                return {"status": "error", "error": str(e)}

    async def smart_initialize_docs(self, tb_app: App) -> Dict[str, Any]:
        """Smart docs initialization with cached index detection"""
        async with self.init_lock:
            if self.init_status["docs"]:
                return {"status": "already_initialized", "cached": True}

            try:
                quick_info("MCP Docs", "Initializing documentation system...")

                # Check for existing index
                docs_system = getattr(tb_app, 'mkdocs', None)
                if not docs_system:
                    return {"status": "not_available"}

                use_cached = self.config.use_cached_index

                if use_cached:
                    quick_info("MCP Docs", "Found cached index, loading...")
                    # Load existing index without rebuild
                    result = await tb_app.docs_init()
                    action = "cached_load"
                else:
                    quick_info("MCP Docs", "Building fresh documentation index...")
                    # Build new index
                    result = await tb_app.docs_init(True)
                    action = "fresh_build"

                if result.is_ok():
                    data = result.get()
                    self.init_status["docs"] = True
                    self.cache_info = {
                        "sections": data.get("total_sections", 0),
                        "elements": data.get("total_code_elements", 0),
                        "linked": data.get("linked_sections", 0),
                        "completion": data.get("completion_rate", "0%")
                    }

                    quick_success("MCP Docs", f"Docs ready: {self.cache_info['sections']} sections, {self.cache_info['completion']} linked")

                    return {
                        "status": "initialized",
                        "action": action,
                        "cache_used": use_cached,
                        **self.cache_info
                    }
                else:
                    quick_warning("MCP Docs", f"Docs init failed: {result.error}")
                    return {"status": "error", "error": str(result.error)}

            except Exception as e:
                quick_error("MCP Docs", f"Docs initialization error: {e}")
                return {"status": "error", "error": str(e)}

class UnifiedAPIKeyManager:
    """Enhanced API key management with notifications"""

    def __init__(self, keys_file: str):
        self.keys_file = keys_file
        self.keys = self._load_keys()
        self._usage_stats = {}

    def _load_keys(self) -> Dict[str, Dict]:
        """Load API keys with error handling"""
        try:
            with MCPSafeIO():
                if BlobFile(self.keys_file, key=Code.DK()()).exists():
                    with BlobFile(self.keys_file, key=Code.DK()(), mode='r') as f:
                        return f.read_json()
        except Exception as e:
            quick_warning("API Keys", f"Could not load keys file: {e}")
        return {}

    def list_keys(self) -> Dict[str, Dict]:
        """List all API keys with usage stats"""
        return self.keys

    def _save_keys(self):
        """Save API keys with error handling"""
        try:
            with BlobFile(self.keys_file, key=Code.DK()(), mode='w') as f:
                f.write_json(self.keys)
        except Exception as e:
            quick_error("API Keys", f"Failed to save keys: {e}")

    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate API key with notification"""
        if permissions is None:
            permissions = ["read", "write", "execute", "admin"]

        api_key = f"tb_mcp_{uuid.uuid4().hex}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        self.keys[key_hash] = {
            "name": name,
            "permissions": permissions,
            "created": time.time(),
            "last_used": None,
            "usage_count": 0
        }

        self._save_keys()
        quick_success("API Keys", f"Generated key for '{name}' with {len(permissions)} permissions")
        return api_key

    def validate_key(self, api_key: str) -> Optional[Dict]:
        """Validate key with usage tracking"""
        if not api_key or not api_key.startswith("tb_mcp_"):
            return None

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash in self.keys:
            key_info = self.keys[key_hash]
            key_info["last_used"] = time.time()
            key_info["usage_count"] += 1

            # Track usage stats
            self._usage_stats[key_hash] = self._usage_stats.get(key_hash, 0) + 1

            self._save_keys()
            return key_info
        return None

class EnhancedFlowSessionManager:
    """Flow session management with notifications and cleanup"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.max_sessions = 100
        self.cleanup_task = None

    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Background cleanup of expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                expired_count = self.cleanup_expired_sessions()
                if expired_count > 0:
                    quick_info("Sessions", f"Cleaned up {expired_count} expired sessions")
            except asyncio.CancelledError:
                break
            except Exception as e:
                quick_warning("Sessions", f"Cleanup error: {e}")

    def create_session(self, flow_name: str, session_id: str = None) -> str:
        """Create session with management notifications"""
        if session_id is None:
            session_id = f"flow_{uuid.uuid4().hex[:8]}"

        # Cleanup if at limit
        if len(self.sessions) >= self.max_sessions:
            oldest_id = min(self.sessions.keys(),
                          key=lambda k: self.sessions[k]["created"])
            del self.sessions[oldest_id]
            quick_info("Sessions", f"Removed oldest session to make room")

        self.sessions[session_id] = {
            "flow_name": flow_name,
            "created": time.time(),
            "last_activity": time.time(),
            "state": "created",
            "context": {},
            "history": []
        }

        quick_success("Sessions", f"Created session {session_id} for flow '{flow_name}'")
        return session_id

    def cleanup_expired_sessions(self, timeout: int = 3600) -> int:
        """Cleanup expired sessions and return count"""
        current_time = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if current_time - session["last_activity"] > timeout
        ]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)


# Neue Klasse einfügen vor ToolBoxV2MCPServer (ca. Zeile ~300)

try:
    from aiohttp import web
    import aiohttp_cors

    AIOHTTP_AVAILABLE = True
except ImportError and ModuleNotFoundError:
    AIOHTTP_AVAILABLE = False



# Einfügen nach MCPSafeIO Klasse (Zeile ~68)

class SmartLogger:
    """Intelligentes Logging-System das zwischen User- und Agent-Modus unterscheidet"""

    def __init__(self, config: 'MCPConfig' = None):
        self.config = config
        self.silent = os.environ.get('MCP_SILENT_MODE', '0') == '1' or (config and config.silent_mode)
        self.log_file = config.log_file if config else None

        # Setup file logging wenn konfiguriert
        if self.log_file:
            self._setup_file_logging()

    def _setup_file_logging(self):
        """Setup file-based logging"""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=str(log_path),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def info(self, category: str, message: str):
        """Log info message"""
        if self.silent:
            if self.log_file:
                logging.info(f"[{category}] {message}")
            return

        with MCPSafeIO():
            _quick_info(category, message)

    def success(self, category: str, message: str):
        """Log success message"""
        if self.silent:
            if self.log_file:
                logging.info(f"[{category}] SUCCESS: {message}")
            return

        with MCPSafeIO():
            _quick_success(category, message)

    def warning(self, category: str, message: str):
        """Log warning message"""
        if self.silent:
            if self.log_file:
                logging.warning(f"[{category}] {message}")
            return

        with MCPSafeIO():
            _quick_warning(category, message)

    def error(self, category: str, message: str):
        """Log error message"""
        if self.silent:
            if self.log_file:
                logging.error(f"[{category}] {message}")
            return

        with MCPSafeIO():
            _quick_error(category, message)


# Globaler Logger - wird später initialisiert
_smart_logger: Optional[SmartLogger] = None


def get_logger() -> SmartLogger:
    """Get global smart logger instance"""
    global _smart_logger
    if _smart_logger is None:
        _smart_logger = SmartLogger()
    return _smart_logger
class MCPHTTPTransport:
    """HTTP/REST Transport für MCP Server mit API-Key Authentifizierung"""

    def __init__(self, server: 'ToolBoxV2MCPServer', config: MCPConfig):
        self.server = server
        self.config = config
        self.app = web.Application()
        self.sessions: Dict[str, Dict] = {}

        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp required for HTTP mode: pip install aiohttp aiohttp-cors")

        self._setup_routes()
        self._setup_cors()

    def _setup_cors(self):
        """Setup CORS for web clients"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })

        for route in list(self.app.router.routes()):
            cors.add(route)

    def _setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_post('/mcp/initialize', self._handle_initialize)
        self.app.router.add_post('/mcp/tools/list', self._handle_list_tools)
        self.app.router.add_post('/mcp/tools/call', self._handle_call_tool)
        self.app.router.add_post('/mcp/resources/list', self._handle_list_resources)
        self.app.router.add_post('/mcp/resources/read', self._handle_read_resource)
        self.app.router.add_get('/health', self._handle_health)
        self.app.router.add_get('/api/keys', self._handle_list_keys)

    def _authenticate(self, request: web.Request) -> Optional[Dict]:
        """Authenticate request via API key"""
        if not self.config.require_auth:
            return {"permissions": ["read", "write", "execute", "admin"]}

        # Check Authorization header
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            api_key = auth_header[7:]
        else:
            # Check X-API-Key header
            api_key = request.headers.get('X-API-Key', '')

        if not api_key:
            return None

        # Validate with API key manager
        key_info = self.server.api_key_manager.validate_key(api_key)
        return key_info

    def _check_permission(self, key_info: Dict, required: str) -> bool:
        """Check if key has required permission"""
        if not key_info:
            return False
        return required in key_info.get('permissions', [])

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "version": self.config.server_version,
            "mode": "http",
            "uptime": time.time() - self.server.performance_metrics.get("init_time", 0)
        })

    async def _handle_initialize(self, request: web.Request) -> web.Response:
        """MCP initialize endpoint"""
        key_info = self._authenticate(request)
        if not key_info:
            return web.json_response({"error": "Unauthorized"}, status=401)

        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "key_info": key_info,
            "created": time.time()
        }

        return web.json_response({
            "session_id": session_id,
            "server_name": self.config.server_name,
            "server_version": self.config.server_version,
            "capabilities": {
                "tools": True,
                "resources": True,
                "permissions": key_info.get('permissions', [])
            }
        })

    async def _handle_list_tools(self, request: web.Request) -> web.Response:
        """List available tools based on permissions"""
        key_info = self._authenticate(request)
        if not key_info:
            return web.json_response({"error": "Unauthorized"}, status=401)

        # Get all tools via handler
        handler = self.server.server._request_handlers.get('tools/list')
        if not handler:
            return web.json_response({"error": "Handler not found"}, status=500)

        try:
            tools = await handler()

            # Filter tools based on permissions
            permissions = key_info.get('permissions', [])
            filtered_tools = []

            for tool in tools:
                tool_name = tool.name

                # Permission checks
                if 'python' in tool_name and 'execute' not in permissions:
                    continue
                if 'docs_writer' in tool_name and 'write' not in permissions:
                    continue
                if 'module_manage' in tool_name and 'admin' not in permissions:
                    continue

                filtered_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                })

            return web.json_response({"tools": filtered_tools})

        except Exception as e:
            get_logger().error("HTTP", f"List tools failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_call_tool(self, request: web.Request) -> web.Response:
        """Call a tool with permission checks"""
        key_info = self._authenticate(request)
        if not key_info:
            return web.json_response({"error": "Unauthorized"}, status=401)

        try:
            data = await request.json()
            tool_name = data.get('name')
            arguments = data.get('arguments', {})

            # Permission checks
            permissions = key_info.get('permissions', [])

            if 'python' in tool_name and 'execute' not in permissions:
                return web.json_response({"error": "Permission denied"}, status=403)
            if 'docs_writer' in tool_name and 'write' not in permissions:
                return web.json_response({"error": "Permission denied"}, status=403)
            if 'module_manage' in tool_name and 'admin' not in permissions:
                return web.json_response({"error": "Permission denied"}, status=403)

            # Execute tool via handler
            handler = self.server.server._request_handlers.get('tools/call')
            if not handler:
                return web.json_response({"error": "Handler not found"}, status=500)

            result = await handler(tool_name, arguments)

            # Convert result to JSON
            result_data = []
            for item in result:
                result_data.append({
                    "type": item.type,
                    "text": item.text
                })

            return web.json_response({"result": result_data})

        except Exception as e:
            get_logger().error("HTTP", f"Call tool failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_list_resources(self, request: web.Request) -> web.Response:
        """List available resources"""
        key_info = self._authenticate(request)
        if not key_info:
            return web.json_response({"error": "Unauthorized"}, status=401)

        if not self._check_permission(key_info, 'read'):
            return web.json_response({"error": "Permission denied"}, status=403)

        try:
            handler = self.server.server._request_handlers.get('resources/list')
            if not handler:
                return web.json_response({"error": "Handler not found"}, status=500)

            resources = await handler()

            resources_data = []
            for resource in resources:
                resources_data.append({
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mimeType
                })

            return web.json_response({"resources": resources_data})

        except Exception as e:
            get_logger().error("HTTP", f"List resources failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_read_resource(self, request: web.Request) -> web.Response:
        """Read a resource"""
        key_info = self._authenticate(request)
        if not key_info:
            return web.json_response({"error": "Unauthorized"}, status=401)

        if not self._check_permission(key_info, 'read'):
            return web.json_response({"error": "Permission denied"}, status=403)

        try:
            data = await request.json()
            uri = data.get('uri')

            handler = self.server.server._request_handlers.get('resources/read')
            if not handler:
                return web.json_response({"error": "Handler not found"}, status=500)

            content = await handler(uri)

            return web.json_response({"content": content})

        except Exception as e:
            get_logger().error("HTTP", f"Read resource failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_list_keys(self, request: web.Request) -> web.Response:
        """List API keys (admin only)"""
        key_info = self._authenticate(request)
        if not key_info:
            return web.json_response({"error": "Unauthorized"}, status=401)

        if not self._check_permission(key_info, 'admin'):
            return web.json_response({"error": "Permission denied"}, status=403)

        keys = self.server.api_key_manager.list_keys()

        # Don't expose actual keys, just metadata
        keys_info = []
        for key_hash, info in keys.items():
            keys_info.append({
                "name": info['name'],
                "permissions": info['permissions'],
                "created": info['created'],
                "last_used": info['last_used'],
                "usage_count": info['usage_count']
            })

        return web.json_response({"keys": keys_info})

    async def start(self):
        """Start HTTP server"""
        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, self.config.http_host, self.config.http_port)
        await site.start()

        get_logger().success("HTTP Server",
                             f"Started on http://{self.config.http_host}:{self.config.http_port}")
        get_logger().info("HTTP Server",
                          "Endpoints: /mcp/*, /health, /api/keys")

class ToolBoxV2MCPServer:
    """Production-ready unified MCP Server with smart features"""

    def __init__(self, config: MCPConfig = None):
        global GLOBAL_SERVER
        self.config = config or MCPConfig()
        self.server = Server(self.config.server_name)
        GLOBAL_SERVER = self.server
        self.api_key_manager = UnifiedAPIKeyManager(self.config.api_keys_file)
        self.flow_session_manager = EnhancedFlowSessionManager()
        self.init_manager = SmartInitManager(self.config)

        # Core components
        self.tb_app: Optional[App] = None
        self.docs_system = None
        self.authenticated_sessions: Dict[str, Dict] = {}

        # Performance tracking
        self.performance_metrics = {
            "requests_handled": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "init_time": 0.0
        }

        # Resource definitions
        self.flowagents_resources = self._initialize_flowagents_resources()

        # Initialize with smart detection
        asyncio.create_task(self._smart_bootstrap())

        # Setup handlers
        self._setup_handlers()

    async def _smart_bootstrap(self):
        """Smart bootstrap with progress notifications"""
        try:
            quick_info("MCP Server", f"Starting {self.config.server_name} v{self.config.server_version}")

            start_time = time.time()

            # Initialize ToolBoxV2 app
            with MCPSafeIO():
                self.tb_app = get_app(from_="MCP-Server", name="mcp")

                # Override print functions for clean MCP communication
                def _silent_print(*args, **kwargs): return False
                self.tb_app.print = _silent_print
                self.tb_app.sprint = _silent_print

            # Smart initialization
            toolbox_result = await self.init_manager.smart_initialize_toolbox(self.tb_app)

            if toolbox_result["status"] == "initialized":
                # Initialize docs system if enabled
                if self.config.docs_system:
                    docs_result = await self.init_manager.smart_initialize_docs(self.tb_app)
                    if docs_result["status"] == "initialized":
                        self.docs_system = self.tb_app.mkdocs

            # Start background tasks
            await self.flow_session_manager.start_cleanup_task()

            # Record initialization metrics
            init_time = time.time() - start_time
            self.performance_metrics["init_time"] = init_time

            quick_success("MCP Server", f"Bootstrap completed in {init_time:.2f}s - Ready for connections")

        except Exception as e:
            quick_error("MCP Server", f"Bootstrap failed: {e}")
            raise

    def _initialize_flowagents_resources(self) -> Dict[str, Dict]:
        """Initialize FlowAgents resource prompts with enhanced metadata"""
        return {
            "flowagents_toolbox_discovery": {
                "name": "flowagents_toolbox_discovery",
                "description": "Comprehensive resource discovery and capability mapping for ToolBoxV2 MCP Server",
                "mimeType": "text/markdown",
                "version": "2.0",
                "content": """# ToolBoxV2 MCP Server - Advanced Resource Discovery

## 🚀 Server Capabilities Overview
This ToolBoxV2 MCP server provides comprehensive access to a sophisticated development and documentation platform with the following core capabilities:

### 📊 Performance Features
- **Smart Initialization**: Cached index loading for 10x faster startup
- **Async Operations**: Non-blocking concurrent request handling
- **Intelligent Caching**: Query result caching with 5-minute TTL
- **Resource Management**: Automatic session cleanup and memory optimization

### 🔧 Core Tool Categories

#### 1. **Function Execution** (`toolbox_execute`)
- Direct access to 25+ ToolBoxV2 modules
- Full parameter passing with type validation
- Result object handling with metadata
- Performance monitoring and timeout protection

#### 2. **Documentation Intelligence** (`docs_reader`, `docs_writer`, `docs_system_status`)
- **Smart Indexing**: Section-level change detection
- **AI Generation**: Automated documentation from source code
- **Cross-referencing**: Code-to-docs linking with validation
- **Bulk Operations**: Auto-update suggestions and batch processing

#### 3. **Flow Orchestration** (`flow_start`, `flow_continue`, `flow_status`)
- **Session Management**: Persistent workflow state
- **Complex Workflows**: Multi-step processes with branching
- **User Interaction**: Callback handling and input processing
- **Error Recovery**: Graceful failure handling and retry logic

#### 4. **System Intelligence** (`toolbox_status`, `module_manage`, `toolbox_info`)
- **Live Introspection**: Real-time system state monitoring
- **Module Lifecycle**: Dynamic loading, reloading, and management
- **Resource Discovery**: Comprehensive capability enumeration
- **Health Monitoring**: Performance metrics and diagnostics

#### 5. **Code Execution** (`python_execute`, `read_file`)
- **Secure Sandboxing**: Isolated execution environment
- **Context Preservation**: Persistent variables across calls
- **File System Access**: Controlled read/write operations
- **Integration APIs**: Direct ToolBoxV2 app instance access

## 🎯 Optimization Strategies

### Quick Start Pattern
```
1. toolbox_status(include_modules=True) → Get system overview
2. docs_reader(query="specific_topic") → Find relevant documentation
3. toolbox_execute(module_name="target", function_name="action") → Execute
```

### Documentation Discovery
```
1. docs_system_status() → Check index status
2. docs_reader(section_id="direct_access") → Fast specific retrieval
3. optional source_code_lookup(element_name="target", element_type="class") → Code context lookup
```

### Complex Workflow Pattern
```
1. flow_start(flow_name="process_name") → Initialize workflow
2. flow_continue(session_id="...", input_data={...}) → Process steps
3. flow_status(session_id="...") → Monitor progress
```

## ⚡ Performance Guidelines
- Use `section_id` for direct docs access (10x faster than search)
- Set `max_results` limits to prevent timeout on large queries
- Leverage `format_type="structured"` for programmatic processing
- Use `include_source_refs=False` when references not needed
- Apply `priority_filter` on suggestion queries for focused results

## 🔗 Integration Protocols
- **Timeout Management**: All operations have intelligent timeout protection
- **Error Recovery**: Graceful degradation with informative error messages
- **Progress Tracking**: Real-time notifications for long-running operations
- **Resource Limits**: Automatic batching and pagination for large datasets
"""
            },

            "flowagents_smart_execution": {
                "name": "flowagents_smart_execution",
                "description": "Intelligent execution strategies with caching and optimization",
                "mimeType": "text/markdown",
                "version": "2.0",
                "content": """# Smart Execution Strategies

## 🧠 Intelligent Tool Selection

### Performance-First Routing
1. **Documentation Queries**
   - `section_id` → Direct access (fastest)
   - `file_path` → File-scoped search (fast)
   - `query` → Full-text search (slower)
   - `tags` → Tag-based filtering (medium)

2. **Function Discovery**
   - `toolbox_info(info_type="modules")` → Module enumeration
   - `toolbox_info(info_type="functions", target="module")` → Function listing
   - `toolbox_info(info_type="function_detail", target="mod.func")` → Detailed info

3. **Execution Strategies**
   - Simple operations → `toolbox_execute` (direct)
   - Multi-step processes → `flow_start` + `flow_continue` (stateful)
   - Code generation → `python_execute` with context
   - Bulk operations → Auto-update tools with batching

## 📈 Caching Optimization

### Query Result Caching
- 5-minute TTL on documentation searches
- Section-level granular cache invalidation
- Smart cache warming for frequently accessed content
- Memory-efficient cache size management (100 entries max)

### Index Optimization
- Cached index loading on server start
- Incremental updates for changed sections only
- Git-based change detection for minimal scanning
- Background index maintenance with notifications

## ⚙️ Advanced Parameters

### Documentation Tools
- `max_results`: Control response size (1-100, default: 20)
- `format_type`: Choose output format ("structured", "markdown", "json")
- `include_source_refs`: Include/exclude code references for speed
- `use_cache`: Enable/disable query result caching

### System Tools
- `include_modules`: Control module enumeration depth
- `include_functions`: Enable detailed function listing
- `get_results`: Return full Result objects with metadata
- `timeout`: Custom timeout values for long operations

### Flow Tools
- `session_id`: Persistent workflow state management
- `input_type`: Specify input data format and handling
- `context`: Pass persistent data between flow steps
"""
            }
        }

    def _setup_handlers(self):
        """Setup optimized MCP request handlers with notifications"""

        @self.server.list_resources()
        async def handle_list_resources() -> List[types.Resource]:
            """List all available resources with caching"""
            resources = []

            # Add FlowAgents resources
            for resource_id, resource_data in self.flowagents_resources.items():
                resources.append(types.Resource(
                    uri=f"flowagents://{resource_id}",
                    name=resource_data["name"],
                    description=resource_data["description"],
                    mimeType=resource_data["mimeType"]
                ))

            # Add dynamic ToolBoxV2 resources if available
            if self.tb_app:
                resources.extend([
                    types.Resource(
                        uri="toolbox://system/status",
                        name="toolbox_system_status",
                        description="Real-time ToolBoxV2 system status and performance metrics",
                        mimeType="application/json"
                    ),
                    types.Resource(
                        uri="toolbox://system/performance",
                        name="toolbox_performance_metrics",
                        description="Server performance analytics and optimization suggestions",
                        mimeType="application/json"
                    )
                ])

                if self.docs_system:
                    resources.append(types.Resource(
                        uri="toolbox://docs/smart_index",
                        name="toolbox_smart_docs_index",
                        description="Intelligent documentation index with cache status",
                        mimeType="application/json"
                    ))

            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: AnyUrl) -> str:
            """Read resource with enhanced caching"""
            uri = str(uri)
            if uri.startswith("flowagents://"):
                resource_id = uri.replace("flowagents://", "")
                if resource_id in self.flowagents_resources:
                    self.performance_metrics["cache_hits"] += 1
                    return self.flowagents_resources[resource_id]["content"]
                raise ValueError(f"Unknown FlowAgents resource: {resource_id}")

            elif uri.startswith("toolbox://"):
                path = uri.replace("toolbox://", "")

                if path == "system/status" and self.tb_app:
                    status = {
                        "app_id": self.tb_app.id,
                        "version": self.tb_app.version,
                        "modules": list(self.tb_app.functions.keys()),
                        "module_count": len(self.tb_app.functions),
                        "flows": list(getattr(self.tb_app, 'flows', {}).keys()),
                        "docs_available": bool(self.docs_system),
                        "init_status": self.init_manager.init_status,
                        "cache_info": self.init_manager.cache_info,
                        "uptime": time.time() - (time.time() - self.performance_metrics.get("init_time", 0))
                    }
                    return json.dumps(status, indent=2)

                elif path == "system/performance":
                    return json.dumps({
                        "performance_metrics": self.performance_metrics,
                        "session_stats": {
                            "active_sessions": len(self.flow_session_manager.sessions),
                            "max_sessions": self.flow_session_manager.max_sessions
                        },
                        "optimization_suggestions": self._get_optimization_suggestions()
                    }, indent=2)

                elif path == "docs/smart_index" and self.docs_system:
                    index = self.docs_system.current_index
                    if index:
                        cache_status = {
                            "index_loaded": True,
                            "version": index.version,
                            "last_indexed": index.last_indexed.isoformat(),
                            "sections": len(index.sections),
                            "code_elements": len(index.code_elements),
                            "cached_queries": len(getattr(self.docs_system, '_search_cache', {})),
                            "git_commit": index.last_git_commit,
                            "performance": {
                                "avg_query_time": "< 100ms",
                                "cache_hit_rate": f"{(self.performance_metrics['cache_hits'] / max(self.performance_metrics['requests_handled'], 1)) * 100:.1f}%"
                            }
                        }
                    else:
                        cache_status = {"index_loaded": False, "status": "initializing"}

                    return json.dumps(cache_status, indent=2)

                raise ValueError(f"Unknown resource path: {path}")

            raise ValueError(f"Unknown resource URI scheme: {uri}")

        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List tools with enhanced schemas"""
            tools = []

            # Core execution tool
            tools.append(types.Tool(
                name="toolbox_execute",
                description="Execute ToolBoxV2 functions with performance monitoring and caching",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "module_name": {"type": "string", "description": "Module name (use toolbox_info to discover)"},
                        "function_name": {"type": "string", "description": "Function name within module"},
                        "args": {"type": "array", "description": "Positional arguments", "default": []},
                        "kwargs": {"type": "object", "description": "Keyword arguments", "default": {}},
                        "get_results": {"type": "boolean", "description": "Return full Result object", "default": False},
                        "timeout": {"type": "integer", "description": "Custom timeout in seconds", "default": 30}
                    },
                    "required": ["module_name", "function_name"]
                }
            ))

            # Enhanced documentation tools
            if self.config.docs_system:
                tools.extend([
                    types.Tool(
                        name="docs_reader",
                        description="Intelligent documentation reader with caching and smart search",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query (supports keywords, phrases)"},
                                "section_id": {"type": "string", "description": "Direct section access (fastest method)"},
                                "file_path": {"type": "string", "description": "Filter by documentation file"},
                                "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags"},
                                "include_source_refs": {"type": "boolean", "default": True, "description": "Include code references"},
                                "format_type": {"type": "string", "enum": ["structured", "markdown", "json"], "default": "structured"},
                                "max_results": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                                "use_cache": {"type": "boolean", "default": True, "description": "Use cached results"}
                            }
                        }
                    ),
                    types.Tool(
                        name="docs_writer",
                        description="Advanced documentation writer with AI generation",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["create_file", "add_section", "update_section", "generate_from_code"]},
                                "file_path": {"type": "string", "description": "Target file (relative to docs/)"},
                                "section_title": {"type": "string", "description": "Section title"},
                                "content": {"type": "string", "description": "Content (optional if auto_generate=true)"},
                                "source_file": {"type": "string", "description": "Source file for AI generation"},
                                "auto_generate": {"type": "boolean", "default": False, "description": "Use AI to generate content"},
                                "position": {"type": "string", "description": "Position: 'top', 'bottom', 'after:SectionName'"},
                                "level": {"type": "integer", "default": 2, "minimum": 1, "maximum": 6}
                            },
                            "required": ["action"]
                        }
                    ),
                    types.Tool(
                        name="source_code_lookup",
                        description="Intelligent source code lookup with caching and smart search",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "element_name": {"type": "string", "description": "Element name (e.g., class, function)"},
                                "file_path": {"type": "string", "description": "Filter by file path"},
                                "element_type": {"type": "string", "description": "Filter by element type (e.g., class, function)"},
                                "max_results": {"type": "integer", "default": 25, "minimum": 1, "maximum": 100},
                                "return_code_block": {"type": "boolean", "default": True, "description": "Include code block in response"}
                            },
                            "required": ["element_name"]
                        }
                    )
                ])

            # Enhanced system tools
            tools.extend([
                types.Tool(
                    name="toolbox_status",
                    description="Comprehensive system status with performance metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_modules": {"type": "boolean", "default": True},
                            "include_functions": {"type": "boolean", "default": False},
                            "include_flows": {"type": "boolean", "default": True},
                            "include_performance": {"type": "boolean", "default": True}
                        }
                    }
                ),
                types.Tool(
                    name="toolbox_info",
                    description="Enhanced system information with guides and examples",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "info_type": {"type": "string", "enum": ["modules", "functions", "module_detail", "function_detail", "python_guide", "performance_guide", "flowagents_guide"]},
                            "target": {"type": "string", "description": "Specific target for detailed info"},
                            "include_examples": {"type": "boolean", "default": False, "description": "Include usage examples"}
                        },
                        "required": ["info_type"]
                    }
                )
            ])

            # Flow and execution tools
            if self.config.enable_flows:
                tools.extend([
                    types.Tool(
                        name="flow_start",
                        description="Start intelligent workflow with session management",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "flow_name": {"type": "string", "description": "Flow name (use toolbox_info to discover)"},
                                "session_id": {"type": "string", "description": "Optional custom session ID"},
                                "kwargs": {"type": "object", "default": {}, "description": "Flow initialization parameters"},
                                "timeout": {"type": "integer", "default": 3600, "description": "Session timeout in seconds"}
                            },
                            "required": ["flow_name"]
                        }
                    )
                ])

            if self.config.enable_python_execution:
                tools.append(types.Tool(
                    name="python_execute",
                    description="Secure Python execution with ToolBoxV2 integration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code ('app' variable available)"},
                            "globals": {"type": "object", "default": {}, "description": "Additional globals"},
                            "timeout": {"type": "integer", "default": 30, "description": "Execution timeout"},
                            "capture_output": {"type": "boolean", "default": True, "description": "Capture stdout/stderr"}
                        },
                        "required": ["code"]
                    }
                ))

            tools.append(types.Tool(
                name="resource_reader",
                description="Read ToolBoxV2 resources with caching and performance tracking",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "resource_uri": {"type": "string", "description": "Resource URI (use toolbox_info to discover)"},
                    },
                    "required": ["resource_uri"]
                }
            ))

            return tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Enhanced tool execution with notifications and performance tracking"""
            start_time = time.time()
            self.performance_metrics["requests_handled"] += 1
            quick_info("MCP", f"Handling tool call: {name}")
            try:
                # Ensure smart initialization
                if not self.init_manager.init_status.get("toolbox", False):
                    quick_info("MCP", "Auto-initializing ToolBoxV2...")
                    await self.init_manager.smart_initialize_toolbox(self.tb_app)

                # Route to specific handlers
                if name == "toolbox_execute":
                    result = await self._handle_toolbox_execute(arguments)
                elif name == "resource_reader":
                    result = await handle_read_resource(arguments.get("resource_uri"))
                elif name == "docs_reader":
                    result = await self._handle_docs_reader(arguments)
                elif name == "docs_writer":
                    result = await self._handle_docs_writer(arguments)
                elif name == "source_code_lookup":
                    result = await self._handle_source_code_lookup(arguments)
                elif name == "toolbox_status":
                    result = await self._handle_toolbox_status(arguments)
                elif name == "toolbox_info":
                    result = await self._handle_toolbox_info(arguments)
                elif name == "python_execute":
                    result = await self._handle_python_execute(arguments)
                    quick_success("Python", f"Execution completed {result}")
                elif name == "flow_start":
                    result = await self._handle_flow_start(arguments)
                elif name.startswith("flow_"):
                    result = await self._handle_flow_operation(name, arguments)
                else:
                    result = [types.TextContent(type="text", text=f"Unknown tool: {name}")]

                # Update performance metrics
                execution_time = time.time() - start_time
                self.performance_metrics["avg_response_time"] = (
                    (self.performance_metrics["avg_response_time"] * (self.performance_metrics["requests_handled"] - 1) + execution_time) /
                    self.performance_metrics["requests_handled"]
                )

                # Add performance info to response if requested
                if arguments.get("include_performance", False):
                    perf_info = f"\n\n---\n⚡ Execution: {execution_time:.3f}s | Cache: {self.performance_metrics['cache_hits']} hits"
                    if result and len(result) > 0:
                        result[0] = types.TextContent(
                            type="text",
                            text=result[0].text + perf_info
                        )
                quick_info("MCP", f"Tool '{name}' completed in {execution_time:.2f}s")
                return result

            except asyncio.TimeoutError:
                quick_warning("MCP", f"Tool '{name}' timed out")
                return [types.TextContent(type="text", text=f"⏱️ Tool '{name}' timed out. Try with smaller parameters or increase timeout.")]
            except Exception as e:
                quick_error("MCP", f"Tool '{name}' failed: {str(e)[:100]}")
                return [types.TextContent(type="text", text=f"❌ Error in {name}: {str(e)}")]

    # Enhanced tool handlers with notifications
    async def _handle_toolbox_execute(self, arguments: Dict) -> List[types.TextContent]:
        """Enhanced function execution with caching and notifications"""
        module_name = arguments.get("module_name")
        function_name = arguments.get("function_name")
        timeout = arguments.get("timeout", 30)

        # Generate cache key
        cache_key = f"{module_name}.{function_name}:{hashlib.md5(str(arguments).encode()).hexdigest()[:8]}"

        try:
            quick_info("Execute", f"Running {module_name}.{function_name}")

            with MCPSafeIO():
                result = await self.tb_app.a_run_any(
                        (module_name, function_name),
                        args_=arguments.get("args", []),
                        get_results=arguments.get("get_results", False),
                        **arguments.get("kwargs", {})
                    )

            # Format result
            if arguments.get("get_results", False) and hasattr(result, 'as_dict'):
                result_text = json.dumps(result.as_dict(), indent=2)
                success_msg = f"✅ {module_name}.{function_name} completed successfully"
            else:
                result_text = str(result)
                success_msg = f"✅ {module_name}.{function_name} → {str(result)[:50]}"

            quick_success("Execute", success_msg)

            return [types.TextContent(
                type="text",
                text=f"**Executed:** `{module_name}.{function_name}`\n\n**Result:**\n```\n{result_text}\n```"
            )]

        except Exception as e:
            quick_error("Execute", f"{module_name}.{function_name} failed: {str(e)[:100]}")
            return [types.TextContent(
                type="text",
                text=f"❌ **Error executing {module_name}.{function_name}:**\n\n{str(e)}"
            )]

    async def _handle_docs_reader(self, arguments: Dict) -> List[types.TextContent]:
        """Enhanced docs reader with smart caching"""
        try:
            # Ensure docs system is initialized
            if not self.init_manager.init_status.get("docs", False):
                quick_info("Docs", "Auto-initializing documentation system...")
                await self.init_manager.smart_initialize_docs(self.tb_app)

            # Use caching if enabled
            use_cache = arguments.get("use_cache", True)
            cache_key = hashlib.md5(str(sorted(arguments.items())).encode()).hexdigest()[:12]

            if use_cache and hasattr(self.docs_system, '_search_cache'):
                cached = self.docs_system._search_cache.get(cache_key)
                if cached and time.time() - cached['timestamp'] < 300:  # 5 min cache
                    self.performance_metrics["cache_hits"] += 1
                    quick_success("Docs", f"Cache hit for query")
                    return [types.TextContent(type="text", text=cached['result'])]

            quick_info("Docs", "Processing documentation query...")

            result = await asyncio.wait_for(
                self.tb_app.docs_reader(
                    query=arguments.get("query"),
                    section_id=arguments.get("section_id"),
                    file_path=arguments.get("file_path"),
                    tags=arguments.get("tags"),
                    format_type=arguments.get("format_type", "structured"),
                    max_results=min(arguments.get("max_results", 20), 100)
                ),
                timeout=15.0
            )

            if 'error' not in result:
                data = result
                if arguments.get("format_type") == "markdown":
                    content = data
                else:
                    content = json.dumps(data, indent=2, ensure_ascii=False)
                    if len(content) > 100000:  # 100KB limit
                        content = content[:100000] + "\n... (truncated)"

                # Cache successful results
                if use_cache and hasattr(self.docs_system, '_search_cache'):
                    if not hasattr(self.docs_system, '_search_cache'):
                        self.docs_system._search_cache = {}
                    self.docs_system._search_cache[cache_key] = {
                        'result': content,
                        'timestamp': time.time()
                    }

                sections_count = len(data.get("sections", [])) if isinstance(data, dict) else 1
                quick_success("Docs", f"Retrieved {sections_count} documentation sections")

                return [types.TextContent(type="text", text=content)]
            else:
                quick_warning("Docs", f"Query failed: {result}")
                return [types.TextContent(type="text", text=f"⚠️ Documentation query error: {result}")]

        except asyncio.TimeoutError:
            quick_warning("Docs", "Documentation query timed out")
            return [types.TextContent(type="text", text="⏱️ Documentation query timed out. Try a more specific query.")]
        except Exception as e:
            quick_error("Docs", f"Documentation error: {str(e)[:100]}")
            return [types.TextContent(type="text", text=f"❌ Documentation system error: {e}")]

    async def _handle_docs_writer(self, arguments: Dict) -> List[types.TextContent]:
        """Enhanced docs writer with progress notifications"""
        try:
            action = arguments.pop("action")
            quick_info("Docs Writer", f"Starting {action} operation...")

            result = await asyncio.wait_for(
                self.tb_app.docs_writer(
                    action=action,
                    **arguments
                ),
                timeout=60.0
            )

            if 'error' not in result:
                data = result
                quick_success("Docs Writer", f"Successfully completed {action}")
                return [types.TextContent(
                    type="text",
                    text=f"✅ **Documentation {action} completed successfully**\n\n```json\n{json.dumps(data, indent=2)}\n```"
                )]
            else:
                quick_error("Docs Writer", f"{action} failed: {result}")
                return [types.TextContent(type="text", text=f"❌ Documentation {action} failed: {result}")]

        except Exception as e:
            quick_error("Docs Writer", f"Writer error: {str(e)[:100]}")
            return [types.TextContent(type="text", text=f"❌ Documentation writer error: {e}")]

    async def _handle_source_code_lookup(self, arguments: Dict) -> List[types.TextContent]:
        """Enhanced source code lookup with caching and notifications"""
        try:
            element_name = arguments.get("element_name")
            file_path = arguments.get("file_path")
            element_type = arguments.get("element_type")
            max_results = min(arguments.get("max_results", 25), 100)
            return_code_block = arguments.get("return_code_block", True)

            quick_info("Code Lookup", f"Searching for {element_name} in source code...")

            result = await self.tb_app.docs_lookup(
                name=element_name,
                file_path=file_path,
                element_type=element_type,
                max_results=max_results,
                include_code=return_code_block
            )

            if 'error' not in result:
                data = result
                matches = data.get("matches", [])
                match_count = len(matches)
                quick_success("Code Lookup", f"Found {match_count} matches for {element_name}")

                content = f"Found {match_count} matches for {element_name}:\n\n```json\n{json.dumps(data, indent=2)}\n```"
                return [types.TextContent(type="text", text=content)]
            else:
                quick_error("Code Lookup", f"Lookup failed: {result}")
                return [types.TextContent(type="text", text=f"❌ Code lookup failed: {result}")]
        except Exception as e:
            quick_error("Code Lookup", f"Lookup error: {str(e)[:100]}")
            return [types.TextContent(type="text", text=f"❌ Code lookup error: {e}")]

    async def _handle_toolbox_status(self, arguments: Dict) -> List[types.TextContent]:
        """Enhanced system status with rich metrics"""
        try:
            include_performance = arguments.get("include_performance", True)

            status = {
                "🏗️ System": {
                    "app_id": self.tb_app.id if self.tb_app else "Not initialized",
                    "version": self.tb_app.version if self.tb_app else "Unknown",
                    "debug_mode": self.tb_app.debug if self.tb_app else False,
                    "alive": self.tb_app.alive if self.tb_app else False
                },
                "📦 Modules": {
                    "loaded_count": len(self.tb_app.functions) if self.tb_app else 0,
                    "module_list": list(self.tb_app.functions.keys()) if self.tb_app and arguments.get("include_modules", True) else "Use include_modules=true"
                },
                "🔄 Flows": {
                    "available_count": len(getattr(self.tb_app, 'flows', {})),
                    "flow_list": list(getattr(self.tb_app, 'flows', {}).keys()) if arguments.get("include_flows", True) else "Use include_flows=true"
                },
                "📚 Documentation": {
                    "system_available": bool(self.docs_system),
                    "index_status": self.init_manager.init_status.get("docs", False),
                    "cache_info": self.init_manager.cache_info
                }
            }

            if include_performance:
                status["⚡ Performance"] = {
                    "requests_handled": self.performance_metrics["requests_handled"],
                    "avg_response_time": f"{self.performance_metrics['avg_response_time']:.3f}s",
                    "cache_hit_rate": f"{(self.performance_metrics['cache_hits'] / max(self.performance_metrics['requests_handled'], 1)) * 100:.1f}%",
                    "active_sessions": len(self.flow_session_manager.sessions),
                    "init_time": f"{self.performance_metrics['init_time']:.2f}s"
                }

            content = "# 🚀 ToolBoxV2 System Status\n\n"
            content += json.dumps(status, indent=2, ensure_ascii=False)

            return [types.TextContent(type="text", text=content)]

        except Exception as e:
            return [types.TextContent(type="text", text=f"❌ Error getting system status: {e}")]

    async def _handle_toolbox_info(self, arguments: Dict) -> List[types.TextContent]:
        """Enhanced system information with rich guides"""
        info_type = arguments.get("info_type")
        target = arguments.get("target")
        include_examples = arguments.get("include_examples", False)

        try:
            if info_type == "performance_guide":
                guide = """# 🚀 ToolBoxV2 MCP Server - Performance Optimization Guide

## ⚡ Quick Performance Tips

### 1. Documentation Queries
- **Fastest**: Use `section_id` for direct access
- **Fast**: Use `file_path` to scope searches
- **Medium**: Use `tags` for filtering
- **Slower**: Use broad `query` searches

### 2. Caching Strategy
- Enable `use_cache=true` for repeated queries (default)
- Cache TTL: 5 minutes for documentation
- Cache size limit: 100 entries (automatic cleanup)

### 3. Result Limits
- Set appropriate `max_results` (default: 20, max: 100)
- Use `include_source_refs=false` when references not needed
- Choose optimal `format_type` for your use case

### 4. System Operations
- Use `include_modules=false` for faster status checks
- Set custom `timeout` values for long operations
- Leverage smart initialization for faster startup

## 📊 Current Performance Metrics
"""
                guide += json.dumps(self.performance_metrics, indent=2)

                return [types.TextContent(type="text", text=guide)]

            elif info_type == "modules":
                if self.tb_app:
                    modules_info = []
                    for mod_name in self.tb_app.functions:
                        modules_info.append(f"📦 **{mod_name}**")
                        if include_examples:
                            # Add function count
                            func_count = len(self.tb_app.functions.get(mod_name, {}))
                            modules_info.append(f"   - Functions: {func_count}")

                    content = "# 📦 Available Modules\n\n" + "\n".join(modules_info)

                    if include_examples:
                        content += "\n\n## 💡 Usage Example\n```\ntoolbox_execute(module_name='target_module', function_name='target_function')\n```"

                    return [types.TextContent(type="text", text=content)]
                else:
                    return [types.TextContent(type="text", text="❌ ToolBoxV2 not initialized")]

            # Handle other info types...
            return [types.TextContent(type="text", text=f"ℹ️ Info type '{info_type}' - Implementation pending")]

        except Exception as e:
            return [types.TextContent(type="text", text=f"❌ Error getting info: {e}")]

    async def _handle_python_execute(self, arguments: Dict) -> List[types.TextContent]:
        """Enhanced Python execution with security and notifications"""
        code = arguments.get("code", "")
        timeout = arguments.get("timeout", 30)

        try:
            quick_info("Python", f"Executing code ({len(code)} chars)")

            # Use ISAA interface if available for enhanced security
            isaa = self.tb_app.get_mod("isaa")
            if isaa and hasattr(isaa, 'get_tools_interface'):
                tools_interface = isaa.get_tools_interface("self")

                with MCPSafeIO():
                    result = await asyncio.wait_for(
                        tools_interface.execute_python(code),
                        timeout=timeout
                    )

                quick_success("Python", f"Code executed successfully")
                return [types.TextContent(type="text", text=f"**Python Execution Result:**\n```\n{result}\n```")]
            else:
                # Fallback execution with safety measures
                execution_globals = {
                    'app': self.tb_app,
                    'tb_app': self.tb_app,
                    **arguments.get("globals", {})
                }

                output_buffer = io.StringIO()

                with contextlib.redirect_stdout(output_buffer):
                    with contextlib.redirect_stderr(output_buffer):
                        result = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, lambda: eval(code, execution_globals)
                            ),
                            timeout=timeout
                        )

                        if result is not None:
                            output_buffer.write(str(result))

                output = output_buffer.getvalue()
                quick_success("Python", "Fallback execution completed")
                return [types.TextContent(type="text", text=f"**Python Output:**\n```\n{output}\n```")]

        except asyncio.TimeoutError:
            quick_warning("Python", f"Execution timed out after {timeout}s")
            return [types.TextContent(type="text", text=f"⏱️ Python execution timed out after {timeout} seconds")]
        except Exception as e:
            quick_error("Python", f"Execution failed: {str(e)[:100]}")
            return [types.TextContent(type="text", text=f"❌ Python execution error: {e}")]

    async def _handle_flow_start(self, arguments: Dict) -> List[types.TextContent]:
        """Enhanced flow management with progress tracking"""
        flow_name = arguments.get("flow_name")
        session_id = arguments.get("session_id")

        try:
            if not self.tb_app or not hasattr(self.tb_app, 'flows'):
                return [types.TextContent(type="text", text="❌ Flow system not available")]

            if flow_name not in self.tb_app.flows:
                available_flows = list(self.tb_app.flows.keys())
                return [types.TextContent(
                    type="text",
                    text=f"❌ Flow '{flow_name}' not found.\n\n**Available flows:**\n" +
                         "\n".join(f"- {flow}" for flow in available_flows)
                )]

            # Create session
            session_id = self.flow_session_manager.create_session(flow_name, session_id)

            return [types.TextContent(
                type="text",
                text=f"🚀 **Flow Started Successfully**\n\n" +
                     f"- **Flow**: {flow_name}\n" +
                     f"- **Session ID**: {session_id}\n" +
                     f"- **Status**: Ready for input\n\n" +
                     f"Use `flow_continue(session_id='{session_id}', input_data={{...}})` to proceed."
            )]

        except Exception as e:
            quick_error("Flow", f"Start failed: {str(e)[:100]}")
            return [types.TextContent(type="text", text=f"❌ Error starting flow: {e}")]

    async def _handle_flow_operation(self, name: str, arguments: Dict) -> List[types.TextContent]:
        """Handle other flow operations"""
        # Implementation for flow_continue, flow_status, etc.
        return [types.TextContent(type="text", text=f"🔄 Flow operation '{name}' - Implementation pending")]

    def _get_optimization_suggestions(self) -> List[str]:
        """Generate performance optimization suggestions"""
        suggestions = []

        if self.performance_metrics["avg_response_time"] > 2.0:
            suggestions.append("Consider using more specific queries to reduce response time")

        if self.performance_metrics["cache_hits"] / max(self.performance_metrics["requests_handled"], 1) < 0.3:
            suggestions.append("Enable caching (use_cache=true) for better performance")

        if len(self.flow_session_manager.sessions) > 50:
            suggestions.append("Consider cleaning up unused flow sessions")

        return suggestions

# Production interface and management
class ProductionMCPInterface:
    """Production-ready MCP server interface with comprehensive management"""

    def __init__(self):
        self.config = MCPConfig()
        self.server_instance: Optional[ToolBoxV2MCPServer] = None
        self.api_key_manager = UnifiedAPIKeyManager(self.config.api_keys_file)

    def generate_api_key(self, name: str, permissions: List[str] = None) -> Dict[str, str]:
        """Generate API key with rich feedback"""
        api_key = self.api_key_manager.generate_api_key(name, permissions)
        quick_success("API Keys", f"Generated key for '{name}'")

        return {
            "api_key": api_key,
            "name": name,
            "permissions": permissions or ["read", "write", "execute", "admin"],
            "usage": "Set as MCP_API_KEY environment variable or in connection config",
            "security_note": "🔐 Store this key securely - it won't be shown again"
        }

    def get_server_config(self) -> Dict:
        """Get comprehensive server configuration"""
        return {
            "server_info": {
                "name": self.config.server_name,
                "version": self.config.server_version,
                "performance_mode": self.config.performance_mode,
                "smart_init": self.config.smart_init
            },
            "features": {
                "flows": self.config.enable_flows,
                "python_execution": self.config.enable_python_execution,
                "system_manipulation": self.config.enable_system_manipulation,
                "docs_system": self.config.docs_system,
                "rich_notifications": self.config.rich_notifications
            },
            "performance": {
                "use_cached_index": self.config.use_cached_index,
                "session_timeout": self.config.session_timeout,
                "max_concurrent_sessions": self.config.max_concurrent_sessions
            },
            "connection": {
                "transport": "stdio",
                "authentication": "api_key",
                "api_key_header": "X-MCP-API-Key"
            }
        }

    async def start_server(self, mode: str = None):
        """Start the production server with full initialization"""
        try:
            # Override mode if specified
            if mode:
                self.config.server_mode = mode

            # Initialize smart logger
            global _smart_logger
            _smart_logger = SmartLogger(self.config)

            get_logger().info("MCP Server",
                              f"🚀 Starting {self.config.server_name} v{self.config.server_version}")
            get_logger().info("MCP Server", f"Mode: {self.config.server_mode.upper()}")

            if self.server_instance is None:
                self.server_instance = ToolBoxV2MCPServer(self.config)

            if self.config.server_mode == "http":
                # HTTP mode
                if not AIOHTTP_AVAILABLE:
                    get_logger().error("MCP Server",
                                       "HTTP mode requires aiohttp: pip install aiohttp aiohttp-cors")
                    raise ImportError("aiohttp not available")

                http_transport = MCPHTTPTransport(self.server_instance, self.config)
                await http_transport.start()

                # Keep running
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    get_logger().info("MCP Server", "🛑 Server stopped by user")

            else:
                # stdio mode (default)
                get_logger().info("MCP Server", "Using stdio transport")

                async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                    await self.server_instance.server.run(
                        read_stream,
                        write_stream,
                        InitializationOptions(
                            server_name=self.config.server_name,
                            server_version=self.config.server_version,
                            capabilities=self.server_instance.server.get_capabilities(
                                notification_options=NotificationOptions(),
                                experimental_capabilities={}
                            )
                        )
                    )

        except KeyboardInterrupt:
            get_logger().info("MCP Server", "🛑 Server stopped by user")
        except Exception as e:
            get_logger().error("MCP Server", f"Server error: {e}")
            raise


# Ersetze die main() Funktion (ca. Zeile ~980)

async def main():
    """Production main entry point with comprehensive CLI"""
    import argparse
    parser = argparse.ArgumentParser(
        description="ToolBoxV2 MCP Server - Production Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='tb mcp',
        epilog="""
Examples:
  # STDIO Mode (für Claude Desktop, etc.)
  %(prog)s                              # Start im stdio Modus

  # HTTP Mode (für Web/REST Clients)
  %(prog)s --mode http                  # Start HTTP server
  %(prog)s --mode http --port 8080      # Custom port
  %(prog)s --mode http --no-auth        # Ohne Authentifizierung

  # API Key Management
  %(prog)s --generate-key admin         # Generate admin API key
  %(prog)s --generate-key agent --permissions read execute
  %(prog)s --list-keys                  # List all API keys

  # Configuration
  %(prog)s --config                     # Show configuration
  %(prog)s --setup                      # Setup wizard
  %(prog)s --performance                # Performance guide
        """
    )

    # Server mode
    parser.add_argument("--mode", choices=["stdio", "http"], default="stdio",
                        help="Server mode: stdio (default) or http")
    parser.add_argument("--show", action="store_false",
                        help="Show console output", default=True)
    parser.add_argument("--log-file", type=str,
                        help="Log to file instead of stdout")

    # HTTP mode options
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="HTTP server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765,
                        help="HTTP server port (default: 8765)")
    parser.add_argument("--no-auth", action="store_true",
                        help="Disable API key authentication (NOT RECOMMENDED)")

    # Management commands
    parser.add_argument("--generate-key", type=str, metavar="NAME",
                        help="Generate new API key with given name")
    parser.add_argument("--permissions", nargs="+",
                        choices=["read", "write", "execute", "admin"],
                        help="Permissions for generated key (default: all)")
    parser.add_argument("--list-keys", action="store_true",
                        help="List all API keys")
    parser.add_argument("--revoke-key", type=str, metavar="NAME",
                        help="Revoke API key by name")
    parser.add_argument("--config", action="store_true",
                        help="Show server configuration")
    parser.add_argument("--setup", action="store_true",
                        help="Run setup wizard")
    parser.add_argument("--performance", action="store_true",
                        help="Show performance guide")

    args = parser.parse_args()

    # Setup config
    config = MCPConfig()
    config.server_mode = args.mode
    config.silent_mode = args.show or (os.environ.get('MCP_SILENT_MODE') == '1')
    config.log_file = args.log_file
    config.http_host = args.host
    config.http_port = args.port
    config.require_auth = not args.no_auth

    interface = ProductionMCPInterface()
    interface.config = config

    # Handle management commands (diese sollten nicht silent sein)
    if args.setup:
        # Temporär silent mode deaktivieren für Setup
        config.silent_mode = False
        _smart_logger = SmartLogger(config)

        print(f"\n{Style.CYAN('┌─')} 🧙 ToolBoxV2 MCP Server Setup Wizard ")
        print(Style.CYAN('│') )

        # Generate initial key if needed
        keys = interface.api_key_manager.list_keys()
        if not keys:
            print(Style.CYAN('│') + "  📝 No API keys found. Generating default admin key...")
            result = interface.generate_api_key("default_admin")
            print(Style.CYAN('│') + f"  🔑 Your API Key: {Style.YELLOW(result['api_key'])}")
            print(Style.CYAN('│') + "  " + Style.RED("⚠️  Save this key securely!"))

        print(Style.CYAN('│'))

        config_data = interface.get_server_config()
        print(Style.CYAN('│') + f"  📋 Server Configuration:")
        print(Style.CYAN('│') + f"     • Name: {config_data['server_info']['name']}")
        print(Style.CYAN('│') + f"     • Version: {config_data['server_info']['version']}")
        print(Style.CYAN('│') + f"     • Default Mode: {Style.GREEN('stdio')}")
        print(Style.CYAN('│') + f"     • HTTP Available: {Style.GREEN('Yes')}")
        print(Style.CYAN('│') )

        print(Style.CYAN('│') + "  " + Style.GREEN("✅ Setup complete!"))
        print(Style.CYAN('│') + f"  Run {Style.YELLOW('tb mcp')} to start the server.")
        print(Style.CYAN('│') )
        print(Style.CYAN('└─') + '─' * 50 + Style.CYAN('\n'))
        return

    if args.generate_key:
        config.silent_mode = False
        _smart_logger = SmartLogger(config)

        permissions = args.permissions if args.permissions else ["read", "write", "execute", "admin"]
        result = interface.generate_api_key(args.generate_key, permissions)

        print(f"\n{Style.GREEN('✓')} API Key Generated:")
        print(f"  {Style.CYAN('Name:')} {result['name']}")
        print(f"  {Style.CYAN('Key:')} {Style.YELLOW(result['api_key'])}")
        print(f"  {Style.CYAN('Permissions:')} {', '.join(result['permissions'])}")
        print(f"\n{Style.RED('⚠')}  {result['security_note']}\n")

        # STDIO Configuration
        print(f"{Style.CYAN('📋 STDIO Configuration (Claude Desktop, etc.):')}")
        print(json.dumps({
            "mcpServers": {
                "toolboxv2": {
                    "command": "tb",
                    "args": [
                        "mcp"
                    ],
                    "env": {
                        "MCP_API_KEY": result['api_key']
                    }
                }
            }
        }, indent=2))

        # HTTP Configuration
        print(f"\n{Style.CYAN('🌐 HTTP Configuration (URL Access):')}")
        print(json.dumps({
            "mcpServers": {
                "toolboxv2-http": {
                    "url": f"http://{config.http_host}:{config.http_port}/mcp",
                    "transport": "http",
                    "headers": {
                        "Authorization": f"Bearer {result['api_key']}"
                    }
                }
            }
        }, indent=2))

        print(f"\n{Style.YELLOW('💡 Usage:')}")
        print(f"  • STDIO: Add first config to your MCP client")
        print(f"  • HTTP: Start server with 'tb mcp --mode http' then use second config")
        return

    if args.list_keys:
        config.silent_mode = False
        _smart_logger = SmartLogger(config)

        keys = interface.api_key_manager.list_keys()
        print(f"\n📋 Found {len(keys)} API keys:\n")

        for key_hash, info in keys.items():
            print(f"  • {Style.CYAN(info['name'])}")
            print(f"    Permissions: {', '.join(info['permissions'])}")
            print(f"    Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info['created']))}")
            print(f"    Usage: {info['usage_count']} calls")
            print()
        return

    if args.config:
        config.silent_mode = False
        _smart_logger = SmartLogger(config)

        config_data = interface.get_server_config()
        print("\n📋 Server Configuration:")
        print(json.dumps(config_data, indent=2))
        print()
        print(json.dumps({
            "mcpServers": {
                "toolboxv2": {
                    "command": "tb",
                    "args": [
                        "mcp"
                    ],
                    "env": {
                        "MCP_API_KEY": "<['api_key']>"
                    }
                }
            }
        }, indent=2))

        # HTTP Configuration
        print(f"\n{Style.CYAN('🌐 HTTP Configuration (URL Access):')}")
        print(json.dumps({
            "mcpServers": {
                "toolboxv2-http": {
                    "url": f"http://{config.http_host}:{config.http_port}/mcp",
                    "transport": "http",
                    "headers": {
                        "Authorization": f"Bearer <['api_key']>"
                    }
                }
            }
        }, indent=2))

        return

    if args.performance:
        config.silent_mode = False
        _smart_logger = SmartLogger(config)

        print("""
🚀 ToolBoxV2 MCP Server - Performance Guide

## Key Features:
- Smart initialization with cached index loading
- Query result caching (5-minute TTL)
- Async operations with timeout protection
- Rich progress notifications
- Memory-efficient session management

## Server Modes:

### STDIO Mode (Default)
- Direct communication via stdin/stdout
- Perfect for Claude Desktop, Cline, etc.
- Zero network overhead
- Usage: tb mcp

### HTTP Mode
- REST API with SSE support
- Multi-client access
- API key authentication
- Usage: tb mcp --mode http --port 8765

## HTTP Endpoints:
- POST /mcp/initialize - Initialize session
- POST /mcp/tools/list - List available tools
- POST /mcp/tools/call - Execute tool
- POST /mcp/resources/list - List resources
- POST /mcp/resources/read - Read resource
- GET /health - Health check
- GET /api/keys - List API keys (admin)

## Authentication:
Include API key in requests:
  Authorization: Bearer YOUR_API_KEY
  or
  X-API-Key: YOUR_API_KEY
        """)
        return

    # Start the server
    if not config.silent_mode:
        print(f"\n{Style.CYAN('═' * 70)}")
        print(f"  {Style.GREEN('🚀 ToolBoxV2 MCP Server')}")
        print(f"  Mode: {Style.YELLOW(args.mode.upper())}")
        if args.mode == "http":
            print(f"  URL: {Style.CYAN(f'http://{args.host}:{args.port}')}")
            auth_status = Style.GREEN("Enabled") if config.require_auth else Style.RED("Disabled")
            print(f"  Auth: {auth_status}")
        print(f"{Style.CYAN('═' * 70)}\n")

    try:
        await interface.start_server(args.mode)
    except KeyboardInterrupt:
        get_logger().info("MCP Server", "🛑 Server stopped by user")
    except Exception as e:
        get_logger().error("Main", f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
