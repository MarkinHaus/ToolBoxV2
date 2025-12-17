"""
ToolBoxV2 MCP Server - Main Server Facade
==========================================
Production-ready MCP server with STDIO and HTTP transport
Following ToolBox V2 Architecture Guidelines (Facade & Workers Pattern)
"""

import asyncio
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Union

# MCP SDK imports
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from pydantic import AnyUrl

# Local imports
from .models import (
    MCPConfig,
    ServerMode,
    FlowState,
    ResponseFormat,
    ToolResult,
    FLOWAGENTS_DISCOVERY_TEMPLATE,
    PYTHON_EXECUTION_TEMPLATE,
    PERFORMANCE_GUIDE_TEMPLATE,
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

# Enhanced Flow Management
from .flow_manager import FlowSessionManager as EnhancedFlowManager
from .flow_handlers import FlowHandlers

# Setup logging to stderr (MCP safe)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='[MCP %(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("mcp.server")


class ToolBoxV2MCPServer:
    """
    Production-ready MCP Server Facade.

    This class orchestrates all components but doesn't implement business logic.
    Following the Facade & Workers pattern:
    - Facade (this class): Routing, initialization, MCP protocol handling
    - Workers: Actual business logic (Python, Docs, Toolbox, System)
    - Managers: State management (API Keys, Sessions, Cache, Performance)

    Features:
    - STDIO and HTTP transport support
    - Lazy loading for fast MCP handshake
    - API key authentication
    - Query caching
    - Performance tracking
    - Flow session management
    """

    def __init__(self, config: Optional[MCPConfig] = None):
        self.config = config or MCPConfig()

        # MCP Server instance
        self.server = Server(self.config.server_name)

        # State
        self._app: Optional[Any] = None  # ToolBoxV2 App (lazy loaded)
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Managers
        self.api_keys = APIKeyManager(self.config.api_keys_file)

        # Enhanced Flow Manager with I/O capture
        self.flow_manager = EnhancedFlowManager(
            max_sessions=self.config.max_sessions,
            timeout=self.config.session_timeout
        )
        self.flow_handlers = FlowHandlers(self.flow_manager)
        self.sessions = self.flow_manager  # Backward compatibility

        self.cache = CacheManager(
            max_size=self.config.max_cache_size,
            default_ttl=self.config.cache_ttl
        )
        self.py_context = PythonContextManager()
        self.performance = PerformanceTracker()

        # Workers
        self.python_worker = PythonWorker(self.py_context)
        self.docs_worker = DocsWorker(self.cache)
        self.toolbox_worker = ToolboxWorker()
        self.system_worker = SystemWorker()

        # HTTP Transport (lazy initialized)
        self._http_transport = None

        # Register MCP handlers
        self._setup_handlers()

        logger.info(f"ToolBoxV2 MCP Server v{self.config.server_version} initialized")

    # =========================================================================
    # LAZY INITIALIZATION
    # =========================================================================

    async def _ensure_app(self) -> None:
        """
        Lazy load ToolBoxV2 app.

        This is called before first tool execution, not at server start.
        This allows the MCP handshake to complete fast (list_tools returns
        static definitions immediately).
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            start_time = time.time()

            try:
                with MCPSafeIO():
                    logger.info("Lazy loading ToolBoxV2...")

                    # Import and initialize ToolBoxV2
                    from toolboxv2 import get_app

                    self._app = get_app(from_="MCP-Server", name="mcp")

                    # Suppress app's print functions
                    self._app.print = lambda *a, **k: None
                    self._app.sprint = lambda *a, **k: None

                    # Set app reference for flow manager
                    self.flow_manager.set_app(self._app)

                    # Initialize flows if available
                    if self.config.enable_flows:
                        try:
                            from toolboxv2.flows import flows_dict as flows_dict_func
                            flows = flows_dict_func(remote=False)
                            self._app.set_flows(flows)
                            logger.info(f"Loaded {len(flows)} flows")
                        except Exception as e:
                            logger.warning(f"Could not load flows: {e}")

                    # Initialize docs if enabled
                    if self.config.enable_docs and hasattr(self._app, 'docs_init'):
                        try:
                            await self._app.docs_init()
                            logger.info("Documentation system initialized")
                        except Exception as e:
                            logger.warning(f"Could not initialize docs: {e}")

                init_time = time.time() - start_time
                await self.performance.set_init_time(init_time)

                self._initialized = True
                logger.info(f"ToolBoxV2 ready in {init_time:.2f}s")

            except Exception as e:
                logger.error(f"Failed to initialize ToolBoxV2: {e}")
                raise

    # =========================================================================
    # TOOL DEFINITIONS (Static - no app loading needed)
    # =========================================================================

    def get_tool_definitions(self) -> List[Dict]:
        """Get static tool definitions for list_tools."""
        tools = []

        # Core execution tools
        tools.append({
            "name": "toolbox_execute",
            "description": "Execute any ToolBoxV2 module function. Use toolbox_info to discover available modules and functions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "description": "Module name (e.g., 'CloudM', 'isaa')"
                    },
                    "function": {
                        "type": "string",
                        "description": "Function name within module"
                    },
                    "args": {
                        "type": "array",
                        "description": "Positional arguments",
                        "default": []
                    },
                    "kwargs": {
                        "type": "object",
                        "description": "Keyword arguments",
                        "default": {}
                    },
                    "get_results": {
                        "type": "boolean",
                        "description": "Return full Result object with metadata",
                        "default": False
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["module", "function"]
            }
        })

        # Python execution
        if self.config.enable_python:
            tools.append({
                "name": "python_execute",
                "description": "Execute Python code with persistent state. Variables persist across calls. 'app' and 'tb' provide ToolBoxV2 access.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute (supports full statements, not just expressions)"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Execution timeout in seconds",
                            "default": 30
                        }
                    },
                    "required": ["code"]
                }
            })

        # Documentation tools
        if self.config.enable_docs:
            tools.extend([
                {
                    "name": "docs_reader",
                    "description": "Search and read documentation with intelligent caching. Use section_id for fastest access.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (keywords, phrases)"
                            },
                            "section_id": {
                                "type": "string",
                                "description": "Direct section access (fastest method)"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Filter by documentation file path"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by tags"
                            },
                            "format_type": {
                                "type": "string",
                                "enum": ["markdown", "json", "structured"],
                                "default": "markdown",
                                "description": "Output format"
                            },
                            "max_results": {
                                "type": "integer",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 100,
                                "description": "Maximum results to return"
                            },
                            "use_cache": {
                                "type": "boolean",
                                "default": True,
                                "description": "Use cached results"
                            }
                        }
                    }
                },
                {
                    "name": "docs_writer",
                    "description": "Create or update documentation sections.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["create_file", "add_section", "update_section", "generate_from_code"],
                                "description": "Action to perform"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Target file path (relative to docs/)"
                            },
                            "section_title": {
                                "type": "string",
                                "description": "Section title"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write"
                            },
                            "source_file": {
                                "type": "string",
                                "description": "Source file for AI generation"
                            },
                            "auto_generate": {
                                "type": "boolean",
                                "default": False,
                                "description": "Use AI to generate content"
                            }
                        },
                        "required": ["action"]
                    }
                },
                {
                    "name": "source_code_lookup",
                    "description": "Find source code elements (classes, functions, etc.) by name.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "element_name": {
                                "type": "string",
                                "description": "Element name to search for"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Filter by file path"
                            },
                            "element_type": {
                                "type": "string",
                                "enum": ["class", "function", "method", "variable"],
                                "description": "Filter by element type"
                            },
                            "max_results": {
                                "type": "integer",
                                "default": 25,
                                "description": "Maximum results"
                            },
                            "include_code": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include code snippets"
                            }
                        },
                        "required": ["element_name"]
                    }
                },
                {
                    "name": "get_task_context",
                    "description": "Get optimized context for an editing task using the graph-based context engine.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Files to analyze"
                            },
                            "intent": {
                                "type": "string",
                                "description": "Task intent/description"
                            }
                        },
                        "required": ["files", "intent"]
                    }
                }
            ])

        # System tools
        if self.config.enable_system:
            tools.extend([
                {
                    "name": "toolbox_status",
                    "description": "Get comprehensive system status including modules, flows, and performance metrics.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "include_modules": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include module list"
                            },
                            "include_flows": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include flow list"
                            },
                            "include_functions": {
                                "type": "boolean",
                                "default": False,
                                "description": "Include detailed function list"
                            }
                        }
                    }
                },
                {
                    "name": "toolbox_info",
                    "description": "Get detailed information about modules, functions, flows, or guides. Use this to discover and understand ToolBoxV2 capabilities.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "info_type": {
                                "type": "string",
                                "enum": [
                                    "modules",
                                    "functions",
                                    "function_detail",
                                    "flows",
                                    "flow_detail",
                                    "python_guide",
                                    "performance_guide",
                                    "discovery"
                                ],
                                "description": "Type of information: 'modules' (list all), 'functions' (list in module), 'function_detail' (specific function), 'flows' (list all), 'flow_detail' (specific flow), 'discovery' (full overview)"
                            },
                            "target": {
                                "type": "string",
                                "description": "Target for detail queries. For 'functions': module name. For 'function_detail': 'Module.function'. For 'flow_detail': flow name."
                            },
                            "include_examples": {
                                "type": "boolean",
                                "default": False,
                                "description": "Include usage examples in output"
                            }
                        },
                        "required": ["info_type"]
                    }
                },
                {
                    "name": "toolbox_search",
                    "description": "Search for functions across all modules by name, module, or docstring content. Great for finding relevant functions when you don't know the exact name.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (matches function names, module names, docstrings)"
                            },
                            "max_results": {
                                "type": "integer",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 50,
                                "description": "Maximum results to return"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "toolbox_tree",
                    "description": "Get a visual tree view of modules and their functions. Useful for exploring the system structure.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "module": {
                                "type": "string",
                                "description": "Specific module to show (omit for all modules)"
                            },
                            "depth": {
                                "type": "integer",
                                "default": 2,
                                "minimum": 1,
                                "maximum": 3,
                                "description": "Tree depth: 1 = modules only, 2 = with functions"
                            }
                        }
                    }
                },
                {
                    "name": "toolbox_callable",
                    "description": "Get a concise summary of all callable API functions in a module with their signatures. Optimized for understanding what you can call.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "module": {
                                "type": "string",
                                "description": "Module name to get callable summary for"
                            }
                        },
                        "required": ["module"]
                    }
                }
            ])

        # Flow tools - Use enhanced FlowHandlers definitions
        if self.config.enable_flows:
            tools.extend(FlowHandlers.get_tool_definitions())

        return tools

    # =========================================================================
    # RESOURCE DEFINITIONS
    # =========================================================================

    def get_resource_definitions(self) -> List[Dict]:
        """Get resource definitions."""
        return [
            {
                "uri": "flowagents://discovery",
                "name": "Resource Discovery Guide",
                "description": "Comprehensive guide to server capabilities and tools",
                "mimeType": "text/markdown"
            },
            {
                "uri": "flowagents://python_guide",
                "name": "Python Execution Guide",
                "description": "Guide for using the Python execution environment",
                "mimeType": "text/markdown"
            },
            {
                "uri": "flowagents://performance",
                "name": "Performance Guide",
                "description": "Performance optimization tips and current metrics",
                "mimeType": "text/markdown"
            },
            {
                "uri": "toolbox://status",
                "name": "System Status",
                "description": "Real-time system status",
                "mimeType": "application/json"
            },
            {
                "uri": "toolbox://performance",
                "name": "Performance Metrics",
                "description": "Server performance analytics",
                "mimeType": "application/json"
            }
        ]

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI."""
        if uri == "flowagents://discovery":
            return FLOWAGENTS_DISCOVERY_TEMPLATE

        elif uri == "flowagents://python_guide":
            return PYTHON_EXECUTION_TEMPLATE

        elif uri == "flowagents://performance":
            m = self.performance.metrics
            return PERFORMANCE_GUIDE_TEMPLATE.format(
                cache_ttl=self.config.cache_ttl,
                max_cache_size=self.config.max_cache_size,
                requests=m.requests_handled,
                avg_time=m.avg_response_time,
                hit_rate=m.cache_hit_rate
            )

        elif uri == "toolbox://status":
            await self._ensure_app()
            return json.dumps({
                "initialized": self._initialized,
                "modules": list(getattr(self._app, 'functions', {}).keys()) if self._app else [],
                "flows": list(getattr(self._app, 'flows', {}).keys()) if self._app else [],
                "docs_available": hasattr(self._app, 'docs_reader') if self._app else False
            }, indent=2)

        elif uri == "toolbox://performance":
            return json.dumps(self.performance.to_dict(), indent=2)

        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    # =========================================================================
    # TOOL EXECUTION
    # =========================================================================

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[ToolResult]:
        """
        Execute a tool and return results.

        This is the main routing method that dispatches to workers.
        """
        start_time = time.time()

        try:
            # Ensure app is loaded before execution
            await self._ensure_app()

            result: ToolResult

            # Route to appropriate worker
            if name == "toolbox_execute":
                result = await self.toolbox_worker.execute(
                    app=self._app,
                    module_name=arguments.get("module"),
                    function_name=arguments.get("function"),
                    args=arguments.get("args"),
                    kwargs=arguments.get("kwargs"),
                    get_results=arguments.get("get_results", False),
                    timeout=arguments.get("timeout", 30)
                )

            elif name == "python_execute":
                result = await self.python_worker.execute(
                    code=arguments.get("code", ""),
                    app=self._app,
                    timeout=arguments.get("timeout", 30)
                )

            elif name == "docs_reader":
                result = await self.docs_worker.reader(
                    app=self._app,
                    query=arguments.get("query"),
                    section_id=arguments.get("section_id"),
                    file_path=arguments.get("file_path"),
                    tags=arguments.get("tags"),
                    format_type=arguments.get("format_type", "markdown"),
                    max_results=arguments.get("max_results", 20),
                    use_cache=arguments.get("use_cache", True)
                )

            elif name == "docs_writer":
                result = await self.docs_worker.writer(
                    app=self._app,
                    action=arguments.get("action"),
                    file_path=arguments.get("file_path"),
                    section_title=arguments.get("section_title"),
                    content=arguments.get("content"),
                    source_file=arguments.get("source_file"),
                    auto_generate=arguments.get("auto_generate", False)
                )

            elif name == "source_code_lookup":
                result = await self.docs_worker.lookup(
                    app=self._app,
                    element_name=arguments.get("element_name"),
                    file_path=arguments.get("file_path"),
                    element_type=arguments.get("element_type"),
                    max_results=arguments.get("max_results", 25),
                    include_code=arguments.get("include_code", True)
                )

            elif name == "get_task_context":
                result = await self.docs_worker.get_task_context(
                    app=self._app,
                    files=arguments.get("files", []),
                    intent=arguments.get("intent", "")
                )

            elif name == "toolbox_status":
                result = await self.system_worker.get_status(
                    app=self._app,
                    include_modules=arguments.get("include_modules", True),
                    include_flows=arguments.get("include_flows", True),
                    include_functions=arguments.get("include_functions", False),
                    metrics=self.performance.to_dict()
                )

            elif name == "toolbox_info":
                result = await self.system_worker.get_info(
                    app=self._app,
                    info_type=arguments.get("info_type"),
                    target=arguments.get("target"),
                    include_examples=arguments.get("include_examples", False)
                )

            elif name == "toolbox_search":
                result = await self.system_worker.search_functions(
                    app=self._app,
                    query=arguments.get("query", ""),
                    max_results=arguments.get("max_results", 20)
                )

            elif name == "toolbox_tree":
                result = await self.system_worker.get_module_tree(
                    app=self._app,
                    module=arguments.get("module"),
                    depth=arguments.get("depth", 2)
                )

            elif name == "toolbox_callable":
                result = await self.system_worker.get_callable_summary(
                    app=self._app,
                    module=arguments.get("module", "")
                )

            # Route all flow tools to enhanced FlowHandlers
            elif name.startswith("flow_"):
                flow_result = await self.flow_handlers.route(name, arguments, self._app)
                result = ToolResult(
                    success=flow_result.success,
                    content=flow_result.content,
                    execution_time=flow_result.execution_time,
                    error=flow_result.error
                )

            else:
                result = ToolResult(
                    success=False,
                    content=f"Unknown tool: {name}",
                    execution_time=time.time() - start_time,
                    error="UnknownTool"
                )

            # Record metrics
            await self.performance.record(
                response_time=result.execution_time,
                cached=result.cached,
                error=not result.success
            )

            return [result]

        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            return [ToolResult(
                success=False,
                content=f"❌ Error executing {name}: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )]

    # =========================================================================
    # MCP HANDLER SETUP
    # =========================================================================

    def _setup_handlers(self):
        """Setup MCP protocol handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """Return static tool definitions immediately (no app loading)."""
            tools = self.get_tool_definitions()
            return [
                types.Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema=t["inputSchema"]
                )
                for t in tools
            ]

        @self.server.list_resources()
        async def handle_list_resources() -> List[types.Resource]:
            """Return resource definitions."""
            resources = self.get_resource_definitions()
            return [
                types.Resource(
                    uri=r["uri"],
                    name=r["name"],
                    description=r["description"],
                    mimeType=r["mimeType"]
                )
                for r in resources
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: AnyUrl) -> str:
            """Read a resource."""
            return await self.read_resource(str(uri))

        @self.server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: Dict[str, Any]
        ) -> List[types.TextContent]:
            """Execute a tool."""
            results = await self.call_tool(name, arguments)

            return [
                types.TextContent(type="text", text=r.content)
                for r in results
            ]

    # =========================================================================
    # SERVER LIFECYCLE
    # =========================================================================

    async def run_stdio(self) -> None:
        """Run server with STDIO transport."""
        logger.info("Starting STDIO transport...")

        # Start background tasks
        await self.sessions.start_cleanup()

        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name=self.config.server_name,
                        server_version=self.config.server_version,
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={}
                        )
                    )
                )
        finally:
            await self.sessions.stop_cleanup()
            self._cleanup()

    async def run_http(self) -> None:
        """Run server with HTTP transport."""
        from .http_transport import HTTPTransport

        logger.info(f"Starting HTTP transport on {self.config.http_host}:{self.config.http_port}")

        # Start background tasks
        await self.sessions.start_cleanup()

        try:
            self._http_transport = HTTPTransport(self)
            await self._http_transport.run_forever()
        finally:
            await self.sessions.stop_cleanup()
            self._cleanup()

    def _cleanup(self):
        """Cleanup resources."""
        self.python_worker.close()
        self.toolbox_worker.close()
        self.api_keys.close()
        logger.info("Server resources cleaned up")
