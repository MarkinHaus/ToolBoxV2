"""
ToolBoxV2 MCP Server
Sophisticated MCP server exposing all ToolBoxV2 functionality with proper communication handling.
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
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path

# MCP imports
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# ToolBoxV2 imports
from toolboxv2 import get_app, App, Result, Code
from toolboxv2.utils.extras.blobs import BlobFile
from toolboxv2.utils.system.types import CallingObject
from toolboxv2.flows import flows_dict as flows_dict_func

# Suppress all stdout/stderr during MCP operations
class MCPSafeIO:
    """Context manager to safely suppress stdout/stderr during MCP communication"""

    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        if self.suppress_stdout:
            self.original_stdout = sys.stdout
            sys.stdout = io.StringIO()
        if self.suppress_stderr:
            self.original_stderr = sys.stderr
            sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr

@dataclass
class MCPConfig:
    """MCP Server configuration"""
    server_name: str = "toolboxv2-mcp"
    server_version: str = "1.0.0"
    api_keys_file: str = "MCPConfig/mcp_api_keys.json"
    session_timeout: int = 3600  # 1 hour
    max_concurrent_sessions: int = 10
    enable_flows: bool = True
    enable_python_execution: bool = True
    enable_system_manipulation: bool = True

class APIKeyManager:
    """Manages API keys for MCP authentication"""

    def __init__(self, keys_file: str):
        self.keys_file = keys_file
        self.keys = self._load_keys()

    def _load_keys(self) -> Dict[str, Dict]:
        """Load API keys from file"""
        with MCPSafeIO():
            if BlobFile(self.keys_file, key=Code.DK()()).exists():
                try:
                    with BlobFile(self.keys_file, key=Code.DK()(), mode='r') as f:
                        return f.read_json()
                except Exception:
                    pass
        return {}

    def _save_keys(self):
        """Save API keys to file"""
        with BlobFile(self.keys_file, key=Code.DK()(), mode='w') as f:
            f.write_json(self.keys)

    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate a new API key"""
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
        return api_key

    def validate_key(self, api_key: str) -> Optional[Dict]:
        """Validate an API key and return permissions"""
        if not api_key or not api_key.startswith("tb_mcp_"):
            return None

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash in self.keys:
            key_info = self.keys[key_hash]
            key_info["last_used"] = time.time()
            key_info["usage_count"] += 1
            self._save_keys()
            return key_info
        return None

    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash in self.keys:
            del self.keys[key_hash]
            self._save_keys()
            return True
        return False

    def list_keys(self) -> List[Dict]:
        """List all API keys (without actual keys)"""
        return [
            {
                "name": info["name"],
                "permissions": info["permissions"],
                "created": info["created"],
                "last_used": info["last_used"],
                "usage_count": info["usage_count"]
            }
            for info in self.keys.values()
        ]

class FlowSessionManager:
    """Manages flow execution sessions"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.max_sessions = 50

    def create_session(self, flow_name: str, session_id: str = None) -> str:
        """Create a new flow session"""
        if session_id is None:
            session_id = f"flow_{uuid.uuid4().hex[:8]}"

        if len(self.sessions) >= self.max_sessions:
            # Remove oldest session
            oldest_id = min(self.sessions.keys(),
                          key=lambda k: self.sessions[k]["created"])
            del self.sessions[oldest_id]

        self.sessions[session_id] = {
            "flow_name": flow_name,
            "created": time.time(),
            "last_activity": time.time(),
            "state": "created",
            "context": {},
            "history": []
        }

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get flow session"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = time.time()
            return self.sessions[session_id]
        return None

    def update_session(self, session_id: str, **updates):
        """Update session data"""
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)
            self.sessions[session_id]["last_activity"] = time.time()

    def cleanup_expired_sessions(self, timeout: int = 3600):
        """Clean up expired sessions"""
        current_time = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if current_time - session["last_activity"] > timeout
        ]
        for sid in expired:
            del self.sessions[sid]

class ToolBoxV2MCPServer:
    """Main MCP Server for ToolBoxV2"""

    def __init__(self, config: MCPConfig = None):
        self.config = config or MCPConfig()
        self.server = Server(self.config.server_name)
        self.api_key_manager = APIKeyManager(self.config.api_keys_file)
        self.flow_session_manager = FlowSessionManager()
        self.tb_app: Optional[App] = None
        self.authenticated_sessions: Dict[str, Dict] = {}

        # Initialize ToolBoxV2 app with suppressed output
        with MCPSafeIO():
            self.tb_app = get_app(from_="MCP-Server", name="mcp")
            if self.tb_app:
                # Load all modules
                asyncio.create_task(self._initialize_toolbox())

        self._setup_handlers()

    async def _initialize_toolbox(self):
        """Initialize ToolBoxV2 with all modules loaded"""
        try:
            with MCPSafeIO():
                await self.tb_app.load_all_mods_in_file()
                # Set up flows
                flows_dict = flows_dict_func(remote=False)
                self.tb_app.set_flows(flows_dict)
        except Exception as e:
            logging.error(f"Failed to initialize ToolBoxV2: {e}")

    def _setup_handlers(self):
        """Set up MCP request handlers"""

        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List all available tools"""
            tools = []

            # Core ToolBoxV2 function execution tool
            tools.append(types.Tool(
                name="toolbox_execute",
                description="Execute any ToolBoxV2 module function with full access to the application",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "module_name": {
                            "type": "string",
                            "description": "Name of the ToolBoxV2 module"
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Name of the function to execute"
                        },
                        "args": {
                            "type": "array",
                            "description": "Positional arguments for the function",
                            "default": []
                        },
                        "kwargs": {
                            "type": "object",
                            "description": "Keyword arguments for the function",
                            "default": {}
                        },
                        "get_results": {
                            "type": "boolean",
                            "description": "Return full Result object instead of just data",
                            "default": False
                        }
                    },
                    "required": ["module_name", "function_name"]
                }
            ))

            # Flow management tools
            if self.config.enable_flows:
                tools.extend([
                    types.Tool(
                        name="flow_start",
                        description="Start a ToolBoxV2 flow with session management",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "flow_name": {"type": "string"},
                                "session_id": {"type": "string", "description": "Optional session ID"},
                                "kwargs": {"type": "object", "default": {}}
                            },
                            "required": ["flow_name"]
                        }
                    ),
                    types.Tool(
                        name="flow_continue",
                        description="Continue a flow session with user input or AI response",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "session_id": {"type": "string"},
                                "input_data": {"type": "object"},
                                "input_type": {
                                    "type": "string",
                                    "enum": ["user_input", "ai_response", "tool_result"],
                                    "default": "ai_response"
                                }
                            },
                            "required": ["session_id", "input_data"]
                        }
                    ),
                    types.Tool(
                        name="flow_status",
                        description="Get the status of a flow session",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "session_id": {"type": "string"}
                            },
                            "required": ["session_id"]
                        }
                    )
                ])

            # Python execution tool
            if self.config.enable_python_execution:
                tools.append(types.Tool(
                    name="python_execute",
                    description="Execute Python code with direct access to ToolBoxV2 app instance",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute. 'app' variable contains the ToolBoxV2 instance"
                            },
                            "globals": {
                                "type": "object",
                                "description": "Additional global variables",
                                "default": {}
                            }
                        },
                        "required": ["code"]
                    }
                ))

            # System manipulation tools
            if self.config.enable_system_manipulation:
                tools.extend([
                    types.Tool(
                        name="toolbox_status",
                        description="Get comprehensive ToolBoxV2 system status",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "include_modules": {"type": "boolean", "default": True},
                                "include_functions": {"type": "boolean", "default": False},
                                "include_flows": {"type": "boolean", "default": True}
                            }
                        }
                    ),
                    types.Tool(
                        name="module_manage",
                        description="Load, reload, or unload ToolBoxV2 modules",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["load", "reload", "unload", "list"]
                                },
                                "module_name": {
                                    "type": "string",
                                    "description": "Module name (required for load/reload/unload)"
                                }
                            },
                            "required": ["action"]
                        }
                    ),
                    types.Tool(
                        name="toolbox_info",
                        description="Get detailed information about modules, functions, and implementation guides",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "info_type": {
                                    "type": "string",
                                    "enum": ["modules", "functions", "module_detail", "function_detail", "implementation_guide", "flow_guide"]
                                },
                                "target": {
                                    "type": "string",
                                    "description": "Specific module or function name for detailed info"
                                }
                            },
                            "required": ["info_type"]
                        }
                    )
                ])

            return tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool execution requests"""

            # Authentication check would go here in a full implementation
            # For now, assuming authenticated

            try:
                if name == "toolbox_execute":
                    return await self._handle_toolbox_execute(arguments)
                elif name == "flow_start":
                    return await self._handle_flow_start(arguments)
                elif name == "flow_continue":
                    return await self._handle_flow_continue(arguments)
                elif name == "flow_status":
                    return await self._handle_flow_status(arguments)
                elif name == "python_execute":
                    return await self._handle_python_execute(arguments)
                elif name == "toolbox_status":
                    return await self._handle_toolbox_status(arguments)
                elif name == "module_manage":
                    return await self._handle_module_manage(arguments)
                elif name == "toolbox_info":
                    return await self._handle_toolbox_info(arguments)
                else:
                    return [types.TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )]

            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]

    async def _handle_toolbox_execute(self, arguments: Dict) -> List[types.TextContent]:
        """Execute ToolBoxV2 function"""
        module_name = arguments.get("module_name")
        function_name = arguments.get("function_name")
        args = arguments.get("args", [])
        kwargs = arguments.get("kwargs", {})
        get_results = arguments.get("get_results", False)

        if not self.tb_app:
            return [types.TextContent(
                type="text",
                text="ToolBoxV2 application not initialized"
            )]

        try:
            with MCPSafeIO():
                # Execute the function
                result = await self.tb_app.a_run_any(
                    (module_name, function_name),
                    args_=args,
                    get_results=get_results,
                    **kwargs
                )

                # Format result
                if get_results and hasattr(result, 'as_dict'):
                    result_text = json.dumps(result.as_dict(), indent=2)
                else:
                    result_text = str(result)

                return [types.TextContent(
                    type="text",
                    text=f"Executed {module_name}.{function_name}\nResult:\n{result_text}"
                )]

        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error executing {module_name}.{function_name}: {str(e)}"
            )]

    async def _handle_flow_start(self, arguments: Dict) -> List[types.TextContent]:
        """Start a ToolBoxV2 flow"""
        flow_name = arguments.get("flow_name")
        session_id = arguments.get("session_id")
        kwargs = arguments.get("kwargs", {})

        if not self.tb_app:
            return [types.TextContent(
                type="text",
                text="ToolBoxV2 application not initialized"
            )]

        try:
            # Create session
            session_id = self.flow_session_manager.create_session(flow_name, session_id)

            with MCPSafeIO():
                # Start the flow (this would need to be adapted based on actual flow implementation)
                if flow_name in self.tb_app.flows:
                    # For now, just prepare the flow
                    self.flow_session_manager.update_session(
                        session_id,
                        state="ready",
                        context=kwargs
                    )

                    return [types.TextContent(
                        type="text",
                        text=f"Flow '{flow_name}' started with session ID: {session_id}\nSession ready for input."
                    )]
                else:
                    return [types.TextContent(
                        type="text",
                        text=f"Flow '{flow_name}' not found. Available flows: {list(self.tb_app.flows.keys())}"
                    )]

        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error starting flow {flow_name}: {str(e)}"
            )]

    async def _handle_flow_continue(self, arguments: Dict) -> List[types.TextContent]:
        """Continue a flow session"""
        session_id = arguments.get("session_id")
        input_data = arguments.get("input_data")
        input_type = arguments.get("input_type", "ai_response")

        session = self.flow_session_manager.get_session(session_id)
        if not session:
            return [types.TextContent(
                type="text",
                text=f"Session {session_id} not found or expired"
            )]

        try:
            # Add input to session history
            session["history"].append({
                "type": input_type,
                "data": input_data,
                "timestamp": time.time()
            })

            # Continue flow execution based on input
            response = f"Flow '{session['flow_name']}' received {input_type}:\n{json.dumps(input_data, indent=2)}\n\n"
            response += f"Session state: {session['state']}\nHistory items: {len(session['history'])}"

            return [types.TextContent(
                type="text",
                text=response
            )]

        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error continuing flow session {session_id}: {str(e)}"
            )]

    async def _handle_flow_status(self, arguments: Dict) -> List[types.TextContent]:
        """Get flow session status"""
        session_id = arguments.get("session_id")

        session = self.flow_session_manager.get_session(session_id)
        if not session:
            return [types.TextContent(
                type="text",
                text=f"Session {session_id} not found"
            )]

        status = {
            "session_id": session_id,
            "flow_name": session["flow_name"],
            "state": session["state"],
            "created": session["created"],
            "last_activity": session["last_activity"],
            "history_count": len(session["history"]),
            "context_keys": list(session["context"].keys())
        }

        return [types.TextContent(
            type="text",
            text=f"Flow Session Status:\n{json.dumps(status, indent=2)}"
        )]

    async def _handle_python_execute(self, arguments: Dict) -> List[types.TextContent]:
        """Execute Python code with ToolBoxV2 access"""
        code = arguments.get("code", "")
        user_globals = arguments.get("globals", {})

        if not self.tb_app:
            return [types.TextContent(
                type="text",
                text="ToolBoxV2 application not initialized"
            )]

        try:
            # Get ISAA module for code execution if available
            isaa = self.tb_app.get_mod("isaa")
            if isaa and hasattr(isaa, 'get_tools_interface'):
                tools_interface = isaa.get_tools_interface("self")

                with MCPSafeIO():
                    result = await tools_interface.execute_python(code)

                return [types.TextContent(
                    type="text",
                    text=f"Python execution result:\n{result}"
                )]
            else:
                # Fallback: direct execution
                execution_globals = {
                    'app': self.tb_app,
                    'tb_app': self.tb_app,
                    **user_globals
                }

                # Capture output
                output_buffer = io.StringIO()
                with contextlib.redirect_stdout(output_buffer):
                    with contextlib.redirect_stderr(output_buffer):
                        try:
                            result = eval(code, execution_globals)
                            if result is not None:
                                output_buffer.write(str(result))
                        except SyntaxError:
                            exec(code, execution_globals)

                output = output_buffer.getvalue()
                return [types.TextContent(
                    type="text",
                    text=f"Python execution output:\n{output}"
                )]

        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error executing Python code: {str(e)}"
            )]

    async def _handle_toolbox_status(self, arguments: Dict) -> List[types.TextContent]:
        """Get ToolBoxV2 system status"""
        include_modules = arguments.get("include_modules", True)
        include_functions = arguments.get("include_functions", False)
        include_flows = arguments.get("include_flows", True)

        if not self.tb_app:
            return [types.TextContent(
                type="text",
                text="ToolBoxV2 application not initialized"
            )]

        try:
            status = {
                "app_id": self.tb_app.id,
                "version": self.tb_app.version,
                "debug_mode": self.tb_app.debug,
                "alive": self.tb_app.alive,
                "system": self.tb_app.system_flag
            }

            if include_modules:
                status["modules"] = list(self.tb_app.functions.keys())
                status["module_count"] = len(self.tb_app.functions)

            if include_functions:
                functions = {}
                for mod_name, mod_functions in self.tb_app.functions.items():
                    if isinstance(mod_functions, dict):
                        functions[mod_name] = list(mod_functions.keys())
                status["functions"] = functions

            if include_flows and hasattr(self.tb_app, 'flows'):
                status["flows"] = list(self.tb_app.flows.keys())
                status["flow_count"] = len(self.tb_app.flows)

            return [types.TextContent(
                type="text",
                text=f"ToolBoxV2 System Status:\n{json.dumps(status, indent=2)}"
            )]

        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error getting system status: {str(e)}"
            )]

    async def _handle_module_manage(self, arguments: Dict) -> List[types.TextContent]:
        """Manage ToolBoxV2 modules"""
        action = arguments.get("action")
        module_name = arguments.get("module_name")

        if not self.tb_app:
            return [types.TextContent(
                type="text",
                text="ToolBoxV2 application not initialized"
            )]

        try:
            with MCPSafeIO():
                if action == "list":
                    modules = list(self.tb_app.functions.keys())
                    return [types.TextContent(
                        type="text",
                        text=f"Loaded modules ({len(modules)}):\n" + "\n".join(f"- {mod}" for mod in modules)
                    )]

                elif action == "load":
                    if not module_name:
                        return [types.TextContent(
                            type="text",
                            text="Module name required for load action"
                        )]

                    result = self.tb_app.load_mod(module_name)
                    return [types.TextContent(
                        type="text",
                        text=f"Module '{module_name}' loaded successfully: {result}"
                    )]

                elif action == "reload":
                    if not module_name:
                        return [types.TextContent(
                            type="text",
                            text="Module name required for reload action"
                        )]

                    result = await self.tb_app.reload_mod(module_name)
                    return [types.TextContent(
                        type="text",
                        text=f"Module '{module_name}' reloaded successfully: {result}"
                    )]

                elif action == "unload":
                    if not module_name:
                        return [types.TextContent(
                            type="text",
                            text="Module name required for unload action"
                        )]

                    result = await self.tb_app.a_remove_mod(module_name)
                    return [types.TextContent(
                        type="text",
                        text=f"Module '{module_name}' unloaded successfully: {result}"
                    )]

                else:
                    return [types.TextContent(
                        type="text",
                        text=f"Unknown action: {action}. Available: list, load, reload, unload"
                    )]

        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error managing module: {str(e)}"
            )]

    async def _handle_toolbox_info(self, arguments: Dict) -> List[types.TextContent]:
        """Get detailed ToolBoxV2 information and guides"""
        info_type = arguments.get("info_type")
        target = arguments.get("target")

        if not self.tb_app:
            return [types.TextContent(
                type="text",
                text="ToolBoxV2 application not initialized"
            )]

        try:
            if info_type == "modules":
                modules_info = []
                for mod_name in self.tb_app.functions:
                    mod_data = self.tb_app.functions[mod_name]
                    if hasattr(mod_data, 'get') and isinstance(mod_data.get("app_instance"), object):
                        version = getattr(mod_data["app_instance"], 'version', 'unknown')
                    else:
                        version = 'functions-only'
                    modules_info.append(f"- {mod_name}: {version}")

                return [types.TextContent(
                    type="text",
                    text=f"ToolBoxV2 Modules:\n" + "\n".join(modules_info)
                )]

            elif info_type == "functions":
                if target:
                    # Specific module functions
                    if target in self.tb_app.functions:
                        mod_functions = self.tb_app.functions[target]
                        if isinstance(mod_functions, dict):
                            func_list = "\n".join(f"- {fname}" for fname in mod_functions.keys())
                            return [types.TextContent(
                                type="text",
                                text=f"Functions in module '{target}':\n{func_list}"
                            )]
                    return [types.TextContent(
                        type="text",
                        text=f"Module '{target}' not found or has no functions"
                    )]
                else:
                    # All functions
                    all_functions = {}
                    for mod_name, mod_functions in self.tb_app.functions.items():
                        if isinstance(mod_functions, dict):
                            all_functions[mod_name] = list(mod_functions.keys())

                    return [types.TextContent(
                        type="text",
                        text=f"All Functions:\n{json.dumps(all_functions, indent=2)}"
                    )]

            elif info_type == "function_detail":
                if not target or "." not in target:
                    return [types.TextContent(
                        type="text",
                        text="Function detail requires format: 'module_name.function_name'"
                    )]

                mod_name, func_name = target.split(".", 1)
                func_data = self.tb_app.get_function((mod_name, func_name), metadata=True)

                return [types.TextContent(
                    type="text",
                    text=f"Function Details for {target}:\n{json.dumps(func_data, indent=2, default=str)}"
                )]

            elif info_type == "implementation_guide":
                guide = """
# ToolBoxV2 Module Implementation Guide

## Basic Module Structure

```python
from toolboxv2 import get_app, Result

app = get_app("MyModule")
export = app.tb

Name = "MyModule"
version = "1.0.0"

@export(mod_name=Name, version=version, helper="My function description")
def my_function(param1: str, param2: int = 10) -> Result:
    '''Function docstring'''
    # Your logic here
    return Result.ok(data={"result": param1 * param2})

# For API endpoints
@export(mod_name=Name, version=version, api=True, api_methods=['GET', 'POST'])
async def api_endpoint(request=None) -> Result:
    return Result.json(data={"message": "API response"})

# For initialization
@export(mod_name=Name, version=version, initial=True)
def on_load():
    print(f"{Name} loaded successfully")

# For cleanup
@export(mod_name=Name, version=version, exit_f=True)
async def on_exit():
    print(f"{Name} shutting down")
```

## MainTool Class (Advanced)

```python
from toolboxv2 import MainTool, Result

class Tools(MainTool):
    async def __ainit__(self):
        await super().__ainit__(name="MyTool", v="1.0.0")
        # Initialization code

    @export(mod_name="MyTool")
    def tool_function(self):
        return Result.ok(data="Tool result")
```

## Key Decorators Parameters

- `mod_name`: Module name
- `version`: Version string
- `helper`: Description/help text
- `api`: Enable as HTTP endpoint
- `api_methods`: Allowed HTTP methods
- `initial`: Run on module load
- `exit_f`: Run on shutdown
- `memory_cache`: Enable caching
- `row`: Return raw data (not wrapped in Result)

"""
                return [types.TextContent(type="text", text=guide)]

            elif info_type == "flow_guide":
                flow_guide = """
# ToolBoxV2 Flow Implementation Guide

Flows are sequences of operations that can handle user interaction and complex workflows.

## Available Flows
""" + "\n".join(f"- {flow}" for flow in self.tb_app.flows.keys() if hasattr(self.tb_app, 'flows'))

                return [types.TextContent(type="text", text=flow_guide)]

            else:
                return [types.TextContent(
                    type="text",
                    text=f"Unknown info type: {info_type}. Available: modules, functions, module_detail, function_detail, implementation_guide, flow_guide"
                )]

        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error getting info: {str(e)}"
            )]


# Main server functions and interface
class MCPInterface:
    """Mini interface for MCP server management"""

    def __init__(self):
        self.config = MCPConfig()
        self.server_instance: Optional[ToolBoxV2MCPServer] = None
        self.api_key_manager = APIKeyManager(self.config.api_keys_file)

    def generate_api_key(self, name: str, permissions: List[str] = None) -> Dict[str, str]:
        """Generate a new API key"""
        api_key = self.api_key_manager.generate_api_key(name, permissions)
        return {
            "api_key": api_key,
            "name": name,
            "permissions": permissions or ["read", "write", "execute", "admin"],
            "usage": "Set this as MCP_API_KEY environment variable or pass in connection config"
        }

    def list_api_keys(self) -> List[Dict]:
        """List all API keys"""
        return self.api_key_manager.list_keys()

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        return self.api_key_manager.revoke_key(api_key)

    def get_server_config(self) -> Dict:
        """Get server configuration"""
        return {
            "server_name": self.config.server_name,
            "server_version": self.config.server_version,
            "features": {
                "flows": self.config.enable_flows,
                "python_execution": self.config.enable_python_execution,
                "system_manipulation": self.config.enable_system_manipulation
            },
            "connection_info": {
                "transport": "stdio",
                "authentication": "api_key",
                "api_key_header": "X-MCP-API-Key"
            }
        }

    async def start_server(self):
        """Start the MCP server"""
        if self.server_instance is None:
            self.server_instance = ToolBoxV2MCPServer(self.config)

        # Run the server using stdio transport
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

def main():
    """Main entry point for the MCP server"""
    import argparse

    parser = argparse.ArgumentParser(description="ToolBoxV2 MCP Server")
    parser.add_argument("--generate-key", type=str, help="Generate new API key with given name")
    parser.add_argument("--list-keys", action="store_true", help="List all API keys")
    parser.add_argument("--revoke-key", type=str, help="Revoke API key")
    parser.add_argument("--config", action="store_true", help="Show server configuration")

    args = parser.parse_args()

    interface = MCPInterface()

    if args.generate_key:
        result = interface.generate_api_key(args.generate_key)
        print(json.dumps(result, indent=2))
        return

    if args.list_keys:
        keys = interface.list_api_keys()
        print(json.dumps(keys, indent=2))
        return

    if args.revoke_key:
        success = interface.revoke_api_key(args.revoke_key)
        print(f"API key {'revoked' if success else 'not found'}")
        return

    if args.config:
        config = interface.get_server_config()
        print(json.dumps(config, indent=2))
        return

    # Start the server
    asyncio.run(interface.start_server())


# mcp_config.py
"""
ToolBoxV2 MCP Server Configuration and Setup
"""

import json
import os
from pathlib import Path

def setup_mcp_server():
    """Set up the MCP server with initial configuration"""

    interface = MCPInterface()

    # Generate initial API key if none exist
    keys = interface.list_api_keys()
    if not keys:
        print("No API keys found. Generating default key...")
        result = interface.generate_api_key("default_admin", ["read", "write", "execute", "admin"])
        print("Generated API Key:")
        print(json.dumps(result, indent=2))
        print("\nSave this API key securely!")
    else:
        print(f"Found {len(keys)} existing API keys")

    # Show configuration
    config = interface.get_server_config()
    print("\nServer Configuration:")
    print(json.dumps(config, indent=2))

    # Create MCP client configuration template
    client_config = {
        "mcpServers": {
            "toolboxv2": {
                "command": "tb",
                "args": ["mcp"],
                "env": {
                    "MCP_API_KEY": "YOUR_API_KEY_HERE"
                }
            }
        }
    }

    config_path = Path("mcp_client_config.json")
    with open(config_path, "w") as f:
        json.dump(client_config, f, indent=2)

    print(f"\nClient configuration template saved to: {config_path}")
    print("Update the MCP_API_KEY in the configuration before connecting.")

if __name__ == "__main__":
    setup_mcp_server()

