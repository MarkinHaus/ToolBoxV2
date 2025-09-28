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
from enum import Enum
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
    docs_system: bool = True
    docs_reader: bool = True
    docs_writer: bool = True

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
        self.docs_system = None

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
                await self.tb_app.get_mod("isaa").init_isaa()
                self.docs_system = self.tb_app.mkdocs
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

            if self.config.docs_system:
                if self.config.docs_reader:
                    tools.append(
                    types.Tool(
                        name="docs_reader",
                        description="Read documentation with advanced filtering and search capabilities",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query for finding specific documentation"
                                },
                                "section_id": {
                                    "type": "string",
                                    "description": "Specific section ID to retrieve (e.g., 'file.md#Section Title')"
                                },
                                "file_path": {
                                    "type": "string",
                                    "description": "Filter by specific documentation file path"
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Filter by tags"
                                },
                                "include_source_refs": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Include source code references"
                                },
                                "format_type": {
                                    "type": "string",
                                    "enum": ["structured", "markdown", "json"],
                                    "default": "structured",
                                    "description": "Output format type"
                                }
                            }
                        }
                    )
                    )
                if self.config.docs_writer and self.config.enable_system_manipulation:
                    tools.extend([
                    types.Tool(
                        name="docs_writer",
                        description="Write, update, or generate documentation with precise control",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["create_file", "add_section", "update_section", "generate_from_code"],
                                    "description": "Type of documentation action to perform"
                                },
                                "file_path": {
                                    "type": "string",
                                    "description": "Target documentation file path (relative to docs/)"
                                },
                                "section_title": {
                                    "type": "string",
                                    "description": "Title for new or updated section"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Content for the section (if not auto-generating)"
                                },
                                "source_file": {
                                    "type": "string",
                                    "description": "Source code file to generate documentation from"
                                },
                                "auto_generate": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Use AI to generate content from source code"
                                },
                                "position": {
                                    "type": "string",
                                    "description": "Position for new sections: 'top', 'bottom', or 'after:SectionName'"
                                },
                                "level": {
                                    "type": "integer",
                                    "default": 2,
                                    "minimum": 1,
                                    "maximum": 6,
                                    "description": "Header level for new sections"
                                }
                            },
                            "required": ["action"]
                        }
                    ),
                    types.Tool(
                        name="auto_update_docs",
                        description="Automatically update documentation based on code changes",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "dry_run": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Preview changes without actually updating files"
                                },
                                "max_updates": {
                                    "type": "integer",
                                    "default": 10,
                                    "minimum": 1,
                                    "description": "Maximum number of updates to process"
                                },
                                "priority_filter": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": ["high", "medium", "low"]
                                    },
                                    "description": "Only process suggestions with these priorities"
                                },
                                "force_scan": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Force full project scan for changes"
                                }
                            }
                        }
                    )])
                tools.extend([
                    types.Tool(
                        name="get_update_suggestions",
                        description="Get suggestions for documentation updates based on code changes",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "force_scan": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Force full project scan instead of git-based change detection"
                                },
                                "priority_filter": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": ["high", "medium", "low"]
                                    },
                                    "description": "Filter suggestions by priority level"
                                }
                            }
                        }
                    ),

                    types.Tool(
                        name="source_code_lookup",
                        description="Look up source code elements and their documentation references",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "element_name": {
                                    "type": "string",
                                    "description": "Name of class, function, or method to search for"
                                },
                                "file_path": {
                                    "type": "string",
                                    "description": "Filter by specific source file path"
                                },
                                "element_type": {
                                    "type": "string",
                                    "enum": ["class", "function", "method", "variable", "module"],
                                    "description": "Type of code element to find"
                                }
                            }
                        }
                    ),
                    types.Tool(
                        name="docs_system_status",
                        description="Get status and statistics of the documentation system",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "include_file_list": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Include list of indexed files"
                                }
                            }
                        }
                    )])

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
                    description="Execute Python code with direct access to ToolBoxV2 app instance use toolbox_info (python_guide) for further usage informations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute. 'app' variable contains the ToolBoxV2 instance in an async ipy like env"
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
                                    "enum": ["modules", "functions", "module_detail", "function_detail", "python_guide", "docs_reader", "flow_guide"]
                                },
                                "target": {
                                    "type": "string",
                                    "description": "Specific module or function name for detailed info"
                                }
                            },
                            "required": ["info_type"]
                        }
                    ),

                ])
                if self.config.docs_system:
                    tools.extend([
types.Tool(
    name="initial_docs_parse",
    description="Parse existing documentation and TOCs, complete the index",
    inputSchema={
        "type": "object",
        "properties": {
            "update_index": {
                "type": "boolean",
                "default": True,
                "description": "Whether to update the documentation index"
            }
        }
    }
),

types.Tool(
    name="auto_adapt_docs_to_index",
    description="Automatically adapt documentation to match current code index",
    inputSchema={
        "type": "object",
        "properties": {
            "create_missing": {
                "type": "boolean",
                "default": True,
                "description": "Create documentation for undocumented code elements"
            },
            "update_existing": {
                "type": "boolean",
                "default": True,
                "description": "Update existing documentation that's outdated"
            }
        }
    }
),

types.Tool(
    name="find_unclear_and_missing",
    description="Find unclear documentation and missing implementations from TOC sections",
    inputSchema={
        "type": "object",
        "properties": {
            "analyze_tocs": {
                "type": "boolean",
                "default": True,
                "description": "Analyze table of contents structure for issues"
            }
        }
    }
),

types.Tool(
    name="rebuild_clean_docs",
    description="Rebuild and clean documentation with options to preserve content",
    inputSchema={
        "type": "object",
        "properties": {
            "keep_unclear": {
                "type": "boolean",
                "default": True,
                "description": "Keep sections marked as unclear"
            },
            "keep_missing": {
                "type": "boolean",
                "default": True,
                "description": "Keep sections with missing implementations"
            },
            "keep_level": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 6,
                "description": "Maximum header level to keep (deeper levels will be removed)"
            },
            "update_mkdocs": {
                "type": "boolean",
                "default": True,
                "description": "Update mkdocs.yml configuration file"
            }
        }
    }
)])

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


                elif name == "initial_docs_parse":
                    result = await self.tb_app.initial_docs_parse(
                        update_index=arguments.get("update_index", True)
                    )
                    return [types.TextContent(type="text", text=json.dumps(
                        result.data if result.is_ok() else {"error": result.error}, indent=2))]

                elif name == "auto_adapt_docs_to_index":
                    result = await self.tb_app.auto_adapt_docs_to_index(
                        create_missing=arguments.get("create_missing", True),
                        update_existing=arguments.get("update_existing", True)
                    )
                    return [types.TextContent(type="text", text=json.dumps(
                        result.data if result.is_ok() else {"error": result.error}, indent=2))]

                elif name == "find_unclear_and_missing":
                    result = await self.tb_app.find_unclear_and_missing(
                        analyze_tocs=arguments.get("analyze_tocs", True)
                    )
                    return [types.TextContent(type="text", text=json.dumps(
                        result.data if result.is_ok() else {"error": result.error}, indent=2))]

                elif name == "rebuild_clean_docs":
                    result = await self.tb_app.rebuild_clean_docs(
                        keep_unclear=arguments.get("keep_unclear", True),
                        keep_missing=arguments.get("keep_missing", True),
                        keep_level=arguments.get("keep_level", 1),
                        update_mkdocs=arguments.get("update_mkdocs", True)
                    )
                    return [types.TextContent(type="text", text=json.dumps(
                        result.data if result.is_ok() else {"error": result.error}, indent=2))]

                elif name == "docs_reader":
                    try:
                        result = await self.tb_app.docs_reader(
                            query=arguments.get("query"),
                            section_id=arguments.get("section_id"),
                            file_path=arguments.get("file_path"),
                            tags=arguments.get("tags"),
                            include_source_refs=arguments.get("include_source_refs", True),
                            format_type=arguments.get("format_type", "structured")
                        )

                        if result.is_ok():
                            if arguments.get("format_type") == "markdown":
                                content = result.data
                            else:
                                content = json.dumps(result.data, indent=2, ensure_ascii=False)
                        else:
                            content = f"Error: {result.error}"

                        return [types.TextContent(type="text", text=content)]

                    except Exception as e:
                        return [types.TextContent(type="text", text=f"Error reading docs: {e}")]

                elif name == "docs_writer":
                    try:
                        result = await self.tb_app.docs_writer(
                            action=arguments["action"],
                            file_path=arguments.get("file_path"),
                            section_title=arguments.get("section_title"),
                            content=arguments.get("content"),
                            source_file=arguments.get("source_file"),
                            auto_generate=arguments.get("auto_generate", False),
                            position=arguments.get("position"),
                            level=arguments.get("level", 2)
                        )

                        if result.is_ok():
                            content = json.dumps(result.data, indent=2, ensure_ascii=False)
                        else:
                            content = f"Error: {result.error}"

                        return [types.TextContent(type="text", text=content)]

                    except Exception as e:
                        return [types.TextContent(type="text", text=f"Error writing docs: {e}")]

                elif name == "get_update_suggestions":
                    try:
                        result = await self.tb_app.get_update_suggestions(
                            force_scan=arguments.get("force_scan", False),
                            priority_filter=arguments.get("priority_filter")
                        )

                        if result.is_ok():
                            # Format suggestions in a readable way
                            data = result.data
                            content = f"""# Documentation Update Suggestions

        ## Summary
        - Total suggestions: {data['total_suggestions']}
        - Force scan used: {data['force_scan_used']}
        - Last indexed: {data.get('index_stats', {}).get('last_indexed', 'Unknown')}

        ## Index Statistics
        - Code elements: {data.get('index_stats', {}).get('code_elements', 0)}
        - Documentation sections: {data.get('index_stats', {}).get('doc_sections', 0)}
        - Import references: {data.get('index_stats', {}).get('import_refs', 0)}

        ## Suggestions
        """
                            for i, suggestion in enumerate(data['suggestions'], 1):
                                content += f"""
        ### {i}. {suggestion['suggestion']}
        - **Priority**: {suggestion['priority']}
        - **Type**: {suggestion['type']}
        - **Action**: {suggestion['action']}
        - **Source file**: {suggestion.get('source_file', 'N/A')}
        - **Doc file**: {suggestion.get('doc_file', 'N/A')}
        - **Change type**: {suggestion.get('change_type', 'N/A')}
        """

                            if data.get('update_notes'):
                                content += "\n## Recent Changes\n"
                                for note in data['update_notes']:
                                    content += f"- {note}\n"

                        else:
                            content = f"Error getting suggestions: {result.error}"

                        return [types.TextContent(type="text", text=content)]

                    except Exception as e:
                        return [types.TextContent(type="text", text=f"Error getting suggestions: {e}")]

                elif name == "auto_update_docs":
                    try:
                        result = await self.tb_app.auto_update_docs(
                            dry_run=arguments.get("dry_run", False),
                            max_updates=arguments.get("max_updates", 10),
                            priority_filter=arguments.get("priority_filter"),
                            force_scan=arguments.get("force_scan", False)
                        )

                        if result.is_ok():
                            data = result.data

                            if data.get("dry_run"):
                                content = f"""# Dry Run Results

        Would update {data['would_update']} documentation sections:

        """
                                for suggestion in data['suggestions']:
                                    content += f"- {suggestion['suggestion']} (Priority: {suggestion['priority']})\n"
                            else:
                                content = f"""# Auto-Update Results

        ## Summary
        - Total suggestions processed: {data['total_suggestions']}
        - Actually processed: {data['processed']}
        - Successful updates: {data['successful_updates']}

        ## Results
        """
                                for i, result_item in enumerate(data['results'], 1):
                                    content += f"""
        ### {i}. {result_item['suggestion']['suggestion']}
        - **Result**: {result_item['result']}
        """
                                    if result_item['result'] == 'error':
                                        content += f"- **Error**: {result_item.get('error', 'Unknown error')}\n"
                                    elif result_item['result'] == 'success':
                                        details = result_item.get('details', {})
                                        content += f"- **Action**: {details.get('action', 'Unknown')}\n"
                        else:
                            content = f"Error in auto-update: {result.error}"

                        return [types.TextContent(type="text", text=content)]

                    except Exception as e:
                        return [types.TextContent(type="text", text=f"Error in auto-update: {e}")]

                elif name == "source_code_lookup":
                    try:
                        result = await self.tb_app.source_code_lookup(
                            element_name=arguments.get("element_name"),
                            file_path=arguments.get("file_path"),
                            element_type=arguments.get("element_type")
                        )

                        if result.is_ok():
                            data = result.data
                            content = f"""# Source Code Lookup Results

        Found {data['total_matches']} matches:

        """
                            for i, match in enumerate(data['matches'], 1):
                                content += f"""
        ## {i}. {match['name']} ({match['type']})
        - **File**: {match['file_path']}:{match['line_start']}-{match['line_end']}
        - **Signature**: `{match['signature']}`
        """
                                if match.get('parent_class'):
                                    content += f"- **Parent Class**: {match['parent_class']}\n"

                                if match.get('docstring'):
                                    content += f"- **Docstring**: {match['docstring'][:100]}...\n"

                                if match.get('related_documentation'):
                                    content += "- **Related Documentation**:\n"
                                    for doc in match['related_documentation']:
                                        content += f"  - [{doc['title']}]({doc['file_path']})\n"
                        else:
                            content = f"Error in lookup: {result.error}"

                        return [types.TextContent(type="text", text=content)]

                    except Exception as e:
                        return [types.TextContent(type="text", text=f"Error in source lookup: {e}")]

                elif name == "docs_system_status":
                    try:
                        if not self.docs_system:
                            return [types.TextContent(type="text", text="Documentation system not initialized")]

                        # Get current index status
                        if self.docs_system.current_index:
                            index = self.docs_system.current_index
                            content = f"""# Documentation System Status

        ## Index Statistics
        - **Version**: {index.version}
        - **Last indexed**: {index.last_indexed.strftime('%Y-%m-%d %H:%M:%S')}
        - **Git commit**: {index.last_git_commit or 'Not tracked'}
        - **Code elements**: {len(index.code_elements)}
        - **Documentation sections**: {len(index.sections)}
        - **Import references**: {len(index.import_refs)}
        - **Tracked files**: {len(index.file_hashes)}

        ## Configuration
        - **Include directories**: {', '.join(self.docs_system.indexer.include_dirs)}
        - **Exclude directories**: {', '.join(self.docs_system.indexer.exclude_dirs)}
        - **Docs root**: {self.docs_system.docs_root}
        - **Project root**: {self.docs_system.project_root}
        """

                            if arguments.get("include_file_list"):
                                content += "\n## Indexed Files\n"
                                for file_path in sorted(index.file_hashes.keys()):
                                    content += f"- {file_path}\n"

                        else:
                            content = "Documentation system initialized but no index loaded yet."

                        return [types.TextContent(type="text", text=content)]

                    except Exception as e:
                        return [types.TextContent(type="text", text=f"Error getting system status: {e}")]

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

            elif info_type == "module_detail":
                # list all functions and their description
                from toolboxv2 import TBEF
                data = [x.lower() for x in TBEF.__dict__.get(target.upper(), Enum).__dict__.get("_member_names_", [])]

                return [types.TextContent(
                    type="text",
                    text=f"Module Details for {target}:\n{data}"
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

            elif info_type == "python_guide":
                python_guide = """
# Python IPy Session Guide: Using the App and Toolbox

# --- Basic Usage ---

# 1. Get the application instance
# You can get a default or a named application instance.
from toolboxv2 import get_app
app = get_app()  # Get the default app
named_app = get_app("my_specific_app")  # Get a named app

# 2. Get a module instance from the app
# This allows you to interact with a specific module's functionality.
mod = app.get_mod("mod_name")

# 3. Run a function from a module
# A direct way to execute a module's function without getting the module instance first.
app.run_any(("mod_name", "function_name"), args_, kwargs_)


# --- Advanced Workflow Example: Working with the 'isaa' Module ---

# This example demonstrates a more complex workflow, including getting a module,
# using a builder pattern, configuring, registering, and retrieving an agent.

import os
from toolboxv2 import init_cwd
from toolboxv2.mods.isaa.module import Tools as Isaa
from toolboxv2.mods.isaa.extras.terminal_progress import ProgressiveTreePrinter, VerbosityMode

# 1. Initialize the app and get the 'isaa' module
app = get_app("isaa_test_app")
isaa: Isaa = app.get_mod("isaa")

# 2. Use a builder to create and configure an agent
agent_builder = isaa.get_agent_builder(name="mcp-agent")
config_path = os.path.join(init_cwd, "mcp.json")
agent_builder.load_mcp_tools_from_config(config_path)

# 3. Register the agent with the module
# This makes the agent available for later use.
# Note: This is an async function, so you would typically 'await' it in an async context.
# await isaa.register_agent(agent_builder)

# 4. Retrieve the registered agent
agent_name = agent_builder.config.name
# agent = await isaa.get_agent(agent_name)

# 5. Interact with the agent
# For example, setting up a progress callback for terminal output.
printer = ProgressiveTreePrinter(mode=VerbosityMode.MINIMAL)
# agent.progress_callback = printer.progress_callback
"""
                return [types.TextContent(type="text", text=python_guide)]
            elif info_type == "docs_reader":
                try:
                    if not self.tb_app or not hasattr(self.tb_app, 'docs_reader'):
                        return [types.TextContent(type="text", text="Documentation system not available")]

                    if target == "list":
                        # List all available documentation sections
                        result = await self.tb_app.docs_reader(format_type="json")
                        if result.is_ok():
                            sections = result.data if isinstance(result.data, list) else result.data.get('sections', [])
                            section_list = []
                            for section in sections:
                                if isinstance(section, dict):
                                    section_list.append(
                                        f"- {section.get('title', 'Unknown')} ({section.get('file_path', 'Unknown')})")
                                else:
                                    section_list.append(f"- {section}")

                            return [types.TextContent(type="text",
                                                      text=f"# Available Documentation Sections\n\n" + "\n".join(
                                                          section_list))]
                        else:
                            return [types.TextContent(type="text", text="Error listing documentation")]
                    else:
                        # Search for specific documentation
                        result = await self.tb_app.docs_reader(query=target, format_type="markdown")
                        if result.is_ok():
                            return [types.TextContent(type="text", text=result.data)]
                        else:
                            return [types.TextContent(type="text", text=f"No documentation found for: {target}")]
                except Exception as e:
                    return [types.TextContent(type="text", text=f"Error accessing documentation: {e}")]

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

