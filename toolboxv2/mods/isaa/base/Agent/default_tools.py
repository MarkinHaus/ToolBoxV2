"""
Default Tools - Always Available Tools for FlowAgent

These tools are ALWAYS available to the agent, even in instant response mode.
They provide:
1. VFS Operations - File management in virtual file system
2. Context Tools - Get context, list tools, request more tools
3. Meta Tools - Agent self-awareness and capability discovery

Author: FlowAgent V2
"""

from typing import Any, Callable, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
    from toolboxv2.mods.isaa.base.Agent.agent_session import AgentSession


# =============================================================================
# DEFAULT TOOLS REGISTRY
# =============================================================================

class DefaultToolCategory(str, Enum):
    """Categories for default tools"""
    VFS = "vfs"           # Virtual File System operations
    CONTEXT = "context"   # Context and memory access
    META = "meta"         # Agent self-awareness
    CONTROL = "control"   # Flow control


@dataclass
class DefaultToolDef:
    """Definition of a default tool"""
    name: str
    description: str
    category: DefaultToolCategory
    parameters: dict  # JSON Schema format
    handler: str  # Method name on DefaultToolsHandler


# =============================================================================
# DEFAULT TOOLS DEFINITIONS
# =============================================================================

DEFAULT_TOOLS_DEFS: list[DefaultToolDef] = [
    # === VFS Tools ===
    DefaultToolDef(
        name="vfs_list",
        description="List all files in the virtual file system with their state (open/closed) and summaries.",
        category=DefaultToolCategory.VFS,
        parameters={"type": "object", "properties": {}, "required": []},
        handler="handle_vfs_list"
    ),
    DefaultToolDef(
        name="vfs_open",
        description="Open a file to see its content. Use line_start/line_end for large files.",
        category=DefaultToolCategory.VFS,
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "File to open"},
                "line_start": {"type": "integer", "description": "Start line (1-indexed)"},
                "line_end": {"type": "integer", "description": "End line (-1 for all)"}
            },
            "required": ["filename"]
        },
        handler="handle_vfs_open"
    ),
    DefaultToolDef(
        name="vfs_close",
        description="Close a file. Creates a summary for later reference.",
        category=DefaultToolCategory.VFS,
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "File to close"}
            },
            "required": ["filename"]
        },
        handler="handle_vfs_close"
    ),
    DefaultToolDef(
        name="vfs_read",
        description="Read file content without opening it (for quick lookups).",
        category=DefaultToolCategory.VFS,
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "File to read"},
                "line_start": {"type": "integer", "description": "Start line"},
                "line_end": {"type": "integer", "description": "End line"}
            },
            "required": ["filename"]
        },
        handler="handle_vfs_read"
    ),
    DefaultToolDef(
        name="vfs_write",
        description="Write/overwrite a file with new content.",
        category=DefaultToolCategory.VFS,
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "File to write"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["filename", "content"]
        },
        handler="handle_vfs_write"
    ),
    DefaultToolDef(
        name="vfs_create",
        description="Create a new file in the virtual file system.",
        category=DefaultToolCategory.VFS,
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "New file name"},
                "content": {"type": "string", "description": "Initial content"}
            },
            "required": ["filename"]
        },
        handler="handle_vfs_create"
    ),

    # === Context Tools ===
    DefaultToolDef(
        name="get_context",
        description="Get relevant context from memory for the current query. Use this when you need more information.",
        category=DefaultToolCategory.CONTEXT,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "max_results": {"type": "integer", "description": "Max results (default 5)"}
            },
            "required": ["query"]
        },
        handler="handle_get_context"
    ),
    DefaultToolDef(
        name="remember",
        description="Store important information for later. Creates a VFS file and adds to memory.",
        category=DefaultToolCategory.CONTEXT,
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Identifier for this memory"},
                "content": {"type": "string", "description": "What to remember"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"}
            },
            "required": ["key", "content"]
        },
        handler="handle_remember"
    ),

    # === Meta Tools ===
    DefaultToolDef(
        name="list_tools",
        description="List all available tools with their descriptions. Use this to discover what you can do.",
        category=DefaultToolCategory.META,
        parameters={
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Filter by category (optional)"},
                "search": {"type": "string", "description": "Search in tool names/descriptions"}
            },
            "required": []
        },
        handler="handle_list_tools"
    ),
    DefaultToolDef(
        name="request_tool",
        description="Request a specific tool to be loaded for this session.",
        category=DefaultToolCategory.META,
        parameters={
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Name of the tool to request"},
                "reason": {"type": "string", "description": "Why you need this tool"}
            },
            "required": ["tool_name"]
        },
        handler="handle_request_tool"
    ),
    DefaultToolDef(
        name="get_capabilities",
        description="Get a summary of your current capabilities, loaded tools, and session state.",
        category=DefaultToolCategory.META,
        parameters={"type": "object", "properties": {}, "required": []},
        handler="handle_get_capabilities"
    ),

    # === Control Tools ===
    DefaultToolDef(
        name="final_answer",
        description="Provide the final answer to the user. Use this when you have completed the task.",
        category=DefaultToolCategory.CONTROL,
        parameters={
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Your final answer"}
            },
            "required": ["answer"]
        },
        handler="handle_final_answer"
    ),
    DefaultToolDef(
        name="need_info",
        description="Indicate that you need more information and cannot proceed.",
        category=DefaultToolCategory.CONTROL,
        parameters={
            "type": "object",
            "properties": {
                "missing": {"type": "string", "description": "What information is missing"}
            },
            "required": ["missing"]
        },
        handler="handle_need_info"
    ),
    DefaultToolDef(
        name="need_human",
        description="Request human assistance when stuck or need confirmation.",
        category=DefaultToolCategory.CONTROL,
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question for the human"}
            },
            "required": ["question"]
        },
        handler="handle_need_human"
    ),
    DefaultToolDef(
        name="think",
        description="Record your thought process. Use this to show your reasoning.",
        category=DefaultToolCategory.CONTROL,
        parameters={
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Your thought/reasoning"}
            },
            "required": ["thought"]
        },
        handler="handle_think"
    ),
]


# =============================================================================
# DEFAULT TOOLS HANDLER
# =============================================================================

class DefaultToolsHandler:
    """
    Handler for all default tools.
    Provides the actual implementation of each tool.
    """

    def __init__(self, agent: 'FlowAgent', session: 'AgentSession'):
        self.agent = agent
        self.session = session
        self._thoughts: list[str] = []

    # === VFS Handlers ===

    def handle_vfs_list(self) -> dict:
        """List all VFS files"""
        return self.session.vfs.list_files()

    def handle_vfs_open(self, filename: str, line_start: int = 1, line_end: int = -1) -> dict:
        """Open a VFS file"""
        return self.session.vfs.open(filename, line_start, line_end)

    async def handle_vfs_close(self, filename: str) -> dict:
        """Close a VFS file"""
        return await self.session.vfs.close(filename)

    def handle_vfs_read(self, filename: str, line_start: int = None, line_end: int = None) -> dict:
        """Read VFS file content"""
        result = self.session.vfs.read(filename)
        if not result.get('success'):
            return result

        content = result['content']
        if line_start or line_end:
            lines = content.split('\n')
            start = (line_start or 1) - 1
            end = line_end or len(lines)
            content = '\n'.join(lines[start:end])

        return {"success": True, "content": content}

    def handle_vfs_write(self, filename: str, content: str) -> dict:
        """Write to VFS file"""
        return self.session.vfs.write(filename, content)

    def handle_vfs_create(self, filename: str, content: str = "") -> dict:
        """Create new VFS file"""
        return self.session.vfs.create(filename, content)

    # === Context Handlers ===

    async def handle_get_context(self, query: str, max_results: int = 5) -> dict:
        """Get context from memory"""
        try:
            context = await self.session.get_reference(query, max_entries=max_results)
            return {"success": True, "context": context}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def handle_remember(self, key: str, content: str, tags: list[str] = None) -> dict:
        """Store something in memory"""
        # Store in VFS
        filename = f"memory_{key}"
        self.session.vfs.create(filename, content)

        # Add to chat history for RAG
        await self.session.add_message({
            "role": "system",
            "content": f"[Memory: {key}] {content}"
        }, tags=tags or [])

        return {"success": True, "stored_as": filename}

    # === Meta Handlers ===

    def handle_list_tools(self, category: str = None, search: str = None) -> dict:
        """List available tools"""
        tools = self.agent.tool_manager.get_all()

        if category:
            tools = [t for t in tools if t.has_category(category)]

        if search:
            search_lower = search.lower()
            tools = [t for t in tools if search_lower in t.name.lower() or search_lower in t.description.lower()]

        # Include default tools
        default_names = [d.name for d in DEFAULT_TOOLS_DEFS]

        return {
            "success": True,
            "default_tools": default_names,
            "registered_tools": [
                {"name": t.name, "description": t.description[:100], "categories": t.category}
                for t in tools
            ],
            "total": len(tools) + len(default_names)
        }

    def handle_request_tool(self, tool_name: str, reason: str = "") -> dict:
        """Request a tool to be loaded"""
        # Check if tool exists
        tool = self.agent.tool_manager.get(tool_name)

        if tool:
            # Tool exists, check if restricted
            if not self.session.is_tool_allowed(tool_name):
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' is restricted in this session"
                }
            return {
                "success": True,
                "tool": tool_name,
                "description": tool.description
            }

        # Tool doesn't exist - check categories
        categories = self.agent.tool_manager.list_categories()

        return {
            "success": False,
            "error": f"Tool '{tool_name}' not found",
            "available_categories": categories,
            "hint": "Use list_tools to see available tools"
        }

    def handle_get_capabilities(self) -> dict:
        """Get current capabilities"""
        stats = self.session.get_stats()
        tool_stats = self.agent.tool_manager.get_stats()

        return {
            "success": True,
            "session": {
                "id": stats['session_id'],
                "vfs_files": stats['vfs_files'],
                "history_length": stats['history_length']
            },
            "tools": {
                "default": len(DEFAULT_TOOLS_DEFS),
                "registered": tool_stats['total_tools'],
                "by_source": tool_stats['by_source']
            },
            "restrictions": self.session.get_restrictions(),
            "current_situation": stats['current_situation']
        }

    # === Control Handlers ===

    def handle_final_answer(self, answer: str) -> dict:
        """Handle final answer"""
        return {
            "type": "final_answer",
            "answer": answer,
            "success": True
        }

    def handle_need_info(self, missing: str) -> dict:
        """Handle need info"""
        return {
            "type": "need_info",
            "missing": missing,
            "success": True
        }

    def handle_need_human(self, question: str) -> dict:
        """Handle need human"""
        return {
            "type": "need_human",
            "question": question,
            "success": True
        }

    def handle_think(self, thought: str) -> dict:
        """Record a thought"""
        self._thoughts.append(thought)
        return {
            "type": "think",
            "thought": thought,
            "thought_count": len(self._thoughts),
            "success": True
        }

    # === Execution ===

    async def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a default tool"""
        # Find the tool definition
        tool_def = next((t for t in DEFAULT_TOOLS_DEFS if t.name == tool_name), None)

        if not tool_def:
            raise ValueError(f"Default tool not found: {tool_name}")

        # Get the handler method
        handler = getattr(self, tool_def.handler, None)

        if not handler:
            raise ValueError(f"Handler not found: {tool_def.handler}")

        # Execute
        import asyncio
        if asyncio.iscoroutinefunction(handler):
            return await handler(**kwargs)
        else:
            return handler(**kwargs)


# =============================================================================
# LITELLM FORMAT CONVERSION
# =============================================================================

def get_default_tools_litellm() -> list[dict]:
    """Get default tools in LiteLLM format"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        }
        for tool in DEFAULT_TOOLS_DEFS
    ]


def get_default_tool_names() -> list[str]:
    """Get list of default tool names"""
    return [tool.name for tool in DEFAULT_TOOLS_DEFS]


def is_default_tool(name: str) -> bool:
    """Check if a tool name is a default tool"""
    return name in get_default_tool_names()


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def create_default_tools_handler(agent: 'FlowAgent', session: 'AgentSession') -> DefaultToolsHandler:
    """Create a default tools handler for a session"""
    return DefaultToolsHandler(agent, session)
