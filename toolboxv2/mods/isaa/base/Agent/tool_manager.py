"""
ToolManager - Unified Tool Registry for FlowAgent

Provides:
- Single registry for all tools (local, MCP, A2A)
- Category-based organization with flags
- Native LiteLLM format support
- RuleSet integration for automatic tool grouping

Author: FlowAgent V2
"""

import asyncio
import inspect
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable
from functools import wraps


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ToolEntry:
    """
    Unified tool entry supporting local functions, MCP tools, and A2A tools.
    """
    name: str
    description: str
    args_schema: str                          # "(arg1: str, arg2: int = 0)"

    # Categorization
    category: list[str] = field(default_factory=list)  # ['local', 'discord', 'mcp_filesystem']
    flags: dict[str, bool] = field(default_factory=dict)  # read, write, dangerous, etc.
    source: str = "local"                     # 'local', 'mcp', 'a2a'

    # Function reference (None when serialized/restored)
    function: Callable | None = None

    # Cached LiteLLM schema
    litellm_schema: dict | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    call_count: int = 0
    last_called: datetime | None = None

    # MCP/A2A specific
    server_name: str | None = None            # For MCP/A2A: which server
    original_name: str | None = None          # Original tool name before prefixing

    def __post_init__(self):
        """Ensure defaults and build schema"""
        if self.flags is None:
            self.flags = {}
        if self.category is None:
            self.category = []
        if self.metadata is None:
            self.metadata = {}

        # Set default flags based on name/description heuristics
        self._infer_flags()

    def _infer_flags(self):
        """Infer flags from tool name and description"""
        name_lower = self.name.lower()
        desc_lower = self.description.lower()

        # Read flag
        if 'read' not in self.flags:
            read_keywords = ['get', 'list', 'fetch', 'query', 'search', 'find', 'show', 'view']
            self.flags['read'] = any(kw in name_lower or kw in desc_lower for kw in read_keywords)

        # Write flag
        if 'write' not in self.flags:
            write_keywords = ['create', 'update', 'set', 'add', 'insert', 'modify', 'change']
            self.flags['write'] = any(kw in name_lower or kw in desc_lower for kw in write_keywords)

        # Save/Permanent write flag
        if 'save_write' not in self.flags:
            save_keywords = ['save', 'store', 'persist', 'permanent', 'commit']
            self.flags['save_write'] = any(kw in name_lower or kw in desc_lower for kw in save_keywords)

        # Dangerous flag
        if 'dangerous' not in self.flags:
            danger_keywords = ['delete', 'remove', 'drop', 'destroy', 'purge', 'clear', 'reset']
            self.flags['dangerous'] = any(kw in name_lower or kw in desc_lower for kw in danger_keywords)

        # Requires confirmation
        if 'requires_confirmation' not in self.flags:
            self.flags['requires_confirmation'] = self.flags.get('dangerous', False)

    def record_call(self):
        """Record that this tool was called"""
        self.call_count += 1
        self.last_called = datetime.now()

    def has_flag(self, flag_name: str) -> bool:
        """Check if tool has a specific flag enabled"""
        return self.flags.get(flag_name, False)

    def has_category(self, category: str) -> bool:
        """Check if tool belongs to category"""
        return category in self.category

    def matches_categories(self, categories: list[str]) -> bool:
        """Check if tool matches any of the given categories"""
        return bool(set(self.category) & set(categories))

    def matches_flags(self, **flags) -> bool:
        """Check if tool matches all given flag conditions"""
        for flag_name, required_value in flags.items():
            if self.flags.get(flag_name, False) != required_value:
                return False
        return True


# =============================================================================
# TOOL MANAGER
# =============================================================================

class ToolManager:
    """
    Unified tool registry managing local, MCP, and A2A tools.

    Features:
    - Single registry for all tool types
    - Category and flag-based filtering
    - Native LiteLLM format support
    - Automatic RuleSet integration
    """

    def __init__(self, rule_set: 'RuleSet | None' = None):
        """
        Initialize ToolManager.

        Args:
            rule_set: Optional RuleSet for automatic tool group registration
        """
        # Main registry
        self._registry: dict[str, ToolEntry] = {}

        # Indexes for fast lookups
        self._category_index: dict[str, set[str]] = {}  # category -> tool names
        self._flags_index: dict[str, set[str]] = {}     # flag -> tool names with flag=True
        self._source_index: dict[str, set[str]] = {}    # source -> tool names

        # RuleSet integration
        self._rule_set = rule_set

        # Statistics
        self._total_calls = 0
        # Pending checkpoint data for tools not yet registered by code
        self._pending_checkpoint_data: dict[str, dict[str, Any]] = {}

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(
        self,
        func: Callable | None,
        name: str | None = None,
        description: str | None = None,
        category: list[str] | str | None = None,
        flags: dict[str, bool] | None = None,
        source: str = "local",
        server_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        args_schema: str | None = None
    ) -> ToolEntry:
        """
        Register a tool in the registry.

        Args:
            func: The callable function (can be None for MCP/A2A stubs)
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            category: Category or list of categories
            flags: Dict of flags (read, write, dangerous, etc.)
            source: Tool source ('local', 'mcp', 'a2a')
            server_name: For MCP/A2A: the server name
            metadata: Additional metadata
            args_schema: Override args schema string

        Returns:
            Created ToolEntry
        """
        # Determine name
        tool_name = name
        if tool_name is None and func is not None:
            tool_name = func.__name__
        if tool_name is None:
            raise ValueError("Tool name required when func is None")

        # Determine description
        tool_description = description
        if tool_description is None and func is not None:
            tool_description = func.__doc__ or f"Tool: {tool_name}"
        if tool_description is None:
            tool_description = f"Tool: {tool_name}"

        # Ensure description is clean
        tool_description = tool_description.strip()[:2000]

        # Determine args schema
        if args_schema is None and func is not None:
            args_schema = self._get_args_schema(func)
        elif args_schema is None:
            args_schema = "()"

        # Normalize category
        if category is None:
            category = [source]
        elif isinstance(category, str):
            category = [category]

        # Ensure source is in category
        if source not in category:
            category.append(source)

        # Wrap sync functions as async
        effective_func = func
        if func is not None and not asyncio.iscoroutinefunction(func):
            # NEU: Check ob Tool GUI-blocking ist und Main Thread braucht
            no_thread = (flags or {}).get("no_thread", False)

            if no_thread:
                @wraps(func)
                async def async_wrapper_direct(*args, **kwargs):
                    """Direkt auf Event-Loop Thread - für GUI/Win32 Calls"""
                    return func(*args, **kwargs)

                effective_func = async_wrapper_direct
            else:
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await asyncio.to_thread(func, *args, **kwargs)

                effective_func = async_wrapper

        # Create entry
        entry = ToolEntry(
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
            category=category,
            flags=flags or {},
            source=source,
            function=effective_func,
            server_name=server_name,
            original_name=name if server_name else None,
            metadata=metadata or {}
        )

        if tool_name in self._pending_checkpoint_data:
            pending_data = self._pending_checkpoint_data.pop(tool_name)

            # Übernehme Laufzeit-Statistiken aus Checkpoint
            entry.call_count = pending_data.get('call_count', 0)

            # Übernehme Timestamps
            if pending_data.get('last_called'):
                entry.last_called = datetime.fromisoformat(pending_data['last_called'])
            if pending_data.get('created_at'):
                entry.created_at = datetime.fromisoformat(pending_data['created_at'])

            # Merge metadata (pending ergänzt das aktuelle metadata)
            pending_metadata = pending_data.get('metadata', {})
            if pending_metadata:
                # Aktuelles metadata hat Priorität, pending füllt Lücken
                merged_metadata = {**pending_metadata, **entry.metadata}
                entry.metadata = merged_metadata

        # Build LiteLLM schema
        entry.litellm_schema = self._build_litellm_schema(entry)

        # Store in registry
        self._registry[tool_name] = entry
        # Update indexes
        self._update_indexes(entry)

        # Sync to RuleSet if available
        if self._rule_set:
            self._sync_tool_to_ruleset(entry)

        return entry


    def un_register(self, name: str) -> bool:
        """Remove a tool from the registry"""
        if name not in self._registry:
            return False

        del self._registry[name]
        return True

    def register_mcp_tools(
        self,
        server_name: str,
        tools: list[dict[str, Any]],
        category_prefix: str = "mcp"
    ):
        """
        Register multiple MCP tools from a server.

        Args:
            server_name: Name of the MCP server
            tools: List of tool configs from MCP server
                   Each should have: name, description, inputSchema
            category_prefix: Prefix for category (default: "mcp")
        """
        for tool_config in tools:
            original_name = tool_config.get('name', 'unknown')
            prefixed_name = f"{server_name}_{original_name}"

            # Extract args schema from inputSchema
            input_schema = tool_config.get('inputSchema', {})
            args_schema = self._schema_to_args_string(input_schema)

            self.register(
                func=None,  # MCP tools don't have local functions
                name=prefixed_name,
                description=tool_config.get('description', f"MCP tool: {original_name}"),
                category=[f"{category_prefix}_{server_name}", category_prefix, server_name],
                source="mcp",
                server_name=server_name,
                args_schema=args_schema,
                metadata={
                    'input_schema': input_schema,
                    'original_config': tool_config
                }
            )

    def register_a2a_tools(
        self,
        server_name: str,
        tools: list[dict[str, Any]],
        category_prefix: str = "a2a"
    ):
        """
        Register multiple A2A tools from a server.

        Args:
            server_name: Name of the A2A server
            tools: List of tool configs from A2A server
            category_prefix: Prefix for category (default: "a2a")
        """
        for tool_config in tools:
            original_name = tool_config.get('name', 'unknown')
            prefixed_name = f"{server_name}_{original_name}"

            self.register(
                func=None,  # A2A tools don't have local functions
                name=prefixed_name,
                description=tool_config.get('description', f"A2A tool: {original_name}"),
                category=[f"{category_prefix}_{server_name}", category_prefix, server_name],
                source="a2a",
                server_name=server_name,
                metadata={
                    'original_config': tool_config
                }
            )

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry"""
        if name not in self._registry:
            return False

        entry = self._registry[name]

        # Remove from indexes
        for cat in entry.category:
            if cat in self._category_index:
                self._category_index[cat].discard(name)

        for flag_name, flag_value in entry.flags.items():
            if flag_value and flag_name in self._flags_index:
                self._flags_index[flag_name].discard(name)

        if entry.source in self._source_index:
            self._source_index[entry.source].discard(name)

        # Remove from registry
        del self._registry[name]

        return True

    def update(self, name: str, **updates) -> bool:
        """Update a tool's attributes"""
        if name not in self._registry:
            return False

        entry = self._registry[name]

        # Store old values for index update
        old_categories = entry.category.copy()
        old_flags = entry.flags.copy()

        # Apply updates
        for key, value in updates.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        # Rebuild LiteLLM schema if description or args changed
        if 'description' in updates or 'args_schema' in updates:
            entry.litellm_schema = self._build_litellm_schema(entry)

        # Update indexes if category or flags changed
        if 'category' in updates or 'flags' in updates:
            # Remove from old indexes
            for cat in old_categories:
                if cat in self._category_index:
                    self._category_index[cat].discard(name)
            for flag_name, flag_value in old_flags.items():
                if flag_value and flag_name in self._flags_index:
                    self._flags_index[flag_name].discard(name)

            # Add to new indexes
            self._update_indexes(entry)

        return True

    def _update_indexes(self, entry: ToolEntry):
        """Update indexes for a tool entry"""
        # Category index
        for cat in entry.category:
            if cat not in self._category_index:
                self._category_index[cat] = set()
            self._category_index[cat].add(entry.name)

        # Flags index
        for flag_name, flag_value in entry.flags.items():
            if flag_value:
                if flag_name not in self._flags_index:
                    self._flags_index[flag_name] = set()
                self._flags_index[flag_name].add(entry.name)

        # Source index
        if entry.source not in self._source_index:
            self._source_index[entry.source] = set()
        self._source_index[entry.source].add(entry.name)

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get(self, name: str) -> ToolEntry | None:
        """Get tool entry by name"""
        return self._registry.get(name)

    def get_function(self, name: str) -> Callable | None:
        """Get tool function by name"""
        entry = self._registry.get(name)
        return entry.function if entry else None

    def get_by_category(self, *categories: str) -> list[ToolEntry]:
        """
        Get tools matching any of the given categories.

        Args:
            *categories: Category names to match

        Returns:
            List of matching ToolEntries
        """
        matching_names: set[str] = set()

        for cat in categories:
            if cat in self._category_index:
                matching_names.update(self._category_index[cat])

        return [self._registry[name] for name in matching_names if name in self._registry]

    def get_by_flags(self, **flags: bool) -> list[ToolEntry]:
        """
        Get tools matching all given flag conditions.

        Args:
            **flags: Flag conditions (e.g., read=True, dangerous=False)

        Returns:
            List of matching ToolEntries
        """
        # Start with all tools
        candidates = set(self._registry.keys())

        for flag_name, required_value in flags.items():
            if required_value:
                # Must have flag = True
                if flag_name in self._flags_index:
                    candidates &= self._flags_index[flag_name]
                else:
                    candidates = set()  # No tools have this flag
            else:
                # Must NOT have flag = True
                if flag_name in self._flags_index:
                    candidates -= self._flags_index[flag_name]

        return [self._registry[name] for name in candidates]

    def get_by_source(self, source: str) -> list[ToolEntry]:
        """Get tools by source (local, mcp, a2a)"""
        if source in self._source_index:
            return [self._registry[name] for name in self._source_index[source] if name in self._registry]
        return []

    def get_all(self) -> list[ToolEntry]:
        """Get all registered tools"""
        return list(self._registry.values())

    def list_names(self) -> list[str]:
        """Get list of all tool names"""
        return list(self._registry.keys())

    def list_categories(self) -> list[str]:
        """Get list of all categories"""
        return list(self._category_index.keys())

    def exists(self, name: str) -> bool:
        """Check if tool exists"""
        return name in self._registry

    def count(self) -> int:
        """Get total number of registered tools"""
        return len(self._registry)

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics"""
        return {
            'total_tools': len(self._registry),
            'by_source': {
                source: len(names)
                for source, names in self._source_index.items()
            },
            'categories': list(self._category_index.keys()),
            'total_calls': self._total_calls
        }

    # =========================================================================
    # LITELLM FORMAT
    # =========================================================================

    def get_litellm_schema(self, name: str) -> dict | None:
        """Get cached LiteLLM schema for a tool"""
        entry = self._registry.get(name)
        if entry:
            return entry.litellm_schema
        return None

    def get_all_litellm(
        self,
        filter_categories: list[str] | None = None,
        filter_flags: dict[str, bool] | None = None,
        exclude_categories: list[str] | None = None,
        max_tools: int | None = None
    ) -> list[dict]:
        """
        Get all tools in LiteLLM format with optional filtering.

        Args:
            filter_categories: Only include tools with these categories
            filter_flags: Only include tools matching these flag conditions
            exclude_categories: Exclude tools with these categories
            max_tools: Maximum number of tools to return

        Returns:
            List of tool schemas in LiteLLM format
        """
        candidates = self.get_all()

        # Filter by categories
        if filter_categories:
            candidates = [e for e in candidates if e.matches_categories(filter_categories)]

        # Exclude categories
        if exclude_categories:
            candidates = [e for e in candidates if not e.matches_categories(exclude_categories)]

        # Filter by flags
        if filter_flags:
            candidates = [e for e in candidates if e.matches_flags(**filter_flags)]

        # Apply limit
        if max_tools and len(candidates) > max_tools:
            candidates = candidates[:max_tools]

        # Return LiteLLM schemas
        return [e.litellm_schema for e in candidates if e.litellm_schema]

    def _build_litellm_schema(self, entry: ToolEntry) -> dict:
        """
        Build LiteLLM/OpenAI function calling schema for a tool.
        """
        # Parse args schema to properties
        properties, required = self._parse_args_schema(entry.args_schema)

        return {
            "type": "function",
            "function": {
                "name": entry.name,
                "description": entry.description[:1024],  # OpenAI limit
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def _get_args_schema(self, func: Callable) -> str:
        """Generate args schema string from function signature"""
        try:
            sig = inspect.signature(func)
            parts = []

            for name, param in sig.parameters.items():
                if name in ('self', 'cls', 'args', 'kwargs'):
                    continue

                # Get type annotation
                ann = ""
                if param.annotation != inspect.Parameter.empty:
                    ann = f": {self._annotation_to_str(param.annotation)}"

                # Get default value
                default = ""
                if param.default != inspect.Parameter.empty:
                    default = f" = {repr(param.default)}"

                # Handle *args and **kwargs
                prefix = ""
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    prefix = "*"
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    prefix = "**"

                parts.append(f"{prefix}{name}{ann}{default}")

            return f"({', '.join(parts)})"
        except Exception:
            return "()"

    def _annotation_to_str(self, annotation) -> str:
        """Convert type annotation to string"""
        import typing

        if isinstance(annotation, str):
            return annotation

        # Handle Optional, Union
        if getattr(annotation, "__origin__", None) is typing.Union:
            args = annotation.__args__
            if len(args) == 2 and type(None) in args:
                non_none = args[0] if args[1] is type(None) else args[1]
                return f"Optional[{self._annotation_to_str(non_none)}]"
            return " | ".join(self._annotation_to_str(a) for a in args)

        # Handle generics
        if hasattr(annotation, "__origin__"):
            origin = getattr(annotation.__origin__, "__name__", str(annotation.__origin__))
            args = getattr(annotation, "__args__", None)
            if args:
                return f"{origin}[{', '.join(self._annotation_to_str(a) for a in args)}]"
            return origin

        # Handle normal types
        if hasattr(annotation, "__name__"):
            return annotation.__name__

        return str(annotation)

    def _parse_args_schema(self, args_schema: str) -> tuple[dict, list]:
        """
        Parse args schema string to LiteLLM properties format.

        Args:
            args_schema: String like "(arg1: str, arg2: int = 0)"

        Returns:
            Tuple of (properties dict, required list)
        """
        properties = {}
        required = []

        if not args_schema or args_schema == "()":
            return properties, required

        # Remove parentheses
        inner = args_schema.strip("()")
        if not inner:
            return properties, required

        # Split by comma (handling nested brackets)
        parts = self._split_args(inner)

        for part in parts:
            part = part.strip()
            if not part or part.startswith('*'):
                continue

            # Parse "name: type = default" format
            has_default = "=" in part

            if ":" in part:
                name_part = part.split(":")[0].strip()
                type_part = part.split(":")[1].strip()

                if "=" in type_part:
                    type_part = type_part.split("=")[0].strip()

                # Map Python types to JSON Schema types
                json_type = self._python_type_to_json(type_part)

                properties[name_part] = {
                    "type": json_type,
                    "description": f"Parameter: {name_part}"
                }

                if not has_default:
                    required.append(name_part)
            else:
                # No type annotation
                name_part = part.split("=")[0].strip() if "=" in part else part.strip()

                properties[name_part] = {
                    "type": "string",
                    "description": f"Parameter: {name_part}"
                }

                if not has_default:
                    required.append(name_part)

        return properties, required

    def _split_args(self, args_str: str) -> list[str]:
        """Split args string by comma, handling nested brackets"""
        parts = []
        current = ""
        bracket_count = 0

        for char in args_str:
            if char in "([{":
                bracket_count += 1
            elif char in ")]}":
                bracket_count -= 1
            elif char == "," and bracket_count == 0:
                parts.append(current)
                current = ""
                continue

            current += char

        if current:
            parts.append(current)

        return parts

    def _python_type_to_json(self, type_str: str) -> str:
        """Map Python type string to JSON Schema type"""
        type_map = {
            "str": "string",
            "string": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "array": "array",
            "dict": "object",
            "object": "object",
            "any": "string",
        }

        type_lower = type_str.lower().split("[")[0]  # Remove generic part
        return type_map.get(type_lower, "string")

    def _schema_to_args_string(self, input_schema: dict) -> str:
        """Convert JSON Schema to args string"""
        if not input_schema:
            return "()"

        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))

        parts = []
        for name, prop in properties.items():
            prop_type = prop.get("type", "string")

            # Map JSON type to Python
            type_map = {
                "string": "str",
                "integer": "int",
                "number": "float",
                "boolean": "bool",
                "array": "list",
                "object": "dict"
            }

            python_type = type_map.get(prop_type, "Any")

            if name in required:
                parts.append(f"{name}: {python_type}")
            else:
                parts.append(f"{name}: {python_type} = None")

        return f"({', '.join(parts)})"

    # =========================================================================
    # EXECUTION
    # =========================================================================

    async def execute(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
            RuntimeError: If tool has no function (MCP/A2A)
        """
        entry = self._registry.get(name)

        if entry is None:
            raise ValueError(f"Tool not found: {name}")

        if entry.function is None:
            raise RuntimeError(
                f"Tool '{name}' has no local function. "
                f"It's a {entry.source} tool from server '{entry.server_name}'. "
                f"Use the appropriate {entry.source} client to execute it."
            )

        # Record call
        entry.record_call()
        self._total_calls += 1

        # Execute
        if asyncio.iscoroutinefunction(entry.function):
            result = await entry.function(**kwargs)
        else:
            result = entry.function(**kwargs)

        # Handle coroutine result
        if asyncio.iscoroutine(result):
            result = await result

        return result

    # =========================================================================
    # RULESET INTEGRATION
    # =========================================================================

    def set_ruleset(self, rule_set: 'RuleSet'):
        """Set RuleSet for automatic tool group registration"""
        self._rule_set = rule_set
        # Sync all existing tools
        self._sync_all_to_ruleset()

    def _sync_all_to_ruleset(self):
        """Sync all tools to RuleSet as tool groups"""
        if not self._rule_set:
            return

        # Group tools by category
        category_tools: dict[str, list[str]] = {}

        for entry in self._registry.values():
            for cat in entry.category:
                if cat not in category_tools:
                    category_tools[cat] = []
                category_tools[cat].append(entry.name)

        # Register as tool groups
        self._rule_set.register_tool_groups_from_categories(category_tools)

    def _sync_tool_to_ruleset(self, entry: ToolEntry):
        """Sync a single tool to RuleSet"""
        if not self._rule_set:
            return

        # Update tool groups for this tool's categories
        for cat in entry.category:
            group_name = f"{cat}_tools"

            if group_name in self._rule_set.tool_groups:
                # Add to existing group
                if entry.name not in self._rule_set.tool_groups[group_name].tool_names:
                    self._rule_set.tool_groups[group_name].tool_names.append(entry.name)
            else:
                # Create new group
                self._rule_set.register_tool_group(
                    name=group_name,
                    display_name=f"{cat.replace('_', ' ').title()} Tools",
                    tool_names=[entry.name],
                    trigger_keywords=[cat],
                    auto_generated=True
                )

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_checkpoint(self) -> dict[str, Any]:
        """
        Serialize registry for checkpoint.
        Note: Function references are NOT serialized.
        """
        return {
            'tools': {
                name: {
                    'name': entry.name,
                    'description': entry.description,
                    'args_schema': entry.args_schema,
                    'category': entry.category,
                    'flags': entry.flags,
                    'source': entry.source,
                    'server_name': entry.server_name,
                    'original_name': entry.original_name,
                    'metadata': entry.metadata,
                    'created_at': entry.created_at.isoformat(),
                    'call_count': entry.call_count,
                    'last_called': entry.last_called.isoformat() if entry.last_called else None,
                    'litellm_schema': entry.litellm_schema
                }
                for name, entry in self._registry.items()
            },
            'stats': {
                'total_calls': self._total_calls
            }
        }

    def from_checkpoint(
        self,
        data: dict[str, Any],
        function_registry: dict[str, Callable] | None = None
    ):
        """
        Restore registry state from checkpoint using hybrid/overlay approach.

        This method does NOT blindly create ToolEntry objects. Instead:
        - For tools already registered by code: overlay checkpoint metadata
        - For tools not yet registered: store in pending data for later merge

        This prevents "ghost tools" (tools without functions) from appearing.

        Args:
            data: Checkpoint data
            function_registry: Optional dict mapping tool names to functions
                              (for restoring local tool functions) - kept for compatibility
        """
        # Restore global stats
        self._total_calls = data.get('stats', {}).get('total_calls', 0)

        # Clear pending data from previous checkpoint loads
        self._pending_checkpoint_data.clear()

        # Process each tool from checkpoint
        for name, tool_data in data.get('tools', {}).items():
            if name in self._registry:
                # Fall A: Tool existiert bereits in Registry (wurde per Code geladen)
                # -> Overlay: Aktualisiere Laufzeit-Metadaten aus Checkpoint
                entry = self._registry[name]

                # Übernehme call statistics
                entry.call_count = tool_data.get('call_count', 0)

                # Übernehme timestamps
                if tool_data.get('last_called'):
                    entry.last_called = datetime.fromisoformat(tool_data['last_called'])
                if tool_data.get('created_at'):
                    entry.created_at = datetime.fromisoformat(tool_data['created_at'])

                # Merge metadata (checkpoint metadata ergänzt, überschreibt nicht komplett)
                checkpoint_metadata = tool_data.get('metadata', {})
                if checkpoint_metadata:
                    entry.metadata.update(checkpoint_metadata)

            else:
                # Fall B: Tool existiert NICHT in Registry
                # -> Speichere in pending data für spätere Registrierung
                # Erstelle KEINEN Registry-Eintrag (verhindert Geister-Tools)
                self._pending_checkpoint_data[name] = tool_data

        # Sync to RuleSet if available (nur für bereits existierende Tools)
        if self._rule_set:
            self._sync_all_to_ruleset()

    def export_for_display(self) -> str:
        """
        Export registry in human-readable format.
        Useful for debugging and status displays.
        """
        lines = ["# Tool Registry", ""]

        # Group by source
        by_source: dict[str, list[ToolEntry]] = {}
        for entry in self._registry.values():
            if entry.source not in by_source:
                by_source[entry.source] = []
            by_source[entry.source].append(entry)

        for source, entries in by_source.items():
            lines.append(f"## {source.upper()} Tools ({len(entries)})")
            lines.append("")

            for entry in sorted(entries, key=lambda e: e.name):
                flags_str = ", ".join(f for f, v in entry.flags.items() if v)
                cats_str = ", ".join(entry.category[:3])

                lines.append(f"- **{entry.name}**")
                lines.append(f"  {entry.description[:80]}...")
                lines.append(f"  Categories: {cats_str}")
                if flags_str:
                    lines.append(f"  Flags: {flags_str}")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_tool_manager(rule_set: 'RuleSet | None' = None) -> ToolManager:
    """Create a ToolManager with optional RuleSet integration"""
    return ToolManager(rule_set=rule_set)
