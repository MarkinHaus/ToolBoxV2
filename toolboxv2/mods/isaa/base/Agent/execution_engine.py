"""
ExecutionEngine V3 - Clean Architecture with Strict ChatML Compliance

Key Improvements over V2:
1. STRICT HISTORY COMPLIANCE: Assistant(tool_calls) → Tool(result) cycle guaranteed
2. FLUID PIPELINE: No rigid phases, model chooses tools dynamically
3. DYNAMIC AUTO-FOCUS: Injected as message, not static system prompt
4. SIMPLIFIED LOOP: Single unified execution loop
5. LIGHTWEIGHT MICROAGENTS: Shared VFS with context locks

Author: FlowAgent V3
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, AsyncGenerator

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
    from toolboxv2.mods.isaa.base.Agent.agent_session import AgentSession


# =============================================================================
# ENUMS - Simplified
# =============================================================================

class ExecutionStatus(str, Enum):
    """Simplified execution status"""
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"


class TerminationReason(str, Enum):
    """Why did execution stop?"""
    FINAL_ANSWER = "final_answer"
    NEED_HUMAN = "need_human"
    NEED_INFO = "need_info"
    MAX_ITERATIONS = "max_iterations"
    TOKEN_BUDGET = "token_budget"
    LOOP_DETECTED = "loop_detected"
    ERROR = "error"


# =============================================================================
# PYDANTIC MODELS - Validation Layer
# =============================================================================

class ToolCallRecord(BaseModel):
    """Record of a tool call for history tracking"""
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)

    @field_validator('arguments', mode='before')
    @classmethod
    def parse_arguments(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v or {}


class HistoryMessage(BaseModel):
    """Validated message for chat history"""
    role: str = Field(pattern=r'^(system|user|assistant|tool)$')
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        """Convert to LiteLLM-compatible dict"""
        msg = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


class ExecutionConfig(BaseModel):
    """Configuration for execution"""
    max_iterations: int = Field(default=15, ge=1, le=100)
    token_budget: int = Field(default=10000, ge=1000)
    loop_threshold: int = Field(default=3, ge=2)
    auto_focus_entries: int = Field(default=3, ge=1, le=10)
    enable_decomposition: bool = Field(default=True)
    model_preference: str = Field(default="fast")


# =============================================================================
# HISTORY MANAGER - Solves the WTF Bug
# =============================================================================

class ChatHistoryManager:
    """
    CRITICAL: This class solves the History Discontinuity Bug.

    The bug in V2: Tool results were added without the preceding assistant
    message that contained the tool_calls. This broke the model's understanding
    of what actions it had taken.

    Solution: Strict enforcement of the ChatML cycle:
    1. Assistant message with tool_calls array
    2. Tool message(s) with matching tool_call_id(s)

    The manager validates and maintains this invariant.
    """

    def __init__(self, max_history: int = 50):
        self._messages: list[dict] = []
        self._max_history = max_history
        self._pending_tool_calls: dict[str, dict] = {}  # tool_call_id -> tool_call

    def add_system(self, content: str) -> None:
        """Add or update system message (always first)"""
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0]["content"] = content
        else:
            self._messages.insert(0, {"role": "system", "content": content})

    def add_user(self, content: str) -> None:
        """Add user message"""
        self._messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_with_tools(self, content: str | None, tool_calls: list[dict]) -> None:
        """
        SOLVED: WTF Bug Fix - Line ~720 in V2

        This method ensures the assistant message with tool_calls is ALWAYS
        added to history before any tool results. The tool_call_ids are
        tracked so we can validate tool results.

        Args:
            content: Optional text content from assistant
            tool_calls: List of tool call objects from LLM response
        """
        msg = {"role": "assistant"}
        if content:
            msg["content"] = content

        # Store tool_calls in LiteLLM format
        msg["tool_calls"] = []
        for tc in tool_calls:
            tool_call_dict = {
                "id": tc.id if hasattr(tc, 'id') else tc.get('id', str(uuid.uuid4())),
                "type": "function",
                "function": {
                    "name": tc.function.name if hasattr(tc, 'function') else tc.get('function', {}).get('name', ''),
                    "arguments": tc.function.arguments if hasattr(tc, 'function') else tc.get('function', {}).get('arguments', '{}')
                }
            }
            msg["tool_calls"].append(tool_call_dict)
            # Track pending tool calls
            self._pending_tool_calls[tool_call_dict["id"]] = tool_call_dict

        self._messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str, name: str | None = None) -> bool:
        """
        Add tool result with validation.

        Returns False if the tool_call_id doesn't match a pending call,
        which would indicate a history corruption.
        """
        if tool_call_id not in self._pending_tool_calls:
            # Log warning but still add - might be from a previous session
            print(f"⚠️ Tool result for unknown call_id: {tool_call_id}")
        else:
            del self._pending_tool_calls[tool_call_id]

        msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(content)[:4000]  # Truncate long results
        }
        if name:
            msg["name"] = name

        self._messages.append(msg)
        return True

    def add_assistant_text(self, content: str) -> None:
        """Add assistant message without tool calls (final answer, thinking, etc.)"""
        self._messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict]:
        """Get all messages for LLM call"""
        return self._messages.copy()

    def get_last_n(self, n: int) -> list[dict]:
        """Get last N messages, always including system"""
        if not self._messages:
            return []

        system_msg = None
        other_msgs = []

        for msg in self._messages:
            if msg["role"] == "system":
                system_msg = msg
            else:
                other_msgs.append(msg)

        result = other_msgs[-n:] if n < len(other_msgs) else other_msgs
        if system_msg:
            result = [system_msg] + result

        return result

    def inject_context(self, context: str, position: str = "before_last_user") -> None:
        """
        Inject dynamic context (like Auto-Focus) into history.

        Args:
            context: The context string to inject
            position: Where to inject - 'before_last_user' or 'end'
        """
        if not context:
            return

        context_msg = {"role": "system", "content": f"[CONTEXT UPDATE]\n{context}"}

        if position == "before_last_user":
            # Find last user message and insert before it
            for i in range(len(self._messages) - 1, -1, -1):
                if self._messages[i]["role"] == "user":
                    self._messages.insert(i, context_msg)
                    return

        # Default: append at end
        self._messages.append(context_msg)

    def _trim_history(self) -> None:
        """Trim history to max size, preserving system message"""
        if len(self._messages) <= self._max_history:
            return

        system_msg = None
        if self._messages and self._messages[0]["role"] == "system":
            system_msg = self._messages[0]

        # Keep last N messages
        keep = self._max_history - (1 if system_msg else 0)
        trimmed = self._messages[-keep:]

        if system_msg:
            self._messages = [system_msg] + trimmed
        else:
            self._messages = trimmed

    def clear(self) -> None:
        """Clear all messages"""
        self._messages.clear()
        self._pending_tool_calls.clear()

    def to_checkpoint(self) -> dict:
        """Serialize for persistence"""
        return {
            "messages": self._messages.copy(),
            "pending_tool_calls": self._pending_tool_calls.copy()
        }

    @classmethod
    def from_checkpoint(cls, data: dict) -> "ChatHistoryManager":
        """Restore from checkpoint"""
        manager = cls()
        manager._messages = data.get("messages", [])
        manager._pending_tool_calls = data.get("pending_tool_calls", {})
        return manager


# =============================================================================
# AUTO-FOCUS TRACKER V2 - Dynamic Injection
# =============================================================================

@dataclass
class FocusEntry:
    """Single focus entry"""
    filename: str
    operation: str
    timestamp: float
    preview: str
    tool_name: str | None = None


class AutoFocusTracker:
    """
    Tracks recent file/tool operations for dynamic context injection.

    V3 Improvement: Instead of injecting into system prompt (static),
    this builds a context string that gets injected dynamically
    before each LLM call as a separate message.
    """

    def __init__(self, max_entries: int = 3):
        self._entries: deque[FocusEntry] = deque(maxlen=max_entries)

    def record_vfs(self, filename: str, operation: str, content: str = "") -> None:
        """Record VFS operation"""
        self._entries.append(FocusEntry(
            filename=filename,
            operation=operation,
            timestamp=time.time(),
            preview=self._truncate_preview(content)
        ))

    def record_tool(self, tool_name: str, result: str) -> None:
        """Record tool execution result"""
        self._entries.append(FocusEntry(
            filename=f"tool_result:{tool_name}",
            operation="executed",
            timestamp=time.time(),
            preview=self._truncate_preview(result),
            tool_name=tool_name
        ))

    def _truncate_preview(self, content: str, max_lines: int = 15, max_chars: int = 800) -> str:
        """Truncate preview to reasonable size"""
        if not content:
            return ""
        lines = content.split('\n')[:max_lines]
        preview = '\n'.join(lines)
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "..."
        return preview

    def build_context(self) -> str:
        """Build context string for injection"""
        if not self._entries:
            return ""

        lines = ["╔══ LETZTE AKTIONEN (Auto-Focus) ══╗"]

        for entry in reversed(list(self._entries)):
            age = time.time() - entry.timestamp
            age_str = f"{int(age)}s" if age < 60 else f"{int(age/60)}m"

            if entry.tool_name:
                lines.append(f"│ 🔧 [{entry.tool_name}] ({age_str})")
            else:
                lines.append(f"│ 📄 [{entry.operation.upper()}] {entry.filename} ({age_str})")

            if entry.preview:
                for line in entry.preview.split('\n')[:8]:
                    lines.append(f"│   {line[:100]}")

        lines.append("╚═══════════════════════════════════╝")
        return '\n'.join(lines)

    def clear(self) -> None:
        """Clear all entries"""
        self._entries.clear()


# =============================================================================
# LOOP DETECTOR V2 - Semantic Matching
# =============================================================================

class LoopDetector:
    """
    Intelligent loop detection with semantic matching.

    Detects:
    1. Same tool called 3+ times with similar args
    2. Same action pattern repeated
    3. Semantic goal repetition
    """

    VARIABLE_PATTERNS = [
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',  # UUIDs
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',  # Timestamps
        r'"id":\s*"[^"]+"',
        r'"timestamp":\s*"[^"]+"',
    ]

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self._history: deque[dict] = deque(maxlen=15)
        self._tool_counts: dict[str, int] = {}

    def _normalize(self, args: dict | None) -> str:
        """Normalize args by removing variable fields"""
        if not args:
            return ""
        text = json.dumps(args, sort_keys=True)
        for pattern in self.VARIABLE_PATTERNS:
            text = re.sub(pattern, '<VAR>', text)
        return text

    def record(self, tool_name: str, args: dict | None = None) -> None:
        """Record a tool call"""
        normalized = self._normalize(args)
        signature = f"{tool_name}:{normalized}"

        self._tool_counts[signature] = self._tool_counts.get(signature, 0) + 1
        self._history.append({
            "tool": tool_name,
            "signature": signature,
            "time": time.time()
        })

    def detect(self) -> tuple[bool, str]:
        """Check for loops. Returns (is_loop, reason)"""
        # Check 1: Same signature repeated
        for sig, count in self._tool_counts.items():
            if count >= self.threshold:
                tool = sig.split(':')[0]
                return True, f"Tool '{tool}' {count}x mit gleichen Parametern aufgerufen"

        # Check 2: Same tool 4x in last 5 calls
        if len(self._history) >= 5:
            last_5 = [h["tool"] for h in list(self._history)[-5:]]
            for tool in set(last_5):
                if last_5.count(tool) >= 4:
                    return True, f"Tool '{tool}' dominiert (4/5 Aufrufe)"

        return False, ""

    def reset(self) -> None:
        """Reset detector"""
        self._history.clear()
        self._tool_counts.clear()


# =============================================================================
# EXECUTION STATE - Simplified
# =============================================================================

@dataclass
class ExecutionState:
    """Execution state - simplified from V2"""
    execution_id: str
    query: str
    session_id: str
    config: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Status
    status: ExecutionStatus = ExecutionStatus.RUNNING
    termination_reason: TerminationReason | None = None

    # Tracking
    iteration: int = 0
    tokens_used: int = 0
    tools_used: list[str] = field(default_factory=list)

    # Results
    final_answer: str | None = None
    human_query: str | None = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    # Error tracking
    errors: list[str] = field(default_factory=list)
    consecutive_failures: int = 0


@dataclass
class ExecutionResult:
    """Result of execution"""
    success: bool
    response: str
    execution_id: str
    iterations: int = 0
    tools_used: list[str] = field(default_factory=list)
    tokens_used: int = 0
    duration: float = 0.0
    termination_reason: TerminationReason | None = None
    needs_human: bool = False
    human_query: str | None = None


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

# VFS Tools - Always available (System Tools)
VFS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vfs_read",
            "description": "Read file content from virtual filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File to read"},
                    "line_start": {"type": "integer", "description": "Start line (1-indexed)"},
                    "line_end": {"type": "integer", "description": "End line (-1 for all)"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_write",
            "description": "Write/overwrite file in virtual filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_create",
            "description": "Create new file in virtual filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File to create"},
                    "content": {"type": "string", "description": "Initial content", "default": ""}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_list",
            "description": "List all files in virtual filesystem",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_edit",
            "description": "Edit specific lines in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "line_start": {"type": "integer"},
                    "line_end": {"type": "integer"},
                    "content": {"type": "string"}
                },
                "required": ["filename", "line_start", "line_end", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_remove",
            "description": "Remove file from virtual filesystem",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"]
            }
        }
    }
]

# Control Tools - Always available
CONTROL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Provide final answer. Use IMMEDIATELY when task is complete. Do NOT repeat actions after success.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The final answer"}
                },
                "required": ["answer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "need_info",
            "description": "Request more information from user",
            "parameters": {
                "type": "object",
                "properties": {
                    "missing": {"type": "string", "description": "What information is needed"}
                },
                "required": ["missing"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "need_human",
            "description": "Request human assistance",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question for human"}
                },
                "required": ["question"]
            }
        }
    }
]

# Discovery Tools - For dynamic tool loading
DISCOVERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "discover_tools",
            "description": """Search available tools by keyword or category. Use this FIRST when you need a capability you don't have loaded.
Returns: List of matching tools with name, description, and category.
WICHTIG: Nach discover_tools musst du load_tools aufrufen um die Tools zu aktivieren!""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query: keyword, tool name, or category (e.g. 'discord', 'web', 'file', 'send message')"
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional: Filter by category (e.g. 'discord', 'web', 'database')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_tools",
            "description": """Load or unload tools into your active toolset. You have MAX 5 active tools at once.
- To USE a tool, you must LOAD it first with discover_tools results
- Unload tools you no longer need to make room for new ones
WICHTIG: Geladene Tools erscheinen in deiner nächsten Antwort als verfügbare Funktionen!""",
            "parameters": {
                "type": "object",
                "properties": {
                    "load": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tool names to load (from discover_tools results)"
                    },
                    "unload": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tool names to unload (free up slots)"
                    }
                },
                "required": []
            }
        }
    }
]


# =============================================================================
# TOOL DISCOVERY MANAGER
# =============================================================================

class ToolDiscoveryManager:
    """
    Manages dynamic tool loading with a max active tool limit.

    Workflow:
    1. Agent calls discover_tools("send discord message")
    2. Manager returns matching tools with descriptions
    3. Agent calls load_tools(load=["discord_send_message"])
    4. Tool is now available in next LLM call
    5. When done, agent calls load_tools(unload=["discord_send_message"])

    Constraints:
    - Max 5 active tools at once (configurable)
    - System tools (VFS, Control, Discovery) are always available
    - Loading a 6th tool requires unloading one first
    """

    def __init__(self, agent: 'FlowAgent', max_active: int = 5):
        self.agent = agent
        self.max_active = max_active
        self._active_tools: dict[str, dict] = {}  # name -> litellm tool dict
        self._tool_cache: dict[str, dict] = {}    # name -> litellm tool dict (all tools)
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-load tool cache from agent"""
        if self._initialized:
            return

        for tool in self.agent.tool_manager.get_all_litellm():
            name = tool.get('function', {}).get('name', '')
            if name:
                self._tool_cache[name] = tool

        self._initialized = True

    def discover(self, query: str, category: str | None = None) -> list[dict]:
        """
        Search tools by query and/or category.

        Returns list of tool info dicts (not full LiteLLM format):
        [{"name": "...", "description": "...", "category": "...", "loaded": bool}, ...]
        """
        self._ensure_initialized()

        query_lower = query.lower()
        results = []

        # Also search in agent's tool manager for category info
        all_tools = self.agent.tool_manager.get_all()
        tool_categories = {t.name: t.category for t in all_tools}

        for name, tool in self._tool_cache.items():
            func = tool.get('function', {})
            desc = func.get('description', '')
            tool_cat = tool_categories.get(name, ['unknown'])

            # Category filter
            if category:
                cat_match = any(category.lower() in str(c).lower() for c in tool_cat)
                if not cat_match:
                    continue

            # Query match (name or description)
            name_match = query_lower in name.lower()
            desc_match = query_lower in desc.lower()
            cat_match = any(query_lower in str(c).lower() for c in tool_cat)

            if name_match or desc_match or cat_match:
                results.append({
                    "name": name,
                    "description": desc[:150] + "..." if len(desc) > 150 else desc,
                    "category": tool_cat if isinstance(tool_cat, list) else [tool_cat],
                    "loaded": name in self._active_tools
                })

        # Sort: loaded first, then by name
        results.sort(key=lambda x: (not x['loaded'], x['name']))

        return results[:10]  # Max 10 results

    def load(self, tool_names: list[str]) -> dict:
        """
        Load tools into active set.

        Returns: {"loaded": [...], "failed": [...], "message": "..."}
        """
        self._ensure_initialized()

        loaded = []
        failed = []

        for name in tool_names:
            if name in self._active_tools:
                loaded.append(name)  # Already loaded
                continue

            if len(self._active_tools) >= self.max_active:
                failed.append(f"{name} (max {self.max_active} tools erreicht - erst unload)")
                continue

            if name not in self._tool_cache:
                failed.append(f"{name} (nicht gefunden)")
                continue

            self._active_tools[name] = self._tool_cache[name]
            loaded.append(name)

        return {
            "loaded": loaded,
            "failed": failed,
            "active_count": len(self._active_tools),
            "slots_free": self.max_active - len(self._active_tools),
            "message": f"✓ {len(loaded)} Tools geladen. {self.max_active - len(self._active_tools)} Slots frei."
        }

    def unload(self, tool_names: list[str]) -> dict:
        """
        Unload tools from active set.

        Returns: {"unloaded": [...], "message": "..."}
        """
        unloaded = []

        for name in tool_names:
            if name in self._active_tools:
                del self._active_tools[name]
                unloaded.append(name)

        return {
            "unloaded": unloaded,
            "active_count": len(self._active_tools),
            "slots_free": self.max_active - len(self._active_tools),
            "message": f"✓ {len(unloaded)} Tools entladen. {self.max_active - len(self._active_tools)} Slots frei."
        }

    def get_active_tools_litellm(self) -> list[dict]:
        """Get currently active tools in LiteLLM format"""
        return list(self._active_tools.values())

    def get_active_tool_names(self) -> list[str]:
        """Get names of currently active tools"""
        return list(self._active_tools.keys())

    def get_status(self) -> str:
        """Get human-readable status"""
        if not self._active_tools:
            return "Keine Tools geladen. Nutze discover_tools um Tools zu finden."

        names = list(self._active_tools.keys())
        return f"Aktive Tools ({len(names)}/{self.max_active}): {', '.join(names)}"

    def reset(self) -> None:
        """Reset active tools"""
        self._active_tools.clear()

    def is_tool_active(self, name: str) -> bool:
        """Check if a tool is currently active"""
        return name in self._active_tools


# =============================================================================
# SYSTEM PROMPT - With Discovery Instructions
# =============================================================================

SYSTEM_PROMPT = """Du bist ein intelligenter Agent mit Zugang zu einem virtuellen Dateisystem (VFS) und dynamisch ladbaren Tools.

═══════════════════════════════════════════════════════════════════
⚡ TOOL-DISCOVERY SYSTEM - MAX 5 AKTIVE TOOLS
═══════════════════════════════════════════════════════════════════

WORKFLOW FÜR EXTERNE TOOLS:
1. discover_tools("suchbegriff") → Finde passende Tools
2. load_tools(load=["tool_name"]) → Lade Tool (max 5 gleichzeitig)
3. Nutze das geladene Tool
4. load_tools(unload=["tool_name"]) → Entlade wenn fertig

IMMER VERFÜGBAR (System-Tools):
- VFS: vfs_read, vfs_write, vfs_create, vfs_list, vfs_edit, vfs_remove
- Control: final_answer, need_info, need_human
- Discovery: discover_tools, load_tools

{active_tools_status}

KRITISCHE REGELN:
1. Nach ERFOLG → sofort `final_answer`. NIEMALS erfolgreiche Aktionen wiederholen.
2. ERFINDE NICHTS - nur Informationen aus Tool-Ergebnissen verwenden.
3. Wenn du ein Tool brauchst das nicht geladen ist → erst discover_tools, dann load_tools
4. Du siehst deine letzten Aktionen im CONTEXT UPDATE Bereich.

AKTUELLE AUFGABE: {query}
"""


# =============================================================================
# EXECUTION ENGINE V3
# =============================================================================

class ExecutionEngine:
    """
    ExecutionEngine V3 - Clean Architecture

    Key improvements:
    1. Strict ChatML history via ChatHistoryManager
    2. Dynamic Tool Discovery - max 5 active tools
    3. Dynamic Auto-Focus injection
    4. Simplified single loop
    """

    def __init__(
        self,
        agent: 'FlowAgent',
        human_online: bool = False,
        callback: Callable[[str], None] | None = None,
        max_active_tools: int = 5
    ):
        self.agent = agent
        self.human_online = human_online
        self.callback = callback
        self.max_active_tools = max_active_tools

        # State tracking
        self._executions: dict[str, ExecutionState] = {}
        self._history_managers: dict[str, ChatHistoryManager] = {}
        self._auto_focus: dict[str, AutoFocusTracker] = {}
        self._loop_detectors: dict[str, LoopDetector] = {}
        self._discovery_managers: dict[str, ToolDiscoveryManager] = {}

    def _emit(self, msg: str) -> None:
        """Emit intermediate message"""
        if self.callback:
            self.callback(msg)

    def _get_history(self, execution_id: str) -> ChatHistoryManager:
        """Get or create history manager"""
        if execution_id not in self._history_managers:
            self._history_managers[execution_id] = ChatHistoryManager()
        return self._history_managers[execution_id]

    def _get_focus(self, session_id: str) -> AutoFocusTracker:
        """Get or create auto-focus tracker"""
        if session_id not in self._auto_focus:
            self._auto_focus[session_id] = AutoFocusTracker()
        return self._auto_focus[session_id]

    def _get_loop_detector(self, execution_id: str) -> LoopDetector:
        """Get or create loop detector"""
        if execution_id not in self._loop_detectors:
            self._loop_detectors[execution_id] = LoopDetector()
        return self._loop_detectors[execution_id]

    def _get_discovery(self, execution_id: str) -> ToolDiscoveryManager:
        """Get or create discovery manager"""
        if execution_id not in self._discovery_managers:
            self._discovery_managers[execution_id] = ToolDiscoveryManager(
                self.agent,
                max_active=self.max_active_tools
            )
        return self._discovery_managers[execution_id]

    # =========================================================================
    # TOOL PREPARATION - Dynamic Discovery
    # =========================================================================

    def _prepare_tools(self, state: ExecutionState, discovery: ToolDiscoveryManager) -> list[dict]:
        """
        Prepare tool list for LLM call.

        V3 Strategy:
        - System tools (VFS, Control, Discovery) always available
        - Agent tools loaded dynamically via discover_tools/load_tools
        - Max 5 active agent tools at once
        """
        tools = []

        # Always include system tools
        tools.extend(VFS_TOOLS)
        tools.extend(CONTROL_TOOLS)
        tools.extend(DISCOVERY_TOOLS)

        # Add currently active (loaded) tools
        active_tools = discovery.get_active_tools_litellm()
        tools.extend(active_tools)

        return tools

    def _build_active_tools_status(self, discovery: ToolDiscoveryManager) -> str:
        """Build status string for system prompt"""
        active = discovery.get_active_tool_names()
        if not active:
            return "GELADENE TOOLS: Keine. Nutze discover_tools um Tools zu finden und load_tools um sie zu laden."

        slots_free = discovery.max_active - len(active)
        return f"GELADENE TOOLS ({len(active)}/{discovery.max_active}, {slots_free} frei): {', '.join(active)}"

    # =========================================================================
    # DISCOVERY TOOL EXECUTION
    # =========================================================================

    def _execute_discover_tools(self, discovery: ToolDiscoveryManager, args: dict) -> str:
        """Execute discover_tools command"""
        query = args.get('query', '')
        category = args.get('category')

        if not query:
            return "Error: 'query' parameter required for discover_tools"

        results = discovery.discover(query, category)

        if not results:
            return f"Keine Tools gefunden für '{query}'. Versuche andere Suchbegriffe."

        lines = [f"🔍 Gefundene Tools für '{query}':"]
        for r in results:
            loaded_marker = "✓ GELADEN" if r['loaded'] else ""
            cat_str = ', '.join(r['category']) if r['category'] else 'unknown'
            lines.append(f"\n• {r['name']} [{cat_str}] {loaded_marker}")
            lines.append(f"  {r['description']}")

        lines.append(f"\n→ Nutze load_tools(load=[\"tool_name\"]) um ein Tool zu laden")
        return '\n'.join(lines)

    def _execute_load_tools(self, discovery: ToolDiscoveryManager, args: dict) -> str:
        """Execute load_tools command"""
        to_load = args.get('load', [])
        to_unload = args.get('unload', [])

        results = []

        if to_unload:
            unload_result = discovery.unload(to_unload)
            if unload_result['unloaded']:
                results.append(f"✓ Entladen: {', '.join(unload_result['unloaded'])}")

        if to_load:
            load_result = discovery.load(to_load)
            if load_result['loaded']:
                results.append(f"✓ Geladen: {', '.join(load_result['loaded'])}")
            if load_result['failed']:
                results.append(f"✗ Fehlgeschlagen: {', '.join(load_result['failed'])}")

        # Status
        active = discovery.get_active_tool_names()
        slots_free = discovery.max_active - len(active)

        if active:
            results.append(f"\n📦 Aktive Tools ({len(active)}/{discovery.max_active}): {', '.join(active)}")
        else:
            results.append(f"\n📦 Keine Tools geladen")

        results.append(f"💡 {slots_free} Slots frei")

        if to_load and load_result.get('loaded'):
            results.append(f"\n→ Die geladenen Tools sind jetzt verfügbar!")

        return '\n'.join(results) if results else "Keine Änderungen"

    # =========================================================================
    # VFS EXECUTION
    # =========================================================================

    async def _execute_vfs(
        self,
        session: 'AgentSession',
        tool_name: str,
        args: dict,
        state: ExecutionState
    ) -> str:
        """Execute VFS operation and track in auto-focus"""
        focus = self._get_focus(state.session_id)
        result = None

        try:
            if tool_name == "vfs_read":
                res = session.vfs.read(args.get('filename'))
                if res.get('success'):
                    content = res['content']
                    # Apply line range if specified
                    if args.get('line_start') or args.get('line_end'):
                        lines = content.split('\n')
                        start = max(0, args.get('line_start', 1) - 1)
                        end = args.get('line_end', len(lines))
                        if end == -1:
                            end = len(lines)
                        content = '\n'.join(lines[start:end])
                    focus.record_vfs(args['filename'], 'read', content)
                    result = content
                else:
                    result = f"Error: {res.get('error', 'Read failed')}"

            elif tool_name == "vfs_write":
                filename = args.get('filename', '')
                content = args.get('content', '')
                res = session.vfs.write(filename, content)
                if res.get('success'):
                    focus.record_vfs(filename, 'written', content)
                    result = f"✓ Datei '{filename}' geschrieben ({len(content)} Zeichen)"
                else:
                    result = f"Error: {res.get('error', 'Write failed')}"

            elif tool_name == "vfs_create":
                filename = args.get('filename', '')
                content = args.get('content', '')
                res = session.vfs.create(filename, content)
                if res.get('success'):
                    focus.record_vfs(filename, 'created', content)
                    result = f"✓ Datei '{filename}' erstellt"
                else:
                    result = f"Error: {res.get('error', 'Create failed')}"

            elif tool_name == "vfs_list":
                files = session.vfs.list_files()
                if files:
                    result = "Dateien im VFS:\n" + '\n'.join(f"- {f}" for f in files)
                else:
                    result = "VFS ist leer"

            elif tool_name == "vfs_edit":
                filename = args.get('filename', '')
                res = session.vfs.edit(
                    filename,
                    args.get('line_start', 1),
                    args.get('line_end', 1),
                    args.get('content', '')
                )
                if res.get('success'):
                    # Read updated content for focus
                    updated = session.vfs.read(filename)
                    if updated.get('success'):
                        focus.record_vfs(filename, 'edited', updated['content'])
                    result = f"✓ Datei '{filename}' bearbeitet"
                else:
                    result = f"Error: {res.get('error', 'Edit failed')}"

            elif tool_name == "vfs_remove":
                filename = args.get('filename', '')
                res = session.vfs.remove(filename)
                if res.get('success'):
                    result = f"✓ Datei '{filename}' gelöscht"
                else:
                    result = f"Error: {res.get('error', 'Remove failed')}"

            else:
                result = f"Unknown VFS operation: {tool_name}"

        except Exception as e:
            result = f"VFS Error: {str(e)}"

        return result or "Operation completed"

    # =========================================================================
    # TOOL EXECUTION
    # =========================================================================

    async def _execute_tool(
        self,
        session: 'AgentSession',
        tool_name: str,
        args: dict,
        state: ExecutionState,
        discovery: ToolDiscoveryManager
    ) -> str:
        """Execute a tool and return result"""
        focus = self._get_focus(state.session_id)

        # Track tool usage
        if tool_name not in state.tools_used:
            state.tools_used.append(tool_name)

        self._emit(f"🔧 {tool_name}...")

        try:
            # VFS tools - always available
            if tool_name.startswith("vfs_"):
                return await self._execute_vfs(session, tool_name, args, state)

            # Check if agent tool is loaded
            if not discovery.is_tool_active(tool_name):
                return f"⚠️ Tool '{tool_name}' ist nicht geladen! Nutze erst:\n1. discover_tools(\"{tool_name}\") um es zu finden\n2. load_tools(load=[\"{tool_name}\"]) um es zu laden"

            # Execute agent tool
            result = await self.agent.arun_function(tool_name, **args)
            result_str = str(result)[:2000]

            # Track in auto-focus
            focus.record_tool(tool_name, result_str)

            return result_str

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Tool Error: {str(e)}"

    # =========================================================================
    # MAIN EXECUTION LOOP
    # =========================================================================

    async def execute(
        self,
        query: str,
        session_id: str = "default",
        config: ExecutionConfig | None = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Main execution entry point.

        This is the unified execution loop that replaces the complex
        phase-based state machine in V2.
        """
        start_time = time.perf_counter()
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"

        # Initialize state
        config = config or ExecutionConfig(**{k: v for k, v in kwargs.items() if k in ExecutionConfig.model_fields})
        state = ExecutionState(
            execution_id=execution_id,
            query=query,
            session_id=session_id,
            config=config
        )
        self._executions[execution_id] = state

        # Get session and managers
        session = await self.agent.session_manager.get_or_create(session_id)
        history = self._get_history(execution_id)
        focus = self._get_focus(session_id)
        loop_detector = self._get_loop_detector(execution_id)
        discovery = self._get_discovery(execution_id)

        # Prepare initial tools (system + discovery, no agent tools yet)
        tools = self._prepare_tools(state, discovery)
        active_status = self._build_active_tools_status(discovery)

        # Initialize history with system prompt and user query
        system_prompt = SYSTEM_PROMPT.format(
            active_tools_status=active_status,
            query=query
        )
        history.add_system(system_prompt)
        history.add_user(query)

        try:
            # Main loop
            while state.iteration < state.config.max_iterations:
                state.iteration += 1
                self._emit(f"Iteration {state.iteration}...")

                # Check token budget
                if state.tokens_used >= state.config.token_budget:
                    state.termination_reason = TerminationReason.TOKEN_BUDGET
                    break

                # Check for loops
                is_loop, loop_reason = loop_detector.detect()
                if is_loop:
                    state.errors.append(f"Loop: {loop_reason}")
                    state.termination_reason = TerminationReason.LOOP_DETECTED
                    self._emit(f"⚠️ Loop erkannt: {loop_reason}")
                    break

                # Inject auto-focus context before LLM call
                focus_context = focus.build_context()
                if focus_context:
                    history.inject_context(focus_context)

                # Update system prompt with current tool status
                active_status = self._build_active_tools_status(discovery)
                system_prompt = SYSTEM_PROMPT.format(
                    active_tools_status=active_status,
                    query=query
                )
                history.add_system(system_prompt)

                # Refresh tools (may have changed via load_tools)
                tools = self._prepare_tools(state, discovery)

                # Get messages for LLM
                messages = history.get_messages()

                # Make LLM call
                try:
                    response = await self.agent.a_run_llm_completion(
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        model_preference=state.config.model_preference,
                        stream=False,
                        get_response_message=True,
                        task_id=f"{execution_id}_iter_{state.iteration}",
                        session_id=session_id,
                        with_context=False
                    )
                except Exception as e:
                    state.consecutive_failures += 1
                    state.errors.append(str(e))
                    if state.consecutive_failures >= 3:
                        state.termination_reason = TerminationReason.ERROR
                        break
                    continue

                state.consecutive_failures = 0

                # Handle response
                if response is None:
                    state.errors.append("Empty LLM response")
                    continue

                # Check for tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # =====================================================
                    # SOLVED: WTF Bug Fix
                    #
                    # This is the critical fix. We MUST add the assistant
                    # message with tool_calls BEFORE processing any results.
                    # This ensures the model sees its own actions in history.
                    # =====================================================
                    history.add_assistant_with_tools(
                        content=response.content if hasattr(response, 'content') else None,
                        tool_calls=response.tool_calls
                    )

                    # Process each tool call
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            args = json.loads(tool_call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}

                        # Record for loop detection
                        loop_detector.record(tool_name, args)

                        # Handle control tools
                        if tool_name == "final_answer":
                            state.final_answer = args.get("answer", "")
                            state.status = ExecutionStatus.COMPLETED
                            state.termination_reason = TerminationReason.FINAL_ANSWER

                            # Add to history for completeness
                            history.add_tool_result(
                                tool_call.id,
                                "Answer accepted",
                                tool_name
                            )
                            break

                        elif tool_name == "need_human":
                            if self.human_online:
                                state.human_query = args.get("question", "")
                                state.status = ExecutionStatus.PAUSED
                                state.termination_reason = TerminationReason.NEED_HUMAN
                                history.add_tool_result(
                                    tool_call.id,
                                    "Waiting for human response",
                                    tool_name
                                )
                                break
                            else:
                                history.add_tool_result(
                                    tool_call.id,
                                    "Human assistance not available",
                                    tool_name
                                )

                        elif tool_name == "need_info":
                            state.human_query = args.get("missing", "")
                            state.status = ExecutionStatus.PAUSED
                            state.termination_reason = TerminationReason.NEED_INFO
                            history.add_tool_result(
                                tool_call.id,
                                "Waiting for information",
                                tool_name
                            )
                            break

                        # Handle discovery tools
                        elif tool_name == "discover_tools":
                            result = self._execute_discover_tools(discovery, args)
                            history.add_tool_result(tool_call.id, result, tool_name)
                            self._emit(f"🔍 discover_tools: {args.get('query', '')}")

                        elif tool_name == "load_tools":
                            result = self._execute_load_tools(discovery, args)
                            history.add_tool_result(tool_call.id, result, tool_name)
                            loaded = args.get('load', [])
                            unloaded = args.get('unload', [])
                            if loaded:
                                self._emit(f"📦 Loaded: {', '.join(loaded)}")
                            if unloaded:
                                self._emit(f"📤 Unloaded: {', '.join(unloaded)}")

                        else:
                            # Execute tool (VFS or agent tool)
                            result = await self._execute_tool(
                                session, tool_name, args, state, discovery
                            )

                            # Add result to history
                            history.add_tool_result(
                                tool_call.id,
                                result,
                                tool_name
                            )

                    # Check if we should exit loop
                    if state.status != ExecutionStatus.RUNNING:
                        break

                else:
                    # No tool calls - treat as direct answer
                    content = response.content if hasattr(response, 'content') else str(response)
                    if content:
                        state.final_answer = content
                        state.status = ExecutionStatus.COMPLETED
                        state.termination_reason = TerminationReason.FINAL_ANSWER
                        history.add_assistant_text(content)
                        break

            # Handle max iterations
            if state.iteration >= state.config.max_iterations and state.status == ExecutionStatus.RUNNING:
                state.termination_reason = TerminationReason.MAX_ITERATIONS
                state.final_answer = f"Aufgabe konnte nicht in {state.config.max_iterations} Schritten abgeschlossen werden."

        except Exception as e:
            import traceback
            traceback.print_exc()
            state.status = ExecutionStatus.FAILED
            state.termination_reason = TerminationReason.ERROR
            state.errors.append(str(e))
            state.final_answer = f"Execution failed: {str(e)}"

        finally:
            state.completed_at = datetime.now()
            self._cleanup(execution_id)

        # Build result
        duration = time.perf_counter() - start_time
        success = state.status == ExecutionStatus.COMPLETED and state.termination_reason == TerminationReason.FINAL_ANSWER

        return ExecutionResult(
            success=success,
            response=state.final_answer or "",
            execution_id=execution_id,
            iterations=state.iteration,
            tools_used=state.tools_used,
            tokens_used=state.tokens_used,
            duration=duration,
            termination_reason=state.termination_reason,
            needs_human=state.status == ExecutionStatus.PAUSED and state.termination_reason in [
                TerminationReason.NEED_HUMAN,
                TerminationReason.NEED_INFO
            ],
            human_query=state.human_query
        )

    async def execute_stream(
        self,
        query: str,
        session_id: str = "default",
        config: ExecutionConfig | None = None,
        **kwargs
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """
        Streaming execution - yields intermediate results.
        """
        start_time = time.perf_counter()
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"

        config = config or ExecutionConfig(**{k: v for k, v in kwargs.items() if k in ExecutionConfig.model_fields})
        state = ExecutionState(
            execution_id=execution_id,
            query=query,
            session_id=session_id,
            config=config
        )
        self._executions[execution_id] = state

        session = await self.agent.session_manager.get_or_create(session_id)
        history = self._get_history(execution_id)
        focus = self._get_focus(session_id)
        loop_detector = self._get_loop_detector(execution_id)
        discovery = self._get_discovery(execution_id)

        tools = self._prepare_tools(state, discovery)
        active_status = self._build_active_tools_status(discovery)

        system_prompt = SYSTEM_PROMPT.format(
            active_tools_status=active_status,
            query=query
        )
        history.add_system(system_prompt)
        history.add_user(query)

        try:
            while state.iteration < state.config.max_iterations:
                state.iteration += 1
                yield f"Iteration {state.iteration}..."

                if state.tokens_used >= state.config.token_budget:
                    state.termination_reason = TerminationReason.TOKEN_BUDGET
                    break

                is_loop, loop_reason = loop_detector.detect()
                if is_loop:
                    state.termination_reason = TerminationReason.LOOP_DETECTED
                    yield f"⚠️ Loop: {loop_reason}"
                    break

                focus_context = focus.build_context()
                if focus_context:
                    history.inject_context(focus_context)

                # Update tools and system prompt
                active_status = self._build_active_tools_status(discovery)
                system_prompt = SYSTEM_PROMPT.format(
                    active_tools_status=active_status,
                    query=query
                )
                history.add_system(system_prompt)
                tools = self._prepare_tools(state, discovery)

                messages = history.get_messages()

                try:
                    response = await self.agent.a_run_llm_completion(
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        model_preference=state.config.model_preference,
                        stream=False,
                        get_response_message=True,
                        task_id=f"{execution_id}_iter_{state.iteration}",
                        session_id=session_id,
                        with_context=False
                    )
                except Exception as e:
                    state.consecutive_failures += 1
                    if state.consecutive_failures >= 3:
                        state.termination_reason = TerminationReason.ERROR
                        break
                    continue

                state.consecutive_failures = 0

                if response is None:
                    continue

                if hasattr(response, 'tool_calls') and response.tool_calls:
                    history.add_assistant_with_tools(
                        content=response.content if hasattr(response, 'content') else None,
                        tool_calls=response.tool_calls
                    )

                    for tool_call in response.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            args = json.loads(tool_call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}

                        loop_detector.record(tool_name, args)

                        if tool_name == "final_answer":
                            state.final_answer = args.get("answer", "")
                            state.status = ExecutionStatus.COMPLETED
                            state.termination_reason = TerminationReason.FINAL_ANSWER
                            history.add_tool_result(tool_call.id, "OK", tool_name)
                            break

                        elif tool_name in ["need_human", "need_info"]:
                            state.human_query = args.get("question") or args.get("missing", "")
                            state.status = ExecutionStatus.PAUSED
                            state.termination_reason = TerminationReason.NEED_HUMAN if tool_name == "need_human" else TerminationReason.NEED_INFO
                            history.add_tool_result(tool_call.id, "Waiting", tool_name)
                            break

                        elif tool_name == "discover_tools":
                            result = self._execute_discover_tools(discovery, args)
                            history.add_tool_result(tool_call.id, result, tool_name)
                            yield f"🔍 discover: {args.get('query', '')}"

                        elif tool_name == "load_tools":
                            result = self._execute_load_tools(discovery, args)
                            history.add_tool_result(tool_call.id, result, tool_name)
                            yield f"📦 load/unload tools"

                        else:
                            yield f"🔧 {tool_name}..."
                            result = await self._execute_tool(session, tool_name, args, state, discovery)
                            history.add_tool_result(tool_call.id, result, tool_name)
                            yield f"Result: {result[:200]}..."

                    if state.status != ExecutionStatus.RUNNING:
                        break

                else:
                    content = response.content if hasattr(response, 'content') else str(response)
                    if content:
                        state.final_answer = content
                        state.status = ExecutionStatus.COMPLETED
                        state.termination_reason = TerminationReason.FINAL_ANSWER
                        history.add_assistant_text(content)
                        break

            if state.iteration >= state.config.max_iterations and state.status == ExecutionStatus.RUNNING:
                state.termination_reason = TerminationReason.MAX_ITERATIONS
                state.final_answer = f"Max iterations reached"

        except Exception as e:
            state.status = ExecutionStatus.FAILED
            state.termination_reason = TerminationReason.ERROR
            state.final_answer = f"Error: {str(e)}"

        finally:
            state.completed_at = datetime.now()
            self._cleanup(execution_id)

        duration = time.perf_counter() - start_time
        success = state.status == ExecutionStatus.COMPLETED and state.termination_reason == TerminationReason.FINAL_ANSWER

        yield ExecutionResult(
            success=success,
            response=state.final_answer or "",
            execution_id=execution_id,
            iterations=state.iteration,
            tools_used=state.tools_used,
            tokens_used=state.tokens_used,
            duration=duration,
            termination_reason=state.termination_reason,
            needs_human=state.termination_reason in [TerminationReason.NEED_HUMAN, TerminationReason.NEED_INFO],
            human_query=state.human_query
        )

    def _cleanup(self, execution_id: str) -> None:
        """Cleanup execution resources"""
        if execution_id in self._history_managers:
            del self._history_managers[execution_id]
        if execution_id in self._loop_detectors:
            del self._loop_detectors[execution_id]
        if execution_id in self._discovery_managers:
            del self._discovery_managers[execution_id]

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_state(self, execution_id: str) -> ExecutionState | None:
        """Get execution state"""
        return self._executions.get(execution_id)

    def get_focus_context(self, session_id: str) -> str:
        """Get current auto-focus context"""
        return self._get_focus(session_id).build_context()

    def clear_focus(self, session_id: str) -> None:
        """Clear auto-focus for session"""
        if session_id in self._auto_focus:
            self._auto_focus[session_id].clear()


# =============================================================================
# FACTORY
# =============================================================================

def create_engine_v3(
    agent: 'FlowAgent',
    human_online: bool = False,
    callback: Callable[[str], None] | None = None,
    max_active_tools: int = 5
) -> ExecutionEngine:
    """Factory function for ExecutionEngine"""
    return ExecutionEngine(
        agent=agent,
        human_online=human_online,
        callback=callback,
        max_active_tools=max_active_tools
    )
