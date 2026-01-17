"""
ExecutionEngine V2 - Battle-Proven Refactoring for FlowAgent V2

Key Improvements:
1. enter_react_mode: Explicit, transparent ReAct transition (replaces use_tools)
2. VFS Tools in Immediate Response: One-shot file operations without ReAct overhead
3. Auto-Focus Context: Last 3 modified files automatically visible
4. Intelligent Loop Detection: Semantic matching, ignores timestamps/UUIDs
5. Crash Resistance: Robust error handling for empty LLM responses

Author: FlowAgent V2
Version: 2.0.0
"""

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING, AsyncGenerator
from collections import deque
try:
    from litellm.types.utils import ModelResponse
except ImportError:
    class ModelResponse: pass
from pydantic import BaseModel, Field

from toolboxv2.mods.isaa.base.Agent import ToolEntry

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
    from toolboxv2.mods.isaa.base.Agent.agent_session import VirtualFileSystem


# =============================================================================
# ENUMS
# =============================================================================

class ExecutionPhase(str, Enum):
    """Current phase of execution"""
    INTENT = "intent"
    CATEGORY_SELECT = "category_select"
    TOOL_SELECT = "tool_select"
    REACT_LOOP = "react_loop"
    DECOMPOSITION = "decomposition"
    VALIDATION = "validation"
    LEARNING = "learning"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"


class ActionType(str, Enum):
    """Types of actions in ReAct loop"""
    TOOL_CALL = "tool_call"
    VFS_OPEN = "vfs_open"
    VFS_CLOSE = "vfs_close"
    VFS_READ = "vfs_read"
    VFS_WRITE = "vfs_write"
    VFS_EDIT = "vfs_edit"
    VFS_VIEW = "vfs_view"
    VFS_LIST = "vfs_list"
    VFS_CREATE = "vfs_create"
    VFS_REMOVE = "vfs_remove"
    FINAL_ANSWER = "final_answer"
    NEED_INFO = "need_info"
    NEED_HUMAN = "need_human"


# =============================================================================
# PYDANTIC MODELS (Atomic - 0.5B compatible)
# =============================================================================

class IntentClassification(BaseModel):
    """Phase 1: What does the user want?"""
    can_answer_directly: bool = Field(description="Can answer without tools?")
    needs_tools: bool = Field(description="Needs tool calls?")
    is_complex_task: bool = Field(description="Multiple steps needed?")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")


class CategorySelection(BaseModel):
    """Which tool categories are relevant?"""
    categories: list[str] = Field(description="Relevant categories like 'discord', 'web', 'file'")
    reasoning: str = Field(max_length=100, description="Brief reason")


class ToolSelection(BaseModel):
    """Which specific tools? (max 5)"""
    tools: list[str] = Field(max_length=5, description="Tool names, max 5")


class TaskDecomposition(BaseModel):
    """For complex tasks: subtasks"""
    subtasks: list[str] = Field(description="Subtask descriptions")
    can_parallel: list[bool] = Field(description="Which subtasks can run parallel with previous")


class ThoughtAction(BaseModel):
    """ReAct: Thought + Action (for a_format_class mode)"""
    thought: str = Field(description="What am I thinking?")
    action: str = Field(description="tool_call|vfs_open|vfs_close|vfs_read|vfs_write|vfs_edit|vfs_view|vfs_list|final_answer|need_info|need_human")
    tool_name: str | None = Field(default=None, description="Tool name if action=tool_call")
    tool_args: dict | None = Field(default=None, description="Tool args if action=tool_call")
    filename: str | None = Field(default=None, description="Filename for VFS ops")
    content: str | None = Field(default=None, description="Content for write/edit")
    line_start: int | None = Field(default=None, description="Start line for view/edit")
    line_end: int | None = Field(default=None, description="End line for view/edit")
    answer: str | None = Field(default=None, description="Final answer if action=final_answer")
    missing_info: str | None = Field(default=None, description="What info is missing")


class ValidationResult(BaseModel):
    """Validation of result"""
    is_valid: bool = Field(description="Is the result valid?")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")
    issues: list[str] = Field(default_factory=list, description="Issues if not valid")


# =============================================================================
# AUTO-FOCUS TRACKER - Solves "Blindness" Problem
# =============================================================================

@dataclass
class AutoFocusEntry:
    """Entry for auto-focus tracking"""
    filename: str
    operation: str  # 'created', 'modified', 'tool_result'
    timestamp: float
    preview: str = ""  # First 500 chars


class AutoFocusTracker:
    """
    Tracks the last N modified/created files for automatic context injection.

    This solves the "blindness" problem where the agent doesn't see results
    of its own actions without explicitly opening files.

    Technical Implementation:
    - Maintains a deque of the last 3 file operations
    - On each VFS write/create or tool result, records the operation
    - Builds a context string that's injected into the system prompt
    - Agent sees results immediately without needing vfs_open
    """

    def __init__(self, max_entries: int = 3):
        self.max_entries = max_entries
        self._entries: deque[AutoFocusEntry] = deque(maxlen=max_entries)

    def record(self, filename: str, operation: str, content: str = ""):
        """Record a file modification or creation"""
        entry = AutoFocusEntry(
            filename=filename,
            operation=operation,
            timestamp=time.time(),
            preview=content[:500] if content else ""
        )
        # Remove existing entry for same file if present
        self._entries = deque(
            [e for e in self._entries if e.filename != filename],
            maxlen=self.max_entries
        )
        self._entries.append(entry)

    def get_focus_context(self) -> str:
        """Build context string for injection into system prompt"""
        if not self._entries:
            return ""

        lines = ["\n═══ AUTO-FOCUS: Letzte Aktionsergebnisse ═══"]

        for entry in reversed(list(self._entries)):
            age_secs = time.time() - entry.timestamp
            age_str = f"{int(age_secs)}s ago" if age_secs < 60 else f"{int(age_secs/60)}m ago"

            lines.append(f"\n📄 [{entry.operation.upper()}] {entry.filename} ({age_str}):")
            if entry.preview:
                preview_lines = entry.preview.split('\n')[:10]
                for i, line in enumerate(preview_lines, 1):
                    lines.append(f"  {i:3}| {line}")
                if len(entry.preview.split('\n')) > 10:
                    lines.append(f"  ... ({len(entry.preview.split(chr(10))) - 10} more lines)")
            else:
                lines.append("  (Keine Vorschau verfügbar)")

        lines.append("\n═══════════════════════════════════════════════════════════════════")
        return '\n'.join(lines)

    def clear(self):
        """Clear all tracked entries"""
        self._entries.clear()


# =============================================================================
# INTELLIGENT LOOP DETECTION
# =============================================================================

class LoopDetector:
    """
    Intelligent loop detection with semantic matching.

    Improvements over simple string matching:
    1. Ignores variable fields (timestamps, UUIDs, sequential IDs)
    2. Tracks tool+goal combinations (semantic repetition)
    3. Hard limit: Same tool 3x with similar parameters → forced exit
    """

    VARIABLE_PATTERNS = [
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',  # UUIDs
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',  # ISO timestamps
        r'task_\d+',  # Sequential task IDs
        r'exec_[a-f0-9]+',  # Execution IDs
        r'"id":\s*"[^"]+"',  # JSON id fields
        r'"timestamp":\s*"[^"]+"',  # JSON timestamp fields
        r'"created_at":\s*"[^"]+"',  # JSON created_at fields
        r'"uuid":\s*"[^"]+"',  # JSON uuid fields
    ]

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self._recent_actions: deque[dict] = deque(maxlen=10)
        self._tool_call_counts: dict[str, int] = {}

    def _normalize_args(self, args: dict | None) -> str:
        """Normalize arguments by removing variable fields"""
        if not args:
            return ""
        args_str = json.dumps(args, sort_keys=True)
        for pattern in self.VARIABLE_PATTERNS:
            args_str = re.sub(pattern, '<VAR>', args_str)
        return args_str

    def _extract_goal(self, args: dict | None) -> str:
        """Extract the semantic goal from tool arguments"""
        if not args:
            return ""
        goal_fields = ['message', 'content', 'query', 'task', 'description',
                       'text', 'reminder', 'action', 'command']
        for field in goal_fields:
            if field in args:
                return str(args[field])[:100]
        return ""

    def record_action(self, action: dict) -> None:
        """Record an action for loop detection"""
        tool_name = action.get('tool') or action.get('type', '')
        normalized_args = self._normalize_args(action.get('args'))
        goal = self._extract_goal(action.get('args'))
        signature = f"{tool_name}:{normalized_args}"
        self._tool_call_counts[signature] = self._tool_call_counts.get(signature, 0) + 1
        self._recent_actions.append({
            'tool': tool_name,
            'normalized_args': normalized_args,
            'goal': goal,
            'signature': signature
        })

    def detect_loop(self) -> tuple[bool, str]:
        """Detect if we're stuck in a loop. Returns (is_loop, reason)"""
        if len(self._recent_actions) < 3:
            return False, ""

        recent = list(self._recent_actions)[-5:]

        # Check 1: Same normalized signature called 3+ times
        for sig, count in self._tool_call_counts.items():
            if count >= self.threshold:
                tool = sig.split(':')[0]
                return True, f"Tool '{tool}' wurde {count}x mit ähnlichen Parametern aufgerufen"

        # Check 2: Same tool with same goal 3x in recent history
        tool_goals = [(a['tool'], a['goal']) for a in recent if a['goal']]
        for tg in set(tool_goals):
            if tool_goals.count(tg) >= 3:
                return True, f"Tool '{tg[0]}' wurde 3x für das gleiche Ziel aufgerufen"

        # Check 3: Exact same action type 4x in a row
        if len(recent) >= 4:
            last_4_tools = [a['tool'] for a in recent[-4:]]
            if len(set(last_4_tools)) == 1:
                return True, f"Gleiche Aktion '{last_4_tools[0]}' 4x hintereinander"

        return False, ""

    def reset(self):
        """Reset loop detection state"""
        self._recent_actions.clear()
        self._tool_call_counts.clear()


# =============================================================================
# EXECUTION STATE
# =============================================================================

@dataclass
class ExecutionState:
    """Serializable state for pause/continue"""
    execution_id: str
    query: str
    session_id: str
    phase: ExecutionPhase = ExecutionPhase.INTENT
    iteration: int = 0
    max_iterations: int = 15
    vfs_snapshot: dict = field(default_factory=dict)
    thoughts: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    selected_categories: list[str] = field(default_factory=list)
    selected_tools: list[str] = field(default_factory=list)
    react_goal: str | None = None  # NEW: Goal for ReAct mode
    react_plan: str | None = None  # NEW: Initial plan for ReAct mode
    subtasks: list[dict] = field(default_factory=list)
    subtask_results: dict[str, Any] = field(default_factory=dict)
    active_microagents: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    retry_count: int = 0
    escalated: bool = False
    consecutive_failures: int = 0  # NEW: Track consecutive failures
    tokens_used: int = 0
    token_budget: int = 10000
    waiting_for_human: bool = False
    human_query: str | None = None
    human_response: str | None = None
    final_answer: str | None = None
    success: bool = False
    started_at: datetime = field(default_factory=datetime.now)
    paused_at: datetime | None = None
    resumed_at: datetime | None = None

    def to_checkpoint(self) -> dict:
        """Serialize for storage"""
        data = asdict(self)
        data['phase'] = self.phase.value
        data['started_at'] = self.started_at.isoformat()
        data['paused_at'] = self.paused_at.isoformat() if self.paused_at else None
        data['resumed_at'] = self.resumed_at.isoformat() if self.resumed_at else None
        return data

    @classmethod
    def from_checkpoint(cls, data: dict) -> 'ExecutionState':
        """Restore from storage"""
        data['phase'] = ExecutionPhase(data['phase'])
        data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data['paused_at']:
            data['paused_at'] = datetime.fromisoformat(data['paused_at'])
        if data['resumed_at']:
            data['resumed_at'] = datetime.fromisoformat(data['resumed_at'])
        return cls(**data)


@dataclass
class ExecutionResult:
    """Result of a_run execution"""
    success: bool
    response: str
    execution_id: str
    path_taken: str
    iterations: int = 0
    tools_used: list[str] = field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    duration: float = 0.0
    learned_patterns: list[str] = field(default_factory=list)
    paused: bool = False
    needs_human: bool = False
    human_query: str | None = None


@dataclass
class MicroagentConfig:
    """Configuration for a decomposition microagent"""
    task_id: str
    task_description: str
    dependencies: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    max_iterations: int = 5
    token_budget: int = 2000
    relevant_files: list[str] = field(default_factory=list)


@dataclass
class MicroagentResult:
    """Result from microagent execution"""
    task_id: str
    success: bool
    result: Any = None
    error: str | None = None
    vfs_changes: dict = field(default_factory=dict)
    learned_patterns: list[str] = field(default_factory=list)
    iterations: int = 0
    tokens_used: int = 0
    duration: float = 0.0
    frozen_state: dict | None = None


# =============================================================================
# VFS TOOLS DEFINITIONS (for LiteLLM native mode)
# =============================================================================

VFS_TOOLS_LITELLM = [
    {
        "type": "function",
        "function": {
            "name": "vfs_open",
            "description": "Open a file in VFS to see its content. Use line_start/line_end for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File to open"},
                    "line_start": {"type": "integer", "description": "Start line (1-indexed, optional)"},
                    "line_end": {"type": "integer", "description": "End line (optional, -1 for all)"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_close",
            "description": "Close a file. Creates a summary for later reference.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "File to close"}},
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_read",
            "description": "Read file content without opening in context. Returns content directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File to read"},
                    "line_start": {"type": "integer", "description": "Start line (1-indexed, optional)"},
                    "line_end": {"type": "integer", "description": "End line (optional)"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_view",
            "description": "Change the visible range of an open file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File to view"},
                    "line_start": {"type": "integer", "description": "Start line"},
                    "line_end": {"type": "integer", "description": "End line"}
                },
                "required": ["filename", "line_start", "line_end"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_write",
            "description": "Write/overwrite a file with new content.",
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
            "description": "Create a new file with initial content.",
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
            "name": "vfs_edit",
            "description": "Edit specific lines in a file. Replaces lines from line_start to line_end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File to edit"},
                    "line_start": {"type": "integer", "description": "First line to replace"},
                    "line_end": {"type": "integer", "description": "Last line to replace"},
                    "content": {"type": "string", "description": "New content"}
                },
                "required": ["filename", "line_start", "line_end", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_list",
            "description": "List all files with their state (open/closed) and summaries.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_remove",
            "description": "Remove/delete a file from VFS.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "File to remove"}},
                "required": ["filename"]
            }
        }
    },
]

CONTROL_TOOLS_LITELLM = [
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Provide the final answer. Use IMMEDIATELY when task succeeds. Do NOT repeat actions after success.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "The final answer"}},
                "required": ["answer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "need_info",
            "description": "Indicate that you need more information.",
            "parameters": {
                "type": "object",
                "properties": {"missing": {"type": "string", "description": "What info is missing"}},
                "required": ["missing"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "need_human",
            "description": "Request human assistance.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Question for the human"}},
                "required": ["question"]
            }
        }
    }
]


# =============================================================================
# SYSTEM PROMPTS V2
# =============================================================================

IMMEDIATE_RESPONSE_SYSTEM_PROMPT = """Du bist ein intelligenter Assistent mit Zugang zu Tools und einem virtuellen Dateisystem (VFS).

DEINE FÄHIGKEITEN:
1. Direkte Antwort: Fragen beantworten ohne Tools wenn möglich
2. VFS Operationen: Dateien lesen, schreiben, bearbeiten (ohne ReAct-Loop)
3. Tool Nutzung: um externe Tools aufzurufen wechele in den ReAct-Modus dieser hat weitere tools zurverfügung.

ENTSCHEIDUNGSBAUM:
- Einfache Frage/Antwort → final_answer
- Datei-Operation (lesen/schreiben/listen) → VFS Tools direkt verwenden
- Komplexe Aufgabe mit mehreren Schritten → enter_react_mode
- Sehr komplexe Aufgabe → complex (Dekomposition)

WICHTIGE REGELN:
1. ERFINDE NICHTS - nur Informationen aus VFS oder Tools verwenden
2. Sei effizient - verwende das einfachste Tool für die Aufgabe
3. VFS Operationen sind "One-Shot" - kein Loop nötig
"""


REACT_SYSTEM_PROMPT_V2 = """Du bist ein Agent der in einem VFS (Virtual File System) arbeitet.

═══════════════════════════════════════════════════════════════════
⚡ AUTO-FOCUS: Ergebnisse deiner Aktionen siehst du SOFORT unten!
═══════════════════════════════════════════════════════════════════

KRITISCH - LOOP-VERMEIDUNG:
1. Wenn ein Tool ERFOLG meldet → SOFORT final_answer, NICHT wiederholen!
2. Du siehst Tool-Ergebnisse automatisch im Kontext (Auto-Focus)
3. Gleiche Aktion wiederholen = FEHLER = Loop = Abbruch
4. Arbiter im muster Observe -> Think -> Act -> Observe

WICHTIGE REGELN:
1. Du darfst NUR Informationen verwenden die du:
   - Aus dem VFS gelesen hast
   - Durch Tool-Calls erhalten hast
   - Im Auto-Focus Bereich siehst

2. Wenn du eine Information NICHT hast:
   - Sage "Ich habe keine Information zu X"
   - ERFINDE NICHTS - niemals!

3. Loop-Erkennung ist AKTIV:
   - 3x gleiches Tool mit ähnlichen Parametern → Zwangs-Abbruch
   - Nach Erfolg immer final_answer!

═══════════════════════════════════════════════════════════════════
AKTUELLES ZIEL: {react_goal}
PLAN: {react_plan}
═══════════════════════════════════════════════════════════════════

Deine verfügbaren Tools: {tools}

{auto_focus_context}
"""


# =============================================================================
# EXECUTION ENGINE V2 - Main Class
# =============================================================================

class ExecutionEngine:
    """
    Main execution engine for a_run() - Battle-Proven V2.

    Key Improvements:
    - enter_react_mode: Explicit ReAct transition
    - VFS Tools in immediate response
    - Auto-Focus context injection
    - Intelligent loop detection
    - Crash-resistant LLM calls
    """

    def __init__(
        self,
        agent: 'FlowAgent',
        use_native_tools: bool = True,
        human_online: bool = False,
        intermediate_callback: Callable[[str], None] | None = None
    ):
        self.agent = agent
        self.use_native_tools = use_native_tools
        self.human_online = human_online
        self.intermediate_callback = intermediate_callback
        self._executions: dict[str, ExecutionState] = {}
        self._frozen_microagents: dict[str, dict] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._auto_focus: dict[str, AutoFocusTracker] = {}
        self._loop_detectors: dict[str, LoopDetector] = {}

    def _emit_intermediate(self, message: str):
        """Send intermediate message to user"""
        if self.intermediate_callback:
            self.intermediate_callback(message)
        else:
            print(f"INTERMEDIATE: {message}")

    def _get_auto_focus(self, session_id: str) -> AutoFocusTracker:
        """Get or create auto-focus tracker for session"""
        if session_id not in self._auto_focus:
            self._auto_focus[session_id] = AutoFocusTracker(max_entries=3)
        return self._auto_focus[session_id]

    def _get_loop_detector(self, execution_id: str) -> LoopDetector:
        """Get or create loop detector for execution"""
        if execution_id not in self._loop_detectors:
            self._loop_detectors[execution_id] = LoopDetector(threshold=3)
        return self._loop_detectors[execution_id]

    async def _safe_llm_call(
        self,
        call_func: Callable,
        fallback_response: Any = None,
        max_retries: int = 2,
        context: str = "LLM call"
    ) -> Any:
        """
        Crash-resistant wrapper for LLM calls.
        Handles: Empty responses, JSON parsing errors, Network timeouts
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                result = await call_func()
                if result is None:
                    raise ValueError("Empty response from LLM")
                if hasattr(result, 'content') and result.content is None:
                    if not (hasattr(result, 'tool_calls') and result.tool_calls):
                        raise ValueError("Empty content and no tool calls")
                return result
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                self._emit_intermediate(f"⚠️ {context}: JSON Fehler, Retry {attempt+1}...")
            except ValueError as e:
                last_error = str(e)
                self._emit_intermediate(f"⚠️ {context}: {e}, Retry {attempt+1}...")
            except asyncio.TimeoutError:
                last_error = "Timeout"
                self._emit_intermediate(f"⚠️ {context}: Timeout, Retry {attempt+1}...")
            except Exception as e:
                last_error = str(e)
                self._emit_intermediate(f"⚠️ {context}: {e}, Retry {attempt+1}...")

            if attempt < max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))

        self._emit_intermediate(f"❌ {context}: Fehlgeschlagen nach {max_retries+1} Versuchen")
        if fallback_response is not None:
            return fallback_response
        raise RuntimeError(f"Failed after {max_retries+1} attempts: {last_error}")

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    async def execute(
        self,
        query: str,
        session_id: str = "default",
        execution_id: str | None = None,
        do_stream: bool = False,
        **kwargs
    ) -> ExecutionResult | tuple[Callable[[ExecutionState], AsyncGenerator[ExecutionResult, Any]], ExecutionState]:
        """Main execution entry point."""
        if execution_id and execution_id in self._executions:
            state = self._executions[execution_id]
            if state.waiting_for_human:
                human_response = kwargs.get('human_response')
                if human_response:
                    state.human_response = human_response
                    state.waiting_for_human = False
                    state.resumed_at = datetime.now()
            return await self._continue_execution(state)

        execution_id = execution_id or f"exec_{uuid.uuid4().hex[:12]}"
        state = ExecutionState(
            execution_id=execution_id,
            query=query,
            session_id=session_id,
            max_iterations=kwargs.get('max_iterations', 15),
            token_budget=kwargs.get('token_budget', 10000)
        )
        self._executions[execution_id] = state
        self._get_loop_detector(execution_id)

        if do_stream:
            return self._run_stream_execution, state
        return await self._run_execution(state)

    async def _run_execution(self, state: ExecutionState) -> ExecutionResult:
        """Run execution from current state"""
        start_time = time.perf_counter()

        try:
            session = await self.agent.session_manager.get_or_create(state.session_id)
            state.vfs_snapshot = session.vfs.to_checkpoint()

            if state.phase == ExecutionPhase.INTENT:
                await session.add_message({"role": "user", "content": state.query})

            if state.phase == ExecutionPhase.INTENT:
                for i in range(8):
                    if state.success:
                        break

                    result = await self._immediate_response(state, session)
                    if result is None:
                        state.final_answer = "Es ist ein Fehler aufgetreten."
                        state.phase = ExecutionPhase.FAILED
                        break

                    if not result.tool_calls:
                        state.final_answer = result.content
                        state.phase = ExecutionPhase.COMPLETED
                        state.success = True
                        break

                    if result.content:
                        await session.add_message({"role": "assistant", "content": result.content})
                        self._emit_intermediate(result.content)

                    await self._tool_runner(result, state, session)

            elif state.phase in [ExecutionPhase.CATEGORY_SELECT, ExecutionPhase.TOOL_SELECT, ExecutionPhase.REACT_LOOP]:
                result = await self._tool_path(state, session)

            elif state.phase == ExecutionPhase.DECOMPOSITION:
                result = await self._decomposition_path(state, session)
            else:
                result = state.final_answer or "Execution completed"

            if state.phase == ExecutionPhase.PAUSED:
                return ExecutionResult(
                    success=False, response="", execution_id=state.execution_id,
                    path_taken="paused", paused=True,
                    needs_human=state.waiting_for_human, human_query=state.human_query
                )

            if state.phase != ExecutionPhase.COMPLETED:
                valid = await self._validate_result(state, result)
                if not valid.is_valid:
                    session.vfs.from_checkpoint(state.vfs_snapshot)
                    state.red_flags.extend(valid.issues)

            if state.final_answer:
                await session.add_message({"role": "assistant", "content": state.final_answer})

            if state.success:
                asyncio.create_task(self._learn_from_execution(state, session))

            duration = time.perf_counter() - start_time
            self._cleanup_execution(state.execution_id)

            return ExecutionResult(
                success=state.success,
                response=state.final_answer or result or "",
                execution_id=state.execution_id,
                path_taken=self._get_path_taken(state),
                iterations=state.iteration,
                tools_used=list(set(a.get('tool') for a in state.actions if a.get('tool'))),
                tokens_used=state.tokens_used,
                duration=duration
            )

        except Exception as e:
            session = self.agent.session_manager.get(state.session_id)
            if session and state.vfs_snapshot:
                session.vfs.from_checkpoint(state.vfs_snapshot)
            state.phase = ExecutionPhase.FAILED
            state.red_flags.append(str(e))
            import traceback
            print(traceback.format_exc())
            self._cleanup_execution(state.execution_id)
            return ExecutionResult(
                success=False, response=f"Execution failed: {str(e)}",
                execution_id=state.execution_id, path_taken="failed",
                duration=time.perf_counter() - start_time
            )

    def _cleanup_execution(self, execution_id: str):
        """Cleanup execution resources"""
        if execution_id in self._loop_detectors:
            del self._loop_detectors[execution_id]

    async def _run_stream_execution(self, state: ExecutionState) -> AsyncGenerator[ExecutionResult, Any]:
        """Run execution with streaming"""
        start_time = time.perf_counter()
        try:
            session = await self.agent.session_manager.get_or_create(state.session_id)
            state.vfs_snapshot = session.vfs.to_checkpoint()

            if state.phase == ExecutionPhase.INTENT:
                await session.add_message({"role": "user", "content": state.query})

            result = None
            if state.phase == ExecutionPhase.INTENT:
                for i in range(8):
                    if state.success:
                        break
                    result = await self._immediate_response(state, session)
                    if result is None:
                        state.final_answer = "Es ist ein Fehler aufgetreten."
                        state.phase = ExecutionPhase.FAILED
                        break
                    if not result.tool_calls:
                        state.final_answer = result.content
                        state.phase = ExecutionPhase.COMPLETED
                        state.success = True
                        break
                    if result.content:
                        await session.add_message({"role": "assistant", "content": result.content})
                        yield result.content
                    await self._tool_runner(result, state, session)
            else:
                result = await self._run_execution(state)
                yield result

            if state.phase == ExecutionPhase.PAUSED:
                yield ExecutionResult(
                    success=False, response="", execution_id=state.execution_id,
                    path_taken="paused", paused=True,
                    needs_human=state.waiting_for_human, human_query=state.human_query
                )

            if state.final_answer:
                await session.add_message({"role": "assistant", "content": state.final_answer})

            if state.success:
                task = asyncio.create_task(self._learn_from_execution(state, session))
                self._background_tasks.add(task)

            duration = time.perf_counter() - start_time
            self._cleanup_execution(state.execution_id)

            yield ExecutionResult(
                success=state.success,
                response=state.final_answer or str(result) or "",
                execution_id=state.execution_id,
                path_taken=self._get_path_taken(state),
                iterations=state.iteration,
                tools_used=list(set(a.get('tool') for a in state.actions if a.get('tool'))),
                tokens_used=state.tokens_used,
                duration=duration
            )
        except Exception as e:
            session = self.agent.session_manager.get(state.session_id)
            if session and state.vfs_snapshot:
                session.vfs.from_checkpoint(state.vfs_snapshot)
            state.phase = ExecutionPhase.FAILED
            state.red_flags.append(str(e))
            self._cleanup_execution(state.execution_id)
            yield ExecutionResult(
                success=False, response=f"Execution failed: {str(e)}",
                execution_id=state.execution_id, path_taken="failed",
                duration=time.perf_counter() - start_time
            )

    # =========================================================================
    # TOOL RUNNER
    # =========================================================================

    async def _tool_runner(self, result, state: ExecutionState, session) -> None:
        """Process tool calls from LLM response"""
        if result.tool_calls is None:
            return

        for tool_call in result.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            tool_name = tool_call.function.name
            self._emit_intermediate(f"Verwende {tool_name}...")

            if tool_name == "final_answer":
                state.final_answer = args.get("final_answer") or args.get("answer")
                state.phase = ExecutionPhase.COMPLETED
                state.success = True
                break

            elif tool_name == "enter_react_mode":
                state.react_goal = args.get("goal", state.query)
                state.react_plan = args.get("initial_plan", "")
                state.phase = ExecutionPhase.CATEGORY_SELECT
                requested_tools = args.get("tools", [])
                tool_result = await self._tool_path(state, session, requested_tools)
                await session.add_message({
                    "role": "tool", "tool_call_id": tool_call.id, "content": tool_result
                })

            elif tool_name == "use_tools":
                state.react_goal = state.query
                state.react_plan = f"Tools: {args.get('tools', [])}"
                state.phase = ExecutionPhase.CATEGORY_SELECT
                tool_result = await self._tool_path(state, session, args.get("tools"))
                await session.add_message({
                    "role": "tool", "tool_call_id": tool_call.id, "content": tool_result
                })

            elif tool_name == "complex":
                state.phase = ExecutionPhase.DECOMPOSITION
                tool_result = await self._decomposition_path(state, session)
                await session.add_message({
                    "role": "tool", "tool_call_id": tool_call.id, "content": tool_result
                })

            elif tool_name == "need_human" and self.human_online:
                state.phase = ExecutionPhase.PAUSED
                state.paused_at = datetime.now()
                state.waiting_for_human = tool_call.id
                state.human_query = args.get("question")
                state.success = True
                break

            elif tool_name == "think":
                state.thoughts.append(args.get("thought"))
                await session.add_message({
                    "role": "tool", "tool_call_id": tool_call.id, "content": args.get("thought")
                })

            elif tool_name == "list_tools":
                tools = self.agent.tool_manager.get_all()
                tools += self._get_system_tool_entries()
                category = args.get("category")
                search = args.get("search")
                if category:
                    tools = [t for t in tools if t.has_category(category)]
                if search:
                    search_lower = search.lower()
                    tools = [t for t in tools if
                             search_lower in t.name.lower() or search_lower in t.description.lower()]
                res = "\n".join(f"{t.name} - {t.description} ({t.category})" for t in tools)
                await session.add_message({
                    "role": "tool", "tool_call_id": tool_call.id, "content": res
                })

            elif tool_name.startswith("vfs_"):
                tool_result = await self._execute_vfs_action(session, tool_name, args, state)
                await session.add_message({
                    "role": "tool", "tool_call_id": tool_call.id, "content": str(tool_result)
                })

            else:
                state.phase = ExecutionPhase.CATEGORY_SELECT
                state.query += f"\nUse tool {tool_name} with infos {args.values()}"
                tool_result = await self._tool_path(state, session, [tool_name])
                await session.add_message({
                    "role": "tool", "tool_call_id": tool_call.id, "content": tool_result
                })

    def _get_system_tool_entries(self) -> list:
        """Get ToolEntry objects for system tools"""
        return [
            ToolEntry(name="enter_react_mode",
                     description="Enter ReAct mode for complex multi-step tasks.",
                     args_schema="(goal: str, initial_plan: str, tools: list[str])", category=["system"]),
            ToolEntry(name="use_tools", description="Legacy: run a ReAct loop",
                     args_schema="(tools: list[str])", category=["system"]),
            ToolEntry(name="complex", description="Request decomposition",
                     args_schema="()", category=["system"]),
            ToolEntry(name="need_human", description="Request human assistance",
                     args_schema="(question: str)", category=["system"]),
            ToolEntry(name="think", description="Reason over data",
                     args_schema="(thought: str)", category=["system"]),
            ToolEntry(name="list_tools", description="List available tools",
                     args_schema="(category: str, search: str)", category=["system"]),
        ] + [
            ToolEntry(name=t['function']['name'], description=t['function']['description'],
                     args_schema=t['function']['parameters'], category=["vfs"])
            for t in VFS_TOOLS_LITELLM
        ]

    async def _execute_vfs_action(self, session, action_type: str, args: dict, state: ExecutionState) -> Any:
        """Execute VFS action and track for auto-focus"""
        auto_focus = self._get_auto_focus(state.session_id)
        result = None

        if action_type == 'vfs_open':
            result = session.vfs.open(args.get('filename'), args.get('line_start', 1), args.get('line_end', -1))
        elif action_type == 'vfs_read':
            res = session.vfs.read(args.get('filename'))
            if res.get('success'):
                content = res['content']
                if args.get('line_start') or args.get('line_end'):
                    lines = content.split('\n')
                    start = args.get('line_start', 1) - 1
                    end = args.get('line_end', len(lines))
                    content = '\n'.join(lines[start:end])
                result = content
                auto_focus.record(args.get('filename'), 'read', content)
            else:
                result = res.get('error', 'Read failed')
        elif action_type == 'vfs_close':
            result = await session.vfs.close(args.get('filename'))
        elif action_type == 'vfs_view':
            f = session.vfs.files.get(args.get('filename'))
            if f:
                f.view_start = max(0, args.get('line_start', 1) - 1)
                f.view_end = args.get('line_end', -1)
                session.vfs._dirty = True
            result = {"success": True, "message": "View range updated"}
        elif action_type == 'vfs_write':
            filename = args.get('filename')
            content = args.get('content', '')
            result = session.vfs.write(filename, content)
            if result.get('success'):
                auto_focus.record(filename, 'modified', content)
        elif action_type == 'vfs_create':
            filename = args.get('filename')
            content = args.get('content', '')
            result = session.vfs.create(filename, content)
            if result.get('success'):
                auto_focus.record(filename, 'created', content)
        elif action_type == 'vfs_edit':
            filename = args.get('filename')
            result = session.vfs.edit(filename, args.get('line_start', 1), args.get('line_end', 1), args.get('content', ''))
            if result.get('success'):
                res = session.vfs.read(filename)
                if res.get('success'):
                    auto_focus.record(filename, 'modified', res['content'])
        elif action_type == 'vfs_list':
            result = session.vfs.list_files()
        elif action_type == 'vfs_remove':
            result = session.vfs.remove(args.get('filename'))
        return result or {"error": f"Unknown VFS action: {action_type}"}

    async def _continue_execution(self, state: ExecutionState) -> ExecutionResult:
        """Continue a paused execution"""
        state.resumed_at = datetime.now()
        return await self._run_execution(state)

    # =========================================================================
    # IMMEDIATE PATH (with VFS tools)
    # =========================================================================

    async def _immediate_response(self, state: ExecutionState, session) -> ModelResponse | None:
        """Direct response with VFS tools available."""
        self._emit_intermediate("Verarbeite Anfrage...")

        auto_focus = self._get_auto_focus(state.session_id)
        auto_focus_context = auto_focus.get_focus_context()

        messages = session.get_history_for_llm(last_n=15)

        system_prompt = IMMEDIATE_RESPONSE_SYSTEM_PROMPT
        if auto_focus_context:
            system_prompt += f"\n{auto_focus_context}"

        self.agent.amd.system_message = system_prompt


        default_tools = [
            {"type": "function", "function": {
                "name": "final_answer",
                "description": "Provide an immediate answer.",
                "parameters": {"type": "object", "properties": {
                    "final_answer": {"type": "string", "description": "Your final answer"}
                }, "required": ["final_answer"]}
            }},
            {"type": "function", "function": {
                "name": "enter_react_mode",
                "description": "Enter ReAct mode.",
                "parameters": {"type": "object", "properties": {
                    "goal": {"type": "string", "description": "The goal to achieve"},
                    "initial_plan": {"type": "string", "description": "Initial plan"},
                    "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools needed"}
                }, "required": ["goal"]}
            }},
            {"type": "function", "function": {
                "name": "complex",
                "description": "Task needs decomposition.",
                "parameters": {"type": "object", "properties": {
                    "is_complex": {"type": "boolean", "description": "True if complex"}
                }, "required": ["is_complex"]}
            }},
            {"type": "function", "function": {
                "name": "think",
                "description": "Reason before answering.",
                "parameters": {"type": "object", "properties": {
                    "thought": {"type": "string", "description": "Your thought"}
                }, "required": ["thought"]}
            }},
            {"type": "function", "function": {
                "name": "list_tools",
                "description": "List available tools.",
                "parameters": {"type": "object", "properties": {
                    "category": {"type": "string", "description": "Category filter"},
                    "search": {"type": "string", "description": "Search query"}
                }}
            }}
        ]

        vfs_immediate_tools = [t for t in VFS_TOOLS_LITELLM if t['function']['name'] in [
            'vfs_read', 'vfs_write', 'vfs_create', 'vfs_list', 'vfs_remove'
        ]]
        default_tools.extend(vfs_immediate_tools)

        if self.human_online:
            default_tools.append({"type": "function", "function": {
                "name": "need_human",
                "description": "Request human assistance.",
                "parameters": {"type": "object", "properties": {
                    "question": {"type": "string", "description": "Question for human"}
                }, "required": ["question"]}
            }})

        async def make_call():
            print(messages)
            return await self.agent.a_run_llm_completion(
                tools=default_tools, tool_choice="auto", messages=messages,
                model_preference="fast", with_context=True, stream=False,
                task_id=f"{state.execution_id}_immediate",
                session_id=state.session_id, get_response_message=True
            )

        try:
            response = await self._safe_llm_call(make_call, fallback_response=None, max_retries=2, context="_immediate_response")
            state.phase = ExecutionPhase.COMPLETED
            return response
        except RuntimeError as e:
            state.red_flags.append(f"Immediate response failed: {e}")
            return None

    # =========================================================================
    # TOOL PATH
    # =========================================================================

    async def _tool_path(self, state: ExecutionState, session, tool_names: list[str] | None = None) -> str:
        """Execute with tools via ReAct loop"""
        if tool_names:
            available_tools = self.agent.tool_manager.list_names()
            if set(tool_names).issubset(set(available_tools)):
                state.selected_tools = tool_names
                state.phase = ExecutionPhase.REACT_LOOP

        if state.phase == ExecutionPhase.CATEGORY_SELECT:
            await self._select_categories(state, session)
            state.phase = ExecutionPhase.TOOL_SELECT

        if state.phase == ExecutionPhase.TOOL_SELECT:
            await self._select_tools(state, session)
            state.phase = ExecutionPhase.REACT_LOOP

        if state.phase == ExecutionPhase.REACT_LOOP:
            result = await self._react_loop(state, session)
            return result

        return state.final_answer or ""

    async def _select_categories(self, state: ExecutionState, session):
        """Select relevant tool categories"""
        self._emit_intermediate("Wähle relevante Tool-Kategorien...")
        categories = self.agent.tool_manager.list_categories()
        if not categories:
            state.red_flags.append("No tool categories available")
            return

        if len(categories) <= 2:
            state.selected_categories = categories
            return

        prompt = f"""Welche Tool-Kategorien sind für diese Aufgabe relevant?
Aufgabe: {state.query}
Verfügbare Kategorien: {', '.join(categories)}
Wähle nur die wirklich relevanten Kategorien."""

        async def make_call():
            return await self.agent.a_format_class(CategorySelection, prompt, model_preference="fast")

        try:
            result = await self._safe_llm_call(
                make_call, fallback_response={'categories': categories[:3], 'reasoning': 'Fallback'},
                max_retries=2, context="_select_categories"
            )
            state.selected_categories = result['categories']
        except Exception as e:
            state.red_flags.append(f"Category selection failed: {e}")
            state.selected_categories = categories[:3]

    async def _select_tools(self, state: ExecutionState, session):
        """Select specific tools (max 5)"""
        tools = self.agent.tool_manager.get_by_category(*state.selected_categories)
        if len(tools) <= 5:
            state.selected_tools = [t.name for t in tools]
            return

        self._emit_intermediate("Wähle die besten Tools aus...")
        tool_list = "\n".join([f"- {t.name}: {t.description[:100]}" for t in tools])

        prompt = f"""Wähle max 5 Tools für diese Aufgabe:
Aufgabe: {state.query}
Verfügbare Tools:
{tool_list}"""

        async def make_call():
            return await self.agent.a_format_class(ToolSelection, prompt, model_preference="fast")

        try:
            result = await self._safe_llm_call(
                make_call, fallback_response={'tools': [t.name for t in tools[:5]]},
                max_retries=2, context="_select_tools"
            )
            state.selected_tools = result['tools'][:5]
        except Exception as e:
            state.red_flags.append(f"Tool selection failed: {e}")
            state.selected_tools = [t.name for t in tools[:5]]

    async def _react_loop(self, state: ExecutionState, session) -> str:
        """Main ReAct loop with Auto-Focus and intelligent loop detection"""
        loop_detector = self._get_loop_detector(state.execution_id)
        auto_focus = self._get_auto_focus(state.session_id)

        while state.iteration < state.max_iterations:
            if state.tokens_used >= state.token_budget:
                state.red_flags.append("Token budget exceeded")
                break
            if state.phase == ExecutionPhase.PAUSED:
                return ""

            state.iteration += 1
            self._emit_intermediate(f"Schritt {state.iteration}...")

            vfs_context = session.build_vfs_context() if hasattr(session, 'build_vfs_context') else ""
            auto_focus_context = auto_focus.get_focus_context()

            action = None
            if self.use_native_tools:
                try:
                    action = await self._get_action_native(state, session, vfs_context, auto_focus_context)
                except Exception as e:
                    self.use_native_tools = False
                    self._emit_intermediate(f"Error using native tools: {e}")

            if action is None:
                action = await self._get_action_format(state, session, vfs_context)

            if action is None:
                state.red_flags.append("Failed to get action")
                state.consecutive_failures += 1
                if state.consecutive_failures >= 3:
                    break
                continue

            state.consecutive_failures = 0
            loop_detector.record_action(action)
            result = await self._execute_action(state, session, action)

            # await session.add_message({
            #     "role": "system", "content": f"action: {action}, result: {result}"
            # })

            state.actions.append(action)
            state.observations.append({"role": "tool", "content": str(result)[:2000]})

            if action.get('type') in [ActionType.FINAL_ANSWER.value, 'final_answer']:
                state.final_answer = action.get('answer', '')
                state.success = True
                state.phase = ExecutionPhase.COMPLETED
                return state.final_answer

            if action.get('type') in [ActionType.NEED_HUMAN.value, 'need_human']:
                state.waiting_for_human = True
                state.human_query = action.get('question', 'Können Sie mir helfen?')
                state.phase = ExecutionPhase.PAUSED
                state.paused_at = datetime.now()
                return ""

            if action.get('type') in [ActionType.NEED_INFO.value, 'need_info']:
                missing = action.get('missing', 'Unbekannt')
                self._emit_intermediate(f"Mir fehlt: {missing}")
                state.phase = ExecutionPhase.PAUSED
                state.paused_at = datetime.now()
                return ""

            is_loop, loop_reason = loop_detector.detect_loop()
            if is_loop:
                state.red_flags.append(f"Loop detected: {loop_reason}")
                self._emit_intermediate(f"⚠️ Loop erkannt: {loop_reason}")
                handled = await self._handle_error(state, session, loop_reason)
                if not handled:
                    if state.observations:
                        state.final_answer = f"Aufgabe teilweise abgeschlossen. Letzte Beobachtung: {state.observations[-1]}"
                    else:
                        state.final_answer = "Konnte die Aufgabe nicht abschließen (Loop erkannt)."
                    state.phase = ExecutionPhase.COMPLETED
                    break

        if not state.final_answer:
            state.final_answer = f"Konnte die Aufgabe nicht in {state.max_iterations} Schritten abschließen."
            state.success = False

        state.phase = ExecutionPhase.COMPLETED
        return state.final_answer

    async def _get_action_native(self, state: ExecutionState, session, vfs_context: str, auto_focus_context: str = "") -> dict | None:
        """Get next action using LiteLLM native tool calling with Auto-Focus"""
        system_prompt = REACT_SYSTEM_PROMPT_V2.format(
            max_open_files=5, tools=", ".join(state.selected_tools or []),
            react_goal=state.react_goal or state.query,
            react_plan=state.react_plan or "Kein initialer Plan",
            auto_focus_context=auto_focus_context,
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": state.query}]
        if state.thoughts:
            for thought, obs in zip(state.thoughts, state.observations):
                messages.append({"role": "assistant", "content": thought})
                messages.append(obs)
        elif state.observations:
            messages.extend(state.observations)

        selected_tools_litellm = [
            t for t in self.agent.tool_manager.get_all_litellm()
            if any(t['function']['name'] == name for name in (state.selected_tools or []))
        ]
        all_tools = VFS_TOOLS_LITELLM + CONTROL_TOOLS_LITELLM + selected_tools_litellm

        async def make_call():
            return await self.agent.a_run_llm_completion(
                messages=messages, tools=all_tools, tool_choice="auto",
                model_preference="fast", stream=False, get_response_message=True,
                task_id=f"{state.execution_id}_react_{state.iteration}",
                session_id=state.session_id, with_context=False
            )

        try:
            response = await self._safe_llm_call(make_call, fallback_response=None, max_retries=2,
                                                  context=f"_get_action_native (iter {state.iteration})")
        except RuntimeError:
            return None

        if response is None:
            return None

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tc = response.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments or "{}")
            except:
                args = {}
            return {'type': tc.function.name, 'tool': tc.function.name, 'args': args, **args}

        if hasattr(response, 'content') and response.content:
            state.thoughts.append(response.content)
            return {'type': 'thinking', 'thought': response.content}

        return None

    async def _get_action_format(self, state: ExecutionState, session, vfs_context: str) -> dict | None:
        """Get next action using a_format_class (fallback)"""
        prompt = f"""Du bearbeitest diese Aufgabe: {state.query}
Aktueller VFS Status: {vfs_context}
Bisherige Aktionen: {len(state.actions)}
{chr(10).join([s.get('content', '') for s in state.observations[-3:]]) if state.observations else "Keine"}
Deine verfügbaren Tools: {', '.join(state.selected_tools or [])}
WICHTIG: Wenn eine Aktion erfolgreich war, verwende final_answer!"""

        async def make_call():
            return await self.agent.a_format_class(ThoughtAction, prompt, model_preference="fast")

        try:
            result = await self._safe_llm_call(
                make_call, fallback_response={'thought': 'Fallback', 'action': 'final_answer', 'answer': 'Konnte nicht fortfahren'},
                max_retries=2, context="_get_action_format"
            )
        except RuntimeError:
            return None

        state.thoughts.append(result.get('thought', ''))
        return {'type': result.get('action'), 'tool': result.get('tool_name'), 'args': result.get('tool_args'), **result}

    async def _execute_action(self, state: ExecutionState, session, action: dict) -> Any:
        """Execute an action with auto-focus tracking"""
        action_type = action.get('type', '')
        auto_focus = self._get_auto_focus(state.session_id)

        if action_type.startswith('vfs_'):
            return await self._execute_vfs_action(session, action_type, action, state)

        elif action_type == 'tool_call' or action.get('tool') in (state.selected_tools or []):
            tool_name = action.get('tool') or action.get('tool_name')
            tool_args = action.get('args') or action.get('tool_args') or {}
            self._emit_intermediate(f"Verwende {tool_name}...")
            try:
                result = await self.agent.arun_function(tool_name, **tool_args)
                result_str = str(result)[:2000]
                result_file = f"tool_result_{state.iteration}"
                session.vfs.create(result_file, result_str)
                auto_focus.record(result_file, 'tool_result', result_str)
                return result
            except Exception as e:

                import traceback
                traceback.print_exc()
                print("⚠️  Error in _execute_action")
                return {"error": str(e)}

        elif action_type == 'final_answer':
            return {"final": action.get('answer')}
        elif action_type in ['need_info', 'need_human']:
            return {"status": action_type, "detail": action.get('missing') or action.get('question')}
        elif action_type == 'thinking':
            return {"status": "thinking"}
        else:
            try:
                return await self.agent.arun_function(action_type, **action)
            except Exception as e:
                return {"error": f"Unknown action type: {action_type}, error: {e}"}

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    async def _handle_error(self, state: ExecutionState, session, error: str) -> bool:
        """Handle errors: B → A → A → D"""
        if not state.escalated:
            self._emit_intermediate("Versuche mit stärkerem Modell...")
            state.escalated = True
            return True

        if state.retry_count < 2:
            self._emit_intermediate(f"Wiederhole... (Versuch {state.retry_count + 1})")
            state.retry_count += 1
            return True

        if self.human_online:
            state.waiting_for_human = True
            state.human_query = f"Ich stecke fest: {error}. Können Sie mir helfen?"
            state.phase = ExecutionPhase.PAUSED
            state.paused_at = datetime.now()
            return True

        return False

    # =========================================================================
    # DECOMPOSITION PATH
    # =========================================================================

    async def _decomposition_path(self, state: ExecutionState, session) -> str:
        """Execute complex task with parallel microagents"""
        if not state.subtasks:
            await self._decompose_task(state, session)

        self._emit_intermediate(f"Führe {len(state.subtasks)} Teilaufgaben aus...")
        parallel_groups = self._group_subtasks(state.subtasks)

        for group_idx, group in enumerate(parallel_groups):
            self._emit_intermediate(f"Gruppe {group_idx + 1}/{len(parallel_groups)}...")
            if state.phase == ExecutionPhase.PAUSED:
                return ""

            results = await self._execute_parallel_group(state, session, group)
            if any(r.frozen_state for r in results):
                state.phase = ExecutionPhase.PAUSED
                state.paused_at = datetime.now()
                return ""

            for result in results:
                state.subtask_results[result.task_id] = result
                if result.vfs_changes:
                    for filename, content in result.vfs_changes.items():
                        session.vfs.write(filename, content)

        final_result = await self._aggregate_results(state, session)
        state.final_answer = final_result
        state.success = True
        state.phase = ExecutionPhase.COMPLETED
        return final_result

    async def _decompose_task(self, state: ExecutionState, session):
        """Decompose complex task into subtasks"""
        self._emit_intermediate("Zerlege Aufgabe in Teilschritte...")
        prompt = f"""Zerlege diese Aufgabe in einfache Teilschritte:
Aufgabe: {state.query}
Erstelle eine Liste von Teilaufgaben. Markiere welche parallel ausgeführt werden können."""

        async def make_call():
            return await self.agent.a_format_class(TaskDecomposition, prompt, model_preference="fast")

        try:
            result = await self._safe_llm_call(make_call, fallback_response={'subtasks': [state.query], 'can_parallel': [False]}, max_retries=2, context="_decompose_task")
        except RuntimeError:
            result = {'subtasks': [state.query], 'can_parallel': [False]}

        for i, (desc, can_par) in enumerate(zip(result['subtasks'], result['can_parallel'])):
            state.subtasks.append({'id': f"task_{i}", 'description': desc, 'can_parallel': can_par, 'status': 'pending'})

    def _group_subtasks(self, subtasks: list[dict]) -> list[list[dict]]:
        """Group subtasks for parallel execution"""
        groups = []
        current_group = []
        for task in subtasks:
            if task.get('can_parallel', False) and current_group:
                current_group.append(task)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [task]
        if current_group:
            groups.append(current_group)
        return groups

    async def _execute_parallel_group(self, state: ExecutionState, session, group: list[dict]) -> list[MicroagentResult]:
        """Execute a group of subtasks in parallel"""
        tasks = []
        for subtask in group:
            task = asyncio.create_task(self._run_microagent(state, session, subtask))
            tasks.append(task)
            state.active_microagents.append(subtask['id'])

        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=60.0)
        except asyncio.TimeoutError:
            results = []
            for task, subtask in zip(tasks, group):
                if task.done():
                    results.append(task.result())
                else:
                    task.cancel()
                    frozen = MicroagentResult(task_id=subtask['id'], success=False, frozen_state={'subtask': subtask})
                    results.append(frozen)
                    self._frozen_microagents[subtask['id']] = frozen.frozen_state

        for subtask in group:
            if subtask['id'] in state.active_microagents:
                state.active_microagents.remove(subtask['id'])

        return [r for r in results if isinstance(r, MicroagentResult)]

    async def _run_microagent(self, state: ExecutionState, session, subtask: dict) -> MicroagentResult:
        """Run a single microagent"""
        start_time = time.perf_counter()
        task_id = subtask['id']

        micro_vfs = await self._create_micro_vfs(session, subtask)
        tools = await self._select_tools_for_subtask(subtask)

        config = MicroagentConfig(task_id=task_id, task_description=subtask['description'], tools=tools, max_iterations=5, token_budget=2000)

        micro_state = ExecutionState(
            execution_id=f"{state.execution_id}_{task_id}", query=subtask['description'],
            session_id=state.session_id, max_iterations=config.max_iterations,
            token_budget=config.token_budget, selected_tools=config.tools
        )
        micro_state.phase = ExecutionPhase.REACT_LOOP

        try:
            class MicroSession:
                def __init__(self, vfs):
                    self.vfs = vfs
                def build_vfs_context(self):
                    return self.vfs.build_context_string()

            micro_session = MicroSession(micro_vfs)
            result = await self._react_loop(micro_state, micro_session)

            vfs_changes = {}
            for filename, file in micro_vfs.files.items():
                if not file.readonly:
                    vfs_changes[f"{task_id}_{filename}"] = file.content

            return MicroagentResult(
                task_id=task_id, success=micro_state.success, result=result,
                vfs_changes=vfs_changes, iterations=micro_state.iteration,
                tokens_used=micro_state.tokens_used, duration=time.perf_counter() - start_time
            )
        except Exception as e:
            return MicroagentResult(task_id=task_id, success=False, error=str(e), duration=time.perf_counter() - start_time)

    async def _create_micro_vfs(self, session, subtask: dict):
        """Create isolated VFS with only relevant files"""
        from toolboxv2.mods.isaa.base.Agent.agent_session import VirtualFileSystem
        micro_vfs = VirtualFileSystem(session_id=f"micro_{subtask['id']}", agent_name="microagent", max_window_lines=100)
        desc_lower = subtask['description'].lower()
        for filename, file in session.vfs.files.items():
            if file.readonly:
                continue
            if filename.lower() in desc_lower or any(word in desc_lower for word in filename.lower().split('_')):
                micro_vfs.create(filename, file.content)
        return micro_vfs

    async def _select_tools_for_subtask(self, subtask: dict) -> list[str]:
        """Select relevant tools for a subtask"""
        prompt = f"""Welche Tools brauche ich für diese Aufgabe?
Aufgabe: {subtask['description']}
Verfügbare Tools: {', '.join(self.agent.tool_manager.list_names())}
Wähle max 3 relevante Tools."""

        async def make_call():
            return await self.agent.a_format_class(ToolSelection, prompt, model_preference="fast")

        try:
            result = await self._safe_llm_call(make_call, fallback_response={'tools': []}, max_retries=1, context="_select_tools_for_subtask")
            return result['tools'][:3]
        except RuntimeError:
            return []

    async def _aggregate_results(self, state: ExecutionState, session) -> str:
        """Aggregate microagent results into final answer"""
        results_summary = "\n".join([f"- {task_id}: {'✓' if r.success else '✗'} {r.result or r.error}" for task_id, r in state.subtask_results.items()])
        prompt = f"""Fasse die Ergebnisse zusammen:
Ursprüngliche Aufgabe: {state.query}
Teilergebnisse:
{results_summary}
Erstelle eine zusammenhängende Antwort."""

        async def make_call():
            return await self.agent.a_run_llm_completion(messages=[{"role": "user", "content": prompt}], model_preference="fast", stream=False, task_id=f"{state.execution_id}_aggregate", session_id=state.session_id)

        try:
            response = await self._safe_llm_call(make_call, fallback_response=results_summary, max_retries=2, context="_aggregate_results")
            return response
        except RuntimeError:
            return results_summary

    # =========================================================================
    # VALIDATION & LEARNING
    # =========================================================================

    async def _validate_result(self, state: ExecutionState, result: str) -> ValidationResult:
        """Validate the execution result"""
        if not result:
            return ValidationResult(is_valid=False, confidence=0.0, issues=["Empty result"])
        if not isinstance(result, str):
            return ValidationResult(is_valid=False, confidence=0.0, issues=["Invalid result type"])

        prompt = f"""Validiere dieses Ergebnis:
Aufgabe: {state.query}
Ergebnis: {result[:2000]}
Ist das Ergebnis vollständig und korrekt?"""

        async def make_call():
            return await self.agent.a_format_class(ValidationResult, prompt, model_preference="fast")

        try:
            valid = await self._safe_llm_call(make_call, fallback_response={'is_valid': True, 'confidence': 0.5, 'issues': []}, max_retries=1, context="_validate_result")
            return ValidationResult(**valid)
        except:
            return ValidationResult(is_valid=True, confidence=0.5, issues=[])

    async def _learn_from_execution(self, state: ExecutionState, session):
        """Non-blocking learning after successful execution"""
        try:
            for action in state.actions:
                tool_name = action.get('tool')
                if tool_name:
                    entry = self.agent.tool_manager.get(tool_name)
                    if entry:
                        entry.record_call()

            if state.success and hasattr(session, 'rule_set') and session.rule_set:
                pattern = f"Query type: {state.query[:50]}... → Path: {self._get_path_taken(state)}"
                session.rule_set.learn_pattern(pattern=pattern, source_situation=state.query[:100], confidence=0.6)
                for rule in session.rule_set.get_active_rules():
                    session.rule_set.record_rule_success(rule.id)
                import random
                if random.random() < 0.1:
                    session.rule_set.prune_low_confidence_patterns(threshold=0.2)
        except Exception:
            import traceback
            traceback.print_exc()
            print("⚠️  Error in _learn_from_execution")
            pass

    def _get_path_taken(self, state: ExecutionState) -> str:
        """Determine which path was taken"""
        if state.subtasks:
            return "decomposition"
        elif state.selected_tools:
            return "tool"
        else:
            return "immediate"

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def pause(self, execution_id: str) -> ExecutionState | None:
        """Pause an execution"""
        state = self._executions.get(execution_id)
        if state and state.phase not in [ExecutionPhase.COMPLETED, ExecutionPhase.FAILED]:
            state.phase = ExecutionPhase.PAUSED
            state.paused_at = datetime.now()
            return state
        return None

    def get_state(self, execution_id: str) -> ExecutionState | None:
        """Get execution state"""
        return self._executions.get(execution_id)

    def list_executions(self) -> list[dict]:
        """List all executions"""
        return [
            {'id': state.execution_id, 'query': state.query[:50], 'phase': state.phase.value,
             'iteration': state.iteration, 'waiting_for_human': state.waiting_for_human}
            for state in self._executions.values()
        ]

    async def cancel(self, execution_id: str) -> bool:
        """Cancel and cleanup an execution"""
        if execution_id in self._executions:
            state = self._executions[execution_id]
            session = self.agent.session_manager.get(state.session_id)
            if session and state.vfs_snapshot:
                session.vfs.from_checkpoint(state.vfs_snapshot)
            state.phase = ExecutionPhase.FAILED
            self._cleanup_execution(execution_id)
            del self._executions[execution_id]
            return True
        return False

    def get_auto_focus_context(self, session_id: str) -> str:
        """Get current auto-focus context for debugging"""
        return self._get_auto_focus(session_id).get_focus_context()

    def clear_auto_focus(self, session_id: str) -> None:
        """Clear auto-focus for a session"""
        if session_id in self._auto_focus:
            self._auto_focus[session_id].clear()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_execution_engine(
    agent: 'FlowAgent',
    use_native_tools: bool = True,
    human_online: bool = False,
    intermediate_callback: Callable[[str], None] | None = None
) -> ExecutionEngine:
    """Factory function to create ExecutionEngine with sensible defaults"""
    return ExecutionEngine(
        agent=agent,
        use_native_tools=use_native_tools,
        human_online=human_online,
        intermediate_callback=intermediate_callback
    )
