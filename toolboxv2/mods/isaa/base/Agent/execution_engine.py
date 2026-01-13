"""
a_run() Execution System for FlowAgent V2

Components:
- IntentDetector: Determines execution path
- ReActLoop: RLM-VFS style execution
- Decomposer: MAKER-style parallel microagents
- TransactionManager: Simple rollback support
- ExecutionState: Pause/Continue support

Author: FlowAgent V2
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING, AsyncGenerator, Coroutine

from litellm.types.utils import ModelResponse
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

    # For tool_call
    tool_name: str | None = Field(default=None, description="Tool name if action=tool_call")
    tool_args: dict | None = Field(default=None, description="Tool args if action=tool_call")

    # For VFS operations
    filename: str | None = Field(default=None, description="Filename for VFS ops")
    content: str | None = Field(default=None, description="Content for write/edit")
    line_start: int | None = Field(default=None, description="Start line for view/edit")
    line_end: int | None = Field(default=None, description="End line for view/edit")

    # For final_answer
    answer: str | None = Field(default=None, description="Final answer if action=final_answer")

    # For need_info/need_human
    missing_info: str | None = Field(default=None, description="What info is missing")


class ValidationResult(BaseModel):
    """Validation of result"""
    is_valid: bool = Field(description="Is the result valid?")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")
    issues: list[str] = Field(default_factory=list, description="Issues if not valid")


# =============================================================================
# EXECUTION STATE (Pause/Continue)
# =============================================================================

@dataclass
class ExecutionState:
    """Serializable state for pause/continue"""

    # Identity
    execution_id: str
    query: str
    session_id: str

    # Current state
    phase: ExecutionPhase = ExecutionPhase.INTENT
    iteration: int = 0
    max_iterations: int = 15

    # VFS Transaction
    vfs_snapshot: dict = field(default_factory=dict)

    # ReAct state
    thoughts: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    selected_categories: list[str] = field(default_factory=list)
    selected_tools: list[str] = field(default_factory=list)

    # Decomposition state
    subtasks: list[dict] = field(default_factory=list)
    subtask_results: dict[str, Any] = field(default_factory=dict)
    active_microagents: list[str] = field(default_factory=list)

    # Error handling
    red_flags: list[str] = field(default_factory=list)
    retry_count: int = 0
    escalated: bool = False

    # Token tracking
    tokens_used: int = 0
    token_budget: int = 10000

    # Human interaction
    waiting_for_human: bool = False
    human_query: str | None = None
    human_response: str | None = None

    # Result
    final_answer: str | None = None
    success: bool = False

    # Timestamps
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
    path_taken: str  # "immediate", "tool", "decomposition"

    # Stats
    iterations: int = 0
    tools_used: list[str] = field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    duration: float = 0.0

    # For learning
    learned_patterns: list[str] = field(default_factory=list)

    # Special states
    paused: bool = False
    needs_human: bool = False
    human_query: str | None = None


# =============================================================================
# MICROAGENT (For Decomposition)
# =============================================================================

@dataclass
class MicroagentConfig:
    """Configuration for a decomposition microagent"""
    task_id: str
    task_description: str
    dependencies: list[str] = field(default_factory=list)

    # Pre-selected tools
    tools: list[str] = field(default_factory=list)

    # Limits (stricter for speed)
    max_iterations: int = 5
    token_budget: int = 2000

    # Relevant files to copy
    relevant_files: list[str] = field(default_factory=list)


@dataclass
class MicroagentResult:
    """Result from microagent execution"""
    task_id: str
    success: bool
    result: Any = None
    error: str | None = None

    # Changes to merge
    vfs_changes: dict = field(default_factory=dict)
    learned_patterns: list[str] = field(default_factory=list)

    # Stats
    iterations: int = 0
    tokens_used: int = 0
    duration: float = 0.0

    # State for freeze/resume
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
                "properties": {
                    "filename": {"type": "string", "description": "File to close"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vfs_view",
            "description": "Change the visible range of an open file. Does not return content directly - updates what you see in context.",
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
            "name": "vfs_edit",
            "description": "Edit specific lines in a file. Replaces lines from line_start to line_end with new content.",
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
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Provide the final answer to the user's query. Use this when you have completed the task.",
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
            "description": "Indicate that you need more information and cannot proceed without it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "missing": {"type": "string", "description": "What information is missing"}
                },
                "required": ["missing"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "need_human",
            "description": "Request human assistance. Use when you're stuck or need confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question for the human"}
                },
                "required": ["question"]
            }
        }
    }
]


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

REACT_SYSTEM_PROMPT = """Du bist ein Agent der in einem VFS (Virtual File System) arbeitet.

WICHTIGE REGELN:
1. Du darfst NUR Informationen verwenden die du:
   - Aus dem VFS gelesen hast
   - Durch Tool-Calls erhalten hast
   - Im RAG Context gefunden hast

2. Wenn du eine Information NICHT hast:
   - Sage "Ich habe keine Information zu X"
   - ERFINDE NICHTS - niemals!

3. VFS Operationen:
   - vfs_open(file, start?, end?) - Datei öffnen
   - vfs_close(file) - Datei schließen (erstellt Summary)
   - vfs_view(file, start, end) - Sichtbaren Bereich ändern (gibt nichts zurück!)
   - vfs_write(file, content) - Datei schreiben
   - vfs_edit(file, start, end, content) - Zeilen ersetzen
   - vfs_list() - Alle Dateien auflisten

4. Du kannst jederzeit sagen:
   - need_info: wenn dir Informationen fehlen
   - need_human: wenn du menschliche Hilfe brauchst
   - final_answer: wenn du fertig bist

5. Limits:
   - Max {max_open_files} Dateien gleichzeitig offen halten
   - Schließe Dateien die du nicht mehr brauchst

6. Deine verfügbaren Tools: {tools}

Aktueller VFS Status wird dir im Context angezeigt."""


MICROAGENT_SYSTEM_PROMPT = """Du bist ein fokussierter Microagent für eine spezifische Aufgabe.

DEINE AUFGABE: {task_description}

REGELN:
1. Fokussiere dich NUR auf deine Aufgabe
2. Verwende NUR Informationen aus VFS oder Tool-Calls
3. ERFINDE NICHTS
4. Wenn du nicht weiterkommst: need_info oder need_human
5. Arbeite schnell und effizient

Deine verfügbaren Tools: {tools}

Wenn du fertig bist: final_answer mit deinem Ergebnis."""

# Confidence threshold for auto-detection
class ExecutionEngine:
    """
    Main execution engine for a_run().

    Supports:
    - Intent detection
    - Tool path (ReAct loop)
    - Decomposition path (parallel microagents)
    - Pause/Continue
    - Transaction rollback
    """

    def __init__(
        self,
        agent: 'FlowAgent',
        use_native_tools: bool = True,  # LiteLLM native vs a_format_class
        human_online: bool = False,
        intermediate_callback: Callable[[str], None] | None = None
    ):
        self.agent = agent
        self.use_native_tools = use_native_tools
        self.human_online = human_online
        self.intermediate_callback = intermediate_callback

        # Active executions (for pause/continue)
        self._executions: dict[str, ExecutionState] = {}

        # Frozen microagent states
        self._frozen_microagents: dict[str, dict] = {}

        # asyncio background tasks
        self._background_tasks: set[asyncio.Task] = set()

    def _emit_intermediate(self, message: str):
        """Send intermediate message to user"""
        if self.intermediate_callback:
            self.intermediate_callback(message)
        else:
            print(f"INTERMEDIATE: {message}")

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    async def execute(
        self,
        query: str,
        session_id: str = "default",
        execution_id: str | None = None,
        do_stream:bool=False,
        **kwargs
    ) -> ExecutionResult | tuple[Callable[[ExecutionState], AsyncGenerator[ExecutionResult, Any]], ExecutionState]:
        """
        Main execution entry point.

        If execution_id provided and exists → Continue
        Else → Start new execution
        """
        # Check for continuation
        if execution_id and execution_id in self._executions:
            state = self._executions[execution_id]

            # Check if waiting for human
            if state.waiting_for_human:
                human_response = kwargs.get('human_response')
                if human_response:
                    state.human_response = human_response
                    state.waiting_for_human = False
                    state.resumed_at = datetime.now()

            return await self._continue_execution(state)

        # Create new execution
        execution_id = execution_id or f"exec_{uuid.uuid4().hex[:12]}"

        state = ExecutionState(
            execution_id=execution_id,
            query=query,
            session_id=session_id,
            max_iterations=kwargs.get('max_iterations', 15),
            token_budget=kwargs.get('token_budget', 10000)
        )

        self._executions[execution_id] = state

        if do_stream:
            return self._run_stream_execution, state

        return await self._run_execution(state)

    async def _run_execution(
        self,
        state: ExecutionState,
    ) -> ExecutionResult:
        """Run execution from current state"""

        start_time = time.perf_counter()

        try:
            # Get or create session
            session = await self.agent.session_manager.get_or_create(state.session_id)

            # Take VFS snapshot for transaction
            state.vfs_snapshot = session.vfs.to_checkpoint()

            # Add user message if remember
            if state.phase == ExecutionPhase.INTENT:
                await session.add_message({"role": "user", "content": state.query})

            # Phase 1: Intent Detection
            if state.phase == ExecutionPhase.INTENT:
                for i in range(8):

                    if state.success:
                        break

                    result = await self._immediate_response(state, session)

                    if not result.tool_calls:
                        state.final_answer = result.content
                        state.phase = ExecutionPhase.COMPLETED
                        state.success = True
                        break

                    if result.content:
                        await session.add_message({
                            "role": "assistant",
                            "content": result.content
                        })
                        self._emit_intermediate(result.content)

                    await self._tool_runner(result, state, session)

            elif state.phase in [ExecutionPhase.CATEGORY_SELECT, ExecutionPhase.TOOL_SELECT, ExecutionPhase.REACT_LOOP]:
                result = await self._tool_path(state, session)

            elif state.phase == ExecutionPhase.DECOMPOSITION:
                result = await self._decomposition_path(state, session)

            else:
                result = state.final_answer or "Execution completed"

            # Check for pause
            if state.phase == ExecutionPhase.PAUSED:
                return ExecutionResult(
                    success=False,
                    response="",
                    execution_id=state.execution_id,
                    path_taken="paused",
                    paused=True,
                    needs_human=state.waiting_for_human,
                    human_query=state.human_query
                )

            # Validation
            if state.phase != ExecutionPhase.COMPLETED:
                valid = await self._validate_result(state, result)
                if not valid.is_valid:
                    # Rollback
                    session.vfs.from_checkpoint(state.vfs_snapshot)
                    state.red_flags.extend(valid.issues)

            # Save assistant response if remember
            if state.final_answer:
                await session.add_message({"role": "assistant", "content": state.final_answer})

            # Learning (non-blocking)
            if state.success:
                asyncio.create_task(self._learn_from_execution(state, session))

            duration = time.perf_counter() - start_time

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
            # Rollback on error
            session = self.agent.session_manager.get(state.session_id)
            if session and state.vfs_snapshot:
                session.vfs.from_checkpoint(state.vfs_snapshot)

            state.phase = ExecutionPhase.FAILED
            state.red_flags.append(str(e))
            import traceback
            print(traceback.format_exc())
            return ExecutionResult(
                success=False,
                response=f"Execution failed: {str(e)}",
                execution_id=state.execution_id,
                path_taken="failed",
                duration=time.perf_counter() - start_time
            )


    async def _run_stream_execution(
        self,
        state: ExecutionState,
    ) -> AsyncGenerator[ExecutionResult, Any]:
        """Run execution from current state"""

        start_time = time.perf_counter()

        try:
            # Get or create session
            session = await self.agent.session_manager.get_or_create(state.session_id)

            # Take VFS snapshot for transaction
            state.vfs_snapshot = session.vfs.to_checkpoint()

            # Add user message if remember
            if state.phase == ExecutionPhase.INTENT:
                await session.add_message({"role": "user", "content": state.query})

            result = None
            # Phase 1: Intent Detection
            if state.phase == ExecutionPhase.INTENT:
                for i in range(8):

                    if state.success:
                        break

                    result = await self._immediate_response(state, session)

                    if not result.tool_calls:
                        state.final_answer = result.content
                        state.phase = ExecutionPhase.COMPLETED
                        state.success = True
                        break

                    if result.content:
                        await session.add_message({
                            "role": "assistant",
                            "content": result.content
                        })
                        yield result.content

                    await self._tool_runner(result, state, session)

            else:
                result = await self._run_execution(state)
                yield result

            # Check for pause
            if state.phase == ExecutionPhase.PAUSED:
                yield ExecutionResult(
                    success=False,
                    response="",
                    execution_id=state.execution_id,
                    path_taken="paused",
                    paused=True,
                    needs_human=state.waiting_for_human,
                    human_query=state.human_query
                )

            # Validation
            if state.phase != ExecutionPhase.COMPLETED and result:
                valid = await self._validate_result(state, result.response)
                if not valid.is_valid:
                    # Rollback
                    session.vfs.from_checkpoint(state.vfs_snapshot)
                    state.red_flags.extend(valid.issues)

            # Save assistant response if remember
            if state.final_answer:
                await session.add_message({"role": "assistant", "content": state.final_answer})

            # Learning (non-blocking)
            if state.success:
                task = asyncio.create_task(self._learn_from_execution(state, session))
                self._background_tasks.add(task)


            duration = time.perf_counter() - start_time

            yield ExecutionResult(
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
            # Rollback on error
            session = self.agent.session_manager.get(state.session_id)
            if session and state.vfs_snapshot:
                session.vfs.from_checkpoint(state.vfs_snapshot)

            state.phase = ExecutionPhase.FAILED
            state.red_flags.append(str(e))
            import traceback
            print(traceback.format_exc())
            yield ExecutionResult(
                success=False,
                response=f"Execution failed: {str(e)}",
                execution_id=state.execution_id,
                path_taken="failed",
                duration=time.perf_counter() - start_time
            )

    async def _tool_runner(self, result, state: ExecutionState, session: 'Session') -> None:
        if result.tool_calls is None:
            return None
        for tool_call in result.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")

            self._emit_intermediate(f"Verwende {tool_call.function.name}...")
            if tool_call.function.name == "final_answer":
                state.final_answer = args.get("final_answer")
                state.phase = ExecutionPhase.COMPLETED
                state.success = True
                break

            elif tool_call.function.name == "need_tool":
                state.phase = ExecutionPhase.CATEGORY_SELECT
                tool_result = await self._tool_path(state, session, args.get("tools"))
                await session.add_message({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            elif tool_call.function.name == "complex":
                state.phase = ExecutionPhase.DECOMPOSITION
                tool_result = await self._decomposition_path(state, session)
                await session.add_message({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            elif tool_call.function.name == "need_human" and self.human_online:
                state.phase = ExecutionPhase.PAUSED
                state.paused_at = datetime.now()
                state.waiting_for_human = tool_call.id
                state.human_query = args.get("question")
                state.success = True
                break

            elif tool_call.function.name == "think":
                state.thoughts.append(args.get("thought"))
                await session.add_message({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": args.get("thought")
                })

            elif tool_call.function.name == "list_tools":
                tools = self.agent.tool_manager.get_all()
                # ["need_tool", "complex", "need_human", "think", "list_tools", "request_tool"]
                tools += [
                    ToolEntry(name="need_tool", description="Request specific tool", args_schema="(tools: list[str])",
                              category=["system"]),
                    ToolEntry(name="complex", description="Request decomposition", args_schema="()",
                              category=["system"]),
                    ToolEntry(name="need_human", description="Request human assistance", args_schema="(question: str)",
                              category=["system"]),
                    ToolEntry(name="think", description="Reason over multiple data sources",
                              args_schema="(thought: str)", category=["system"]),
                    ToolEntry(name="list_tools", description="List available tools",
                              args_schema="(category: str, search: str)", category=["system"]),
                    ToolEntry(name="request_tool", description="Request tool definition",
                              args_schema="(tool_name: str)", category=["system"])]

                # vfs tools
                tools += [ToolEntry(name=t['function']['name'], description=t['function']['description'],
                                    args_schema=t['function']['parameters'], category=["vfs"]) for t in
                          VFS_TOOLS_LITELLM]
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
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": res
                })
            elif tool_call.function.name == "request_tool":
                tool_name = args.get("tool_name")

                tool_result = f"Tool '{tool_name}' not found"
                if tool_name in ["need_tool", "complex", "need_human", "think", "list_tools",
                                 "request_tool"] or tool_name.startswith("vfs_"):
                    tool_result = "is a system tool"

                tool = self.agent.tool_manager.get(tool_name)

                if tool:
                    # Tool exists, check if restricted
                    if not session.is_tool_allowed(tool_name):
                        tool_result = f"Tool '{tool_name}' is restricted in this session"
                    else:
                        tool_result = f"{tool.description} {tool.args_schema}"

                await session.add_message({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            else:
                args['type'] = tool_call.function.name
                tool_result = await self._execute_action(state, session, args)

                await session.add_message({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })
    async def _continue_execution(
        self,
        state: ExecutionState
    ) -> ExecutionResult:
        """Continue a paused execution"""
        state.resumed_at = datetime.now()

        # Unfreeze any frozen microagents
        if state.active_microagents:
            for agent_id in state.active_microagents:
                if agent_id in self._frozen_microagents:
                    # Will be resumed in decomposition_path
                    pass

        return await self._run_execution(state)

    # =========================================================================
    # IMMEDIATE PATH
    # =========================================================================

    async def _immediate_response(
        self,
        state: ExecutionState,
        session
    ) -> ModelResponse:
        """Direct response without tools"""

        self._emit_intermediate("Antworte direkt...")

        messages = session.get_history_for_llm(last_n=15)
        # list of 4 tools with description and args_schema 1 finale response 2 need tools 3 is complex
        default_tools = [
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": "Provide an immediate answer to the user based on the available information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "final_answer": {
                                "type": "string",
                                "description": "Your final answer"
                            }
                        },
                        "required": ["final_answer"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "need_tool",
                    "description": "Indicate that you need a tool to proceed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tools": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of tools you need"
                            }
                        },
                        "required": ["tools"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "complex",
                    "description": "Indicate that the task is too complex to answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_complex": {
                                "type": "boolean",
                                "description": "True if the task needs to be broken down"
                            }
                        },
                        "required": ["is_complex"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "think",
                    "description": "Reason over multiple data sources. for an final answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "Your thought"
                            }
                        },
                        "required": ["thought"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_tools",
                    "description": "List available tools. sort by category. or search by name and description.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Category to filter by"
                            },
                            "search": {
                                "type": "string",
                                "description": "Search query"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "request_tool",
                    "description": "Request a specific tool by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "description": "Name of the tool to request"
                            }
                        },
                        "required": ["tool_name"]
                    }
                }
            }
            # think list_tools request_tool
        ]

        if self.human_online:
            default_tools.append({
                "type": "function",
                "function": {
                    "name": "need_human",
                    "description": "Request human assistance when stuck or need confirmation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question for the human"
                            }
                        },
                        "required": ["question"]
                    }
                }
            })

        response = await self.agent.a_run_llm_completion(
            tools=default_tools,
            tool_choice="auto",
            messages=messages,
            model_preference="fast",
            with_context=True,
            stream=False,
            task_id=f"{state.execution_id}_immediate",
            session_id=state.session_id,
            get_response_message=True
        )

        return response

    # =========================================================================
    # TOOL PATH
    # =========================================================================

    async def _tool_path(
        self,
        state: ExecutionState,
        session,
        tool_names:list[str]|None=None
    ) -> str:
        """Execute with tools via ReAct loop"""

        if tool_names:
            # Check if tool_names is a sub set of the available tools
            available_tools = self.agent.tool_manager.list_names()
            if set(tool_names).issubset(set(available_tools)):
                state.selected_tools = tool_names
                state.phase = ExecutionPhase.REACT_LOOP

        # Category selection
        if state.phase == ExecutionPhase.CATEGORY_SELECT:
            await self._select_categories(state, session)
            state.phase = ExecutionPhase.TOOL_SELECT

        # Tool selection
        if state.phase == ExecutionPhase.TOOL_SELECT:
            await self._select_tools(state, session)
            state.phase = ExecutionPhase.REACT_LOOP

        # ReAct loop
        if state.phase == ExecutionPhase.REACT_LOOP:
            result = await self._react_loop(state, session)
            return result

        return state.final_answer or ""

    async def _select_categories(self, state: ExecutionState, session):
        """Select relevant tool categories"""

        self._emit_intermediate("Wähle relevante Tool-Kategorien...")

        # Get available categories
        categories = self.agent.tool_manager.list_categories()

        if not categories:
            state.red_flags.append("No tool categories available")
            return

        prompt = f"""Welche Tool-Kategorien sind für diese Aufgabe relevant?

Aufgabe: {state.query}

Verfügbare Kategorien: {', '.join(categories)}

Wähle nur die wirklich relevanten Kategorien."""

        result = await self.agent.a_format_class(
            CategorySelection,
            prompt,
            model_preference="fast"
        )

        state.selected_categories = result['categories']

    async def _select_tools(self, state: ExecutionState, session):
        """Select specific tools (max 5)"""

        # Get tools from selected categories
        tools = self.agent.tool_manager.get_by_category(*state.selected_categories)

        if len(tools) <= 5:
            state.selected_tools = [t.name for t in tools]
            return

        self._emit_intermediate("Wähle die besten Tools aus...")

        tool_list = "\n".join([f"- {t.name}: {t.description[:100]}" for t in tools])

        prompt = f"""Wähle max 5 Tools für diese Aufgabe:

Aufgabe: {state.query}

Verfügbare Tools:
{tool_list}

Wähle nur die wichtigsten Tools."""

        result = await self.agent.a_format_class(
            ToolSelection,
            prompt,
            model_preference="fast"
        )

        state.selected_tools = result['tools'][:5]

    async def _react_loop(
        self,
        state: ExecutionState,
        session,
    ) -> str:
        """Main ReAct loop (RLM-VFS style)"""

        while state.iteration < state.max_iterations:

            # Check token budget
            if state.tokens_used >= state.token_budget:
                state.red_flags.append("Token budget exceeded")
                break

            # Check for paused state
            if state.phase == ExecutionPhase.PAUSED:
                return ""

            state.iteration += 1

            self._emit_intermediate(f"Schritt {state.iteration}...")

            # Build VFS context
            vfs_context = session.build_vfs_context()

            # Get next action
            action = None
            if self.use_native_tools:
                try:
                    action = await self._get_action_native(state, session, vfs_context)
                except Exception as e:
                    self.use_native_tools = False
                    self._emit_intermediate(f"Error using native tools: {e}, switching to format class")

            if action is None:
                action = await self._get_action_format(state, session, vfs_context)

            if action is None:
                state.red_flags.append("Failed to get action")
                break

            # Execute action
            result = await self._execute_action(state, session, action)

            # Record
            state.actions.append(action)
            state.observations.append(str(result)[:500])

            # Check for completion
            if action.get('type') == ActionType.FINAL_ANSWER.value:
                state.final_answer = action.get('answer', '')
                state.success = True
                state.phase = ExecutionPhase.COMPLETED
                return state.final_answer

            # Check for need_human
            if action.get('type') == ActionType.NEED_HUMAN.value:
                state.waiting_for_human = True
                state.human_query = action.get('question', 'Können Sie mir helfen?')
                state.phase = ExecutionPhase.PAUSED
                state.paused_at = datetime.now()
                return ""

            # Check for need_info
            if action.get('type') == ActionType.NEED_INFO.value:
                missing = action.get('missing', 'Unbekannt')
                self._emit_intermediate(f"Mir fehlt: {missing}")
                state.phase = ExecutionPhase.PAUSED
                state.paused_at = datetime.now()
                return ""

            # Loop detection
            if self._detect_loop(state):
                state.red_flags.append("Loop detected")

                # Error handling: B → A → A → D
                handled = await self._handle_error(state, session, "Loop detected")
                if not handled:
                    break

        # Max iterations reached
        if not state.final_answer:
            state.final_answer = f"Konnte die Aufgabe nicht in {state.max_iterations} Schritten abschließen."
            state.success = False

        state.phase = ExecutionPhase.COMPLETED
        return state.final_answer

    async def _get_action_native(
        self,
        state: ExecutionState,
        session,
        vfs_context: str
    ) -> dict | None:
        """Get next action using LiteLLM native tool calling"""

        # Build messages
        system_prompt = REACT_SYSTEM_PROMPT.format(
            max_open_files=5,
            tools=", ".join(state.selected_tools)
        )

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{vfs_context}"},
            {"role": "user", "content": state.query}
        ]

        # Add history
        for i, (thought, obs) in enumerate(zip(state.thoughts, state.observations)):
            messages.append({"role": "assistant", "content": thought})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        # Get selected tools in LiteLLM format
        selected_tools_litellm = [
            t for t in self.agent.tool_manager.get_all_litellm()
            if any(t['function']['name'] == name for name in state.selected_tools)
        ]

        # Add VFS tools
        all_tools = VFS_TOOLS_LITELLM + selected_tools_litellm

        # Make LLM call
        response = await self.agent.a_run_llm_completion(
            messages=messages,
            tools=all_tools,
            tool_choice="auto",
            model_preference="fast",
            stream=False,
            get_response_message=True,
            task_id=f"{state.execution_id}_react_{state.iteration}",
            session_id=state.session_id
        )

        # Parse response
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tc = response.tool_calls[0]  # Take first tool call

            try:
                args = json.loads(tc.function.arguments or "{}")
            except:
                args = {}

            # Map to action dict
            action_type = tc.function.name

            return {
                'type': action_type,
                'tool': tc.function.name,
                'args': args,
                **args  # Spread args for easy access
            }

        # No tool call - check for text response
        if hasattr(response, 'content') and response.content:
            state.thoughts.append(response.content)
            # If just text, assume need more thinking
            return {
                'type': 'thinking',
                'thought': response.content
            }

        return None

    async def _get_action_format(
        self,
        state: ExecutionState,
        session,
        vfs_context: str
    ) -> dict | None:
        """Get next action using a_format_class"""

        prompt = f"""Du bearbeitest diese Aufgabe: {state.query}

Aktueller VFS Status:
{vfs_context}

Bisherige Aktionen: {len(state.actions)}
{chr(10).join(state.observations[-3:]) if state.observations else "Keine"}

Deine verfügbaren Tools: {', '.join(state.selected_tools)}

Was ist der nächste Schritt?"""

        result = await self.agent.a_format_class(
            ThoughtAction,
            prompt,
            model_preference="fast"
        )

        state.thoughts.append(result.get('thought', ''))

        return {
            'type': result.get('action'),
            'tool': result.get('tool_name'),
            'args': result.get('tool_args'),
            **result
        }

    async def _execute_action(
        self,
        state: ExecutionState,
        session,
        action: dict
    ) -> Any:
        """Execute an action"""

        action_type = action.get('type', '')

        # VFS operations
        if action_type == 'vfs_open':
            return session.vfs.open(
                action.get('filename'),
                action.get('line_start', 1),
                action.get('line_end', -1)
            )
        elif action_type == 'vfs_read':
            res = session.vfs.read(action.get('filename'))
            if res['success']:
                content = res['content']
                if action.get('line_start') or action.get('line_end'):
                    lines = res['content'].split('\n')
                    start = action.get('line_start', 1) - 1
                    end = action.get('line_end', len(lines))
                    content = '\n'.join(lines[start:end])
                return content
            return res['error']

        elif action_type == 'vfs_close':
            return await session.vfs.close(action.get('filename'))

        elif action_type == 'vfs_view':
            # Just changes view range, doesn't return content
            f = session.vfs.files.get(action.get('filename'))
            if f:
                f.view_start = max(0, action.get('line_start', 1) - 1)
                f.view_end = action.get('line_end', -1)
                session.vfs._dirty = True
            return {"success": True, "message": "View range updated"}

        elif action_type == 'vfs_write':
            return session.vfs.write(action.get('filename'), action.get('content', ''))

        elif action_type == 'vfs_edit':
            return session.vfs.edit(
                action.get('filename'),
                action.get('line_start', 1),
                action.get('line_end', 1),
                action.get('content', '')
            )

        elif action_type == 'vfs_list':
            return session.vfs.list_files()

        # Tool call
        elif action_type == 'tool_call' or action.get('tool') in state.selected_tools:
            tool_name = action.get('tool') or action.get('tool_name')
            tool_args = action.get('args') or action.get('tool_args') or {}

            self._emit_intermediate(f"Verwende {tool_name}...")

            try:
                result = await self.agent.arun_function(tool_name, **tool_args)

                # Store result in VFS
                session.vfs.create(f"tool_result_{state.iteration}", str(result)[:2000])

                return result
            except Exception as e:
                return {"error": str(e)}

        # Final answer
        elif action_type == 'final_answer':
            return {"final": action.get('answer')}

        # Need info/human
        elif action_type in ['need_info', 'need_human']:
            return {"status": action_type, "detail": action.get('missing') or action.get('question')}

        # Thinking (no action)
        elif action_type == 'thinking':
            return {"status": "thinking"}

        else:
            return await self.agent.arun_function(action_type, **action)

        return {"error": f"Unknown action type: {action_type}"}

    def _detect_loop(self, state: ExecutionState) -> bool:
        """Detect if we're stuck in a loop"""
        if len(state.actions) < 3:
            return False

        # Check last 3 actions for repetition
        last_3 = state.actions[-3:]

        # Simple check: same action type repeated
        if len(set(a.get('type') for a in last_3)) == 1:
            return True

        # Check for same tool called repeatedly with same args
        if len(state.actions) >= 4:
            last_4 = state.actions[-4:]
            tool_calls = [
                (a.get('tool'), str(a.get('args')))
                for a in last_4
                if a.get('tool')
            ]
            if len(tool_calls) >= 3 and len(set(tool_calls)) == 1:
                return True

        return False

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    async def _handle_error(
        self,
        state: ExecutionState,
        session,
        error: str
    ) -> bool:
        """
        Handle errors: B → A → A → D
        B: Escalate to complex model
        A: Retry (2x)
        D: Human (if online)

        Returns True if handled, False if should abort
        """

        # B: Escalate to complex model (first attempt)
        if not state.escalated:
            self._emit_intermediate("Versuche mit stärkerem Modell...")
            state.escalated = True
            # Continue loop will use complex model
            return True

        # A: Retry (max 2 times)
        if state.retry_count < 2:
            self._emit_intermediate(f"Wiederhole... (Versuch {state.retry_count + 1})")
            state.retry_count += 1
            return True

        # D: Human (if online)
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

    async def _decomposition_path(
        self,
        state: ExecutionState,
        session
    ) -> str:
        """Execute complex task with parallel microagents"""

        # Decompose if not already done
        if not state.subtasks:
            await self._decompose_task(state, session)

        self._emit_intermediate(f"Führe {len(state.subtasks)} Teilaufgaben aus...")

        # Group subtasks by parallelization
        parallel_groups = self._group_subtasks(state.subtasks)

        # Execute groups
        for group_idx, group in enumerate(parallel_groups):
            self._emit_intermediate(f"Gruppe {group_idx + 1}/{len(parallel_groups)}...")

            # Check for pause
            if state.phase == ExecutionPhase.PAUSED:
                return ""

            # Execute group in parallel
            results = await self._execute_parallel_group(state, session, group)

            # Check for human needed
            if any(r.frozen_state for r in results):
                state.phase = ExecutionPhase.PAUSED
                state.paused_at = datetime.now()
                return ""

            # Merge results
            for result in results:
                state.subtask_results[result.task_id] = result

                if result.vfs_changes:
                    for filename, content in result.vfs_changes.items():
                        session.vfs.write(filename, content)

        # Aggregate final result
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

Erstelle eine Liste von Teilaufgaben.
Markiere welche parallel ausgeführt werden können."""

        result = await self.agent.a_format_class(
            TaskDecomposition,
            prompt,
            model_preference="fast"
        )

        # Create subtask entries
        for i, (desc, can_par) in enumerate(zip(result['subtasks'], result['can_parallel'])):
            state.subtasks.append({
                'id': f"task_{i}",
                'description': desc,
                'can_parallel': can_par,
                'status': 'pending'
            })

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

    async def _execute_parallel_group(
        self,
        state: ExecutionState,
        session,
        group: list[dict]
    ) -> list[MicroagentResult]:
        """Execute a group of subtasks in parallel"""

        # Create microagent tasks
        tasks = []
        for subtask in group:
            task = asyncio.create_task(
                self._run_microagent(state, session, subtask)
            )
            tasks.append(task)
            state.active_microagents.append(subtask['id'])

        # Wait for all with timeout for freeze
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0  # 60 second timeout
            )
        except asyncio.TimeoutError:
            # Freeze remaining agents
            results = []
            for task, subtask in zip(tasks, group):
                if task.done():
                    results.append(task.result())
                else:
                    # Freeze
                    task.cancel()
                    frozen = MicroagentResult(
                        task_id=subtask['id'],
                        success=False,
                        frozen_state={'subtask': subtask}
                    )
                    results.append(frozen)
                    self._frozen_microagents[subtask['id']] = frozen.frozen_state

        # Clean up active list
        for subtask in group:
            if subtask['id'] in state.active_microagents:
                state.active_microagents.remove(subtask['id'])

        return [r for r in results if isinstance(r, MicroagentResult)]

    async def _run_microagent(
        self,
        state: ExecutionState,
        session,
        subtask: dict,
    ) -> MicroagentResult:
        """Run a single microagent"""

        start_time = time.perf_counter()
        task_id = subtask['id']

        # Create isolated VFS with relevant files only
        micro_vfs = await self._create_micro_vfs(session, subtask)

        # Select tools for this subtask
        tools = await self._select_tools_for_subtask(subtask)

        config = MicroagentConfig(
            task_id=task_id,
            task_description=subtask['description'],
            tools=tools,
            max_iterations=5,
            token_budget=2000
        )

        # Run mini ReAct loop
        micro_state = ExecutionState(
            execution_id=f"{state.execution_id}_{task_id}",
            query=subtask['description'],
            session_id=state.session_id,
            max_iterations=config.max_iterations,
            token_budget=config.token_budget,
            selected_tools=config.tools
        )
        micro_state.phase = ExecutionPhase.REACT_LOOP

        try:
            # Create temporary session-like object with micro_vfs
            class MicroSession:
                def __init__(self, vfs):
                    self.vfs = vfs
                def build_vfs_context(self):
                    return self.vfs.build_context_string()

            micro_session = MicroSession(micro_vfs)

            result = await self._react_loop(micro_state, micro_session, with_context=False)

            # Collect VFS changes
            vfs_changes = {}
            for filename, file in micro_vfs.files.items():
                if not file.readonly:
                    vfs_changes[f"{task_id}_{filename}"] = file.content

            return MicroagentResult(
                task_id=task_id,
                success=micro_state.success,
                result=result,
                vfs_changes=vfs_changes,
                iterations=micro_state.iteration,
                tokens_used=micro_state.tokens_used,
                duration=time.perf_counter() - start_time
            )

        except Exception as e:
            return MicroagentResult(
                task_id=task_id,
                success=False,
                error=str(e),
                duration=time.perf_counter() - start_time
            )

    async def _create_micro_vfs(self, session, subtask: dict):
        """Create isolated VFS with only relevant files"""
        from toolboxv2.mods.isaa.base.Agent.agent_session import VirtualFileSystem

        micro_vfs = VirtualFileSystem(
            session_id=f"micro_{subtask['id']}",
            agent_name="microagent",
            max_window_lines=100
        )

        # Copy only relevant files (simple heuristic: check if mentioned in description)
        desc_lower = subtask['description'].lower()

        for filename, file in session.vfs.files.items():
            if file.readonly:
                continue
            if filename.lower() in desc_lower or any(
                word in desc_lower for word in filename.lower().split('_')
            ):
                micro_vfs.create(filename, file.content)

        return micro_vfs

    async def _select_tools_for_subtask(self, subtask: dict) -> list[str]:
        """Select relevant tools for a subtask"""

        prompt = f"""Welche Tools brauche ich für diese Aufgabe?

Aufgabe: {subtask['description']}

Verfügbare Kategorien: {', '.join(self.agent.tool_manager.list_categories())}

Wähle max 3 relevante Tools."""

        result = await self.agent.a_format_class(
            ToolSelection,
            prompt,
            model_preference="fast"
        )

        return result['tools'][:3]

    async def _aggregate_results(self, state: ExecutionState, session) -> str:
        """Aggregate microagent results into final answer"""

        results_summary = "\n".join([
            f"- {task_id}: {'✓' if r.success else '✗'} {r.result or r.error}"
            for task_id, r in state.subtask_results.items()
        ])

        prompt = f"""Fasse die Ergebnisse zusammen:

Ursprüngliche Aufgabe: {state.query}

Teilergebnisse:
{results_summary}

Erstelle eine zusammenhängende Antwort."""

        response = await self.agent.a_run_llm_completion(
            messages=[{"role": "user", "content": prompt}],
            model_preference="fast",
            stream=False,
            task_id=f"{state.execution_id}_aggregate",
            session_id=state.session_id
        )

        return response

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
Ergebnis: {result[:2000]+ (f'...{len(result) - 2000}...{result[-2000:]}' if len(result) > 2000 else '')}

Ist das Ergebnis vollständig und korrekt?"""

        try:
            valid = await self.agent.a_format_class(
                ValidationResult,
                prompt,
                model_preference="fast"
            )
            return ValidationResult(**valid)
        except:
            return ValidationResult(is_valid=True, confidence=0.5, issues=[])

    async def _learn_from_execution(self, state: ExecutionState, session):
        """Non-blocking learning after successful execution"""

        try:
            # Update tool usage stats
            for action in state.actions:
                tool_name = action.get('tool')
                if tool_name:
                    entry = self.agent.tool_manager.get(tool_name)
                    if entry:
                        entry.record_call()

            # Update RuleSet patterns
            if state.success and session.rule_set:
                # Record successful patterns
                pattern = f"Query type: {state.query[:50]}... → Path: {self._get_path_taken(state)}"
                session.rule_set.learn_pattern(
                    pattern=pattern,
                    source_situation=state.query[:100],
                    confidence=0.6
                )

                # Update existing rule success
                for rule in session.rule_set.get_active_rules():
                    session.rule_set.record_rule_success(rule.id)

                # Prune occasionally
                import random
                if random.random() < 0.1:
                    session.rule_set.prune_low_confidence_patterns(threshold=0.2)

        except Exception as e:
            # Learning failures should not affect execution
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
    # PAUSE / CONTINUE API
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
            {
                'id': state.execution_id,
                'query': state.query[:50],
                'phase': state.phase.value,
                'iteration': state.iteration,
                'waiting_for_human': state.waiting_for_human
            }
            for state in self._executions.values()
        ]

    async def cancel(self, execution_id: str) -> bool:
        """Cancel and cleanup an execution"""
        if execution_id in self._executions:
            state = self._executions[execution_id]

            # Rollback VFS
            session = self.agent.session_manager.get(state.session_id)
            if session and state.vfs_snapshot:
                session.vfs.from_checkpoint(state.vfs_snapshot)

            state.phase = ExecutionPhase.FAILED
            del self._executions[execution_id]

            return True
        return False
