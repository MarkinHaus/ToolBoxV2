"""
Sub-Agent System - Parallel Task Execution for FlowAgent

Features:
- spawn_sub_agent: Create isolated sub-agents for focused tasks
- wait_for: Await completion of one or more sub-agents
- Parallel execution with async support
- VFS write isolation (sub-agent can only write to output_dir)
- Fixed token budget per sub-agent
- Max depth = 1 (NO sub-sub-agents!)

Architecture:
- Sub-agents share tools with parent (no tool loading overhead)
- Sub-agents have isolated history (clean context)
- Sub-agents can READ entire VFS
- Sub-agents can ONLY WRITE to their output_dir
- Results are injected into parent's AutoFocus

Author: FlowAgent V3
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SubAgentStatus(Enum):
    """Sub-agent execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MAX_ITERATIONS = "max_iterations"  # NEW: Hit max iterations
    PAUSED = "paused"  # NEW: Manually paused


@dataclass
class SubAgentConfig:
    """Configuration for sub-agent execution"""
    max_tokens: int = 5000  # Token budget for this sub-agent
    max_iterations: int = 10  # Max execution iterations
    timeout_seconds: int = 300  # Timeout in seconds

    # Inherited from parent (set at spawn time)
    model_preference: str = "fast"


@dataclass
class SubAgentState:
    """Runtime state of a sub-agent"""
    id: str
    task: str
    output_dir: str
    config: SubAgentConfig

    # Status
    status: SubAgentStatus = SubAgentStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Results
    result: str | None = None
    error: str | None = None
    tokens_used: int = 0
    iterations_used: int = 0
    files_written: list[str] = field(default_factory=list)

    # Async handling
    _task: asyncio.Task | None = field(default=None, repr=False)
    
    # NEW: Resume support
    execution_context: Optional['ExecutionContext'] = None  # Preserved context
    resumable: bool = False  # Can this be resumed?
    original_budget: int = 5000  # Original token budget
    session_id: str = ""  # Session ID
    engine: Optional[Any] = None  # Sub-agent engine instance


@dataclass
class SubAgentResult:
    """Result returned from wait_for"""
    id: str
    success: bool
    status: SubAgentStatus
    result: str | None
    error: str | None
    output_dir: str
    files_written: list[str]
    tokens_used: int
    duration_seconds: float
    task: str = ""  # NEW: Original task
    
    # NEW: Resume support fields
    max_iterations_reached: bool = False  # NEW: Hit max iterations?
    resumable: bool = False  # NEW: Can be resumed?
    iterations_used: int = 0  # NEW: Total iterations used
    execution_context: Optional['ExecutionContext'] = None  # NEW: Preserved context


# =============================================================================
# SUB-AGENT MANAGER
# =============================================================================

class SubAgentManager:
    """
    Manages sub-agent lifecycle and execution.

    Responsibilities:
    - Spawn sub-agents with isolated contexts
    - Track running sub-agents
    - Handle wait_for logic (single or multiple)
    - Enforce token budgets
    - Collect and report results

    CRITICAL: Enforces max_depth=1 (no sub-sub-agents)
    """

    def __init__(
        self,
        parent_engine: Any,  # ExecutionEngine
        parent_session: Any,  # AgentSessionV2
        is_sub_agent: bool = False
    ):
        """
        Initialize SubAgentManager.

        Args:
            parent_engine: Parent ExecutionEngine instance
            parent_session: Parent session for VFS access
            is_sub_agent: If True, spawn is DISABLED (prevents sub-sub-agents)
        """
        self.parent_engine = parent_engine
        self.parent_session = parent_session
        self.is_sub_agent = is_sub_agent

        # Active sub-agents
        self._sub_agents: dict[str, SubAgentState] = {}

        # Completed results (kept for reference)
        self._completed: dict[str, SubAgentResult] = {}

    def can_spawn(self) -> bool:
        """Check if spawning is allowed (False if already a sub-agent)"""
        return not self.is_sub_agent

    async def spawn(
        self,
        task: str,
        output_dir: str,
        wait: bool = True,
        budget: int = 5000,
        timeout: int = 300
    ) -> str | SubAgentResult:
        """
        Spawn a new sub-agent.

        Args:
            task: Task description for the sub-agent
            output_dir: VFS directory where sub-agent can write (e.g., /sub/research)
            wait: If True, wait for completion. If False, return immediately with ID.
            budget: Token budget for this sub-agent
            timeout: Timeout in seconds

        Returns:
            If wait=True: SubAgentResult
            If wait=False: sub_agent_id (str)

        Raises:
            RuntimeError: If called from a sub-agent (depth > 1)
        """
        if self.is_sub_agent:
            raise RuntimeError(
                "Sub-agents cannot spawn other sub-agents! "
                "Max depth is 1 to prevent infinite delegation."
            )

        # Generate unique ID
        sub_id = f"sub_{uuid.uuid4().hex[:8]}"

        # Normalize output_dir
        if not output_dir.startswith("/"):
            output_dir = f"/{output_dir}"
        if not output_dir.startswith("/sub/"):
            # Force sub-agent outputs under /sub/
            output_dir = f"/sub{output_dir}"

        # Create config
        config = SubAgentConfig(
            max_tokens=budget,
            timeout_seconds=timeout,
            model_preference=getattr(self.parent_engine, 'model_preference', 'fast')
        )

        # Create state
        state = SubAgentState(
            id=sub_id,
            task=task,
            output_dir=output_dir,
            config=config
        )

        self._sub_agents[sub_id] = state

        # Ensure output directory exists in VFS
        try:
            self.parent_session.vfs.mkdir(output_dir, parents=True)
        except Exception:
            pass  # Directory might already exist

        # Start execution
        state._task = asyncio.create_task(
            self._run_sub_agent(state)
        )

        if wait:
            # Wait for completion and return result
            result = await self._wait_single(sub_id, timeout)
            return result
        else:
            # Return ID immediately
            return sub_id

    async def wait_for(
        self,
        sub_agent_ids: str | list[str],
        timeout: int = 300
    ) -> dict[str, SubAgentResult]:
        """
        Wait for one or more sub-agents to complete.

        Args:
            sub_agent_ids: Single ID or list of IDs
            timeout: Timeout in seconds (applies to all)

        Returns:
            Dict mapping sub_agent_id to SubAgentResult
        """
        if isinstance(sub_agent_ids, str):
            ids = [sub_agent_ids]
        else:
            ids = list(sub_agent_ids)

        results = {}

        # Wait for all in parallel
        tasks = []
        for sub_id in ids:
            if sub_id in self._completed:
                # Already completed
                results[sub_id] = self._completed[sub_id]
            elif sub_id in self._sub_agents:
                tasks.append((sub_id, self._wait_single(sub_id, timeout)))
            else:
                # Unknown ID
                results[sub_id] = SubAgentResult(
                    id=sub_id,
                    success=False,
                    status=SubAgentStatus.FAILED,
                    result=None,
                    error=f"Unknown sub-agent ID: {sub_id}",
                    output_dir="",
                    files_written=[],
                    tokens_used=0,
                    duration_seconds=0
                )

        # Await all pending
        if tasks:
            gathered = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True
            )

            for (sub_id, _), result in zip(tasks, gathered):
                if isinstance(result, Exception):
                    results[sub_id] = SubAgentResult(
                        id=sub_id,
                        success=False,
                        status=SubAgentStatus.FAILED,
                        result=None,
                        error=str(result),
                        output_dir=self._sub_agents.get(sub_id,
                                                        SubAgentState(sub_id, "", "", SubAgentConfig())).output_dir,
                        files_written=[],
                        tokens_used=0,
                        duration_seconds=0
                    )
                else:
                    results[sub_id] = result

        return results

    async def _wait_single(self, sub_id: str, timeout: int) -> SubAgentResult:
        """Wait for a single sub-agent to complete"""
        state = self._sub_agents.get(sub_id)

        if not state:
            return SubAgentResult(
                id=sub_id,
                success=False,
                status=SubAgentStatus.FAILED,
                result=None,
                error=f"Unknown sub-agent: {sub_id}",
                output_dir="",
                files_written=[],
                tokens_used=0,
                duration_seconds=0
            )

        if state._task is None:
            return SubAgentResult(
                id=sub_id,
                success=False,
                status=SubAgentStatus.FAILED,
                result=None,
                error="Sub-agent task not started",
                output_dir=state.output_dir,
                files_written=[],
                tokens_used=0,
                duration_seconds=0
            )

        try:
            # Wait with timeout
            await asyncio.wait_for(state._task, timeout=timeout)
        except asyncio.TimeoutError:
            state.status = SubAgentStatus.TIMEOUT
            state.error = f"Timeout after {timeout} seconds"
            state.completed_at = datetime.now()

            # Cancel the task
            state._task.cancel()
            try:
                await state._task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            state.status = SubAgentStatus.FAILED
            state.error = str(e)
            state.completed_at = datetime.now()

        # Build result
        duration = 0.0
        if state.started_at and state.completed_at:
            duration = (state.completed_at - state.started_at).total_seconds()

        result = SubAgentResult(
            id=sub_id,
            success=state.status == SubAgentStatus.COMPLETED,
            status=state.status,
            result=state.result,
            error=state.error,
            output_dir=state.output_dir,
            files_written=state.files_written,
            tokens_used=state.tokens_used,
            duration_seconds=duration,
            task=state.task,  # NEW
            max_iterations_reached=(state.status == SubAgentStatus.MAX_ITERATIONS),  # NEW
            resumable=state.resumable,  # NEW
            iterations_used=state.iterations_used,  # NEW
            execution_context=state.execution_context if state.resumable else None  # NEW
        )

        # Move to completed
        self._completed[sub_id] = result
        del self._sub_agents[sub_id]

        return result

    async def _run_sub_agent(self, state: SubAgentState):
        """
        Execute sub-agent in isolated context.

        Creates a new ExecutionEngine instance with:
        - is_sub_agent=True (prevents further spawning)
        - VFS write restricted to output_dir
        - Shared tools from parent
        - Fresh history (isolated context)
        """
        state.status = SubAgentStatus.RUNNING
        state.started_at = datetime.now()

        try:
            # Import here to avoid circular imports
            from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

            # Create sub-agent engine
            # CRITICAL: is_sub_agent=True prevents spawn_sub_agent from being available
            sub_engine = ExecutionEngine(
                agent=self.parent_engine.agent,
                human_online=False,
                is_sub_agent=True,
                sub_agent_output_dir=state.output_dir,
                sub_agent_budget=state.config.max_tokens
            )
            
            # Store engine and session_id for resume
            state.engine = sub_engine
            state.session_id = f"{self.parent_session.session_id}__sub__{state.id}"
            state.original_budget = state.config.max_tokens

            # Execute with sub-agent context - GET CONTEXT BACK!
            result, ctx = await sub_engine.execute(
                query=state.task,
                session_id=state.session_id,
                max_iterations=state.config.max_iterations,
                get_ctx=True  # IMPORTANT: Get context for resume!
            )

            # Collect results
            state.result = result
            state.tokens_used = getattr(sub_engine, '_tokens_used', 0)
            state.iterations_used = ctx.current_iteration
            
            # NEW: Check if max iterations reached
            if ctx.current_iteration >= state.config.max_iterations:
                # Max iterations hit - check if resumable
                state.status = SubAgentStatus.MAX_ITERATIONS
                
                # Resumable if tools were used (progress was made)
                if len(ctx.tools_used) > 0:
                    state.resumable = True
                    state.execution_context = ctx  # PRESERVE CONTEXT!
                    print(f"[SubAgent {state.id}] Max iterations reached, but resumable (tools used: {len(ctx.tools_used)})")
                else:
                    state.resumable = False
                    print(f"[SubAgent {state.id}] Max iterations reached, no progress (not resumable)")
            else:
                # Completed successfully
                state.status = SubAgentStatus.COMPLETED
                state.resumable = False

            # Collect files written
            # Check VFS for files in output_dir
            try:
                ls_result = self.parent_session.vfs.ls(state.output_dir, recursive=True)
                if ls_result.get("success"):
                    state.files_written = [
                        f["path"] for f in ls_result.get("files", [])
                    ]
            except Exception:
                pass

            # Write result to output_dir/result.md if not already done
            result_path = f"{state.output_dir}/result.md"
            try:
                existing = self.parent_session.vfs.read(result_path)
                if not existing.get("success"):
                    # Write result summary
                    self.parent_session.vfs.write(
                        result_path,
                        f"# Sub-Agent Result\n\n"
                        f"**Task:** {state.task}\n\n"
                        f"**Status:** {state.status.value}\n\n"
                        f"**Result:**\n{result}\n"
                    )
                    if result_path not in state.files_written:
                        state.files_written.append(result_path)
            except Exception:
                pass

        except Exception as e:
            state.status = SubAgentStatus.FAILED
            state.error = str(e)

            # Write error to output_dir
            try:
                error_path = f"{state.output_dir}/error.log"
                self.parent_session.vfs.write(
                    error_path,
                    f"# Sub-Agent Error\n\n"
                    f"**Task:** {state.task}\n\n"
                    f"**Error:** {str(e)}\n"
                )
            except Exception:
                pass

        finally:
            state.completed_at = datetime.now()

    def get_status(self, sub_id: str) -> dict | None:
        """Get status of a sub-agent"""
        if sub_id in self._sub_agents:
            state = self._sub_agents[sub_id]
            return {
                "id": sub_id,
                "status": state.status.value,
                "task": state.task,
                "output_dir": state.output_dir,
                "started_at": state.started_at.isoformat() if state.started_at else None,
                "tokens_used": state.tokens_used
            }
        elif sub_id in self._completed:
            result = self._completed[sub_id]
            return {
                "id": sub_id,
                "status": result.status.value,
                "success": result.success,
                "output_dir": result.output_dir,
                "files_written": result.files_written,
                "tokens_used": result.tokens_used,
                "duration_seconds": result.duration_seconds
            }
        return None

    def get_all_status(self) -> dict:
        """Get status of all sub-agents"""
        return {
            "running": {
                sid: self.get_status(sid)
                for sid in self._sub_agents
            },
            "completed": {
                sid: self.get_status(sid)
                for sid in self._completed
            }
        }

    def format_results_for_auto_focus(self, results: dict[str, SubAgentResult]) -> str:
        """Format sub-agent results for AutoFocus injection"""
        lines = ["SUB-AGENT ERGEBNISSE:"]

        for sub_id, result in results.items():
            status_icon = "✅" if result.success else "❌"
            lines.append(
                f"• {status_icon} [{sub_id}]: {result.status.value} "
                f"→ {result.output_dir}"
            )
            
            # NEW: Show resume info for MAX_ITERATIONS
            if result.max_iterations_reached:
                lines.append(f"  ⏱️  Max Iterations erreicht!")
                if result.resumable:
                    lines.append(f"  ℹ️  Resumable: Ja (nutze resume_sub_agent('{sub_id}'))")
                else:
                    lines.append(f"  ℹ️  Resumable: Nein (keine Tools verwendet)")

            if result.files_written:
                files_str = ", ".join(result.files_written[:3])
                if len(result.files_written) > 3:
                    files_str += f" (+{len(result.files_written) - 3})"
                lines.append(f"  Files: {files_str}")

            if result.error:
                lines.append(f"  Error: {result.error[:60]}...")

        return "\n".join(lines)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

SUB_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "spawn_sub_agent",
            "description": """Spawn a sub-agent to handle a focused task in parallel.

USE WHEN:
- Task can be split into independent parts
- You want to run multiple searches/analyses in parallel
- A sub-task needs focused attention without context pollution

SUB-AGENT LIMITATIONS:
- Can READ entire VFS
- Can ONLY WRITE to its output_dir (e.g., /sub/research_x/)
- CANNOT spawn further sub-agents (max depth = 1)
- Has fixed token budget
- Gets task in query, no access to current conversation

PROVIDE CLEAR TASKS: Sub-agents cannot ask clarifying questions!""",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear, complete task description. Include ALL context needed."
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory for sub-agent output (e.g., 'research_topic1'). Will be created under /sub/"
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "True = wait for completion. False = run async, use wait_for later.",
                        "default": True
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Token budget for sub-agent (default: 5000)",
                        "default": 5000
                    }
                },
                "required": ["task", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for",
            "description": """Wait for one or more sub-agents to complete.

Use after spawning sub-agents with wait=False.
Returns results from all specified sub-agents.
Results are automatically added to your context.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agent_ids": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ],
                        "description": "Single sub-agent ID or list of IDs to wait for"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 300)",
                        "default": 300
                    }
                },
                "required": ["sub_agent_ids"]
            }
        }
    }
]


# =============================================================================
# VFS WRITE RESTRICTION WRAPPER
# =============================================================================

class RestrictedVFSWrapper:
    """
    Wraps VFS to restrict write operations to a specific directory.

    Used by sub-agents to enforce output isolation.
    """

    def __init__(self, vfs: Any, allowed_write_dir: str):
        """
        Args:
            vfs: Original VFS instance
            allowed_write_dir: Only directory where writes are allowed
        """
        self._vfs = vfs
        self._allowed_dir = allowed_write_dir.rstrip("/")

    def _is_write_allowed(self, path: str) -> bool:
        """Check if write to path is allowed"""
        # Normalize path
        if not path.startswith("/"):
            path = f"/{path}"

        # Allow writes only under allowed_dir
        return path.startswith(self._allowed_dir + "/") or path == self._allowed_dir

    # Read operations - pass through
    def read(self, path: str) -> dict:
        return self._vfs.read(path)

    def list_files(self) -> dict:
        return self._vfs.list_files()

    def ls(self, path: str = "/", recursive: bool = False) -> dict:
        return self._vfs.ls(path, recursive)

    def open(self, path: str, line_start: int = 1, line_end: int = -1) -> dict:
        return self._vfs.open(path, line_start, line_end)

    # Write operations - restricted
    def write(self, path: str, content: str) -> dict:
        if not self._is_write_allowed(path):
            return {
                "success": False,
                "error": f"Sub-agent can only write to {self._allowed_dir}. "
                         f"Attempted: {path}"
            }
        return self._vfs.write(path, content)

    def create(self, path: str, content: str = "") -> dict:
        if not self._is_write_allowed(path):
            return {
                "success": False,
                "error": f"Sub-agent can only create files in {self._allowed_dir}. "
                         f"Attempted: {path}"
            }
        return self._vfs.create(path, content)

    def mkdir(self, path: str, parents: bool = False) -> dict:
        if not self._is_write_allowed(path):
            return {
                "success": False,
                "error": f"Sub-agent can only create directories in {self._allowed_dir}. "
                         f"Attempted: {path}"
            }
        return self._vfs.mkdir(path, parents)

    def rmdir(self, path: str, force: bool = False) -> dict:
        if not self._is_write_allowed(path):
            return {
                "success": False,
                "error": f"Sub-agent can only remove directories in {self._allowed_dir}. "
                         f"Attempted: {path}"
            }
        return self._vfs.rmdir(path, force)

    def mv(self, source: str, destination: str) -> dict:
        # Both source and destination must be in allowed dir for move
        if not self._is_write_allowed(destination):
            return {
                "success": False,
                "error": f"Sub-agent can only move files within {self._allowed_dir}. "
                         f"Destination: {destination}"
            }
        return self._vfs.mv(source, destination)

    # Pass through other methods
    def __getattr__(self, name: str):
        return getattr(self._vfs, name)


# =============================================================================
# SKILL FOR SUB-AGENT USAGE
# =============================================================================

PARALLEL_SUBTASKS_SKILL = {
    "id": "parallel_subtasks",
    "name": "Parallel Sub-Task Execution",
    "triggers": [
        "parallel", "gleichzeitig", "recherchiere mehrere",
        "vergleiche", "sammle von verschiedenen", "und dann zusammen",
        "mehrere quellen", "verschiedene aspekte"
    ],
    "instruction": """Für parallelisierbare Aufgaben:
1. Identifiziere UNABHÄNGIGE Teilaufgaben (die nicht aufeinander warten)
2. Für jede Teilaufgabe:
   spawn_sub_agent(
     task="KLARE, VOLLSTÄNDIGE Beschreibung - Sub-Agent kann nicht nachfragen!",
     output_dir="aufgabe_name",
     wait=False
   )
3. Sammle alle sub_agent_ids
4. wait_for([alle_ids]) - wartet auf alle parallel
5. Lese Ergebnisse aus /sub/[name]/result.md
6. Kombiniere/Vergleiche die Ergebnisse

WICHTIG:
- Sub-Agents können NUR in ihren output_dir schreiben
- Sub-Agents können NICHT zurückfragen - gib ALLE Infos mit
- Sub-Agents können KEINE weiteren Sub-Agents spawnen
- Bei Fehlern: Prüfe /sub/[name]/error.log""",
    "tools_used": ["think", "spawn_sub_agent", "wait_for", "vfs_read", "final_answer"],
    "tool_groups": ["vfs"],
    "source": "predefined"
}
