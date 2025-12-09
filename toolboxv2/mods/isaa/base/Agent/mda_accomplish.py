"""
MAKER Framework Implementation for FlowAgent
============================================

Implements "Massively Decomposed Agentic Processes" (MDAPs) based on the paper:
"Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025)

Key Components:
1. DivideNode - Recursive task decomposition with complexity estimation
2. TaskTreeBuilderNode - Builds execution tree with parallel groups
3. AtomicConquerNode - Executes atomic tasks with k-voting and red-flagging
4. ResultAggregatorNode - Aggregates partial results
5. MDAFlow - Orchestrates the complete MDAP process

Features:
- First-to-ahead-by-k voting for error correction
- Red-flagging to discard unreliable responses
- Stop/Resume with compact checkpoint serialization
- Integration with FlowAgent's existing checkpoint system

Author: Integration with ToolBoxV2 FlowAgent
"""

import asyncio
import functools
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field

from toolboxv2.mods.isaa.base.Agent.types import ProgressEvent, TaskPlan, NodeStatus
# Import from existing framework
from toolboxv2.mods.isaa.base.tbpocketflow import AsyncFlow, AsyncNode

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ============================================================================

class TaskComplexity(BaseModel):
    """Complexity assessment of a task"""
    score: int = Field(ge=0, le=10, description="Complexity 0-10")
    reasoning: str = Field(description="Reasoning for the assessment")
    is_atomic: bool = Field(description="True if cannot be further decomposed")
    estimated_steps: int = Field(ge=1, description="Estimated number of atomic steps")


class SubTask(BaseModel):
    """Single subtask after decomposition"""
    id: str = Field(description="Unique ID")
    description: str = Field(description="Task description")
    relevant_context: str = Field(description="Relevant context for this task")
    complexity: int = Field(ge=0, le=10, description="Complexity 0-10")
    dependencies: list[str] = Field(default_factory=list, description="IDs of predecessor tasks")
    is_atomic: bool = Field(default=False)
    output_schema: Optional[str] = Field(default=None, description="Expected output format")
    # NEW: Action hints
    requires_tools: bool = Field(default=False, description="Whether this task needs tools")
    suggested_tools: list[str] = Field(default_factory=list, description="Tools that might be needed")
    requires_external_context: bool = Field(default=False, description="Needs external context")


class DivisionResult(BaseModel):
    """Result of task division"""
    can_divide: bool = Field(description="Can be further divided")
    subtasks: list[SubTask] = Field(default_factory=list)
    division_reasoning: str = Field(description="Explanation of the division")
    preserved_context: str = Field(description="Context passed to subtasks")
    context_mappings: dict[str, str] = Field(default_factory=dict, description="Context flow between dependent tasks")


class ActionType(str, Enum):
    """Type of action for an atomic task"""
    REASONING = "reasoning"           # Pure LLM reasoning
    TOOL_CALL = "tool_call"           # Execute external tool
    CONTEXT_FETCH = "context_fetch"   # Fetch external context
    MULTI_ACTION = "multi_action"     # Multiple actions in sequence


class ToolCallSpec(BaseModel):
    """Specification for a tool call"""
    tool_name: str = Field(description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    purpose: str = Field(description="Why this tool is needed")
    fallback_on_error: Optional[str] = Field(default=None, description="Fallback action if tool fails")


class ContextFetchSpec(BaseModel):
    """Specification for context fetching"""
    source_type: Literal["variable", "session", "tool", "world_model"] = Field(description="Source type")
    source_path: str = Field(description="Path or identifier for the source")
    query: Optional[str] = Field(default=None, description="Query for filtered fetch")
    transform: Optional[str] = Field(default=None, description="Transformation to apply")


class AtomicAction(BaseModel):
    """Single atomic action within a task"""
    action_type: ActionType
    reasoning_prompt: Optional[str] = Field(default=None, description="Prompt for reasoning")
    tool_call: Optional[ToolCallSpec] = Field(default=None, description="Tool call specification")
    context_fetch: Optional[ContextFetchSpec] = Field(default=None, description="Context fetch specification")
    depends_on_action: Optional[int] = Field(default=None, description="Index of action this depends on")


class TaskActionPlan(BaseModel):
    """Plan of actions for an atomic task"""
    requires_tools: bool = Field(default=False, description="Whether tools are needed")
    requires_context: bool = Field(default=False, description="Whether external context is needed")
    actions: list[AtomicAction] = Field(default_factory=list, description="Sequence of actions")
    final_synthesis: bool = Field(default=True, description="Whether to synthesize results")
    available_tools_used: list[str] = Field(default_factory=list, description="Tools that will be used")


class AtomicResult(BaseModel):
    """Result of an atomic execution"""
    success: bool
    result: str = Field(description="Partial solution or result")
    context_for_next: str = Field(description="Context for subsequent tasks")
    confidence: float = Field(ge=0, le=1)
    red_flags: list[str] = Field(default_factory=list, description="Detected warning signs")
    execution_time_ms: float = Field(default=0)
    # NEW: Action tracking
    actions_executed: list[dict] = Field(default_factory=list, description="Actions that were executed")
    tool_results: dict[str, Any] = Field(default_factory=dict, description="Results from tool calls")
    context_fetched: dict[str, Any] = Field(default_factory=dict, description="Context that was fetched")


class VotingCandidate(BaseModel):
    """Candidate for voting"""
    result: AtomicResult
    hash: str = Field(description="Hash for comparison")
    votes: int = Field(default=1)


class AggregatedResult(BaseModel):
    """Final aggregated result"""
    success: bool
    final_result: str
    partial_results: dict[str, str] = Field(default_factory=dict)
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_voting_rounds: int
    red_flags_caught: int


# ============================================================================
# CHECKPOINT DATA STRUCTURES (Compact & Serializable)
# ============================================================================

class MDATaskStatus(str, Enum):
    """Status of an MDA task"""
    PENDING = "pending"
    DIVIDING = "dividing"
    READY = "ready"
    EXECUTING = "executing"
    VOTING = "voting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class MDATaskNode:
    """Compact task node for checkpoint serialization"""
    id: str
    description: str
    context: str
    complexity: int
    dependencies: list[str]
    is_atomic: bool
    status: MDATaskStatus
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    result: Optional[dict] = None
    votes: list[dict] = field(default_factory=list)
    execution_attempts: int = 0
    parallel_group: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    # NEW: Action-related fields
    requires_tools: bool = False
    suggested_tools: list[str] = field(default_factory=list)
    requires_external_context: bool = False
    action_plan: Optional[dict] = None  # Serialized TaskActionPlan
    tool_results: dict = field(default_factory=dict)
    fetched_context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "description": self.description[:500],  # Truncate for compactness
            "context": self.context[:1000],  # Truncate context
            "complexity": self.complexity,
            "dependencies": self.dependencies,
            "is_atomic": self.is_atomic,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "result": self.result,
            "votes": self.votes[-10:] if self.votes else [],  # Keep last 10 votes
            "execution_attempts": self.execution_attempts,
            "parallel_group": self.parallel_group,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            # NEW fields
            "requires_tools": self.requires_tools,
            "suggested_tools": self.suggested_tools,
            "requires_external_context": self.requires_external_context,
            "action_plan": self.action_plan,
            "tool_results": self.tool_results,
            "fetched_context": self.fetched_context
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MDATaskNode":
        """Create from dictionary"""
        data["status"] = MDATaskStatus(data["status"])
        # Handle new fields with defaults for backwards compatibility
        data.setdefault("requires_tools", False)
        data.setdefault("suggested_tools", [])
        data.setdefault("requires_external_context", False)
        data.setdefault("action_plan", None)
        data.setdefault("tool_results", {})
        data.setdefault("fetched_context", {})
        return cls(**data)


@dataclass
class MDACheckpoint:
    """Compact checkpoint for MDA process - integrates with AgentCheckpoint"""
    # Identification
    checkpoint_id: str
    original_task: str
    original_context: str
    session_id: str

    # Configuration
    config: dict  # min_complexity, max_parallel, k_margin, etc.

    # Task Tree State (compact)
    task_nodes: dict[str, dict]  # id -> MDATaskNode.to_dict()
    root_task_id: str

    # Execution State
    current_parallel_group: int
    completed_groups: list[int]
    pending_task_ids: list[str]
    executing_task_ids: list[str]
    completed_task_ids: list[str]
    failed_task_ids: list[str]

    # Results (compact)
    results: dict[str, dict]  # task_id -> {result, context_for_next}

    # Statistics
    stats: dict  # total_divisions, voting_rounds, red_flags, etc.

    # Timestamps
    created_at: str
    last_updated: str
    paused_at: Optional[str] = None

    # Version for compatibility
    version: str = "1.0"

    def to_dict(self) -> dict:
        """Serialize to compact dictionary"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "original_task": self.original_task[:500],
            "original_context": self.original_context[:1000],
            "session_id": self.session_id,
            "config": self.config,
            "task_nodes": self.task_nodes,
            "root_task_id": self.root_task_id,
            "current_parallel_group": self.current_parallel_group,
            "completed_groups": self.completed_groups,
            "pending_task_ids": self.pending_task_ids,
            "executing_task_ids": self.executing_task_ids,
            "completed_task_ids": self.completed_task_ids,
            "failed_task_ids": self.failed_task_ids,
            "results": {k: {
                "result": v.get("result", "")[:500],
                "context_for_next": v.get("context_for_next", "")[:300]
            } for k, v in self.results.items()},
            "stats": self.stats,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "paused_at": self.paused_at,
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MDACheckpoint":
        """Deserialize from dictionary"""
        return cls(**data)

    def get_resumable_tasks(self) -> list[str]:
        """Get tasks that can be resumed"""
        resumable = []
        for task_id in self.pending_task_ids + self.executing_task_ids:
            task_data = self.task_nodes.get(task_id)
            if task_data:
                # Check if dependencies are satisfied
                deps_satisfied = all(
                    dep_id in self.completed_task_ids
                    for dep_id in task_data.get("dependencies", [])
                )
                if deps_satisfied:
                    resumable.append(task_id)
        return resumable


# ============================================================================
# ASYNC NODES FOR MDA PROCESS
# ============================================================================
def with_progress_tracking(cls):
    """
    Ein Klassendekorator, der die Methoden run_async, prep_async, exec_async,
    und exec_fallback_async automatisch mit umfassendem Progress-Tracking umwickelt.
    """

    # --- Wrapper für run_async ---
    original_run = getattr(cls, 'run_async', None)
    if original_run:
        @functools.wraps(original_run)
        async def wrapped_run_async(self, shared):
            progress_tracker = shared.get("progress_tracker")
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_run(self, shared)

            timer_key = f"{node_name}_total"
            progress_tracker.start_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_enter",
                timestamp=time.time(),
                node_name=node_name,
                session_id=shared.get("session_id"),
                task_id=shared.get("current_task_id"),
                plan_id=shared.get("current_plan", TaskPlan(id="none", name="none", description="none")).id if shared.get("current_plan") else None,
                status=NodeStatus.RUNNING,
                success=None
            ))

            try:
                # Hier wird die ursprüngliche Methode aufgerufen
                result = await original_run(self, shared)

                total_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_exit",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.COMPLETED,
                    success=True,
                    node_duration=total_duration,
                    routing_decision=result,
                    session_id=shared.get("session_id"),
                    task_id=shared.get("current_task_id"),
                    metadata={"success": True}
                ))

                return result
            except Exception as e:
                total_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="error",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.FAILED,
                    success=False,
                    node_duration=total_duration,
                    session_id=shared.get("session_id"),
                    metadata={"error": str(e), "error_type": type(e).__name__}
                ))
                raise

        cls.run_async = wrapped_run_async

    # --- Wrapper für prep_async ---
    original_prep = getattr(cls, 'prep_async', None)
    if original_prep:
        @functools.wraps(original_prep)
        async def wrapped_prep_async(self, shared):
            progress_tracker = shared.get("progress_tracker")
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_prep(self, shared)
            timer_key = f"{node_name}_total_p"
            progress_tracker.start_timer(timer_key)
            timer_key = f"{node_name}_prep"
            progress_tracker.start_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.STARTING,
                node_phase="prep",
                session_id=shared.get("session_id")
            ))

            try:
                result = await original_prep(self, shared)

                prep_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_phase",
                    timestamp=time.time(),
                    status=NodeStatus.RUNNING,
                    success=True,
                    node_name=node_name,
                    node_phase="prep_complete",
                    node_duration=prep_duration,
                    session_id=shared.get("session_id")
                ))
                return result
            except Exception as e:
                progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="error",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.FAILED,
                    success=False,
                    metadata={"error": str(e), "error_type": type(e).__name__},
                    node_phase="prep_failed"
                ))
                raise


        cls.prep_async = wrapped_prep_async

    # --- Wrapper für exec_async ---
    original_exec = getattr(cls, 'exec_async', None)
    if original_exec:
        @functools.wraps(original_exec)
        async def wrapped_exec_async(self, prep_res):
            progress_tracker = prep_res.get("progress_tracker") if isinstance(prep_res, dict) else None
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_exec(self, prep_res)

            timer_key = f"{node_name}_exec"
            progress_tracker.start_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.RUNNING,
                node_phase="exec",
                session_id=prep_res.get("session_id") if isinstance(prep_res, dict) else None
            ))

            # In exec gibt es normalerweise keine Fehlerbehandlung, da diese von run_async übernommen wird
            result = await original_exec(self, prep_res)

            exec_duration = progress_tracker.end_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.RUNNING,
                success=True,
                node_phase="exec_complete",
                node_duration=exec_duration,
                session_id=prep_res.get("session_id") if isinstance(prep_res, dict) else None
            ))
            return result

        cls.exec_async = wrapped_exec_async

    # --- Wrapper für post_async ---
    original_post = getattr(cls, 'post_async', None)
    if original_post:
        @functools.wraps(original_post)
        async def wrapped_post_async(self, shared, prep_res, exec_res):
            if isinstance(exec_res, str):
                print("exec_res is string:", exec_res)
            progress_tracker = shared.get("progress_tracker")
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_post(self, shared, prep_res, exec_res)

            timer_key_post = f"{node_name}_post"
            progress_tracker.start_timer(timer_key_post)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.COMPLETING,  # Neue Phase "completing"
                node_phase="post",
                session_id=shared.get("session_id")
            ))

            try:
                # Die eigentliche post_async Methode aufrufen
                result = await original_post(self, shared, prep_res, exec_res)

                post_duration = progress_tracker.end_timer(timer_key_post)
                total_duration = progress_tracker.end_timer(f"{node_name}_total_p")  # Gesamtdauer stoppen

                # Sende das entscheidende "node_exit" Event nach erfolgreicher post-Phase
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_exit",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.COMPLETED,
                    success=True,
                    node_duration=total_duration,
                    routing_decision=result,
                    session_id=shared.get("session_id"),
                    task_id=shared.get("current_task_id"),
                    metadata={
                        "success": True,
                        "post_duration": post_duration
                    }
                ))

                return result
            except Exception as e:
                # Fehler in der post-Phase

                post_duration = progress_tracker.end_timer(timer_key_post)
                total_duration = progress_tracker.end_timer(f"{node_name}_total")
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="error",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.FAILED,
                    success=False,
                    node_duration=total_duration,
                    metadata={"error": str(e), "error_type": type(e).__name__, "phase": "post"},
                    node_phase="post_failed"
                ))
                raise

        cls.post_async = wrapped_post_async

    # --- Wrapper für exec_fallback_async ---
    original_fallback = getattr(cls, 'exec_fallback_async', None)
    if original_fallback:
        @functools.wraps(original_fallback)
        async def wrapped_fallback_async(self, prep_res, exc):
            progress_tracker = prep_res.get("progress_tracker") if isinstance(prep_res, dict) else None
            node_name = self.__class__.__name__

            if progress_tracker:
                timer_key = f"{node_name}_exec"
                exec_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_phase",
                    timestamp=time.time(),
                    node_name=node_name,
                    node_phase="exec_fallback",
                    node_duration=exec_duration,
                    status=NodeStatus.FAILED,
                    success=False,
                    session_id=prep_res.get("session_id") if isinstance(prep_res, dict) else None,
                    metadata={"error": str(exc), "error_type": type(exc).__name__},
                ))

            return await original_fallback(self, prep_res, exc)

        cls.exec_fallback_async = wrapped_fallback_async

    return cls


@with_progress_tracking
class DivideNode(AsyncNode):
    """
    Recursively divides tasks until minimum complexity is reached.
    Implements MAD (Maximal Agentic Decomposition) from MAKER paper.

    NEW: Detects when subtasks require external tools or context.
    """

    def __init__(self,
                 min_complexity: int = 2,
                 max_subtasks: int = 5,
                 model_strength: Literal["weak", "medium", "strong"] = "medium"):
        super().__init__()
        self.min_complexity = min_complexity
        self.max_subtasks_map = {"weak": 2, "medium": 3, "strong": 5}
        self.max_subtasks = self.max_subtasks_map.get(model_strength, 3)
        self.model_strength = model_strength

    async def prep_async(self, shared) -> dict:
        """Prepare for division"""
        # Get available tools for context
        agent = shared.get("agent_instance")
        available_tools = []
        if agent and hasattr(agent, '_tool_registry'):
            available_tools = list(agent._tool_registry.keys())

        return {
            "task_node": shared.get("current_task_node"),
            "agent_instance": agent,
            "mda_state": shared.get("mda_state"),
            "depth": shared.get("division_depth", 0),
            "max_depth": shared.get("max_division_depth", 10),
            "session_id": shared.get("session_id"),
            "is_paused": shared.get("mda_paused", False),
            "available_tools": available_tools
        }

    async def exec_async(self, prep_res) -> dict:
        """Execute task division"""
        if prep_res.get("is_paused"):
            return {"action": "paused", "reason": "MDA process paused"}

        task_node: MDATaskNode = prep_res["task_node"]
        agent = prep_res["agent_instance"]
        depth = prep_res["depth"]
        max_depth = prep_res["max_depth"]
        available_tools = prep_res.get("available_tools", [])

        # Check depth limit
        if depth >= max_depth:
            return {
                "action": "force_atomic",
                "task_node": task_node,
                "reason": f"Max depth {max_depth} reached"
            }

        # 1. Estimate complexity (with tool awareness)
        complexity = await self._estimate_complexity(
            task_node.description,
            task_node.context,
            agent,
            prep_res.get("session_id"),
            available_tools
        )

        # 2. Check if atomic
        if complexity.is_atomic or complexity.score <= self.min_complexity:
            task_node.is_atomic = True
            task_node.complexity = complexity.score
            task_node.status = MDATaskStatus.READY
            return {
                "action": "atomic",
                "task_node": task_node,
                "complexity": complexity.model_dump()
            }

        # 3. Divide task (with tool detection)
        task_node.status = MDATaskStatus.DIVIDING
        division = await self._divide_task(
            task_node,
            complexity,
            agent,
            prep_res.get("session_id"),
            available_tools
        )

        return {
            "action": "divided",
            "task_node": task_node,
            "division": division.model_dump(),
            "subtasks": [st.model_dump() for st in division.subtasks]
        }

    async def _estimate_complexity(self, task: str, context: str,
                                    agent, session_id: str,
                                    available_tools: list) -> TaskComplexity:
        """Estimate task complexity using LLM"""
        # Include tool info for better estimation
        tools_hint = ""
        if available_tools:
            tools_hint = f"\n\nVERFÜGBARE TOOLS (können Komplexität reduzieren): {', '.join(available_tools[:10])}"

        prompt = f"""Bewerte die Komplexität dieser Aufgabe auf einer Skala von 0-10:

AUFGABE: {task}

KONTEXT: {context[:800]}{tools_hint}

BEWERTUNGSKRITERIEN:
- 0-2: Trivial, kann direkt mit einer Antwort gelöst werden
- 3-4: Einfach, erfordert 1-2 logische Schritte
- 5-6: Mittel, erfordert mehrere Schritte oder Informationssammlung
- 7-8: Komplex, viele abhängige Schritte erforderlich
- 9-10: Sehr komplex, erfordert umfangreiche Zerlegung

WICHTIG:
- is_atomic = true wenn die Aufgabe NICHT weiter sinnvoll zerlegbar ist
- estimated_steps = geschätzte Anzahl atomarer Ausführungsschritte
- Wenn ein Tool die Aufgabe direkt lösen kann, ist sie oft atomar"""

        try:
            result = await agent.a_format_class(
                pydantic_model=TaskComplexity,
                prompt=prompt,
                model_preference="fast",
                max_retries=2,
                auto_context=False,
                session_id=session_id
            )
            return TaskComplexity(**result)
        except Exception as e:
            # Fallback: assume medium complexity
            return TaskComplexity(
                score=5,
                reasoning=f"Fallback due to error: {str(e)}",
                is_atomic=False,
                estimated_steps=3
            )

    async def _divide_task(self, task_node: MDATaskNode,
                           complexity: TaskComplexity,
                           agent, session_id: str,
                           available_tools: list) -> DivisionResult:
        """Divide task into subtasks with tool detection"""

        # Build tools info for prompt
        tools_info = ""
        if available_tools:
            tools_info = f"""

VERFÜGBARE TOOLS:
{chr(10).join(['- ' + t for t in available_tools[:15]])}

Wenn eine Unteraufgabe ein Tool verwenden könnte:
- Setze requires_tools = true
- Liste die passenden Tools in suggested_tools
- Beispiel: Aufgabe "Lies Datei X" → requires_tools=true, suggested_tools=["file_read"]"""

        prompt = f"""Zerlege diese Aufgabe in maximal {self.max_subtasks} Unteraufgaben:

HAUPTAUFGABE: {task_node.description}

KONTEXT: {task_node.context[:1000]}

KOMPLEXITÄT: {complexity.score}/10 ({complexity.reasoning}){tools_info}

REGELN FÜR DIE ZERLEGUNG:

1. UNABHÄNGIGKEIT: Jede Unteraufgabe muss so unabhängig wie möglich sein
2. ABHÄNGIGKEITEN: Wenn eine Aufgabe das Ergebnis einer anderen benötigt:
   - Markiere die Abhängigkeit explizit in dependencies
   - Definiere welcher Kontext weitergegeben werden muss
3. KONTEXT: Jede Unteraufgabe braucht ihren eigenen relevanten Kontext
4. ATOMARITÄT: Unteraufgaben sollten möglichst einfach sein (Komplexität < 5)
5. TOOLS: Wenn eine Aufgabe Tools verwenden sollte:
   - requires_tools = true
   - suggested_tools = ["tool_name1", "tool_name2"]
6. EXTERNE DATEN: Wenn externe Daten benötigt werden:
   - requires_external_context = true

WICHTIG für context_mappings:
- Format: {{"task_id_abhängig": "Beschreibung welcher Kontext von welcher Aufgabe kommt"}}
- Beispiel: {{"task_2": "Ergebnis von task_1 als Input"}}"""

        try:
            result = await agent.a_format_class(
                pydantic_model=DivisionResult,
                prompt=prompt,
                model_preference="fast" if complexity.score < 7 else "complex",
                max_retries=2,
                auto_context=False,
                session_id=session_id
            )

            # Ensure subtask IDs are unique
            division = DivisionResult(**result)
            for i, subtask in enumerate(division.subtasks):
                if not subtask.id or subtask.id in [st.id for st in division.subtasks[:i]]:
                    subtask.id = f"{task_node.id}_sub_{i}_{uuid.uuid4().hex[:6]}"

                # Validate suggested_tools against available tools
                if subtask.suggested_tools:
                    subtask.suggested_tools = [
                        t for t in subtask.suggested_tools
                        if t in available_tools
                    ]

            return division

        except Exception as e:
            # Fallback: create single atomic subtask
            return DivisionResult(
                can_divide=False,
                subtasks=[SubTask(
                    id=f"{task_node.id}_atomic",
                    description=task_node.description,
                    relevant_context=task_node.context,
                    complexity=complexity.score,
                    is_atomic=True
                )],
                division_reasoning=f"Fallback to atomic due to: {str(e)}",
                preserved_context=task_node.context
            )

    async def post_async(self, shared, prep_res, exec_res) -> str:
        """Update state after division"""
        mda_state: MDAState = shared.get("mda_state")

        if exec_res["action"] == "paused":
            return "paused"

        task_node = exec_res["task_node"]

        if exec_res["action"] in ["atomic", "force_atomic"]:
            # Task is atomic, ready for execution
            mda_state.mark_task_ready(task_node.id)
            shared["atomic_tasks_ready"] = shared.get("atomic_tasks_ready", []) + [task_node.id]

            # Check if all divisions complete
            if not mda_state.has_pending_divisions():
                return "all_divided"
            return "continue_division"

        elif exec_res["action"] == "divided":
            # Create child task nodes
            division = exec_res["division"]
            subtasks_data = exec_res["subtasks"]

            child_ids = []
            for st_data in subtasks_data:
                child_node = MDATaskNode(
                    id=st_data["id"],
                    description=st_data["description"],
                    context=st_data["relevant_context"],
                    complexity=st_data["complexity"],
                    dependencies=st_data["dependencies"],
                    is_atomic=st_data["is_atomic"],
                    status=MDATaskStatus.PENDING,
                    parent_id=task_node.id,
                    # NEW: Tool-related fields
                    requires_tools=st_data.get("requires_tools", False),
                    suggested_tools=st_data.get("suggested_tools", []),
                    requires_external_context=st_data.get("requires_external_context", False)
                )
                mda_state.add_task_node(child_node)
                child_ids.append(child_node.id)

                # Add to pending divisions if not atomic
                if not child_node.is_atomic:
                    mda_state.pending_divisions.append(child_node.id)

            # Update parent
            task_node.children_ids = child_ids
            task_node.status = MDATaskStatus.COMPLETED
            mda_state.update_task_node(task_node)
            mda_state.stats["total_divisions"] += 1

            # Continue with next pending division
            if mda_state.has_pending_divisions():
                next_task_id = mda_state.pending_divisions.pop(0)
                shared["current_task_node"] = mda_state.get_task_node(next_task_id)
                shared["division_depth"] = prep_res["depth"] + 1
                return "continue_division"

            return "all_divided"

        return "error"

@with_progress_tracking
class TaskTreeBuilderNode(AsyncNode):
    """
    Builds execution tree with parallel groups from atomic tasks.
    Identifies independent tasks for parallel execution.
    """

    async def prep_async(self, shared) -> dict:
        return {
            "mda_state": shared.get("mda_state"),
            "max_parallel": shared.get("max_parallel", 5),
            "is_paused": shared.get("mda_paused", False)
        }

    async def exec_async(self, prep_res) -> dict:
        if prep_res.get("is_paused"):
            return {"action": "paused"}

        mda_state: MDAState = prep_res["mda_state"]
        max_parallel = prep_res["max_parallel"]

        # Get all atomic tasks
        atomic_tasks = mda_state.get_atomic_tasks()

        if not atomic_tasks:
            return {"action": "no_tasks", "parallel_groups": []}

        # Build dependency graph
        dep_graph = {}
        for task in atomic_tasks:
            dep_graph[task.id] = task.dependencies

        # Topological sort with parallel groups
        parallel_groups = self._build_parallel_groups(atomic_tasks, dep_graph, max_parallel)

        # Assign parallel group to each task
        for group_idx, group in enumerate(parallel_groups):
            for task_id in group:
                task = mda_state.get_task_node(task_id)
                if task:
                    task.parallel_group = group_idx
                    mda_state.update_task_node(task)

        return {
            "action": "tree_built",
            "parallel_groups": parallel_groups,
            "total_groups": len(parallel_groups),
            "total_tasks": len(atomic_tasks),
            "max_parallelism": max(len(g) for g in parallel_groups) if parallel_groups else 0
        }

    def _build_parallel_groups(self, tasks: list[MDATaskNode],
                                dep_graph: dict, max_parallel: int) -> list[list[str]]:
        """Build groups of tasks that can execute in parallel"""
        task_ids = {t.id for t in tasks}
        completed = set()
        groups = []

        while len(completed) < len(tasks):
            # Find tasks with all dependencies satisfied
            ready = []
            for task in tasks:
                if task.id not in completed:
                    # Filter dependencies to only include tasks in our set
                    relevant_deps = [d for d in dep_graph.get(task.id, []) if d in task_ids]
                    if all(d in completed for d in relevant_deps):
                        ready.append(task.id)

            if not ready:
                # Deadlock detection - force remaining tasks
                remaining = [t.id for t in tasks if t.id not in completed]
                if remaining:
                    ready = remaining[:max_parallel]

            # Limit group size
            group = ready[:max_parallel]
            groups.append(group)
            completed.update(group)

        return groups

    async def post_async(self, shared, prep_res, exec_res) -> str:
        if exec_res["action"] == "paused":
            return "paused"

        mda_state: MDAState = shared.get("mda_state")

        if exec_res["action"] == "no_tasks":
            return "no_tasks"

        mda_state.parallel_groups = exec_res["parallel_groups"]
        mda_state.current_group_index = 0
        shared["parallel_groups"] = exec_res["parallel_groups"]

        return "tree_built"

@with_progress_tracking
class AtomicConquerNode(AsyncNode):
    """
    Executes atomic tasks with k-voting and red-flagging.
    Implements error correction from MAKER paper.

    NEW: Supports external tool calls and context fetching
    for tasks that require interaction with the outside world.
    """

    def __init__(self,
                 num_attempts: int = 3,
                 k_margin: int = 2,
                 max_response_tokens: int = 750,
                 red_flag_patterns: list[str] = None,
                 enable_tools: bool = True,
                 enable_context_fetch: bool = True):
        super().__init__()
        self.num_attempts = num_attempts
        self.k_margin = k_margin
        self.max_response_tokens = max_response_tokens
        self.enable_tools = enable_tools
        self.enable_context_fetch = enable_context_fetch
        self.red_flag_patterns = red_flag_patterns or [
            r"(?i)ich bin (mir )?nicht sicher",
            r"(?i)das ist (sehr )?komplex",
            r"(?i)ich kann (das )?nicht",
            r"(?i)es ist schwierig",
            r"(?i)möglicherweise",
            r"(?i)vielleicht",
            r"(?i)i('m| am) not sure",
            r"(?i)that is (very )?complex",
            r"(?i)i can('t|not)( do this)?",
            r"(?i)it('s| is) difficult",
            r"(?i)possibly",
            r"(?i)maybe"
        ]

    async def prep_async(self, shared) -> dict:
        mda_state: MDAState = shared.get("mda_state")

        # Get current group to execute
        parallel_groups = mda_state.parallel_groups
        current_idx = mda_state.current_group_index

        if current_idx >= len(parallel_groups):
            return {"action": "all_complete", "tasks": []}

        current_group = parallel_groups[current_idx]
        tasks_to_execute = []

        for task_id in current_group:
            task = mda_state.get_task_node(task_id)
            if task and task.status in [MDATaskStatus.READY, MDATaskStatus.PENDING]:
                tasks_to_execute.append(task)

        # Get available tools from agent
        agent = shared.get("agent_instance")
        available_tools = []
        tool_descriptions = {}

        if agent and self.enable_tools:
            available_tools = list(agent._tool_registry.keys()) if hasattr(agent, '_tool_registry') else []
            # Get tool descriptions for LLM context
            for tool_name in available_tools[:20]:  # Limit to 20 tools
                tool_info = agent._tool_registry.get(tool_name, {})
                tool_descriptions[tool_name] = {
                    "description": tool_info.get("description", ""),
                    "args_schema": tool_info.get("args_schema", "()")
                }

        return {
            "tasks": tasks_to_execute,
            "agent_instance": agent,
            "mda_state": mda_state,
            "session_id": shared.get("session_id"),
            "is_paused": shared.get("mda_paused", False),
            "group_index": current_idx,
            "available_tools": available_tools,
            "tool_descriptions": tool_descriptions,
            "variable_manager": shared.get("variable_manager")
        }

    async def exec_async(self, prep_res) -> dict:
        if prep_res.get("is_paused"):
            return {"action": "paused", "results": []}

        if prep_res.get("action") == "all_complete":
            return {"action": "all_complete", "results": []}

        tasks = prep_res["tasks"]
        if not tasks:
            return {"action": "group_empty", "results": []}

        agent = prep_res["agent_instance"]
        mda_state = prep_res["mda_state"]
        session_id = prep_res["session_id"]
        available_tools = prep_res["available_tools"]
        tool_descriptions = prep_res["tool_descriptions"]
        variable_manager = prep_res["variable_manager"]

        # Execute tasks in parallel
        execution_tasks = [
            self._execute_with_voting(
                task, agent, mda_state, session_id,
                available_tools, tool_descriptions, variable_manager
            )
            for task in tasks
        ]

        results = await asyncio.gather(*execution_tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                processed_results.append({
                    "task_id": task.id,
                    "success": False,
                    "error": str(result),
                    "result": None
                })
            else:
                processed_results.append({
                    "task_id": task.id,
                    "success": result.success,
                    "result": result.model_dump(),
                    "error": None
                })

        return {
            "action": "group_executed",
            "results": processed_results,
            "group_index": prep_res["group_index"]
        }

    async def _execute_with_voting(self, task: MDATaskNode, agent,
                                    mda_state: "MDAState", session_id: str,
                                    available_tools: list, tool_descriptions: dict,
                                    variable_manager) -> AtomicResult:
        """Execute task with k-voting and red-flagging, including tool support"""
        task.status = MDATaskStatus.EXECUTING
        mda_state.update_task_node(task)

        # Build context from dependencies
        base_context = self._build_execution_context(task, mda_state)

        # Step 1: Plan actions if task might need tools/context
        action_plan = None
        if self.enable_tools and (task.requires_tools or task.suggested_tools):
            action_plan = await self._plan_actions(
                task, base_context, agent, session_id,
                available_tools, tool_descriptions
            )
            task.action_plan = action_plan.model_dump() if action_plan else None

        # Step 2: Execute pre-actions (context fetch, tool calls)
        enriched_context = base_context
        tool_results = {}
        fetched_context = {}

        if action_plan and action_plan.actions:
            pre_result = await self._execute_pre_actions(
                action_plan, task, agent, session_id, variable_manager, mda_state
            )
            enriched_context = pre_result["enriched_context"]
            tool_results = pre_result["tool_results"]
            fetched_context = pre_result["fetched_context"]

            # Store in task for checkpoint
            task.tool_results = tool_results
            task.fetched_context = fetched_context

        # Step 3: Collect votes with enriched context
        votes: list[VotingCandidate] = []
        valid_results = []

        for attempt in range(self.num_attempts * 2):  # Allow extra attempts for red-flagged
            if len(valid_results) >= self.num_attempts:
                break

            result = await self._execute_single_attempt(
                task, enriched_context, agent, session_id, attempt,
                tool_results, fetched_context
            )

            # Red-flag check
            if self._has_red_flags(result):
                mda_state.stats["red_flags_caught"] += 1
                continue

            valid_results.append(result)

            # Add to voting
            result_hash = self._hash_result(result)
            existing = next((v for v in votes if v.hash == result_hash), None)

            if existing:
                existing.votes += 1
            else:
                votes.append(VotingCandidate(
                    result=result,
                    hash=result_hash,
                    votes=1
                ))

            # Check k-margin victory
            winner = self._check_k_margin_victory(votes)
            if winner:
                mda_state.stats["voting_rounds"] += len(valid_results)
                return winner.result

        # No clear winner - return best candidate
        if votes:
            best = max(votes, key=lambda v: (v.votes, v.result.confidence))
            mda_state.stats["voting_rounds"] += len(valid_results)
            return best.result

        # All attempts failed
        return AtomicResult(
            success=False,
            result="All attempts failed or were red-flagged",
            context_for_next="",
            confidence=0.0,
            red_flags=["all_attempts_failed"],
            tool_results=tool_results,
            context_fetched=fetched_context
        )

    async def _plan_actions(self, task: MDATaskNode, context: str,
                            agent, session_id: str,
                            available_tools: list, tool_descriptions: dict) -> Optional[TaskActionPlan]:
        """Plan what actions are needed for this task"""

        # Build tool description string
        tools_info = "\n".join([
            f"- {name}{desc.get('args_schema', '()')}: {desc.get('description', 'No description')}"
            for name, desc in list(tool_descriptions.items())[:15]
        ])

        prompt = f"""Analysiere diese atomare Aufgabe und plane die notwendigen Aktionen:

AUFGABE: {task.description}

KONTEXT: {context[:800]}

VERFÜGBARE TOOLS:
{tools_info}

ANALYSE:
1. Kann diese Aufgabe NUR durch Reasoning gelöst werden?
2. Werden externe Daten oder Tools benötigt?
3. Welche Aktionen sind in welcher Reihenfolge nötig?

REGELN:
- requires_tools = true NUR wenn ein Tool-Aufruf NOTWENDIG ist
- Wenn kein Tool nötig: actions = [] und requires_tools = false
- Tool-Aufrufe müssen die exakten Tool-Namen aus der Liste verwenden
- Jede Aktion muss atomar und unabhängig testbar sein"""

        try:
            result = await agent.a_format_class(
                pydantic_model=TaskActionPlan,
                prompt=prompt,
                model_preference="fast",
                max_retries=1,
                auto_context=False,
                session_id=session_id
            )
            return TaskActionPlan(**result)
        except Exception:
            # Default: no special actions needed
            return TaskActionPlan(
                requires_tools=False,
                requires_context=False,
                actions=[],
                final_synthesis=True
            )

    async def _execute_pre_actions(self, action_plan: TaskActionPlan,
                                    task: MDATaskNode, agent,
                                    session_id: str, variable_manager,
                                    mda_state: "MDAState") -> dict:
        """Execute tool calls and context fetches before main reasoning"""
        tool_results = {}
        fetched_context = {}
        enriched_context_parts = [task.context]

        for i, action in enumerate(action_plan.actions):
            try:
                if action.action_type == ActionType.TOOL_CALL and action.tool_call:
                    # Execute tool call atomically
                    tool_result = await self._execute_tool_call(
                        action.tool_call, agent, session_id
                    )
                    tool_results[action.tool_call.tool_name] = tool_result
                    enriched_context_parts.append(
                        f"\n[Tool {action.tool_call.tool_name}]: {str(tool_result)[:500]}"
                    )
                    mda_state.stats["tool_calls"] = mda_state.stats.get("tool_calls", 0) + 1

                elif action.action_type == ActionType.CONTEXT_FETCH and action.context_fetch:
                    # Fetch external context
                    fetch_result = await self._execute_context_fetch(
                        action.context_fetch, agent, variable_manager, session_id
                    )
                    fetched_context[action.context_fetch.source_path] = fetch_result
                    enriched_context_parts.append(
                        f"\n[Context {action.context_fetch.source_path}]: {str(fetch_result)[:500]}"
                    )
                    mda_state.stats["context_fetches"] = mda_state.stats.get("context_fetches", 0) + 1

            except Exception as e:
                # Log error but continue - the main reasoning might still work
                error_msg = f"Action {i} failed: {str(e)}"
                enriched_context_parts.append(f"\n[Error]: {error_msg}")

        return {
            "enriched_context": "\n".join(enriched_context_parts),
            "tool_results": tool_results,
            "fetched_context": fetched_context
        }

    async def _execute_tool_call(self, tool_spec: ToolCallSpec,
                                  agent, session_id: str) -> Any:
        """Execute a single tool call atomically"""
        try:
            # Use agent's arun_function for tool execution
            result = await agent.arun_function(
                tool_spec.tool_name,
                **tool_spec.arguments
            )
            return result
        except Exception as e:
            if tool_spec.fallback_on_error:
                return f"Tool failed, fallback: {tool_spec.fallback_on_error}"
            raise e

    async def _execute_context_fetch(self, fetch_spec: ContextFetchSpec,
                                      agent, variable_manager,
                                      session_id: str) -> Any:
        """Fetch external context atomically"""
        try:
            if fetch_spec.source_type == "variable":
                # Fetch from variable manager
                if variable_manager:
                    return variable_manager.get(fetch_spec.source_path)
                return None

            elif fetch_spec.source_type == "session":
                # Fetch from session context
                if agent and hasattr(agent, 'context_manager'):
                    context = await agent.get_context(
                        session_id=session_id,
                        format_for_llm=True
                    )
                    return context
                return None

            elif fetch_spec.source_type == "world_model":
                # Fetch from world model
                if agent and hasattr(agent, 'world_model'):
                    return agent.world_model.get(fetch_spec.source_path)
                return None

            elif fetch_spec.source_type == "tool":
                # Use a tool to fetch context (e.g., web_search, file_read)
                if agent and fetch_spec.query:
                    result = await agent.arun_function(
                        fetch_spec.source_path,  # Tool name
                        query=fetch_spec.query
                    )
                    return result
                return None

        except Exception as e:
            return f"Context fetch failed: {str(e)}"

    def _build_execution_context(self, task: MDATaskNode, mda_state: "MDAState") -> str:
        """Build context from task dependencies"""
        context_parts = [task.context]

        for dep_id in task.dependencies:
            dep_result = mda_state.results.get(dep_id)
            if dep_result:
                context_parts.append(
                    f"\n[Ergebnis von {dep_id}]: {dep_result.get('context_for_next', dep_result.get('result', ''))}"
                )

            # Also include tool results from dependencies
            dep_task = mda_state.get_task_node(dep_id)
            if dep_task and dep_task.tool_results:
                for tool_name, tool_result in dep_task.tool_results.items():
                    context_parts.append(
                        f"\n[Tool {tool_name} von {dep_id}]: {str(tool_result)[:300]}"
                    )

        return "\n".join(context_parts)

    async def _execute_single_attempt(self, task: MDATaskNode, context: str,
                                       agent, session_id: str, attempt: int,
                                       tool_results: dict = None,
                                       fetched_context: dict = None) -> AtomicResult:
        """Single execution attempt with tool results included"""
        start_time = time.perf_counter()

        # Build enhanced prompt with tool results
        tool_info = ""
        if tool_results:
            tool_info = "\n\nTOOL-ERGEBNISSE:\n" + "\n".join([
                f"- {name}: {str(result)[:300]}"
                for name, result in tool_results.items()
            ])

        context_info = ""
        if fetched_context:
            context_info = "\n\nZUSÄTZLICHER KONTEXT:\n" + "\n".join([
                f"- {path}: {str(data)[:300]}"
                for path, data in fetched_context.items()
            ])

        prompt = f"""Führe diese atomare Aufgabe aus:

AUFGABE: {task.description}

KONTEXT: {context}{tool_info}{context_info}

ANWEISUNGEN:
1. Nutze die bereitgestellten Tool-Ergebnisse und Kontextdaten
2. Löse die Aufgabe präzise und direkt
3. Gib das Ergebnis klar an
4. Beschreibe welcher Kontext für nachfolgende Aufgaben relevant ist
5. Sei sicher in deiner Antwort

VERSUCH: {attempt + 1}"""

        try:
            result = await agent.a_format_class(
                pydantic_model=AtomicResult,
                prompt=prompt,
                model_preference="fast",
                max_retries=1,
                auto_context=False,
                session_id=session_id,
                llm_kwargs={
                    "max_tokens": self.max_response_tokens,
                    "temperature": 0.1 if attempt == 0 else 0.3
                }
            )

            result_obj = AtomicResult(**result)
            result_obj.execution_time_ms = (time.perf_counter() - start_time) * 1000
            result_obj.tool_results = tool_results or {}
            result_obj.context_fetched = fetched_context or {}
            result_obj.actions_executed = [
                {"type": "reasoning", "attempt": attempt}
            ]
            if tool_results:
                result_obj.actions_executed.extend([
                    {"type": "tool_call", "tool": name}
                    for name in tool_results.keys()
                ])

            return result_obj

        except Exception as e:
            return AtomicResult(
                success=False,
                result=f"Execution error: {str(e)}",
                context_for_next="",
                confidence=0.0,
                red_flags=["execution_error"],
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                tool_results=tool_results or {},
                context_fetched=fetched_context or {}
            )

    def _has_red_flags(self, result: AtomicResult) -> bool:
        """Check for red flags as in MAKER paper"""
        # 1. Response too long
        if len(result.result) > self.max_response_tokens * 4:
            return True

        # 2. Pattern-based red flags
        for pattern in self.red_flag_patterns:
            if re.search(pattern, result.result):
                return True

        # 3. Low confidence
        if result.confidence < 0.3:
            return True

        # 4. Explicit red flags
        if result.red_flags and len(result.red_flags) > 0:
            return True

        return False

    def _hash_result(self, result: AtomicResult) -> str:
        """Create hash for result comparison"""
        # Normalize and hash
        normalized = result.result.strip().lower()[:200]
        return hashlib.md5(normalized.encode()).hexdigest()

    def _check_k_margin_victory(self, votes: list[VotingCandidate]) -> Optional[VotingCandidate]:
        """Check if any candidate has k-margin victory"""
        if len(votes) < 2:
            if votes and votes[0].votes >= self.k_margin:
                return votes[0]
            return None

        sorted_votes = sorted(votes, key=lambda v: v.votes, reverse=True)
        first, second = sorted_votes[0], sorted_votes[1]

        if first.votes - second.votes >= self.k_margin:
            return first

        return None

    async def post_async(self, shared, prep_res, exec_res) -> str:
        if exec_res["action"] == "paused":
            return "paused"

        if exec_res["action"] == "all_complete":
            return "all_complete"

        mda_state: MDAState = shared.get("mda_state")

        # Update task states and store results
        for result_data in exec_res["results"]:
            task_id = result_data["task_id"]
            task = mda_state.get_task_node(task_id)

            if task:
                if result_data["success"]:
                    task.status = MDATaskStatus.COMPLETED
                    task.result = result_data["result"]
                    task.completed_at = datetime.now().isoformat()
                    mda_state.results[task_id] = {
                        "result": result_data["result"]["result"],
                        "context_for_next": result_data["result"]["context_for_next"],
                        "tool_results": result_data["result"].get("tool_results", {}),
                        "context_fetched": result_data["result"].get("context_fetched", {})
                    }
                    mda_state.completed_task_ids.append(task_id)
                else:
                    task.status = MDATaskStatus.FAILED
                    task.result = {"error": result_data["error"]}
                    mda_state.failed_task_ids.append(task_id)

                mda_state.update_task_node(task)

        # Move to next group
        mda_state.current_group_index += 1
        mda_state.completed_groups.append(exec_res["group_index"])

        if mda_state.current_group_index >= len(mda_state.parallel_groups):
            return "all_complete"

        return "continue_execution"

@with_progress_tracking
class ResultAggregatorNode(AsyncNode):
    """Aggregates partial results into final result"""

    async def prep_async(self, shared) -> dict:
        return {
            "mda_state": shared.get("mda_state"),
            "agent_instance": shared.get("agent_instance"),
            "original_task": shared.get("original_task"),
            "session_id": shared.get("session_id"),
            "is_paused": shared.get("mda_paused", False)
        }

    async def exec_async(self, prep_res) -> dict:
        if prep_res.get("is_paused"):
            return {"action": "paused"}

        mda_state: MDAState = prep_res["mda_state"]
        agent = prep_res["agent_instance"]
        original_task = prep_res["original_task"]
        session_id = prep_res["session_id"]

        # Collect all results
        results = mda_state.results
        completed = len(mda_state.completed_task_ids)
        failed = len(mda_state.failed_task_ids)
        total = completed + failed

        if not results:
            return {
                "action": "no_results",
                "aggregated": AggregatedResult(
                    success=False,
                    final_result="No results to aggregate",
                    total_tasks=total,
                    successful_tasks=completed,
                    failed_tasks=failed,
                    total_voting_rounds=mda_state.stats.get("voting_rounds", 0),
                    red_flags_caught=mda_state.stats.get("red_flags_caught", 0)
                ).model_dump()
            }

        # Synthesize final result
        final_result = await self._synthesize_results(
            original_task, results, agent, session_id
        )

        aggregated = AggregatedResult(
            success=completed > 0 and failed == 0,
            final_result=final_result,
            partial_results={k: v.get("result", "") for k, v in results.items()},
            total_tasks=total,
            successful_tasks=completed,
            failed_tasks=failed,
            total_voting_rounds=mda_state.stats.get("voting_rounds", 0),
            red_flags_caught=mda_state.stats.get("red_flags_caught", 0)
        )

        return {
            "action": "aggregated",
            "aggregated": aggregated.model_dump()
        }

    async def _synthesize_results(self, original_task: str,
                                   results: dict, agent, session_id: str) -> str:
        """Synthesize partial results into final answer"""
        # Build results summary
        results_text = "\n".join([
            f"[{task_id}]: {data.get('result', 'N/A')}"
            for task_id, data in results.items()
        ])

        prompt = f"""Fasse die Teilergebnisse zu einer vollständigen Antwort zusammen:

URSPRÜNGLICHE AUFGABE: {original_task}

TEILERGEBNISSE:
{results_text}

ANWEISUNGEN:
1. Kombiniere alle relevanten Informationen
2. Beantworte die ursprüngliche Aufgabe vollständig
3. Sei präzise und strukturiert
4. Vermeide Wiederholungen"""

        try:
            response = await agent.a_run_llm_completion(
                node_name="ResultAggregator",
                task_id="synthesize_results",
                model_preference="fast",
                with_context=False,
                messages=[{"role": "user", "content": prompt}],
                session_id=session_id,
                max_tokens=2000
            )
            return response.strip()
        except Exception as e:
            # Fallback: concatenate results
            return f"Zusammengefasste Ergebnisse:\n{results_text}\n\n(Synthesefehler: {e})"

    async def post_async(self, shared, prep_res, exec_res) -> str:
        if exec_res["action"] == "paused":
            return "paused"

        shared["final_aggregated_result"] = exec_res["aggregated"]
        shared["mda_state"].final_result = exec_res["aggregated"]

        return "aggregated"


# ============================================================================
# MDA STATE MANAGER
# ============================================================================

class MDAState:
    """
    Manages the complete state of an MDA process.
    Supports checkpointing for stop/resume.
    """

    def __init__(self, original_task: str, original_context: str,
                 session_id: str, config: dict):
        self.checkpoint_id = f"mda_{uuid.uuid4().hex[:12]}"
        self.original_task = original_task
        self.original_context = original_context
        self.session_id = session_id
        self.config = config

        # Task tree
        self.task_nodes: dict[str, MDATaskNode] = {}
        self.root_task_id: Optional[str] = None

        # Execution state
        self.pending_divisions: list[str] = []
        self.parallel_groups: list[list[str]] = []
        self.current_group_index: int = 0
        self.completed_groups: list[int] = []
        self.completed_task_ids: list[str] = []
        self.failed_task_ids: list[str] = []

        # Results
        self.results: dict[str, dict] = {}
        self.final_result: Optional[dict] = None

        # Statistics
        self.stats = {
            "total_divisions": 0,
            "voting_rounds": 0,
            "red_flags_caught": 0,
            "total_execution_time_ms": 0
        }

        # Timestamps
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        self.paused_at: Optional[str] = None

    def create_root_task(self) -> MDATaskNode:
        """Create the root task node"""
        root = MDATaskNode(
            id=f"root_{uuid.uuid4().hex[:8]}",
            description=self.original_task,
            context=self.original_context,
            complexity=10,  # Will be estimated
            dependencies=[],
            is_atomic=False,
            status=MDATaskStatus.PENDING
        )
        self.task_nodes[root.id] = root
        self.root_task_id = root.id
        self.pending_divisions.append(root.id)
        return root

    def add_task_node(self, node: MDATaskNode):
        """Add a task node"""
        self.task_nodes[node.id] = node
        self.last_updated = datetime.now().isoformat()

    def get_task_node(self, task_id: str) -> Optional[MDATaskNode]:
        """Get task node by ID"""
        return self.task_nodes.get(task_id)

    def update_task_node(self, node: MDATaskNode):
        """Update a task node"""
        self.task_nodes[node.id] = node
        self.last_updated = datetime.now().isoformat()

    def mark_task_ready(self, task_id: str):
        """Mark task as ready for execution"""
        node = self.get_task_node(task_id)
        if node:
            node.status = MDATaskStatus.READY
            self.update_task_node(node)

    def has_pending_divisions(self) -> bool:
        """Check if there are pending divisions"""
        return len(self.pending_divisions) > 0

    def get_atomic_tasks(self) -> list[MDATaskNode]:
        """Get all atomic tasks"""
        return [
            node for node in self.task_nodes.values()
            if node.is_atomic and node.status in [MDATaskStatus.READY, MDATaskStatus.PENDING]
        ]

    def to_checkpoint(self) -> MDACheckpoint:
        """Create checkpoint from current state"""
        return MDACheckpoint(
            checkpoint_id=self.checkpoint_id,
            original_task=self.original_task,
            original_context=self.original_context,
            session_id=self.session_id,
            config=self.config,
            task_nodes={tid: node.to_dict() for tid, node in self.task_nodes.items()},
            root_task_id=self.root_task_id or "",
            current_parallel_group=self.current_group_index,
            completed_groups=self.completed_groups,
            pending_task_ids=self.pending_divisions,
            executing_task_ids=[
                tid for tid, node in self.task_nodes.items()
                if node.status == MDATaskStatus.EXECUTING
            ],
            completed_task_ids=self.completed_task_ids,
            failed_task_ids=self.failed_task_ids,
            results=self.results,
            stats=self.stats,
            created_at=self.created_at,
            last_updated=datetime.now().isoformat(),
            paused_at=self.paused_at
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: MDACheckpoint) -> "MDAState":
        """Restore state from checkpoint"""
        state = cls(
            original_task=checkpoint.original_task,
            original_context=checkpoint.original_context,
            session_id=checkpoint.session_id,
            config=checkpoint.config
        )

        state.checkpoint_id = checkpoint.checkpoint_id
        state.root_task_id = checkpoint.root_task_id
        state.current_group_index = checkpoint.current_parallel_group
        state.completed_groups = checkpoint.completed_groups
        state.pending_divisions = checkpoint.pending_task_ids
        state.completed_task_ids = checkpoint.completed_task_ids
        state.failed_task_ids = checkpoint.failed_task_ids
        state.results = checkpoint.results
        state.stats = checkpoint.stats
        state.created_at = checkpoint.created_at
        state.last_updated = checkpoint.last_updated
        state.paused_at = checkpoint.paused_at

        # Restore task nodes
        for tid, node_dict in checkpoint.task_nodes.items():
            state.task_nodes[tid] = MDATaskNode.from_dict(node_dict)

        return state


# ============================================================================
# MDA FLOW - MAIN ORCHESTRATOR
# ============================================================================
@with_progress_tracking
class MDAFlow(AsyncFlow):
    """
    Massively Decomposed Agentic Process Flow.
    Implements the complete MAKER framework with stop/resume support.

    NEW: Supports external tool calls and context fetching.
    """

    def __init__(self,
                 min_complexity: int = 2,
                 max_parallel: int = 5,
                 k_margin: int = 2,
                 num_attempts: int = 3,
                 model_strength: Literal["weak", "medium", "strong"] = "medium",
                 max_division_depth: int = 10,
                 enable_tools: bool = True,
                 enable_context_fetch: bool = True):

        self.config = {
            "min_complexity": min_complexity,
            "max_parallel": max_parallel,
            "k_margin": k_margin,
            "num_attempts": num_attempts,
            "model_strength": model_strength,
            "max_division_depth": max_division_depth,
            "enable_tools": enable_tools,
            "enable_context_fetch": enable_context_fetch
        }

        # Initialize nodes
        self.divide_node = DivideNode(
            min_complexity=min_complexity,
            max_subtasks={"weak": 2, "medium": 3, "strong": 5}.get(model_strength, 3),
            model_strength=model_strength
        )
        self.tree_builder = TaskTreeBuilderNode()
        self.atomic_conquer = AtomicConquerNode(
            num_attempts=num_attempts,
            k_margin=k_margin,
            enable_tools=enable_tools,
            enable_context_fetch=enable_context_fetch
        )
        self.aggregator = ResultAggregatorNode()

        # Define flow connections
        self.divide_node - "continue_division" >> self.divide_node
        self.divide_node - "all_divided" >> self.tree_builder
        self.divide_node - "paused" >> None  # Exit for pause

        self.tree_builder - "tree_built" >> self.atomic_conquer
        self.tree_builder - "no_tasks" >> self.aggregator
        self.tree_builder - "paused" >> None

        self.atomic_conquer - "continue_execution" >> self.atomic_conquer
        self.atomic_conquer - "all_complete" >> self.aggregator
        self.atomic_conquer - "paused" >> None

        #self.aggregator - "aggregated" >> None
        #self.aggregator - "paused" >> None

        super().__init__(start=self.divide_node)

    async def run_async(self, shared) -> str:
        """Execute the MDA flow"""
        return await super().run_async(shared)


# ============================================================================
# FLOWAGENT INTEGRATION - a_accomplish METHOD
# ============================================================================

async def a_accomplish(
    agent,  # FlowAgent instance
    task: str,
    context: str = "",
    min_complexity: int = 2,
    max_parallel: int = 5,
    k_margin: int = 2,
    num_attempts: int = 3,
    model_strength: Literal["weak", "medium", "strong"] = "medium",
    max_division_depth: int = 10,
    session_id: str = None,
    progress_callback: Callable = None,
    resume_checkpoint: MDACheckpoint = None,
    # NEW: Tool configuration
    enable_tools: bool = True,
    enable_context_fetch: bool = True,
    allowed_tools: list[str] = None,  # None = all tools allowed
    **kwargs
) -> dict[str, Any]:
    """
    Massively Decomposed Agentic Process (MDAP) for complex tasks.

    Implements the MAKER framework from:
    "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025)

    Args:
        agent: FlowAgent instance
        task: Main task to accomplish
        context: Additional context
        min_complexity: Minimum complexity before stopping decomposition (0-10)
        max_parallel: Maximum parallel executions
        k_margin: Required vote margin for k-voting
        num_attempts: Attempts per atomic task
        model_strength: Model strength ("weak", "medium", "strong")
        max_division_depth: Maximum decomposition depth
        session_id: Session ID
        progress_callback: Callback for progress updates
        resume_checkpoint: Checkpoint to resume from
        enable_tools: Whether to allow tool calls in atomic tasks
        enable_context_fetch: Whether to allow context fetching
        allowed_tools: List of allowed tool names (None = all)

    Returns:
        dict with:
            - success: bool
            - result: Final aggregated result
            - checkpoint: MDACheckpoint for resume
            - stats: Execution statistics (including tool_calls, context_fetches)
            - cost_info: Cost information

    Example:
        # With tool access
        result = await agent.a_accomplish(
            task="Read config.json and update the database settings",
            context="Project root is /home/user/project",
            enable_tools=True,
            allowed_tools=["file_read", "file_write", "db_query"]
        )

        # Pure reasoning (no tools)
        result = await agent.a_accomplish(
            task="Analyze this algorithm's complexity",
            context="def sort(arr): ...",
            enable_tools=False
        )
    """
    session_id = session_id or agent.active_session or f"mda_{uuid.uuid4().hex[:8]}"

    # Configuration
    config = {
        "min_complexity": min_complexity,
        "max_parallel": max_parallel,
        "k_margin": k_margin,
        "num_attempts": num_attempts,
        "model_strength": model_strength,
        "max_division_depth": max_division_depth,
        "enable_tools": enable_tools,
        "enable_context_fetch": enable_context_fetch,
        "allowed_tools": allowed_tools
    }

    # Track costs
    start_cost = agent.total_cost_accumulated
    start_tokens_in = agent.total_tokens_in
    start_tokens_out = agent.total_tokens_out
    start_time = time.perf_counter()

    try:
        # Initialize or restore state
        if resume_checkpoint:
            mda_state = MDAState.from_checkpoint(resume_checkpoint)
            mda_state.paused_at = None  # Clear pause state
        else:
            mda_state = MDAState(
                original_task=task,
                original_context=context,
                session_id=session_id,
                config=config
            )
            root_task = mda_state.create_root_task()

        # Initialize MDA Flow with tool support
        mda_flow = MDAFlow(
            min_complexity=min_complexity,
            max_parallel=max_parallel,
            k_margin=k_margin,
            num_attempts=num_attempts,
            model_strength=model_strength,
            max_division_depth=max_division_depth,
            enable_tools=enable_tools,
            enable_context_fetch=enable_context_fetch
        )

        # Prepare shared state
        shared = {
            "mda_state": mda_state,
            "agent_instance": agent,
            "session_id": session_id,
            "original_task": task,
            "max_parallel": max_parallel,
            "max_division_depth": max_division_depth,
            "mda_paused": False,
            "progress_tracker": agent.progress_tracker if progress_callback else None,
            "variable_manager": agent.variable_manager if hasattr(agent, 'variable_manager') else None,
            # Tool configuration
            "enable_tools": enable_tools,
            "enable_context_fetch": enable_context_fetch,
            "allowed_tools": allowed_tools
        }

        # Set initial task for division
        if not resume_checkpoint and mda_state.pending_divisions:
            first_task_id = mda_state.pending_divisions.pop(0)
            shared["current_task_node"] = mda_state.get_task_node(first_task_id)
            shared["division_depth"] = 0

        # Execute flow
        result = await mda_flow.run_async(shared)

        # Get final result
        final_result = shared.get("final_aggregated_result", {})

        # Update stats
        mda_state.stats["total_execution_time_ms"] = (time.perf_counter() - start_time) * 1000

        # Create final checkpoint
        checkpoint = mda_state.to_checkpoint()

        return {
            "success": final_result.get("success", False),
            "result": final_result.get("final_result", ""),
            "partial_results": final_result.get("partial_results", {}),
            "checkpoint": checkpoint.to_dict(),
            "stats": {
                **mda_state.stats,
                "total_tasks": final_result.get("total_tasks", 0),
                "successful_tasks": final_result.get("successful_tasks", 0),
                "failed_tasks": final_result.get("failed_tasks", 0)
            },
            "cost_info": {
                "total_cost": agent.total_cost_accumulated - start_cost,
                "tokens_in": agent.total_tokens_in - start_tokens_in,
                "tokens_out": agent.total_tokens_out - start_tokens_out,
                "execution_time_s": (time.perf_counter() - start_time)
            }
        }

    except Exception as e:
        # Create checkpoint even on failure for resume
        checkpoint = mda_state.to_checkpoint() if 'mda_state' in locals() else None
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "checkpoint": checkpoint.to_dict() if checkpoint else None,
            "stats": mda_state.stats if 'mda_state' in locals() else {},
            "cost_info": {
                "total_cost": agent.total_cost_accumulated - start_cost,
                "tokens_in": agent.total_tokens_in - start_tokens_in,
                "tokens_out": agent.total_tokens_out - start_tokens_out,
                "execution_time_s": (time.perf_counter() - start_time)
            }
        }


async def pause_accomplish(agent, session_id: str = None) -> dict[str, Any]:
    """
    Pause an ongoing MDA process and return checkpoint.

    Args:
        agent: FlowAgent instance
        session_id: Session ID of the MDA process

    Returns:
        dict with checkpoint data for resume
    """
    # Set pause flag
    if hasattr(agent, 'shared') and agent.shared:
        agent.shared["mda_paused"] = True

        mda_state = agent.shared.get("mda_state")
        if mda_state:
            mda_state.paused_at = datetime.now().isoformat()
            checkpoint = mda_state.to_checkpoint()

            return {
                "success": True,
                "checkpoint": checkpoint.to_dict(),
                "message": f"MDA process paused at {mda_state.paused_at}",
                "resumable_tasks": checkpoint.get_resumable_tasks()
            }

    return {
        "success": False,
        "error": "No active MDA process found"
    }


def integrate_mda_checkpoint(agent_checkpoint: dict, mda_checkpoint: dict) -> dict:
    """
    Integrate MDA checkpoint into agent checkpoint for unified storage.

    Args:
        agent_checkpoint: Agent's checkpoint dictionary
        mda_checkpoint: MDA checkpoint dictionary

    Returns:
        Updated agent checkpoint with MDA data
    """
    if "mda_checkpoints" not in agent_checkpoint:
        agent_checkpoint["mda_checkpoints"] = {}

    checkpoint_id = mda_checkpoint.get("checkpoint_id", f"mda_{uuid.uuid4().hex[:8]}")
    agent_checkpoint["mda_checkpoints"][checkpoint_id] = mda_checkpoint

    return agent_checkpoint


def extract_mda_checkpoint(agent_checkpoint: dict, checkpoint_id: str = None) -> Optional[MDACheckpoint]:
    """
    Extract MDA checkpoint from agent checkpoint.

    Args:
        agent_checkpoint: Agent's checkpoint dictionary
        checkpoint_id: Specific checkpoint ID, or None for latest

    Returns:
        MDACheckpoint or None
    """
    mda_checkpoints = agent_checkpoint.get("mda_checkpoints", {})

    if not mda_checkpoints:
        return None

    if checkpoint_id:
        data = mda_checkpoints.get(checkpoint_id)
    else:
        # Get latest by timestamp
        latest = max(mda_checkpoints.values(), key=lambda x: x.get("last_updated", ""))
        data = latest

    if data:
        return MDACheckpoint.from_dict(data)

    return None


"""
FlowAgent Integration for MDA (a_accomplish)
=============================================

This module provides the integration mixin that adds the a_accomplish method
to FlowAgent, enabling MAKER-style task decomposition with stop/resume support.

Usage:
    # Add to FlowAgent class or use as mixin
    from mda_accomplish import a_accomplish, MDACheckpoint

    # Bind method to agent
    agent.a_accomplish = lambda *args, **kwargs: a_accomplish(agent, *args, **kwargs)

    # Or use the mixin
    class EnhancedFlowAgent(FlowAgentMDAMixin, FlowAgent):
        pass
"""

import json
import pickle
import os
from datetime import datetime
from typing import Any, Callable, Literal, Optional

_a_accomplish=a_accomplish
_pause_accomplish=pause_accomplish


class FlowAgentMDAMixin:
    """
    Mixin class that adds a_accomplish capability to FlowAgent.

    This mixin integrates the MAKER framework for massively decomposed
    agentic processes with full stop/resume support.

    Usage:
        class EnhancedFlowAgent(FlowAgentMDAMixin, FlowAgent):
            pass

        agent = EnhancedFlowAgent(amd)
        result = await agent.a_accomplish("Complex task...")
    """

    # MDA-specific attributes
    _mda_active_checkpoints: dict[str, dict] = {}
    _mda_current_session: Optional[str] = None

    async def a_accomplish(
        self,
        task: str,
        context: str = "",
        min_complexity: int = 2,
        max_parallel: int = 5,
        k_margin: int = 2,
        num_attempts: int = 3,
        model_strength: Literal["weak", "medium", "strong"] = "medium",
        max_division_depth: int = 10,
        session_id: str = None,
        progress_callback: Callable = None,
        auto_checkpoint: bool = True,
        checkpoint_interval: int = 60,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute a complex task using Massively Decomposed Agentic Processes (MDAP).

        Implements the MAKER framework from:
        "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025)

        Key Features:
        - Recursive task decomposition based on complexity
        - First-to-ahead-by-k voting for error correction
        - Red-flagging to discard unreliable responses
        - Full stop/resume with compact checkpoints
        - Integration with FlowAgent checkpoint system

        Args:
            task: Main task to accomplish
            context: Additional context for the task
            min_complexity: Minimum complexity threshold (0-10) before stopping decomposition
            max_parallel: Maximum number of parallel task executions
            k_margin: Required vote margin for k-voting (higher = more reliable, slower)
            num_attempts: Number of attempts per atomic task for voting
            model_strength: Model capability assumption ("weak", "medium", "strong")
                - weak: Max 2 subtasks per division
                - medium: Max 3 subtasks per division
                - strong: Max 5 subtasks per division
            max_division_depth: Maximum recursion depth for decomposition
            session_id: Session identifier for tracking
            progress_callback: Optional callback for progress updates
            auto_checkpoint: Whether to auto-save checkpoints
            checkpoint_interval: Seconds between auto-checkpoints
            **kwargs: Additional arguments

        Returns:
            dict containing:
                - success: bool - Whether the task completed successfully
                - result: str - Final aggregated result
                - partial_results: dict - Individual task results
                - checkpoint: dict - Checkpoint data for resume
                - stats: dict - Execution statistics
                    - total_divisions: Number of task divisions
                    - voting_rounds: Total voting rounds used
                    - red_flags_caught: Number of red-flagged responses
                    - total_tasks: Total atomic tasks
                    - successful_tasks: Successfully completed tasks
                    - failed_tasks: Failed tasks
                - cost_info: dict - Cost and token information
                    - total_cost: Accumulated cost
                    - tokens_in: Input tokens used
                    - tokens_out: Output tokens used
                    - execution_time_s: Total execution time

        Example:
            # Simple usage
            result = await agent.a_accomplish(
                task="Analyze the uploaded codebase and create comprehensive documentation",
                context="Python FastAPI project with SQLAlchemy ORM",
                min_complexity=3
            )

            if result["success"]:
                print(result["result"])
            else:
                print(f"Failed: {result.get('error')}")
                # Can resume later with checkpoint
                saved_checkpoint = result["checkpoint"]

            # Resume from checkpoint
            result = await agent.a_accomplish(
                task="...",  # Same task
                resume_checkpoint=MDACheckpoint.from_dict(saved_checkpoint)
            )
        """
        # Store current MDA session
        self._mda_current_session = (
            session_id or f"mda_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Check for existing checkpoint to resume
        resume_checkpoint = kwargs.pop("resume_checkpoint", None)

        # Execute MDA
        result = await _a_accomplish(
            agent=self,
            task=task,
            context=context,
            min_complexity=min_complexity,
            max_parallel=max_parallel,
            k_margin=k_margin,
            num_attempts=num_attempts,
            model_strength=model_strength,
            max_division_depth=max_division_depth,
            session_id=self._mda_current_session,
            progress_callback=progress_callback,
            resume_checkpoint=resume_checkpoint,
            **kwargs,
        )

        # Store checkpoint for potential resume
        if result.get("checkpoint"):
            self._mda_active_checkpoints[self._mda_current_session] = result["checkpoint"]

            # Auto-save to agent checkpoint if enabled
            if auto_checkpoint:
                await self._save_mda_checkpoint(result["checkpoint"])

        return result

    async def pause_accomplish(self) -> dict[str, Any]:
        """
        Pause the current MDA process and get checkpoint.

        Returns:
            dict with:
                - success: bool
                - checkpoint: MDACheckpoint data for resume
                - message: Status message
                - resumable_tasks: List of tasks that can be resumed

        Example:
            # During execution, pause the process
            pause_result = await agent.pause_accomplish()

            if pause_result["success"]:
                # Save checkpoint for later
                checkpoint_data = pause_result["checkpoint"]
                with open("mda_checkpoint.json", "w") as f:
                    json.dump(checkpoint_data, f)
        """
        result = await _pause_accomplish(self, self._mda_current_session)

        if result.get("success") and result.get("checkpoint"):
            await self._save_mda_checkpoint(result["checkpoint"])

        return result

    async def resume_accomplish(self, checkpoint_id: str = None) -> dict[str, Any]:
        """
        Resume an MDA process from checkpoint.

        Args:
            checkpoint_id: Specific checkpoint ID, or None for latest

        Returns:
            Result dict from a_accomplish

        Example:
            # Resume from latest checkpoint
            result = await agent.resume_accomplish()

            # Resume from specific checkpoint
            result = await agent.resume_accomplish(checkpoint_id="mda_abc123...")
        """
        # Try to get checkpoint from active checkpoints
        checkpoint_data = None

        if checkpoint_id:
            checkpoint_data = self._mda_active_checkpoints.get(checkpoint_id)
        elif self._mda_active_checkpoints:
            # Get latest
            checkpoint_data = list(self._mda_active_checkpoints.values())[-1]

        # If not found, try loading from agent checkpoint
        if not checkpoint_data:
            checkpoint_data = await self._load_mda_checkpoint(checkpoint_id)

        if not checkpoint_data:
            return {
                "success": False,
                "error": f"No checkpoint found for ID: {checkpoint_id or 'latest'}",
            }

        # Resume
        checkpoint = MDACheckpoint.from_dict(checkpoint_data)
        return await self.a_accomplish(
            task=checkpoint.original_task,
            context=checkpoint.original_context,
            session_id=checkpoint.session_id,
            resume_checkpoint=checkpoint,
            **checkpoint.config,
        )

    def list_mda_checkpoints(self) -> list[dict]:
        """
        List available MDA checkpoints.

        Returns:
            List of checkpoint summaries
        """
        checkpoints = []

        for session_id, data in self._mda_active_checkpoints.items():
            checkpoints.append(
                {
                    "session_id": session_id,
                    "checkpoint_id": data.get("checkpoint_id"),
                    "created_at": data.get("created_at"),
                    "last_updated": data.get("last_updated"),
                    "paused_at": data.get("paused_at"),
                    "task_preview": data.get("original_task", "")[:100],
                    "stats": data.get("stats", {}),
                }
            )

        return sorted(checkpoints, key=lambda x: x.get("last_updated", ""), reverse=True)

    def clear_mda_checkpoint(self, checkpoint_id: str = None):
        """
        Clear MDA checkpoint(s).

        Args:
            checkpoint_id: Specific checkpoint to clear, or None for all
        """
        if checkpoint_id:
            self._mda_active_checkpoints.pop(checkpoint_id, None)
        else:
            self._mda_active_checkpoints.clear()

    async def _save_mda_checkpoint(self, checkpoint_data: dict):
        """Save MDA checkpoint integrated with agent checkpoint"""
        try:
            # If agent has checkpoint system, integrate
            if hasattr(self, "_create_checkpoint") and hasattr(self, "_save_checkpoint"):
                # Create agent checkpoint
                agent_checkpoint = await self._create_checkpoint()

                # Convert to dict if needed
                if hasattr(agent_checkpoint, "__dict__"):
                    cp_dict = agent_checkpoint.__dict__.copy()
                else:
                    cp_dict = dict(agent_checkpoint) if agent_checkpoint else {}

                # Add MDA checkpoint
                if "mda_checkpoints" not in cp_dict:
                    cp_dict["mda_checkpoints"] = {}

                cp_id = checkpoint_data.get("checkpoint_id", self._mda_current_session)
                cp_dict["mda_checkpoints"][cp_id] = checkpoint_data

                # Save updated checkpoint
                # Note: This requires modification of AgentCheckpoint to accept mda_checkpoints
                # For now, save separately
                await self._save_mda_checkpoint_file(checkpoint_data)
            else:
                await self._save_mda_checkpoint_file(checkpoint_data)

        except Exception as e:
            print(f"Warning: Could not save MDA checkpoint: {e}")

    async def _save_mda_checkpoint_file(self, checkpoint_data: dict):
        """Save MDA checkpoint to separate file"""
        try:
            from toolboxv2 import get_app

            folder = str(get_app().data_dir) + "/Agents/mda_checkpoints/" + self.amd.name
            os.makedirs(folder, exist_ok=True)

            cp_id = checkpoint_data.get("checkpoint_id", "unknown")
            filepath = os.path.join(folder, f"{cp_id}.json")

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            print(f"Warning: Could not save MDA checkpoint file: {e}")

    async def _load_mda_checkpoint(self, checkpoint_id: str = None) -> Optional[dict]:
        """Load MDA checkpoint from file"""
        try:
            from toolboxv2 import get_app

            folder = str(get_app().data_dir) + "/Agents/mda_checkpoints/" + self.amd.name

            if not os.path.exists(folder):
                return None

            if checkpoint_id:
                filepath = os.path.join(folder, f"{checkpoint_id}.json")
                if os.path.exists(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        return json.load(f)
            else:
                # Get latest
                files = [f for f in os.listdir(folder) if f.endswith(".json")]
                if files:
                    # Sort by modification time
                    files.sort(
                        key=lambda x: os.path.getmtime(os.path.join(folder, x)),
                        reverse=True,
                    )
                    filepath = os.path.join(folder, files[0])
                    with open(filepath, "r", encoding="utf-8") as f:
                        return json.load(f)

            return None

        except Exception as e:
            print(f"Warning: Could not load MDA checkpoint: {e}")
            return None

    def get_mda_stats(self) -> dict[str, Any]:
        """
        Get aggregated MDA statistics across all sessions.

        Returns:
            dict with aggregated statistics
        """
        total_stats = {
            "total_sessions": len(self._mda_active_checkpoints),
            "total_divisions": 0,
            "total_voting_rounds": 0,
            "total_red_flags_caught": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            "sessions": [],
        }

        for session_id, data in self._mda_active_checkpoints.items():
            stats = data.get("stats", {})
            total_stats["total_divisions"] += stats.get("total_divisions", 0)
            total_stats["total_voting_rounds"] += stats.get("voting_rounds", 0)
            total_stats["total_red_flags_caught"] += stats.get("red_flags_caught", 0)

            completed = len(data.get("completed_task_ids", []))
            failed = len(data.get("failed_task_ids", []))
            total_stats["total_tasks_completed"] += completed
            total_stats["total_tasks_failed"] += failed

            total_stats["sessions"].append(
                {
                    "session_id": session_id,
                    "completed": completed,
                    "failed": failed,
                    "divisions": stats.get("total_divisions", 0),
                }
            )

        return total_stats


async def bind_accomplish_to_agent(agent, and_as_tool=True):
    """
    Bind a_accomplish method to an existing FlowAgent instance.

    This function adds the MDA capabilities to an agent without requiring
    inheritance or class modification.

    Args:
        agent: FlowAgent instance

    Example:
        from flowagent_mda import bind_accomplish_to_agent

        agent = FlowAgent(amd)
        bind_accomplish_to_agent(agent)

        # Now can use a_accomplish
        result = await agent.a_accomplish("Complex task...")
    """
    import types

    # Add MDA attributes
    agent._mda_active_checkpoints = {}
    agent._mda_current_session = None

    # Bind methods from mixin
    mixin_methods = [
        "a_accomplish",
        "pause_accomplish",
        "resume_accomplish",
        "list_mda_checkpoints",
        "clear_mda_checkpoint",
        "_save_mda_checkpoint",
        "_save_mda_checkpoint_file",
        "_load_mda_checkpoint",
        "get_mda_stats",
    ]

    for method_name in mixin_methods:
        method = getattr(FlowAgentMDAMixin, method_name)
        bound_method = types.MethodType(method, agent)
        setattr(agent, method_name, bound_method)

    if and_as_tool:
        async def accomplish_background_wrapper(
            task: str,
            context: str = "",
            min_complexity: int = 2,
            max_parallel: int = 5,
            model_strength: str = "medium",
            enable_tools: bool = True,
            **kwargs,
        ) -> str:

            session_id = agent.active_session or "default"
            res = await agent.a_accomplish(
                        task=task,
                        context=context,
                        min_complexity=min_complexity,
                        max_parallel=max_parallel,
                        model_strength=model_strength,
                        enable_tools=enable_tools,
                        session_id=session_id,  # Wichtig: Gleiche Session nutzen
                        **kwargs,
                    )

            res['checkpoint'] = {}
            return res.get("result", str(res)) if res.get("success") else f"Error: {res.get('error', str(res))}"


        # Das Tool registrieren
        # Hinweis: add_tool muss in deiner Implementierung existieren
        # und idealerweise awaitable sein.
        agent.add_first_class_tool(
            accomplish_background_wrapper,
            "MAKER",
            description="""**META_TOOL_CALL: MAKER(task: str, context: str, min_complexity: int, enable_tools: bool)**
        - **Purpose:** Orchestrate massive, high-complexity missions using the MDAP (Massively Decomposed Agentic Process). Splits tasks recursively, executes parallelly, and uses consensus voting.
        - **Use for:** Complex coding, deep research, "Zero Error" analysis, tasks requiring >10 steps.
        - **Do NOT use for:** Simple linear tasks (use `create_and_execute_plan` , `delegate_to_llm_tool_node`), or tasks with **irreversible side effects** (sending emails/payments) as voting executes actions multiple times.
        - **Example:** `MAKER(task="Refactor entire auth module", context="Use JWT", min_complexity=7, enable_tools=True)`""",
        )

    return agent


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def quick_accomplish(agent, task: str, **kwargs) -> str:
    """
    Quick wrapper that returns just the result string.

    Args:
        agent: FlowAgent instance
        task: Task to accomplish
        **kwargs: Additional arguments for a_accomplish

    Returns:
        Result string or error message
    """
    # Ensure agent has a_accomplish
    if not hasattr(agent, "a_accomplish"):
        await bind_accomplish_to_agent(agent)

    result = await agent.a_accomplish(task, **kwargs)

    if result.get("success"):
        return result.get("result", "Task completed.")
    else:
        return f"Error: {result.get('error', 'Unknown error')}"


# ============================================================================
# EXPORTS
# ============================================================================

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
"FlowAgentMDAMixin", "bind_accomplish_to_agent", "quick_accomplish",

    # Main functions
    "a_accomplish",
    "pause_accomplish",
    "integrate_mda_checkpoint",
    "extract_mda_checkpoint",

    # Pydantic Models
    "TaskComplexity",
    "SubTask",
    "DivisionResult",
    "AtomicResult",
    "AggregatedResult",
    # NEW: Action Models
    "ActionType",
    "ToolCallSpec",
    "ContextFetchSpec",
    "AtomicAction",
    "TaskActionPlan",

    # State Management
    "MDAState",
    "MDACheckpoint",
    "MDATaskNode",
    "MDATaskStatus",

    # AsyncNodes
    "DivideNode",
    "TaskTreeBuilderNode",
    "AtomicConquerNode",
    "ResultAggregatorNode",

    # Flow
    "MDAFlow"
]
