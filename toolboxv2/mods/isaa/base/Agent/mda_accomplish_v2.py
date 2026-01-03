"""
MAKER V2 Framework - Massively Decomposed Agentic Processes with Virtual Workspace
===================================================================================

A complete rewrite of the MAKER framework with:

1. **Virtual Workspace (Sandboxing)**: Agents work in isolated environments
   - File operations are staged before commit
   - Voting on actual diffs, not text outputs
   - Only committed after consensus

2. **Safe Tool Registry**: Clear separation of tools
   - Information-gathering tools (safe for voting)
   - Side-effect tools (blocked during voting)
   - Virtual overrides for file operations

3. **Incremental Aggregation**: Real-time result processing
   - After each parallel batch, aggregate and check
   - Dynamic abort on impossible tasks
   - Progressive response building

4. **Dynamic Recursion**: Self-correcting decomposition
   - Tasks can signal NEEDS_DECOMPOSITION
   - Re-planning triggered automatically
   - Fail-fast on impossible branches

5. **Response Type Manager**: Flexible output formats
   - TEXT: Simple text response
   - REPORT: Structured detailed report
   - STATUS: Compact status update
   - FINAL: Complete synthesized result

Based on: "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025)

Author: ToolBoxV2 FlowAgent Integration
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Literal, Optional, TypeVar, Generic

from pydantic import BaseModel, Field

from toolboxv2.mods.isaa.base.Agent.types import (
    ProgressEvent,
    TaskPlan,
    NodeStatus,
    ProgressTracker,
)
from toolboxv2.mods.isaa.base.tbpocketflow import AsyncFlow, AsyncNode

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================


class ResponseType(str, Enum):
    """Type of response to generate"""
    TEXT = "text"           # Simple text answer
    REPORT = "report"       # Detailed structured report
    STATUS = "status"       # Compact status update
    FINAL = "final"         # Complete synthesized result
    STREAM = "stream"       # Streaming progressive output


class ToolCategory(str, Enum):
    """Category of tool for safety classification"""
    SAFE_READ = "safe_read"         # Read-only, no side effects
    SAFE_SEARCH = "safe_search"     # Search/query, idempotent
    SAFE_COMPUTE = "safe_compute"   # Computation, deterministic
    UNSAFE_WRITE = "unsafe_write"   # Writes to filesystem
    UNSAFE_API = "unsafe_api"       # External API calls with effects
    UNSAFE_EXEC = "unsafe_exec"     # Code execution
    VIRTUAL = "virtual"             # Virtualized for sandboxing


class MDATaskStatus(str, Enum):
    """Status of an MDA task"""
    PENDING = "pending"
    DIVIDING = "dividing"
    READY = "ready"
    EXECUTING = "executing"
    VOTING = "voting"
    STAGED = "staged"       # Has staged changes awaiting commit
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    IMPOSSIBLE = "impossible"


class AggregationAction(str, Enum):
    """Action to take after aggregation"""
    CONTINUE = "continue"
    ABORT = "abort"
    REPLAN = "replan"
    COMPLETE = "complete"
    RETRY = "retry"


class ActionType(str, Enum):
    """Type of action for an atomic task"""
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    CONTEXT_FETCH = "context_fetch"
    VIRTUAL_WRITE = "virtual_write"
    MULTI_ACTION = "multi_action"


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class TaskComplexity(BaseModel):
    """Complexity assessment of a task"""
    score: int = Field(ge=0, le=10, description="Complexity 0-10")
    reasoning: str = Field(description="Reasoning for the assessment")
    is_atomic: bool = Field(description="True if cannot be further decomposed")
    estimated_steps: int = Field(ge=1, description="Estimated number of atomic steps")
    requires_tools: bool = Field(default=False, description="Whether tools are needed")
    suggested_tools: list[str] = Field(default_factory=list)


class SubTask(BaseModel):
    """Single subtask after decomposition"""
    id: str = Field(description="Unique ID")
    description: str = Field(description="Task description")
    relevant_context: str = Field(description="Relevant context for this task")
    complexity: int = Field(ge=0, le=10, description="Complexity 0-10")
    dependencies: list[str] = Field(default_factory=list)
    is_atomic: bool = Field(default=False)
    output_schema: Optional[str] = Field(default=None)
    requires_tools: bool = Field(default=False)
    suggested_tools: list[str] = Field(default_factory=list)
    requires_external_context: bool = Field(default=False)
    expected_response_type: ResponseType = Field(default=ResponseType.TEXT)
    can_fail: bool = Field(default=False, description="If true, failure doesn't block pipeline")


class DivisionResult(BaseModel):
    """Result of task division"""
    can_divide: bool = Field(description="Can be further divided")
    subtasks: list[SubTask] = Field(default_factory=list)
    preserved_context: str = Field(description="Context passed to subtasks")
    context_mappings: dict[str, str] = Field(default_factory=dict)
    division_strategy: str = Field(default="parallel", description="parallel|sequential|mixed")


class ToolCallSpec(BaseModel):
    """Specification for a tool call"""
    tool_name: str = Field(description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = Field(description="Why this tool is needed")
    fallback_on_error: Optional[str] = Field(default=None)
    is_safe: bool = Field(default=True, description="Whether tool is safe for voting")


class StagedChange(BaseModel):
    """A staged file change in the virtual workspace"""
    path: str
    original_content: Optional[str] = None
    new_content: str
    change_type: Literal["create", "modify", "delete"]
    timestamp: float = Field(default_factory=time.time)
    task_id: str = ""


class AtomicResult(BaseModel):
    """Result of an atomic execution with staging support"""
    success: bool
    result: str = Field(description="Partial solution or result")
    context_for_next: str = Field(description="Context for subsequent tasks")
    confidence: float = Field(ge=0, le=1)
    red_flags: list[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0)

    # Staging information
    staged_changes: list[StagedChange] = Field(default_factory=list)
    staged_diff_summary: str = Field(default="")

    # Tool tracking
    tool_results: dict[str, Any] = Field(default_factory=dict)
    context_fetched: dict[str, Any] = Field(default_factory=dict)
    actions_executed: list[dict] = Field(default_factory=list)

    # Response type
    response_type: ResponseType = Field(default=ResponseType.TEXT)

    # Signals for dynamic control
    needs_decomposition: bool = Field(default=False)
    is_impossible: bool = Field(default=False)
    abort_reason: Optional[str] = Field(default=None)


class VotingCandidate(BaseModel):
    """Candidate for voting with staging"""
    result: AtomicResult
    workspace_hash: str = Field(description="Hash of staged changes")
    text_hash: str = Field(description="Hash of text result")
    combined_hash: str = Field(description="Combined hash for comparison")
    votes: int = Field(default=1)


class IncrementalStatus(BaseModel):
    """Status update after each batch"""
    action: AggregationAction
    completed_tasks: int
    total_tasks: int
    failed_tasks: int
    current_summary: str
    can_continue: bool
    replan_needed: bool = False
    new_subtasks: list[SubTask] = Field(default_factory=list)
    abort_reason: Optional[str] = None


class AggregatedResult(BaseModel):
    """Final aggregated result"""
    success: bool
    final_result: str
    response_type: ResponseType
    partial_results: dict[str, str] = Field(default_factory=dict)
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_voting_rounds: int
    red_flags_caught: int
    commits_made: int = 0
    files_modified: list[str] = Field(default_factory=list)


# ============================================================================
# VIRTUAL WORKSPACE - Core Sandboxing Component
# ============================================================================


class VirtualWorkspace:
    """
    Manages a virtual file system layer for safe, reversible operations.

    Key Features:
    - All writes go to staging area first
    - Reads check staging, then cache, then real FS
    - Diff-based voting for consensus
    - Atomic commit only after voting success
    """

    def __init__(self, variable_manager, task_id: str, base_path: str = "/"):
        self.vm = variable_manager
        self.task_id = task_id
        self.base_path = base_path

        # Staging area: path -> StagedChange
        self.staged_changes: dict[str, StagedChange] = {}

        # Read tracking for dependency analysis
        self.reads: set[str] = set()

        # Original content cache (for diff generation)
        self._original_cache: dict[str, str] = {}

    async def read_file(self, path: str, real_fs_tool: Callable) -> str:
        """
        Read from staging if modified, else from cache, else from real FS.
        Implements the layered read strategy for consistency.
        """
        self.reads.add(path)

        # Layer 1: Check staging (uncommitted changes)
        if path in self.staged_changes:
            return self.staged_changes[path].new_content

        # Layer 2: Check Variable Manager cache (current world state)
        cached = self._get_from_vm_cache(path)
        if cached is not None:
            return cached

        # Layer 3: Read from real FS and cache
        try:
            content = await self._execute_real_read(real_fs_tool, path)
            self._cache_in_vm(path, content, "real_fs")
            self._original_cache[path] = content
            return content
        except Exception as e:
            return f"[ERROR reading {path}: {str(e)}]"

    def virtual_write_file(self, path: str, content: str) -> str:
        """
        Write to staging area only - no real FS modification.
        Returns confirmation message for agent feedback.
        """
        # Get original for diff
        original = self._original_cache.get(path) or self._get_from_vm_cache(path)

        change_type = "create" if original is None else "modify"

        self.staged_changes[path] = StagedChange(
            path=path,
            original_content=original,
            new_content=content,
            change_type=change_type,
            task_id=self.task_id
        )

        return f"✓ File '{path}' staged for commit. ({len(content)} bytes, {change_type})"

    def virtual_delete_file(self, path: str) -> str:
        """Stage a file deletion"""
        original = self._original_cache.get(path) or self._get_from_vm_cache(path)

        self.staged_changes[path] = StagedChange(
            path=path,
            original_content=original,
            new_content="",
            change_type="delete",
            task_id=self.task_id
        )

        return f"✓ File '{path}' staged for deletion."

    def get_diff_summary(self) -> str:
        """Returns a human-readable summary of staged changes for voting"""
        if not self.staged_changes:
            return "No file changes staged."

        lines = ["═══ STAGED CHANGES ═══"]

        for path, change in self.staged_changes.items():
            if change.change_type == "create":
                lines.append(f"+ CREATE: {path} ({len(change.new_content)} bytes)")
            elif change.change_type == "delete":
                lines.append(f"- DELETE: {path}")
            else:
                old_size = len(change.original_content or "")
                new_size = len(change.new_content)
                diff = new_size - old_size
                sign = "+" if diff >= 0 else ""
                lines.append(f"~ MODIFY: {path} ({old_size} → {new_size} bytes, {sign}{diff})")

                # Include first few lines of diff for context
                if change.original_content:
                    lines.append(self._generate_mini_diff(
                        change.original_content,
                        change.new_content,
                        max_lines=5
                    ))

        lines.append("═════════════════════")
        return "\n".join(lines)

    def get_staging_hash(self) -> str:
        """Generate hash of all staged changes for voting comparison"""
        if not self.staged_changes:
            return "empty"

        # Sort for deterministic hashing
        sorted_changes = sorted(self.staged_changes.items())
        content = json.dumps([
            {
                "path": path,
                "content": change.new_content,
                "type": change.change_type
            }
            for path, change in sorted_changes
        ], sort_keys=True)

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def commit_to_real_fs(
        self,
        real_write_tool: Callable,
        real_delete_tool: Optional[Callable] = None
    ) -> list[dict]:
        """
        Apply staged changes to real file system.
        Called only after voting consensus is reached.
        """
        results = []

        for path, change in self.staged_changes.items():
            try:
                if change.change_type == "delete" and real_delete_tool:
                    res = await real_delete_tool(path=path)
                    # Clear from VM cache
                    self._clear_vm_cache(path)
                else:
                    res = await self._execute_real_write(real_write_tool, path, change.new_content)
                    # Update VM cache with new content
                    self._cache_in_vm(path, change.new_content, f"committed_{self.task_id}")

                results.append({
                    "path": path,
                    "action": change.change_type,
                    "success": True,
                    "result": res
                })
            except Exception as e:
                results.append({
                    "path": path,
                    "action": change.change_type,
                    "success": False,
                    "error": str(e)
                })

        # Clear staging after commit
        self.staged_changes.clear()

        return results

    def rollback(self):
        """Discard all staged changes without committing"""
        self.staged_changes.clear()

    def get_staged_files(self) -> list[str]:
        """Get list of files with staged changes"""
        return list(self.staged_changes.keys())

    def has_changes(self) -> bool:
        """Check if there are any staged changes"""
        return len(self.staged_changes) > 0

    # === Private Helpers ===

    def _get_from_vm_cache(self, path: str) -> Optional[str]:
        """Get file content from Variable Manager cache"""
        if self.vm is None:
            return None
        cached = self.vm.get(f"files.{path}")
        if cached and isinstance(cached, dict):
            return cached.get('content')
        return None

    def _cache_in_vm(self, path: str, content: str, source: str):
        """Cache file content in Variable Manager"""
        if self.vm is None:
            return
        self.vm.set(f"files.{path}", {
            "content": content,
            "timestamp": time.time(),
            "source": source
        })

    def _clear_vm_cache(self, path: str):
        """Clear file from Variable Manager cache"""
        if self.vm is None:
            return
        try:
            self.vm.delete(f"files.{path}")
        except Exception:
            pass

    async def _execute_real_read(self, tool: Callable, path: str) -> str:
        """Execute the actual file read"""
        if asyncio.iscoroutinefunction(tool):
            return await tool(path=path)
        return tool(path=path)

    async def _execute_real_write(self, tool: Callable, path: str, content: str) -> Any:
        """Execute the actual file write"""
        if asyncio.iscoroutinefunction(tool):
            return await tool(path=path, content=content)
        return tool(path=path, content=content)

    def _generate_mini_diff(self, old: str, new: str, max_lines: int = 5) -> str:
        """Generate a minimal diff for display"""
        old_lines = old.split('\n')
        new_lines = new.split('\n')

        diff_lines = []
        for i, (o, n) in enumerate(zip(old_lines[:max_lines], new_lines[:max_lines])):
            if o != n:
                diff_lines.append(f"  L{i+1}: {o[:50]}... → {n[:50]}...")

        if len(new_lines) > len(old_lines):
            diff_lines.append(f"  +{len(new_lines) - len(old_lines)} new lines")
        elif len(old_lines) > len(new_lines):
            diff_lines.append(f"  -{len(old_lines) - len(new_lines)} lines removed")

        return "\n".join(diff_lines[:max_lines])


# ============================================================================
# SAFE TOOL REGISTRY - Tool Classification and Virtualization
# ============================================================================


class SafeToolRegistry:
    """
    Manages tool classification and virtualization for safe voting.

    Categories:
    - SAFE_*: Can be used freely during voting (idempotent, no side effects)
    - UNSAFE_*: Blocked during voting or virtualized
    - VIRTUAL: Overridden to use VirtualWorkspace
    """

    # Default classifications for common tools
    DEFAULT_CLASSIFICATIONS: dict[str, ToolCategory] = {
        # Safe read tools
        "read_file": ToolCategory.SAFE_READ,
        "file_read": ToolCategory.SAFE_READ,
        "list_directory": ToolCategory.SAFE_READ,
        "list_files": ToolCategory.SAFE_READ,
        "get_file_info": ToolCategory.SAFE_READ,
        "view_file": ToolCategory.SAFE_READ,

        # Safe search tools
        "google_search": ToolCategory.SAFE_SEARCH,
        "web_search": ToolCategory.SAFE_SEARCH,
        "search": ToolCategory.SAFE_SEARCH,
        "find_files": ToolCategory.SAFE_SEARCH,
        "grep": ToolCategory.SAFE_SEARCH,

        # Safe compute tools
        "calculator": ToolCategory.SAFE_COMPUTE,
        "math_eval": ToolCategory.SAFE_COMPUTE,
        "json_parse": ToolCategory.SAFE_COMPUTE,
        "format_code": ToolCategory.SAFE_COMPUTE,

        # Unsafe write tools (will be virtualized)
        "write_file": ToolCategory.UNSAFE_WRITE,
        "file_write": ToolCategory.UNSAFE_WRITE,
        "create_file": ToolCategory.UNSAFE_WRITE,
        "delete_file": ToolCategory.UNSAFE_WRITE,
        "move_file": ToolCategory.UNSAFE_WRITE,
        "rename_file": ToolCategory.UNSAFE_WRITE,

        # Unsafe API tools (blocked during voting)
        "send_email": ToolCategory.UNSAFE_API,
        "post_request": ToolCategory.UNSAFE_API,
        "api_call": ToolCategory.UNSAFE_API,
        "webhook": ToolCategory.UNSAFE_API,
        "publish": ToolCategory.UNSAFE_API,

        # Unsafe execution tools
        "run_command": ToolCategory.UNSAFE_EXEC,
        "execute_code": ToolCategory.UNSAFE_EXEC,
        "shell": ToolCategory.UNSAFE_EXEC,
        "bash": ToolCategory.UNSAFE_EXEC,
        "python_exec": ToolCategory.UNSAFE_EXEC,
    }

    def __init__(self, custom_classifications: Optional[dict[str, ToolCategory]] = None):
        self.classifications = {**self.DEFAULT_CLASSIFICATIONS}
        if custom_classifications:
            self.classifications.update(custom_classifications)

        # Track virtualized tools
        self._virtualized_tools: dict[str, Callable] = {}

    def classify(self, tool_name: str) -> ToolCategory:
        """Get the category of a tool"""
        # Check exact match
        if tool_name in self.classifications:
            return self.classifications[tool_name]

        # Check pattern matches
        name_lower = tool_name.lower()
        if any(x in name_lower for x in ["read", "get", "list", "view", "show"]):
            return ToolCategory.SAFE_READ
        if any(x in name_lower for x in ["search", "find", "query", "lookup"]):
            return ToolCategory.SAFE_SEARCH
        if any(x in name_lower for x in ["write", "create", "update", "delete", "modify"]):
            return ToolCategory.UNSAFE_WRITE
        if any(x in name_lower for x in ["send", "post", "publish", "notify"]):
            return ToolCategory.UNSAFE_API
        if any(x in name_lower for x in ["exec", "run", "shell", "command"]):
            return ToolCategory.UNSAFE_EXEC

        # Default to unsafe for unknown tools
        return ToolCategory.UNSAFE_EXEC

    def is_safe_for_voting(self, tool_name: str) -> bool:
        """Check if tool can be used during voting"""
        category = self.classify(tool_name)
        return category in {
            ToolCategory.SAFE_READ,
            ToolCategory.SAFE_SEARCH,
            ToolCategory.SAFE_COMPUTE,
            ToolCategory.VIRTUAL
        }

    def get_safe_tool_names(self, agent) -> list[str]:
        """Get list of tool names that are safe for voting"""
        all_tools = self._get_agent_tools(agent)
        safe_names = []
        for name in all_tools.keys():
            category = self.classify(name)
            if category in {ToolCategory.SAFE_READ, ToolCategory.SAFE_SEARCH,
                           ToolCategory.SAFE_COMPUTE, ToolCategory.UNSAFE_WRITE}:
                # Include write tools - they'll be virtualized
                safe_names.append(name)
        return safe_names

    def create_virtual_tool_executor(
        self,
        agent,
        workspace: VirtualWorkspace,
        allowed_unsafe: Optional[list[str]] = None
    ) -> "VirtualToolExecutor":
        """
        Create a VirtualToolExecutor that wraps agent's arun_function
        with virtualization for write operations.
        """
        return VirtualToolExecutor(
            agent=agent,
            workspace=workspace,
            registry=self,
            allowed_unsafe=allowed_unsafe or []
        )

    def _get_agent_tools(self, agent) -> dict[str, Callable]:
        """Extract tools from agent"""
        tools = {}

        if hasattr(agent, '_tool_registry'):
            for name, info in agent._tool_registry.items():
                if callable(info):
                    tools[name] = info
                elif isinstance(info, dict) and 'func' in info:
                    tools[name] = info['func']
                elif isinstance(info, dict) and 'function' in info:
                    tools[name] = info['function']

        if hasattr(agent, 'tools'):
            for tool in agent.tools:
                if hasattr(tool, 'name') and callable(tool):
                    tools[tool.name] = tool
                elif hasattr(tool, '__name__'):
                    tools[tool.__name__] = tool

        return tools


class VirtualToolExecutor:
    """
    Executes tools with virtualization layer.

    - Safe tools: Executed normally via agent.arun_function
    - Write tools: Intercepted and redirected to VirtualWorkspace
    - Unsafe tools: Blocked
    """

    def __init__(
        self,
        agent,
        workspace: VirtualWorkspace,
        registry: SafeToolRegistry,
        allowed_unsafe: list[str] = None
    ):
        self.agent = agent
        self.workspace = workspace
        self.registry = registry
        self.allowed_unsafe = allowed_unsafe or []
        self.execution_log: list[dict] = []

    async def execute(self, tool_name: str, arguments: dict) -> dict:
        """
        Execute a tool with virtualization.

        Returns:
            dict with keys: success, result, error, virtualized
        """
        category = self.registry.classify(tool_name)
        start_time = time.time()

        try:
            # Handle virtualized write operations
            if category == ToolCategory.UNSAFE_WRITE:
                result = await self._execute_virtual_write(tool_name, arguments)
                self._log_execution(tool_name, arguments, result, True, None)
                return {
                    "success": True,
                    "result": result,
                    "virtualized": True,
                    "tool_name": tool_name
                }

            # Block unsafe tools (unless explicitly allowed)
            elif category in {ToolCategory.UNSAFE_API, ToolCategory.UNSAFE_EXEC}:
                if tool_name not in self.allowed_unsafe:
                    error = f"Tool '{tool_name}' is blocked during atomic execution (category: {category.value})"
                    self._log_execution(tool_name, arguments, None, False, error)
                    return {
                        "success": False,
                        "error": error,
                        "virtualized": False,
                        "tool_name": tool_name
                    }

            # Execute safe tools normally via agent
            result = await self.agent.arun_function(tool_name, **arguments)
            self._log_execution(tool_name, arguments, result, True, None)
            return {
                "success": True,
                "result": result,
                "virtualized": False,
                "tool_name": tool_name
            }

        except Exception as e:
            error = str(e)
            self._log_execution(tool_name, arguments, None, False, error)
            return {
                "success": False,
                "error": error,
                "virtualized": False,
                "tool_name": tool_name
            }

    async def _execute_virtual_write(self, tool_name: str, arguments: dict) -> str:
        """Execute a write operation virtually"""
        name_lower = tool_name.lower()

        # Determine operation type and extract path/content
        path = arguments.get("path") or arguments.get("file_path") or arguments.get("filepath")
        content = arguments.get("content") or arguments.get("text") or arguments.get("data", "")

        if not path:
            raise ValueError(f"No path found in arguments for {tool_name}: {arguments}")

        if "delete" in name_lower or "remove" in name_lower:
            return self.workspace.virtual_delete_file(path)
        elif "write" in name_lower or "create" in name_lower or "save" in name_lower:
            return self.workspace.virtual_write_file(path, str(content))
        elif "read" in name_lower:
            # Read operations - use workspace's layered read
            real_read = self.agent.get_tool_by_name(tool_name)
            if real_read:
                return await self.workspace.read_file(path, real_read)
            return f"[Cannot read {path}: no read tool available]"
        else:
            # Unknown write operation - stage as modify
            return self.workspace.virtual_write_file(path, str(content))

    def _log_execution(self, tool_name: str, arguments: dict, result: Any,
                       success: bool, error: Optional[str]):
        """Log tool execution for tracking"""
        self.execution_log.append({
            "tool_name": tool_name,
            "arguments": {k: str(v)[:100] for k, v in arguments.items()},
            "success": success,
            "error": error,
            "result_preview": str(result)[:200] if result else None,
            "timestamp": time.time()
        })

    def get_execution_summary(self) -> str:
        """Get summary of all executions"""
        if not self.execution_log:
            return "No tools executed."

        lines = [f"Executed {len(self.execution_log)} tool calls:"]
        for log in self.execution_log[-10:]:  # Last 10
            status = "✓" if log["success"] else "✗"
            lines.append(f"  {status} {log['tool_name']}")
        return "\n".join(lines)


# ============================================================================
# INCREMENTAL AGGREGATOR - Real-time Result Processing
# ============================================================================


class IncrementalAggregator:
    """
    Processes results after each parallel batch execution.

    Features:
    - Progressive response building
    - Dynamic abort on impossible tasks
    - Re-planning triggers
    - Response type detection
    """

    def __init__(
        self,
        agent,
        session_id: str,
        original_task: str,
        response_type: ResponseType = ResponseType.TEXT
    ):
        self.agent = agent
        self.session_id = session_id
        self.original_task = original_task
        self.response_type = response_type

        # Progressive state
        self.accumulated_results: list[dict] = []
        self.current_summary: str = ""
        self.iteration_count: int = 0

        # Signals
        self.abort_requested: bool = False
        self.abort_reason: Optional[str] = None
        self.replan_needed: bool = False
        self.new_subtasks: list[SubTask] = []

    async def process_batch(
        self,
        batch_results: list[dict],
        mda_state: "MDAStateV2"
    ) -> IncrementalStatus:
        """
        Process results from a parallel batch execution.
        Returns status with action to take next.
        """
        self.iteration_count += 1
        self.accumulated_results.extend(batch_results)

        # 1. Check for impossible tasks
        impossible_tasks = [
            r for r in batch_results
            if r.get("result", {}).get("is_impossible", False)
        ]
        if impossible_tasks:
            reasons = [r["result"].get("abort_reason", "Unknown") for r in impossible_tasks]
            return IncrementalStatus(
                action=AggregationAction.ABORT,
                completed_tasks=len(mda_state.completed_task_ids),
                total_tasks=len(mda_state.task_nodes),
                failed_tasks=len(mda_state.failed_task_ids),
                current_summary=self.current_summary,
                can_continue=False,
                abort_reason=f"Impossible tasks detected: {'; '.join(reasons)}"
            )

        # 2. Check for decomposition requests
        needs_decomposition = [
            r for r in batch_results
            if r.get("result", {}).get("needs_decomposition", False)
        ]
        if needs_decomposition:
            # Generate new subtasks
            new_subs = await self._generate_subtasks_for_decomposition(needs_decomposition, mda_state)
            return IncrementalStatus(
                action=AggregationAction.REPLAN,
                completed_tasks=len(mda_state.completed_task_ids),
                total_tasks=len(mda_state.task_nodes),
                failed_tasks=len(mda_state.failed_task_ids),
                current_summary=self.current_summary,
                can_continue=True,
                replan_needed=True,
                new_subtasks=new_subs
            )

        # 3. Update progressive summary
        await self._update_summary(batch_results, mda_state)

        # 4. Determine if we should continue or are complete
        remaining = len(mda_state.parallel_groups) - mda_state.current_group_index

        if remaining <= 0:
            return IncrementalStatus(
                action=AggregationAction.COMPLETE,
                completed_tasks=len(mda_state.completed_task_ids),
                total_tasks=len(mda_state.task_nodes),
                failed_tasks=len(mda_state.failed_task_ids),
                current_summary=self.current_summary,
                can_continue=False
            )

        # 5. Check failure threshold
        failure_rate = len(mda_state.failed_task_ids) / max(1, len(mda_state.task_nodes))
        if failure_rate > 0.5:  # More than 50% failed
            return IncrementalStatus(
                action=AggregationAction.ABORT,
                completed_tasks=len(mda_state.completed_task_ids),
                total_tasks=len(mda_state.task_nodes),
                failed_tasks=len(mda_state.failed_task_ids),
                current_summary=self.current_summary,
                can_continue=False,
                abort_reason=f"High failure rate: {failure_rate:.1%}"
            )

        return IncrementalStatus(
            action=AggregationAction.CONTINUE,
            completed_tasks=len(mda_state.completed_task_ids),
            total_tasks=len(mda_state.task_nodes),
            failed_tasks=len(mda_state.failed_task_ids),
            current_summary=self.current_summary,
            can_continue=True
        )

    async def generate_final_response(
        self,
        mda_state: "MDAStateV2",
        response_type: Optional[ResponseType] = None
    ) -> str:
        """Generate the final aggregated response"""
        rtype = response_type or self.response_type

        if rtype == ResponseType.STATUS:
            return await self._generate_status_response(mda_state)
        elif rtype == ResponseType.REPORT:
            return await self._generate_report_response(mda_state)
        elif rtype == ResponseType.FINAL:
            return await self._generate_final_synthesis(mda_state)
        else:  # TEXT
            return await self._generate_text_response(mda_state)

    async def _update_summary(self, batch_results: list[dict], mda_state: "MDAStateV2"):
        """Update the progressive summary with new results"""
        successful = [r for r in batch_results if r.get("success")]

        if not successful:
            return

        # Build incremental summary
        result_texts = [r["result"].get("result", "")[:200] for r in successful if "result" in r]

        if self.response_type == ResponseType.STATUS:
            # Compact status update
            self.current_summary = f"Progress: {len(mda_state.completed_task_ids)}/{len(mda_state.task_nodes)} tasks"
        else:
            # Append key findings
            if result_texts:
                self.current_summary += f"\n\n[Batch {self.iteration_count}]:\n" + "\n".join(result_texts[:3])

    async def _generate_subtasks_for_decomposition(
        self,
        needs_decomposition: list[dict],
        mda_state: "MDAStateV2"
    ) -> list[SubTask]:
        """Generate new subtasks for tasks that need further decomposition"""
        new_subtasks = []

        for result_data in needs_decomposition:
            task_id = result_data.get("task_id")
            task = mda_state.get_task_node(task_id)
            if not task:
                continue

            # Use agent to generate new subtasks
            prompt = f"""The task "{task.description}" needs to be broken down further.

Current result indicated: {result_data.get('result', {}).get('result', 'Unknown')}

Generate 2-3 simpler subtasks that together accomplish the original goal."""

            try:
                result = await self.agent.a_format_class(
                    pydantic_model=DivisionResult,
                    prompt=prompt,
                    model_preference="fast",
                    max_retries=1,
                    session_id=self.session_id
                )
                division = DivisionResult(**result)
                new_subtasks.extend(division.subtasks)
            except Exception:
                # Fallback: mark as failed
                pass

        return new_subtasks

    async def _generate_status_response(self, mda_state: "MDAStateV2") -> str:
        """Generate compact status response"""
        completed = len(mda_state.completed_task_ids)
        total = len(mda_state.task_nodes)
        failed = len(mda_state.failed_task_ids)

        return f"""Status: {'✓ Complete' if failed == 0 else '⚠ Partial'}
Tasks: {completed}/{total} completed, {failed} failed
Files: {len(mda_state.committed_files)} modified
Summary: {self.current_summary[:200]}"""

    async def _generate_report_response(self, mda_state: "MDAStateV2") -> str:
        """Generate detailed structured report"""
        sections = ["# Execution Report\n"]

        # Overview
        sections.append("## Overview")
        sections.append(f"- Original Task: {self.original_task[:200]}")
        sections.append(f"- Total Tasks: {len(mda_state.task_nodes)}")
        sections.append(f"- Completed: {len(mda_state.completed_task_ids)}")
        sections.append(f"- Failed: {len(mda_state.failed_task_ids)}")

        # Results by task
        sections.append("\n## Task Results")
        for task_id, result in list(mda_state.results.items())[:10]:
            task = mda_state.get_task_node(task_id)
            sections.append(f"\n### {task.description[:50] if task else task_id}")
            sections.append(result.get("result", "No result")[:300])

        # Files modified
        if mda_state.committed_files:
            sections.append("\n## Files Modified")
            for f in mda_state.committed_files[:20]:
                sections.append(f"- {f}")

        return "\n".join(sections)

    async def _generate_final_synthesis(self, mda_state: "MDAStateV2") -> str:
        """Generate complete synthesized final result"""
        # Collect all results
        all_results = "\n".join([
            f"[{tid}]: {data.get('result', '')[:300]}"
            for tid, data in mda_state.results.items()
        ])

        prompt = f"""Synthesize these partial results into a complete response:

ORIGINAL TASK: {self.original_task}

PARTIAL RESULTS:
{all_results}

Create a comprehensive, well-structured response that fully addresses the original task."""

        try:
            response = await self.agent.a_run_llm_completion(
                node_name="FinalSynthesis",
                task_id="final",
                model_preference="complex",
                with_context=False,
                messages=[{"role": "user", "content": prompt}],
                session_id=self.session_id,
                max_tokens=3000
            )
            return response.strip()
        except Exception as e:
            return f"Synthesis failed: {str(e)}\n\nRaw results:\n{all_results}"

    async def _generate_text_response(self, mda_state: "MDAStateV2") -> str:
        """Generate simple text response"""
        if len(mda_state.results) == 1:
            # Single result - return directly
            return list(mda_state.results.values())[0].get("result", "")

        # Multiple results - brief synthesis
        results = [r.get("result", "")[:200] for r in mda_state.results.values()]

        prompt = f"""Briefly summarize these results in 2-3 sentences:

{chr(10).join(results[:5])}"""

        try:
            response = await self.agent.a_run_llm_completion(
                node_name="TextSummary",
                task_id="summary",
                model_preference="fast",
                with_context=False,
                messages=[{"role": "user", "content": prompt}],
                session_id=self.session_id,
                max_tokens=500
            )
            return response.strip()
        except Exception:
            return "\n\n".join(results)


# ============================================================================
# PROGRESS TRACKING DECORATOR
# ============================================================================


def with_progress_tracking(cls):
    """Decorator for automatic progress tracking on async nodes"""

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
            ))

            try:
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
                    metadata={"error": str(e)},
                ))
                raise

        cls.run_async = wrapped_run_async

    # Similar wrappers for prep_async, exec_async, post_async...
    # (Keeping compact for readability, same pattern as original)

    return cls


# ============================================================================
# MDA STATE V2 - Enhanced State Management
# ============================================================================


@dataclass
class MDATaskNodeV2:
    """Enhanced task node with staging support"""
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

    # V2: Enhanced fields
    requires_tools: bool = False
    suggested_tools: list[str] = field(default_factory=list)
    response_type: ResponseType = ResponseType.TEXT
    staged_changes: list[dict] = field(default_factory=list)
    can_fail: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description[:500],
            "context": self.context[:1000],
            "complexity": self.complexity,
            "dependencies": self.dependencies,
            "is_atomic": self.is_atomic,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "result": self.result,
            "votes": self.votes[-5:],
            "execution_attempts": self.execution_attempts,
            "parallel_group": self.parallel_group,
            "requires_tools": self.requires_tools,
            "suggested_tools": self.suggested_tools,
            "response_type": self.response_type.value,
            "can_fail": self.can_fail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MDATaskNodeV2":
        data["status"] = MDATaskStatus(data.get("status", "pending"))
        data["response_type"] = ResponseType(data.get("response_type", "text"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class MDAStateV2:
    """Enhanced state management with virtual workspace support"""

    def __init__(
        self,
        original_task: str,
        original_context: str,
        session_id: str,
        config: dict,
        variable_manager = None
    ):
        self.checkpoint_id = f"mda2_{uuid.uuid4().hex[:12]}"
        self.original_task = original_task
        self.original_context = original_context
        self.session_id = session_id
        self.config = config
        self.variable_manager = variable_manager

        # Task tree
        self.task_nodes: dict[str, MDATaskNodeV2] = {}
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

        # V2: Commit tracking
        self.committed_files: list[str] = []
        self.commit_history: list[dict] = []

        # Statistics
        self.stats = {
            "total_divisions": 0,
            "voting_rounds": 0,
            "red_flags_caught": 0,
            "commits_made": 0,
            "files_modified": 0,
            "tool_calls": 0,
            "context_fetches": 0,
            "total_execution_time_ms": 0
        }

        # Timestamps
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        self.paused_at: Optional[str] = None

    def create_root_task(self) -> MDATaskNodeV2:
        """Create the root task node"""
        root = MDATaskNodeV2(
            id=f"root_{uuid.uuid4().hex[:8]}",
            description=self.original_task,
            context=self.original_context,
            complexity=10,
            dependencies=[],
            is_atomic=False,
            status=MDATaskStatus.PENDING
        )
        self.task_nodes[root.id] = root
        self.root_task_id = root.id
        self.pending_divisions.append(root.id)
        return root

    def add_task_node(self, node: MDATaskNodeV2):
        self.task_nodes[node.id] = node
        self.last_updated = datetime.now().isoformat()

    def get_task_node(self, task_id: str) -> Optional[MDATaskNodeV2]:
        return self.task_nodes.get(task_id)

    def update_task_node(self, node: MDATaskNodeV2):
        self.task_nodes[node.id] = node
        self.last_updated = datetime.now().isoformat()

    def mark_task_ready(self, task_id: str):
        node = self.get_task_node(task_id)
        if node:
            node.status = MDATaskStatus.READY
            self.update_task_node(node)

    def has_pending_divisions(self) -> bool:
        return len(self.pending_divisions) > 0

    def get_atomic_tasks(self) -> list[MDATaskNodeV2]:
        return [
            node for node in self.task_nodes.values()
            if node.is_atomic and node.status in [MDATaskStatus.READY, MDATaskStatus.PENDING]
        ]

    def inject_tasks(self, new_subtasks: list[SubTask], parent_id: str = None):
        """Inject new tasks for dynamic recursion"""
        for st in new_subtasks:
            node = MDATaskNodeV2(
                id=st.id or f"dyn_{uuid.uuid4().hex[:8]}",
                description=st.description,
                context=st.relevant_context,
                complexity=st.complexity,
                dependencies=st.dependencies,
                is_atomic=st.is_atomic,
                status=MDATaskStatus.PENDING,
                parent_id=parent_id,
                requires_tools=st.requires_tools,
                suggested_tools=st.suggested_tools,
                response_type=st.expected_response_type,
                can_fail=st.can_fail
            )
            self.add_task_node(node)
            if not node.is_atomic:
                self.pending_divisions.append(node.id)

    def record_commit(self, task_id: str, files: list[str]):
        """Record a successful commit"""
        self.stats["commits_made"] += 1
        self.stats["files_modified"] += len(files)
        self.committed_files.extend(files)
        self.commit_history.append({
            "task_id": task_id,
            "files": files,
            "timestamp": datetime.now().isoformat()
        })

    def to_checkpoint(self) -> dict:
        """Create checkpoint from current state"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "original_task": self.original_task[:500],
            "original_context": self.original_context[:1000],
            "session_id": self.session_id,
            "config": self.config,
            "task_nodes": {tid: node.to_dict() for tid, node in self.task_nodes.items()},
            "root_task_id": self.root_task_id,
            "current_group_index": self.current_group_index,
            "completed_groups": self.completed_groups,
            "pending_divisions": self.pending_divisions,
            "completed_task_ids": self.completed_task_ids,
            "failed_task_ids": self.failed_task_ids,
            "results": {k: {
                "result": v.get("result", "")[:500],
                "context_for_next": v.get("context_for_next", "")[:300]
            } for k, v in self.results.items()},
            "committed_files": self.committed_files,
            "stats": self.stats,
            "created_at": self.created_at,
            "last_updated": datetime.now().isoformat(),
            "version": "2.0"
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, variable_manager=None) -> "MDAStateV2":
        """Restore state from checkpoint"""
        state = cls(
            original_task=checkpoint["original_task"],
            original_context=checkpoint["original_context"],
            session_id=checkpoint["session_id"],
            config=checkpoint["config"],
            variable_manager=variable_manager
        )

        state.checkpoint_id = checkpoint["checkpoint_id"]
        state.root_task_id = checkpoint["root_task_id"]
        state.current_group_index = checkpoint["current_group_index"]
        state.completed_groups = checkpoint["completed_groups"]
        state.pending_divisions = checkpoint["pending_divisions"]
        state.completed_task_ids = checkpoint["completed_task_ids"]
        state.failed_task_ids = checkpoint["failed_task_ids"]
        state.results = checkpoint["results"]
        state.committed_files = checkpoint.get("committed_files", [])
        state.stats = checkpoint["stats"]
        state.created_at = checkpoint["created_at"]

        for tid, node_dict in checkpoint["task_nodes"].items():
            state.task_nodes[tid] = MDATaskNodeV2.from_dict(node_dict)

        return state


# ============================================================================
# ASYNC NODES V2
# ============================================================================


COMPLEXITY_PATTERNS = {
    r"berechne|calculate|compute": 2,
    r"liste|list|enumerate": 3,
    r"lese|read|fetch": 3,
    r"analysiere|analyze|examine": 6,
    r"erstelle.*plan|design|architect": 7,
    r"refactor|rewrite|restructure": 8,
    r"implement|build|create.*system": 8,
}


@with_progress_tracking
class DivideNodeV2(AsyncNode):
    """Enhanced division node with response type detection"""

    def __init__(
        self,
        min_complexity: int = 2,
        max_subtasks: int = 5,
        model_strength: Literal["weak", "medium", "strong"] = "medium"
    ):
        super().__init__()
        self.min_complexity = min_complexity
        self.max_subtasks = {"weak": 2, "medium": 3, "strong": 5}.get(model_strength, 3)
        self.model_strength = model_strength

    async def prep_async(self, shared) -> dict:
        agent = shared.get("agent_instance")
        tool_registry = SafeToolRegistry()
        available_tools = list(agent._tool_registry.keys()) if hasattr(agent, '_tool_registry') else []

        return {
            "task_node": shared.get("current_task_node"),
            "agent_instance": agent,
            "mda_state": shared.get("mda_state"),
            "depth": shared.get("division_depth", 0),
            "max_depth": shared.get("max_division_depth", 10),
            "session_id": shared.get("session_id"),
            "is_paused": shared.get("mda_paused", False),
            "available_tools": available_tools,
            "tool_registry": tool_registry
        }

    async def exec_async(self, prep_res) -> dict:
        if prep_res.get("is_paused"):
            return {"action": "paused"}

        task_node: MDATaskNodeV2 = prep_res["task_node"]
        agent = prep_res["agent_instance"]
        depth = prep_res["depth"]
        max_depth = prep_res["max_depth"]
        available_tools = prep_res["available_tools"]

        if depth >= max_depth:
            return {
                "action": "force_atomic",
                "task_node": task_node,
                "reason": f"Max depth {max_depth} reached"
            }

        # Estimate complexity
        complexity = await self._estimate_complexity(
            task_node.description,
            task_node.context,
            agent,
            prep_res.get("session_id"),
            available_tools
        )

        if complexity.is_atomic or complexity.score <= self.min_complexity:
            task_node.is_atomic = True
            task_node.complexity = complexity.score
            task_node.status = MDATaskStatus.READY
            task_node.requires_tools = complexity.requires_tools
            task_node.suggested_tools = complexity.suggested_tools
            return {
                "action": "atomic",
                "task_node": task_node,
                "complexity": complexity.model_dump()
            }

        # Divide task
        task_node.status = MDATaskStatus.DIVIDING
        division = await self._divide_task(
            task_node, complexity, agent,
            prep_res.get("session_id"), available_tools
        )

        return {
            "action": "divided",
            "task_node": task_node,
            "division": division.model_dump(),
            "subtasks": [st.model_dump() for st in division.subtasks]
        }

    async def _estimate_complexity(
        self, task: str, context: str, agent, session_id: str, available_tools: list
    ) -> TaskComplexity:
        # Pattern-based fast check
        for pattern, score in COMPLEXITY_PATTERNS.items():
            if re.search(pattern, task, re.IGNORECASE):
                needs_tools = any(t in task.lower() for t in ["file", "read", "write", "search", "fetch"])
                return TaskComplexity(
                    score=score,
                    reasoning="Pattern-matched",
                    is_atomic=score <= self.min_complexity,
                    estimated_steps=max(1, score // 2),
                    requires_tools=needs_tools,
                    suggested_tools=[t for t in available_tools if t in task.lower()][:3]
                )

        # LLM estimation
        tools_hint = f"\nAvailable tools: {', '.join(available_tools[:10])}" if available_tools else ""

        prompt = f"""Rate task complexity 0-10:
Task: {task[:200]}
Context: {context[:200]}{tools_hint}

0-2=trivial, 3-4=simple, 5-6=medium, 7+=complex
is_atomic=true if cannot be divided further
requires_tools=true if external tools needed"""

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
            return TaskComplexity(
                score=5, reasoning=f"Fallback: {e}",
                is_atomic=False, estimated_steps=3
            )

    async def _divide_task(
        self, task_node: MDATaskNodeV2, complexity: TaskComplexity,
        agent, session_id: str, available_tools: list
    ) -> DivisionResult:

        tools_info = "\n".join([f"- {t}" for t in available_tools[:15]]) if available_tools else "None"

        prompt = f"""Divide this task into max {self.max_subtasks} subtasks:

TASK: {task_node.description}
CONTEXT: {task_node.context[:800]}
COMPLEXITY: {complexity.score}/10

AVAILABLE TOOLS:
{tools_info}

RULES:
1. Each subtask should be as independent as possible
2. Mark dependencies explicitly
3. Set requires_tools=true if a tool is needed
4. Set can_fail=true for optional/non-critical subtasks
5. Choose expected_response_type (text/status/report)"""

        try:
            result = await agent.a_format_class(
                pydantic_model=DivisionResult,
                prompt=prompt,
                model_preference="fast" if complexity.score < 7 else "complex",
                max_retries=2,
                auto_context=False,
                session_id=session_id
            )

            division = DivisionResult(**result)

            # Ensure unique IDs
            for i, subtask in enumerate(division.subtasks):
                if not subtask.id:
                    subtask.id = f"{task_node.id}_sub_{i}_{uuid.uuid4().hex[:6]}"

            return division

        except Exception as e:
            return DivisionResult(
                can_divide=False,
                subtasks=[SubTask(
                    id=f"{task_node.id}_atomic",
                    description=task_node.description,
                    relevant_context=task_node.context,
                    complexity=complexity.score,
                    is_atomic=True
                )],
                preserved_context=task_node.context
            )

    async def post_async(self, shared, prep_res, exec_res) -> str:
        mda_state: MDAStateV2 = shared.get("mda_state")

        if exec_res["action"] == "paused":
            return "paused"

        task_node = exec_res["task_node"]

        if exec_res["action"] in ["atomic", "force_atomic"]:
            mda_state.mark_task_ready(task_node.id)
            shared["atomic_tasks_ready"] = shared.get("atomic_tasks_ready", []) + [task_node.id]

            if not mda_state.has_pending_divisions():
                return "all_divided"
            return "continue_division"

        elif exec_res["action"] == "divided":
            subtasks_data = exec_res["subtasks"]
            child_ids = []

            for st_data in subtasks_data:
                child_node = MDATaskNodeV2(
                    id=st_data["id"],
                    description=st_data["description"],
                    context=st_data["relevant_context"],
                    complexity=st_data["complexity"],
                    dependencies=st_data["dependencies"],
                    is_atomic=st_data["is_atomic"],
                    status=MDATaskStatus.PENDING,
                    parent_id=task_node.id,
                    requires_tools=st_data.get("requires_tools", False),
                    suggested_tools=st_data.get("suggested_tools", []),
                    response_type=ResponseType(st_data.get("expected_response_type", "text")),
                    can_fail=st_data.get("can_fail", False)
                )
                mda_state.add_task_node(child_node)
                child_ids.append(child_node.id)

                if not child_node.is_atomic:
                    mda_state.pending_divisions.append(child_node.id)

            task_node.children_ids = child_ids
            task_node.status = MDATaskStatus.COMPLETED
            mda_state.update_task_node(task_node)
            mda_state.stats["total_divisions"] += 1

            if mda_state.has_pending_divisions():
                next_task_id = mda_state.pending_divisions.pop(0)
                shared["current_task_node"] = mda_state.get_task_node(next_task_id)
                shared["division_depth"] = prep_res["depth"] + 1
                return "continue_division"

            return "all_divided"

        return "error"


@with_progress_tracking
class TaskTreeBuilderNodeV2(AsyncNode):
    """Builds execution tree with parallel groups"""

    async def prep_async(self, shared) -> dict:
        return {
            "mda_state": shared.get("mda_state"),
            "max_parallel": shared.get("max_parallel", 5),
            "is_paused": shared.get("mda_paused", False)
        }

    async def exec_async(self, prep_res) -> dict:
        if prep_res.get("is_paused"):
            return {"action": "paused"}

        mda_state: MDAStateV2 = prep_res["mda_state"]
        max_parallel = prep_res["max_parallel"]

        atomic_tasks = mda_state.get_atomic_tasks()

        if not atomic_tasks:
            return {"action": "no_tasks", "parallel_groups": []}

        # Build dependency graph
        dep_graph = {task.id: task.dependencies for task in atomic_tasks}

        # Topological sort with parallel groups
        parallel_groups = self._build_parallel_groups(atomic_tasks, dep_graph, max_parallel)

        # Assign groups
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
            "total_tasks": len(atomic_tasks)
        }

    def _build_parallel_groups(
        self, tasks: list[MDATaskNodeV2], dep_graph: dict, max_parallel: int
    ) -> list[list[str]]:
        task_ids = {t.id for t in tasks}
        completed = set()
        groups = []

        while len(completed) < len(tasks):
            ready = []
            for task in tasks:
                if task.id not in completed:
                    relevant_deps = [d for d in dep_graph.get(task.id, []) if d in task_ids]
                    if all(d in completed for d in relevant_deps):
                        ready.append(task.id)

            if not ready:
                remaining = [t.id for t in tasks if t.id not in completed]
                ready = remaining[:max_parallel]

            group = ready[:max_parallel]
            groups.append(group)
            completed.update(group)

        return groups

    async def post_async(self, shared, prep_res, exec_res) -> str:
        if exec_res["action"] == "paused":
            return "paused"

        mda_state: MDAStateV2 = shared.get("mda_state")

        if exec_res["action"] == "no_tasks":
            return "no_tasks"

        mda_state.parallel_groups = exec_res["parallel_groups"]
        mda_state.current_group_index = 0
        shared["parallel_groups"] = exec_res["parallel_groups"]

        return "tree_built"


@with_progress_tracking
class AtomicConquerNodeV2(AsyncNode):
    """
    Enhanced atomic execution with Virtual Workspace sandboxing.

    Key Features:
    - Safe toolset with virtualized writes
    - Diff-based voting
    - Commit only after consensus
    - Incremental aggregation hooks
    """

    def __init__(
        self,
        num_attempts: int = 3,
        k_margin: int = 2,
        max_response_tokens: int = 750,
        red_flag_patterns: list[str] = None,
        enable_tools: bool = True,
        benchmark_mode: bool = False
    ):
        super().__init__()

        if benchmark_mode:
            self.num_attempts = 1
            self.k_margin = 1
        else:
            self.num_attempts = num_attempts
            self.k_margin = k_margin

        self.benchmark_mode = benchmark_mode
        self.max_response_tokens = max_response_tokens
        self.enable_tools = enable_tools

        self.red_flag_patterns = red_flag_patterns or [
            r"(?i)ich bin (mir )?nicht sicher",
            r"(?i)i('m| am) not sure",
            r"(?i)impossible|cannot be done",
        ]

        self.tool_registry = SafeToolRegistry()

    async def prep_async(self, shared) -> dict:
        mda_state: MDAStateV2 = shared.get("mda_state")

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

        agent = shared.get("agent_instance")
        variable_manager = shared.get("variable_manager")

        return {
            "tasks": tasks_to_execute,
            "agent_instance": agent,
            "mda_state": mda_state,
            "session_id": shared.get("session_id"),
            "is_paused": shared.get("mda_paused", False),
            "group_index": current_idx,
            "variable_manager": variable_manager,
            "aggregator": shared.get("aggregator")
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
        variable_manager = prep_res["variable_manager"]

        # Execute tasks in parallel with virtual workspaces
        execution_tasks = [
            self._execute_with_voting_and_sandbox(
                task, agent, mda_state, session_id, variable_manager
            )
            for task in tasks
        ]

        results = await asyncio.gather(*execution_tasks, return_exceptions=True)

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

    async def _execute_with_voting_and_sandbox(
        self,
        task: MDATaskNodeV2,
        agent,
        mda_state: MDAStateV2,
        session_id: str,
        variable_manager
    ) -> AtomicResult:
        """Execute task with Virtual Workspace sandboxing and diff-based voting"""
        task.status = MDATaskStatus.EXECUTING
        mda_state.update_task_node(task)

        base_context = self._build_execution_context(task, mda_state)

        votes: list[VotingCandidate] = []
        valid_results: list[AtomicResult] = []
        winning_workspace: Optional[VirtualWorkspace] = None
        winning_executor: Optional[VirtualToolExecutor] = None

        for attempt in range(self.num_attempts * 2):
            if len(valid_results) >= self.num_attempts:
                break

            # Create isolated workspace for this attempt
            workspace = VirtualWorkspace(
                variable_manager,
                f"{task.id}_{attempt}",
                base_path="/"
            )

            # Create virtual tool executor
            tool_executor = self.tool_registry.create_virtual_tool_executor(
                agent, workspace
            )

            # Execute with ReAct loop using virtual executor
            result = await self._execute_react_loop(
                task, base_context, agent, session_id, attempt,
                workspace, tool_executor
            )

            # Red-flag check
            if self._has_red_flags(result):
                mda_state.stats["red_flags_caught"] += 1
                workspace.rollback()
                continue

            valid_results.append(result)

            # Create voting candidate with both text and diff hash
            workspace_hash = workspace.get_staging_hash()
            text_hash = self._hash_text(result.result)
            combined_hash = f"{text_hash}_{workspace_hash}"

            existing = next((v for v in votes if v.combined_hash == combined_hash), None)

            if existing:
                existing.votes += 1
            else:
                votes.append(VotingCandidate(
                    result=result,
                    workspace_hash=workspace_hash,
                    text_hash=text_hash,
                    combined_hash=combined_hash,
                    votes=1
                ))
                # Store workspace and executor for potential commit
                votes[-1]._workspace = workspace  # type: ignore
                votes[-1]._executor = tool_executor  # type: ignore

            # Check k-margin victory
            winner = self._check_k_margin_victory(votes)
            if winner:
                winning_workspace = getattr(winner, '_workspace', None)
                winning_executor = getattr(winner, '_executor', None)
                mda_state.stats["voting_rounds"] += len(valid_results)

                # Track tool calls
                if winning_executor:
                    mda_state.stats["tool_calls"] = mda_state.stats.get("tool_calls", 0) + len(winning_executor.execution_log)

                # COMMIT PHASE: Apply winner's changes to real FS
                if winning_workspace and winning_workspace.has_changes():
                    await self._commit_workspace(winning_workspace, agent, task, mda_state)

                return winner.result

        # No clear winner - use best candidate
        if votes:
            best = max(votes, key=lambda v: (v.votes, v.result.confidence))
            winning_workspace = getattr(best, '_workspace', None)
            mda_state.stats["voting_rounds"] += len(valid_results)

            # Commit if we have changes
            if winning_workspace and winning_workspace.has_changes():
                await self._commit_workspace(winning_workspace, agent, task, mda_state)

            return best.result

        return AtomicResult(
            success=False,
            result="All attempts failed or were red-flagged",
            context_for_next="",
            confidence=0.0,
            red_flags=["all_attempts_failed"]
        )

    async def _execute_react_loop(
        self,
        task: MDATaskNodeV2,
        context: str,
        agent,
        session_id: str,
        attempt: int,
        workspace: VirtualWorkspace,
        tool_executor: VirtualToolExecutor
    ) -> AtomicResult:
        """
        Execute a full ReAct loop with tool calling via LiteLLM.
        Uses the VirtualToolExecutor for safe execution.
        """
        start_time = time.perf_counter()
        tool_results: dict[str, Any] = {}
        actions_executed: list[dict] = []

        # Get available tools for this task
        safe_tool_names = self.tool_registry.get_safe_tool_names(agent)

        # Prepare tools in LiteLLM format
        litellm_tools = self._prepare_tools_for_litellm(agent, safe_tool_names)

        # Add final_answer tool
        litellm_tools.append({
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": "Provide the final answer when the task is complete. Call this to finish.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "description": "Whether the task succeeded"},
                        "result": {"type": "string", "description": "The final result/answer"},
                        "context_for_next": {"type": "string", "description": "Context for subsequent tasks"},
                        "confidence": {"type": "number", "description": "Confidence 0-1"},
                        "needs_decomposition": {"type": "boolean", "description": "If task needs breakdown"},
                        "is_impossible": {"type": "boolean", "description": "If task is impossible"}
                    },
                    "required": ["success", "result"]
                }
            }
        })

        # Build initial prompt
        system_prompt = f"""You are executing an atomic task. Use the available tools to complete it.

RULES:
1. Use tools to gather information or make changes
2. File writes are STAGED (not immediately applied) - this is safe
3. When done, call 'final_answer' with your result
4. Be confident and precise
5. If task is impossible, set is_impossible=true in final_answer

AVAILABLE TOOLS: {', '.join(safe_tool_names)}

CURRENT STAGED CHANGES:
{workspace.get_diff_summary()}"""

        user_prompt = f"""TASK: {task.description}

CONTEXT: {context}

Complete this task. Use tools as needed, then call final_answer."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # ReAct Loop
        max_iterations = 10
        final_result = None

        for iteration in range(max_iterations):
            try:
                # Call LLM with tools
                response = await self._call_llm_with_tools(
                    agent, messages, litellm_tools, session_id
                )

                if not response:
                    break

                # Check for tool calls
                tool_calls = self._extract_tool_calls(response)

                if not tool_calls:
                    # No tool calls - check if there's a text response
                    text_content = self._extract_text_content(response)
                    if text_content:
                        final_result = AtomicResult(
                            success=True,
                            result=text_content,
                            context_for_next=text_content[:200],
                            confidence=0.7,
                            tool_results=tool_results,
                            actions_executed=actions_executed,
                            response_type=task.response_type
                        )
                    break

                # Process tool calls
                assistant_msg = {"role": "assistant", "content": None, "tool_calls": tool_calls}
                messages.append(assistant_msg)

                tool_responses = []
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    try:
                        arguments = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}

                    # Check for final_answer
                    if tool_name == "final_answer":
                        final_result = AtomicResult(
                            success=arguments.get("success", True),
                            result=arguments.get("result", ""),
                            context_for_next=arguments.get("context_for_next", ""),
                            confidence=arguments.get("confidence", 0.8),
                            needs_decomposition=arguments.get("needs_decomposition", False),
                            is_impossible=arguments.get("is_impossible", False),
                            tool_results=tool_results,
                            actions_executed=actions_executed,
                            response_type=task.response_type
                        )
                        break

                    # Execute tool via VirtualToolExecutor
                    exec_result = await tool_executor.execute(tool_name, arguments)

                    actions_executed.append({
                        "iteration": iteration,
                        "tool": tool_name,
                        "arguments": arguments,
                        "success": exec_result["success"],
                        "virtualized": exec_result.get("virtualized", False)
                    })

                    if exec_result["success"]:
                        result_str = str(exec_result["result"])[:2000]
                        tool_results[tool_name] = exec_result["result"]
                    else:
                        result_str = f"ERROR: {exec_result['error']}"

                    tool_responses.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str
                    })

                # Add tool responses to messages
                messages.extend(tool_responses)

                if final_result:
                    break

            except Exception as e:
                # Log error but continue
                actions_executed.append({
                    "iteration": iteration,
                    "error": str(e)
                })
                break

        # Build final result if not set
        if not final_result:
            final_result = AtomicResult(
                success=len(tool_results) > 0,
                result=f"Completed with {len(actions_executed)} actions. {workspace.get_diff_summary()}",
                context_for_next="",
                confidence=0.5,
                tool_results=tool_results,
                actions_executed=actions_executed,
                response_type=task.response_type
            )

        final_result.execution_time_ms = (time.perf_counter() - start_time) * 1000
        final_result.staged_changes = [
            change.model_dump() for change in workspace.staged_changes.values()
        ]
        final_result.staged_diff_summary = workspace.get_diff_summary()

        return final_result

    def _prepare_tools_for_litellm(
        self,
        agent,
        tool_names: list[str]
    ) -> list[dict]:
        """Prepare tools in LiteLLM format"""
        tools = []

        if not hasattr(agent, '_tool_registry'):
            return tools

        for name in tool_names:
            if name not in agent._tool_registry:
                continue

            tool_info = agent._tool_registry[name]

            # Build parameters schema
            params = {"type": "object", "properties": {}, "required": []}

            if isinstance(tool_info, dict):
                # Get schema from tool info
                if "args_schema" in tool_info:
                    schema = tool_info["args_schema"]
                    if isinstance(schema, dict):
                        params = schema
                    elif hasattr(schema, "schema"):
                        params = schema.schema()

                description = tool_info.get("description", f"Tool: {name}")
            else:
                description = getattr(tool_info, "__doc__", f"Tool: {name}") or f"Tool: {name}"

            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description[:500],
                    "parameters": params
                }
            })

        return tools

    async def _call_llm_with_tools(
        self,
        agent,
        messages: list[dict],
        tools: list[dict],
        session_id: str
    ) -> Optional[dict]:
        """Call LLM with tools using litellm"""
        try:
            import litellm

            model = agent.amd.fast_llm_model
            # TODO use ratlimiter
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                max_tokens=self.max_response_tokens,
                temperature=0.2
            )

            return response

        except Exception as e:
            # Fallback: Try without tools
            try:
                response = await agent.a_run_llm_completion(
                    node_name="AtomicConquer",
                    task_id="react_loop",
                    model_preference="fast",
                    with_context=False,
                    messages=messages,
                    session_id=session_id,
                    max_tokens=self.max_response_tokens
                )
                return {"choices": [{"message": {"content": response}}]}
            except Exception:
                return None

    def _extract_tool_calls(self, response) -> list[dict]:
        """Extract tool calls from LLM response"""
        tool_calls = []

        try:
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tc in message.tool_calls:
                        tool_calls.append({
                            "id": tc.id if hasattr(tc, 'id') else f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        })
            elif isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    if "tool_calls" in message:
                        tool_calls = message["tool_calls"]
        except Exception:
            pass

        return tool_calls

    def _extract_text_content(self, response) -> str:
        """Extract text content from LLM response"""
        try:
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                if hasattr(message, 'content') and message.content:
                    return message.content
            elif isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
        except Exception:
            pass
        return ""

    async def _commit_workspace(
        self,
        workspace: VirtualWorkspace,
        agent,
        task: MDATaskNodeV2,
        mda_state: MDAStateV2
    ):
        """Commit workspace changes to real filesystem"""
        # Get real write tool from agent
        real_write = None
        if hasattr(agent, '_tool_registry'):
            for name in ["write_file", "file_write", "create_file"]:
                if name in agent._tool_registry:
                    real_write = agent._tool_registry[name]
                    if isinstance(real_write, dict):
                        real_write = real_write.get('func') or real_write.get('function')
                    break

        if real_write:
            commit_results = await workspace.commit_to_real_fs(real_write)

            # Record commit
            committed_files = [r["path"] for r in commit_results if r["success"]]
            if committed_files:
                mda_state.record_commit(task.id, committed_files)
                task.staged_changes = [r for r in commit_results]

    def _build_execution_context(self, task: MDATaskNodeV2, mda_state: MDAStateV2) -> str:
        """Build context from task dependencies"""
        context_parts = [task.context]

        for dep_id in task.dependencies:
            dep_result = mda_state.results.get(dep_id)
            if dep_result:
                context_parts.append(
                    f"\n[Result from {dep_id}]: {dep_result.get('context_for_next', dep_result.get('result', ''))[:300]}"
                )

        return "\n".join(context_parts)

    def _has_red_flags(self, result: AtomicResult) -> bool:
        """Check for red flags"""
        if len(result.result) > self.max_response_tokens * 4:
            return True

        for pattern in self.red_flag_patterns:
            if re.search(pattern, result.result):
                return True

        if result.confidence < 0.3:
            return True

        if result.red_flags:
            return True

        return False

    def _hash_text(self, text: str) -> str:
        """Hash text result for comparison"""
        normalized = text.strip().lower()[:200]
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def _check_k_margin_victory(self, votes: list[VotingCandidate]) -> Optional[VotingCandidate]:
        """Check for k-margin victory"""
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

        mda_state: MDAStateV2 = shared.get("mda_state")
        aggregator: IncrementalAggregator = shared.get("aggregator")

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
                        "response_type": result_data["result"].get("response_type", "text"),
                        "staged_changes": result_data["result"].get("staged_changes", [])
                    }
                    mda_state.completed_task_ids.append(task_id)
                else:
                    if task.can_fail:
                        # Non-critical task - mark as completed with failure note
                        task.status = MDATaskStatus.COMPLETED
                        mda_state.results[task_id] = {
                            "result": f"[Optional task failed: {result_data['error']}]",
                            "context_for_next": ""
                        }
                        mda_state.completed_task_ids.append(task_id)
                    else:
                        task.status = MDATaskStatus.FAILED
                        task.result = {"error": result_data["error"]}
                        mda_state.failed_task_ids.append(task_id)

                mda_state.update_task_node(task)

        # Incremental aggregation
        if aggregator:
            status = await aggregator.process_batch(exec_res["results"], mda_state)

            if status.action == AggregationAction.ABORT:
                shared["abort_reason"] = status.abort_reason
                return "abort"

            if status.action == AggregationAction.REPLAN:
                # Inject new subtasks
                mda_state.inject_tasks(status.new_subtasks)
                # Rebuild tree
                return "replan"

            if status.action == AggregationAction.COMPLETE:
                return "all_complete"

        # Move to next group
        mda_state.current_group_index += 1
        mda_state.completed_groups.append(exec_res["group_index"])

        if mda_state.current_group_index >= len(mda_state.parallel_groups):
            return "all_complete"

        return "continue_execution"


@with_progress_tracking
class ResultAggregatorNodeV2(AsyncNode):
    """Enhanced result aggregation with flexible response types"""

    async def prep_async(self, shared) -> dict:
        return {
            "mda_state": shared.get("mda_state"),
            "agent_instance": shared.get("agent_instance"),
            "original_task": shared.get("original_task"),
            "session_id": shared.get("session_id"),
            "is_paused": shared.get("mda_paused", False),
            "aggregator": shared.get("aggregator"),
            "response_type": shared.get("response_type", ResponseType.TEXT)
        }

    async def exec_async(self, prep_res) -> dict:
        if prep_res.get("is_paused"):
            return {"action": "paused"}

        mda_state: MDAStateV2 = prep_res["mda_state"]
        aggregator: IncrementalAggregator = prep_res["aggregator"]
        response_type = prep_res["response_type"]

        completed = len(mda_state.completed_task_ids)
        failed = len(mda_state.failed_task_ids)
        total = len(mda_state.task_nodes)

        if not mda_state.results:
            return {
                "action": "no_results",
                "aggregated": AggregatedResult(
                    success=False,
                    final_result="No results to aggregate",
                    response_type=response_type,
                    total_tasks=total,
                    successful_tasks=completed,
                    failed_tasks=failed,
                    total_voting_rounds=mda_state.stats.get("voting_rounds", 0),
                    red_flags_caught=mda_state.stats.get("red_flags_caught", 0)
                ).model_dump()
            }

        # Generate final response using aggregator
        if aggregator:
            final_result = await aggregator.generate_final_response(mda_state, response_type)
        else:
            final_result = "\n\n".join([
                r.get("result", "") for r in mda_state.results.values()
            ])

        aggregated = AggregatedResult(
            success=completed > 0 and (failed == 0 or all(
                mda_state.get_task_node(tid).can_fail for tid in mda_state.failed_task_ids
                if mda_state.get_task_node(tid)
            )),
            final_result=final_result,
            response_type=response_type,
            partial_results={k: v.get("result", "") for k, v in mda_state.results.items()},
            total_tasks=total,
            successful_tasks=completed,
            failed_tasks=failed,
            total_voting_rounds=mda_state.stats.get("voting_rounds", 0),
            red_flags_caught=mda_state.stats.get("red_flags_caught", 0),
            commits_made=mda_state.stats.get("commits_made", 0),
            files_modified=mda_state.committed_files
        )

        return {
            "action": "aggregated",
            "aggregated": aggregated.model_dump()
        }

    async def post_async(self, shared, prep_res, exec_res) -> str:
        if exec_res["action"] == "paused":
            return "paused"

        shared["final_aggregated_result"] = exec_res["aggregated"]
        shared["mda_state"].final_result = exec_res["aggregated"]

        return "aggregated"


# ============================================================================
# MDA FLOW V2 - Main Orchestrator
# ============================================================================


@with_progress_tracking
class MDAFlowV2(AsyncFlow):
    """
    MAKER V2 Flow with Virtual Workspace and Incremental Aggregation.

    Features:
    - Sandboxed execution with diff-based voting
    - Dynamic recursion and re-planning
    - Flexible response types
    - Full stop/resume support
    """

    def __init__(
        self,
        min_complexity: int = 2,
        max_parallel: int = 5,
        k_margin: int = 2,
        num_attempts: int = 3,
        model_strength: Literal["weak", "medium", "strong"] = "medium",
        max_division_depth: int = 10,
        enable_tools: bool = True,
        response_type: ResponseType = ResponseType.TEXT
    ):
        self.config = {
            "min_complexity": min_complexity,
            "max_parallel": max_parallel,
            "k_margin": k_margin,
            "num_attempts": num_attempts,
            "model_strength": model_strength,
            "max_division_depth": max_division_depth,
            "enable_tools": enable_tools,
            "response_type": response_type.value
        }

        # Initialize nodes
        self.divide_node = DivideNodeV2(
            min_complexity=min_complexity,
            max_subtasks={"weak": 2, "medium": 3, "strong": 5}.get(model_strength, 3),
            model_strength=model_strength
        )
        self.tree_builder = TaskTreeBuilderNodeV2()
        self.atomic_conquer = AtomicConquerNodeV2(
            num_attempts=num_attempts,
            k_margin=k_margin,
            enable_tools=enable_tools
        )
        self.aggregator_node = ResultAggregatorNodeV2()

        # Define flow connections
        self.divide_node - "continue_division" >> self.divide_node
        self.divide_node - "all_divided" >> self.tree_builder
        self.divide_node - "paused" >> None

        self.tree_builder - "tree_built" >> self.atomic_conquer
        self.tree_builder - "no_tasks" >> self.aggregator_node
        self.tree_builder - "paused" >> None

        self.atomic_conquer - "continue_execution" >> self.atomic_conquer
        self.atomic_conquer - "all_complete" >> self.aggregator_node
        self.atomic_conquer - "replan" >> self.tree_builder  # Dynamic recursion
        self.atomic_conquer - "abort" >> self.aggregator_node
        self.atomic_conquer - "paused" >> None

        super().__init__(start=self.divide_node)

    async def run_async(self, shared) -> str:
        return await super().run_async(shared)


# ============================================================================
# MAIN API - a_accomplish_v2
# ============================================================================


async def a_accomplish_v2(
    agent,
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
    resume_checkpoint: dict = None,
    enable_tools: bool = True,
    response_type: ResponseType = ResponseType.TEXT,
    **kwargs
) -> dict[str, Any]:
    """
    MAKER V2: Massively Decomposed Agentic Process with Virtual Workspace.

    Key Improvements over V1:
    - Virtual Workspace: All file operations sandboxed, committed only after voting
    - Diff-based Voting: Vote on actual changes, not just text
    - Incremental Aggregation: Process results after each batch
    - Dynamic Recursion: Tasks can request further decomposition
    - Flexible Response Types: TEXT, REPORT, STATUS, FINAL

    Args:
        agent: FlowAgent instance
        task: Main task to accomplish
        context: Additional context
        min_complexity: Minimum complexity threshold (0-10)
        max_parallel: Maximum parallel executions
        k_margin: Required vote margin for k-voting
        num_attempts: Attempts per atomic task
        model_strength: Model strength ("weak", "medium", "strong")
        max_division_depth: Maximum decomposition depth
        session_id: Session ID
        progress_callback: Callback for progress updates
        resume_checkpoint: Checkpoint to resume from
        enable_tools: Whether to allow tool calls
        response_type: Type of final response

    Returns:
        dict with:
            - success: bool
            - result: Final result string
            - response_type: Type of response
            - checkpoint: Checkpoint data for resume
            - stats: Execution statistics
            - cost_info: Cost information
            - files_modified: List of modified files
    """
    session_id = session_id or agent.active_session or f"mda2_{uuid.uuid4().hex[:8]}"

    config = {
        "min_complexity": min_complexity,
        "max_parallel": max_parallel,
        "k_margin": k_margin,
        "num_attempts": num_attempts,
        "model_strength": model_strength,
        "max_division_depth": max_division_depth,
        "enable_tools": enable_tools,
        "response_type": response_type.value
    }

    # Track costs
    start_cost = agent.total_cost_accumulated
    start_tokens_in = agent.total_tokens_in
    start_tokens_out = agent.total_tokens_out
    start_time = time.perf_counter()

    try:
        # Get variable manager
        variable_manager = getattr(agent, 'variable_manager', None)

        # Initialize or restore state
        if resume_checkpoint:
            mda_state = MDAStateV2.from_checkpoint(resume_checkpoint, variable_manager)
            mda_state.paused_at = None
        else:
            mda_state = MDAStateV2(
                original_task=task,
                original_context=context,
                session_id=session_id,
                config=config,
                variable_manager=variable_manager
            )
            mda_state.create_root_task()

        # Initialize incremental aggregator
        aggregator = IncrementalAggregator(
            agent=agent,
            session_id=session_id,
            original_task=task,
            response_type=response_type
        )

        # Initialize flow
        mda_flow = MDAFlowV2(
            min_complexity=min_complexity,
            max_parallel=max_parallel,
            k_margin=k_margin,
            num_attempts=num_attempts,
            model_strength=model_strength,
            max_division_depth=max_division_depth,
            enable_tools=enable_tools,
            response_type=response_type
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
            "progress_tracker": ProgressTracker(progress_callback) if progress_callback else None,
            "variable_manager": variable_manager,
            "aggregator": aggregator,
            "response_type": response_type,
            "enable_tools": enable_tools
        }

        # Set initial task
        if not resume_checkpoint and mda_state.pending_divisions:
            first_task_id = mda_state.pending_divisions.pop(0)
            shared["current_task_node"] = mda_state.get_task_node(first_task_id)
            shared["division_depth"] = 0

        # Execute flow
        await mda_flow.run_async(shared)

        # Get final result
        final_result = shared.get("final_aggregated_result", {})
        mda_state.stats["total_execution_time_ms"] = (time.perf_counter() - start_time) * 1000

        checkpoint = mda_state.to_checkpoint()

        return {
            "success": final_result.get("success", False),
            "result": final_result.get("final_result", ""),
            "response_type": final_result.get("response_type", response_type.value),
            "partial_results": final_result.get("partial_results", {}),
            "files_modified": final_result.get("files_modified", []),
            "checkpoint": checkpoint,
            "stats": {
                **mda_state.stats,
                "total_tasks": final_result.get("total_tasks", 0),
                "successful_tasks": final_result.get("successful_tasks", 0),
                "failed_tasks": final_result.get("failed_tasks", 0),
                "commits_made": final_result.get("commits_made", 0)
            },
            "cost_info": {
                "total_cost": agent.total_cost_accumulated - start_cost,
                "tokens_in": agent.total_tokens_in - start_tokens_in,
                "tokens_out": agent.total_tokens_out - start_tokens_out,
                "execution_time_s": (time.perf_counter() - start_time)
            }
        }

    except Exception as e:
        checkpoint = mda_state.to_checkpoint() if 'mda_state' in locals() else None
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "checkpoint": checkpoint,
            "stats": mda_state.stats if 'mda_state' in locals() else {},
            "cost_info": {
                "total_cost": agent.total_cost_accumulated - start_cost,
                "tokens_in": agent.total_tokens_in - start_tokens_in,
                "tokens_out": agent.total_tokens_out - start_tokens_out,
                "execution_time_s": (time.perf_counter() - start_time)
            }
        }


# ============================================================================
# FLOWAGENT MIXIN V2
# ============================================================================


class FlowAgentMDAMixinV2:
    """
    Mixin that adds MAKER V2 capabilities to FlowAgent.
    """

    _mda_v2_checkpoints: dict[str, dict] = {}
    _mda_v2_current_session: Optional[str] = None

    async def a_accomplish_v2(
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
        response_type: ResponseType = ResponseType.TEXT,
        **kwargs
    ) -> dict[str, Any]:
        """
        Execute complex task using MAKER V2 with Virtual Workspace.
        """
        self._mda_v2_current_session = session_id or f"mda2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        resume_checkpoint = kwargs.pop("resume_checkpoint", None)

        result = await a_accomplish_v2(
            agent=self,
            task=task,
            context=context,
            min_complexity=min_complexity,
            max_parallel=max_parallel,
            k_margin=k_margin,
            num_attempts=num_attempts,
            model_strength=model_strength,
            max_division_depth=max_division_depth,
            session_id=self._mda_v2_current_session,
            progress_callback=progress_callback,
            resume_checkpoint=resume_checkpoint,
            response_type=response_type,
            **kwargs
        )

        if result.get("checkpoint"):
            self._mda_v2_checkpoints[self._mda_v2_current_session] = result["checkpoint"]

        return result


async def bind_accomplish_v2_to_agent(agent, and_as_tool: bool = True):
    """
    Bind MAKER V2 capabilities to an existing FlowAgent instance.
    """
    import types

    agent._mda_v2_checkpoints = {}
    agent._mda_v2_current_session = None

    # Bind methods
    for method_name in ["a_accomplish_v2"]:
        method = getattr(FlowAgentMDAMixinV2, method_name)
        bound_method = types.MethodType(method, agent)
        setattr(agent, method_name, bound_method)

    if and_as_tool:
        async def maker_v2_wrapper(
            task: str,
            context: str = "",
            min_complexity: int = 2,
            max_parallel: int = 5,
            model_strength: str = "medium",
            response_type: str = "text",
            **kwargs
        ) -> str:
            session_id = agent.active_session or "default"
            res = await agent.a_accomplish_v2(
                task=task,
                context=context,
                min_complexity=min_complexity,
                max_parallel=max_parallel,
                model_strength=model_strength,
                response_type=ResponseType(response_type),
                session_id=session_id,
                **kwargs
            )
            res['checkpoint'] = {}
            return res.get("result", str(res)) if res.get("success") else f"Error: {res.get('error', str(res))}"

        agent.add_first_class_tool(
            maker_v2_wrapper,
            "MAKER_V2",
            description="""**META_TOOL: MAKER_V2(task, context, min_complexity, response_type)**
- **Purpose:** Advanced MDAP with Virtual Workspace sandboxing
- **Features:**
  - Sandboxed file operations (staged, voted on, then committed)
  - Diff-based voting for reliable consensus
  - Incremental aggregation with dynamic abort
  - Dynamic recursion for complex tasks
- **Response Types:** text, report, status, final
- **Use for:** Complex coding, refactoring, multi-file operations
- **NOT for:** Simple queries, irreversible external actions
- **Example:** `MAKER_V2(task="Refactor auth module", min_complexity=5, response_type="report")`"""
        )

    return agent


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Main API
    "a_accomplish_v2",
    "bind_accomplish_v2_to_agent",
    "FlowAgentMDAMixinV2",

    # Enums
    "ResponseType",
    "ToolCategory",
    "MDATaskStatus",
    "AggregationAction",
    "ActionType",

    # Models
    "TaskComplexity",
    "SubTask",
    "DivisionResult",
    "AtomicResult",
    "AggregatedResult",
    "StagedChange",
    "VotingCandidate",
    "IncrementalStatus",
    "ToolCallSpec",

    # Core Components
    "VirtualWorkspace",
    "SafeToolRegistry",
    "IncrementalAggregator",

    # State Management
    "MDAStateV2",
    "MDATaskNodeV2",

    # Nodes
    "DivideNodeV2",
    "TaskTreeBuilderNodeV2",
    "AtomicConquerNodeV2",
    "ResultAggregatorNodeV2",

    # Flow
    "MDAFlowV2",
]
