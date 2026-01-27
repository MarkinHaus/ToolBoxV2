"""
ExecutionEngine V3 - Intelligent Agent Orchestration

Features:
- Dynamic Tool Loading with keyword-based relevance scoring
- Working/Permanent History separation with rule-based compression
- AutoFocusTracker for context continuity
- LoopDetector for autonomous safety
- Skills integration for learned behaviors
- Graceful max iterations handling with honest communication
- Intelligent tool slot management (auto-remove lowest relevance)

Compression Triggers:
- TRIGGER 1: final_answer → Always rule-based compression
- TRIGGER 2: load_tools + category change + len(working) > 3 → Rule-based compression

Author: FlowAgent V3
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from litellm.types.utils import ModelResponseStream

# Import Skills System
from toolboxv2.mods.isaa.base.Agent.skills import (
    Skill,
    SkillsManager,
    auto_group_tools_by_name_pattern,
)

# Import Sub-Agent System
from toolboxv2.mods.isaa.base.Agent.sub_agent import (
    PARALLEL_SUBTASKS_SKILL,
    SUB_AGENT_TOOLS,
    RestrictedVFSWrapper,
    SubAgentManager,
    SubAgentResult,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Function
# =============================================================================
# STATIC TOOL DEFINITIONS (separate from dynamic limit)
# =============================================================================

STATIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Use this to think through a problem step by step. Write your reasoning here before taking action. This helps you plan and avoid mistakes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your reasoning, plan, or analysis",
                    }
                },
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Call this when you have completed the task or have the final answer. Be HONEST - if you couldn't complete the task, explain why.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your final response to the user",
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether you successfully completed the task (true/false)",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

DISCOVERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tools",
            "description": "List available tools, optionally filtered by category. Use this to discover what tools you can load.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g., 'discord', 'vfs', 'memory')",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_tools",
            "description": "Load specific tools into your active context. You can only use tools after loading them. Old tools are auto-removed when limit is reached.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tools": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": "Tool name or list of tool names to load",
                    }
                },
                "required": ["tools"],
            },
        },
    },
]

# VFS Tools - always available for file navigation (part of static)
VFS_TOOL_NAMES = ["vfs_read", "vfs_write", "vfs_list", "vfs_navigate", "vfs_control"]


# =============================================================================
# HELPER COMPONENTS
# =============================================================================


@dataclass
class AutoFocusTracker:
    """
    Tracks recent actions for context continuity.
    Injected as system message BEFORE user query in working_history.

    This prevents "I forgot what I just did" errors in small models.
    """

    max_actions: int = 5
    max_chars: int = 500
    actions: List[str] = field(default_factory=list)

    def record(self, tool_name: str, args: dict, result: str):
        """Compress and store action"""
        summary = self._compress(tool_name, args, result)
        self.actions.append(summary)

        if len(self.actions) > self.max_actions:
            self.actions.pop(0)

    def _compress(self, tool_name: str, args: dict, result: str) -> str:
        """Intelligent compression based on tool type"""
        tool_lower = tool_name.lower()
        result_lower = result.lower() if result else ""

        # File operations
        if "write" in tool_lower or "create" in tool_lower:
            path = args.get("path", args.get("filename", args.get("name", "?")))
            lines = len(result.split("\n")) if result else 0
            return f"✏️ Wrote {path} ({lines} lines)"

        elif "read" in tool_lower:
            path = args.get("path", "?")
            chars = len(result) if result else 0
            return f"📖 Read {path} ({chars} chars)"

        elif "list" in tool_lower or "navigate" in tool_lower:
            count = result.count("\n") + 1 if result else 0
            return f"📋 Listed {count} items"

        # Execution
        elif "execute" in tool_lower or "run" in tool_lower or "shell" in tool_lower:
            status = "✅" if "error" not in result_lower else "❌"
            return f"{status} Executed command"

        # Search/Query
        elif "search" in tool_lower or "query" in tool_lower:
            count = result.count("\n") + 1 if result else 0
            return f"🔍 Searched, found {count} results"

        # Memory
        elif "memory" in tool_lower or "inject" in tool_lower:
            return f"💾 Memory operation: {tool_name}"

        # Think
        elif tool_name == "think":
            thought_preview = args.get("thought", "")[:40]
            return f"💭 Thought: {thought_preview}..."

        # Discovery
        elif tool_name == "list_tools":
            category = args.get("category", "all")
            return f"📋 Listed tools (category: {category})"

        elif tool_name == "load_tools":
            tools = args.get("tools", [])
            if isinstance(tools, str):
                tools = [tools]
            return f"📦 Loaded: {', '.join(tools[:3])}"

        # Default
        else:
            return f"🔧 {tool_name}"

    def get_focus_message(self) -> Optional[dict]:
        """Get system message for injection into working_history"""
        if not self.actions:
            return None

        content = "LETZTE AKTIONEN (zur Erinnerung):\n" + "\n".join(
            f"- {a}" for a in self.actions
        )

        if len(content) > self.max_chars:
            content = content[: self.max_chars] + "..."

        return {"role": "system", "content": content}

    def clear(self):
        """Clear all tracked actions"""
        self.actions.clear()


@dataclass
class LoopDetector:
    """
    Detects when agent is stuck in a loop.

    Patterns detected:
    - Same tool 3x with same args (exact repeat)
    - Ping-pong pattern (A-B-A-B)
    """

    max_repeats: int = 3
    history: List[Tuple[str, int]] = field(default_factory=list)  # (tool_name, args_hash)

    def record(self, tool_name: str, args: dict) -> bool:
        """
        Record call and return True if loop detected.
        """
        try:
            args_hash = hash(json.dumps(args, sort_keys=True, default=str))
        except:
            args_hash = hash(str(args))

        entry = (tool_name, args_hash)
        self.history.append(entry)

        # Keep last 10
        if len(self.history) > 10:
            self.history.pop(0)

        # Check: Exact same call N times in a row
        if len(self.history) >= self.max_repeats:
            last_n = self.history[-self.max_repeats :]
            if all(e == entry for e in last_n):
                return True

        # Check: Ping-pong pattern (A-B-A-B)
        if len(self.history) >= 4:
            last4 = self.history[-4:]
            if last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
                return True

        return False

    def get_intervention_message(self) -> str:
        """Message for agent when loop detected"""
        last_tool = self.history[-1][0] if self.history else "unknown"

        return f"""⚠️ LOOP ERKANNT: Du hast '{last_tool}' mehrfach mit gleichen Argumenten aufgerufen.

OPTIONEN:
1. Falls du blockiert bist → Nutze final_answer um das Problem zu erklären
2. Falls du andere Daten brauchst → Ändere deinen Ansatz oder die Argumente
3. Falls du auf User-Input wartest → Sage das ehrlich in final_answer

WIEDERHOLE NICHT die gleiche Aktion. Sei ehrlich wenn du nicht weiterkommst."""

    def reset(self):
        """Clear history"""
        self.history.clear()


@dataclass
class ToolSlot:
    """Represents a dynamically loaded tool slot with relevance tracking"""

    name: str
    relevance_score: float
    category: str = "unknown"
    loaded_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================


@dataclass
class ExecutionContext:
    """
    Complete state for one execution run.
    Separates working_history (temporary) from permanent history.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # Execution metadata (for pause/resume)
    session_id: str = ""
    query: str = ""
    status: str = "running"  # "running", "paused", "completed", "cancelled"

    # Tool Management (dynamic tools, separate from static)
    dynamic_tools: List[ToolSlot] = field(default_factory=list)
    max_dynamic_tools: int = 5
    tool_relevance_cache: Dict[str, float] = field(default_factory=dict)
    tool_category_cache: Dict[str, Set[str]] = field(default_factory=dict)

    # History Management
    working_history: List[dict] = field(default_factory=list)

    # Trackers
    auto_focus: AutoFocusTracker = field(default_factory=AutoFocusTracker)
    loop_detector: LoopDetector = field(default_factory=LoopDetector)

    # Run State
    tools_used: List[str] = field(default_factory=list)
    current_iteration: int = 0
    matched_skills: List[Skill] = field(default_factory=list)
    loop_warning_given: bool = False

    def get_dynamic_tool_names(self) -> List[str]:
        """Get names of currently loaded dynamic tools"""
        return [slot.name for slot in self.dynamic_tools]

    def get_least_relevant_tool(self) -> Optional[str]:
        """Get the tool with lowest relevance score"""
        if not self.dynamic_tools:
            return None

        sorted_tools = sorted(self.dynamic_tools, key=lambda t: t.relevance_score)
        return sorted_tools[0].name if sorted_tools else None

    def get_majority_category(self) -> Optional[str]:
        """Get the most common category among loaded tools"""
        if not self.dynamic_tools:
            return None

        categories = [
            slot.category for slot in self.dynamic_tools if slot.category != "unknown"
        ]
        if not categories:
            return None

        from collections import Counter

        return Counter(categories).most_common(1)[0][0]

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from dynamic slots"""
        for i, slot in enumerate(self.dynamic_tools):
            if slot.name == name:
                self.dynamic_tools.pop(i)
                return True
        return False

    def add_tool(
        self, name: str, relevance_score: float, category: str = "unknown"
    ) -> bool:
        """Add a tool to dynamic slots"""
        # Check if already loaded
        if name in self.get_dynamic_tool_names():
            return False

        self.dynamic_tools.append(
            ToolSlot(name=name, relevance_score=relevance_score, category=category)
        )
        if len(self.dynamic_tools) >= self.max_dynamic_tools:
            _ = self.dynamic_tools.pop(1)
        return True

    def to_checkpoint(self) -> dict:
        """Serialize context for pause/resume"""
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "query": self.query,
            "status": self.status,
            "dynamic_tools": [
                {
                    "name": t.name,
                    "relevance_score": t.relevance_score,
                    "category": t.category,
                }
                for t in self.dynamic_tools
            ],
            "max_dynamic_tools": self.max_dynamic_tools,
            "tool_relevance_cache": self.tool_relevance_cache,
            "working_history": self.working_history,
            "tools_used": self.tools_used,
            "current_iteration": self.current_iteration,
            "loop_warning_given": self.loop_warning_given,
            "auto_focus_actions": self.auto_focus.actions,
            "loop_detector_history": self.loop_detector.history,
        }

    @classmethod
    def from_checkpoint(cls, data: dict) -> "ExecutionContext":
        """Restore context from checkpoint"""
        ctx = cls(run_id=data.get("run_id", uuid.uuid4().hex[:8]))
        ctx.session_id = data.get("session_id", "")
        ctx.query = data.get("query", "")
        ctx.status = data.get("status", "paused")
        ctx.dynamic_tools = [
            ToolSlot(
                name=t["name"],
                relevance_score=t["relevance_score"],
                category=t.get("category", "unknown"),
            )
            for t in data.get("dynamic_tools", [])
        ]
        ctx.max_dynamic_tools = data.get("max_dynamic_tools", 5)
        ctx.tool_relevance_cache = data.get("tool_relevance_cache", {})
        ctx.working_history = data.get("working_history", [])
        ctx.tools_used = data.get("tools_used", [])
        ctx.current_iteration = data.get("current_iteration", 0)
        ctx.loop_warning_given = data.get("loop_warning_given", False)
        ctx.auto_focus.actions = data.get("auto_focus_actions", [])
        ctx.loop_detector.history = data.get("loop_detector_history", [])
        return ctx


# =============================================================================
# HISTORY COMPRESSOR
# =============================================================================


class HistoryCompressor:
    """
    Rule-based compression for working history.

    Always rule-based (no LLM calls) for predictability and speed.
    """

    @staticmethod
    def compress_to_summary(working_history: List[dict], run_id: str) -> Optional[dict]:
        """
        Compress working history to a single summary message.

        Returns system message with action summary (goes BEFORE final_answer).
        """
        if not working_history:
            return None

        files_created = []
        files_read = []
        files_modified = []
        tools_used = set()
        errors = []
        thoughts = []
        searches = []

        for msg in working_history:
            role = msg.get("role", "")

            # Track tool calls from assistant messages
            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    if hasattr(tc, "function"):
                        tools_used.add(tc.function.name)
                    elif isinstance(tc, dict) and "function" in tc:
                        tools_used.add(tc["function"].get("name", ""))

            # Analyze tool results
            elif role == "tool":
                tool_name = msg.get("name", "")
                content = msg.get("content", "")
                tools_used.add(tool_name)

                # Categorize by tool type
                tool_lower = tool_name.lower()
                content_lower = content.lower()

                if "error" in content_lower or "failed" in content_lower:
                    errors.append(f"{tool_name}: {content[:80]}")
                elif "write" in tool_lower or "create" in tool_lower:
                    files_created.append(tool_name)
                elif "read" in tool_lower:
                    files_read.append(tool_name)
                elif (
                    "modify" in tool_lower
                    or "edit" in tool_lower
                    or "update" in tool_lower
                ):
                    files_modified.append(tool_name)
                elif "search" in tool_lower or "query" in tool_lower:
                    result_count = content.count("\n") + 1 if content else 0
                    searches.append(f"{tool_name} ({result_count} results)")
                elif tool_name == "think":
                    # Extract thought preview
                    thought_preview = content[:60] if content else ""
                    thoughts.append(thought_preview)

        # Build summary
        lines = ["ABGESCHLOSSENE AKTIONEN:"]

        if files_created:
            lines.append(f"• Erstellt: {len(files_created)} Datei(en)")
        if files_modified:
            lines.append(f"• Bearbeitet: {len(files_modified)} Datei(en)")
        if files_read:
            lines.append(f"• Gelesen: {len(files_read)} Datei(en)")
        if searches:
            lines.append(f"• Suchen: {len(searches)}x")
        if thoughts:
            lines.append(f"• Überlegungen: {len(thoughts)}x")
        if errors:
            lines.append(f"• ⚠️ Fehler: {len(errors)}")
            for err in errors[:2]:  # Max 2 errors
                lines.append(f"  - {err[:60]}...")

        # Tool summary
        meaningful_tools = tools_used - {
            "think",
            "final_answer",
            "list_tools",
            "load_tools",
        }
        if meaningful_tools:
            lines.append(f"• Tools genutzt: {', '.join(list(meaningful_tools)[:5])}")

        lines.append(f"• Gesamt Tool-Calls: {len(tools_used)}")

        return {
            "role": "system",
            "content": "\n".join(lines),
            "metadata": {"type": "action_summary", "run_id": run_id},
        }

    @staticmethod
    def compress_partial(
        working_history: List[dict], keep_last_n: int = 3
    ) -> Tuple[Optional[dict], List[dict]]:
        """
        Partially compress working history, keeping last N messages.
        Used when loading new tools mid-run to prevent context overflow.

        Returns: (summary_message, remaining_messages)
        """
        if len(working_history) <= keep_last_n + 1:  # +1 for system prompt
            return None, working_history

        # Find where to split (keep system prompt + last N)
        system_msg = (
            working_history[0] if working_history[0].get("role") == "system" else None
        )

        if system_msg:
            to_compress = working_history[1:-keep_last_n]
            to_keep = [system_msg] + working_history[-keep_last_n:]
        else:
            to_compress = working_history[:-keep_last_n]
            to_keep = working_history[-keep_last_n:]

        # Simple summary for partial compression
        tool_names = []
        for msg in to_compress:
            if msg.get("role") == "tool":
                tool_names.append(msg.get("name", "unknown"))

        if not tool_names:
            return None, working_history

        # Deduplicate while keeping order
        seen = set()
        unique_tools = []
        for t in tool_names:
            if t not in seen:
                seen.add(t)
                unique_tools.append(t)

        summary = {
            "role": "system",
            "content": f"FRÜHERE AKTIONEN: {', '.join(unique_tools[:5])}{'...' if len(unique_tools) > 5 else ''} ({len(tool_names)} calls)",
        }

        # Insert summary after system prompt
        if system_msg:
            return summary, [system_msg, summary] + to_keep[1:]
        else:
            return summary, [summary] + to_keep


# =============================================================================
# EXECUTION ENGINE V3
# =============================================================================


class ExecutionEngine:
    """
    Main orchestration engine for FlowAgent.

    Features:
    - Dynamic tool loading with keyword-based relevance scoring
    - Working/Permanent history separation with rule-based compression
    - Skills integration for learned behaviors
    - Loop detection and intervention
    - Graceful max iterations handling with honest communication
    - Intelligent tool slot management (auto-remove lowest relevance)

    Compression Triggers:
    - TRIGGER 1: final_answer → Always compress working → summary → permanent
    - TRIGGER 2: load_tools + category change + len > 3 → Partial compression
    """

    def __init__(
        self,
        agent,
        human_online: bool = False,
        callback: Any = None,
        # Sub-agent parameters
        is_sub_agent: bool = False,
        sub_agent_output_dir: str | None = None,
        sub_agent_budget: int = 5000,
    ):
        """
        Initialize ExecutionEngine.

        Args:
            agent: FlowAgent instance
            human_online: Whether human is actively monitoring
            is_sub_agent: If True, this is a sub-agent (cannot spawn further sub-agents)
            sub_agent_output_dir: If sub-agent, the only directory where writes are allowed
            sub_agent_budget: Token budget for sub-agent execution
        """
        self.agent = agent
        self.human_online = human_online

        # Sub-agent state
        self.is_sub_agent = is_sub_agent
        self.sub_agent_output_dir = sub_agent_output_dir
        self.sub_agent_budget = sub_agent_budget
        self._tokens_used = 0
        self._iterations_used = 0

        # SubAgentManager (None for sub-agents, they can't spawn)
        self._sub_agent_manager: SubAgentManager | None = None

        # Active executions for pause/resume
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._current_session = None

        # Get or create SkillsManager
        if hasattr(agent, "skills_manager") and agent.skills_manager:
            self.skills_manager = agent.skills_manager
        else:
            self.skills_manager = SkillsManager(
                agent_name=agent.amd.name, memory_instance=self._get_memory_instance()
            )
            # Store back on agent
            agent.skills_manager = self.skills_manager

        # Add parallel_subtasks skill if not present
        if "parallel_subtasks" not in self.skills_manager.skills:
            from toolboxv2.mods.isaa.base.Agent.skills import Skill

            self.skills_manager.skills["parallel_subtasks"] = Skill(
                **PARALLEL_SUBTASKS_SKILL
            )

        # Auto-group tools if not done yet
        if not self.skills_manager.tool_groups:
            self._auto_setup_tool_groups()

    def _get_memory_instance(self) -> Any:
        """Get memory instance from agent's session manager"""
        try:
            return self.agent.session_manager._get_memory()
        except:
            return None

    def _auto_setup_tool_groups(self):
        """Automatically create tool groups from tool names"""
        try:
            groups = auto_group_tools_by_name_pattern(
                tool_manager=self.agent.tool_manager,
                skills_manager=self.skills_manager,
                min_group_size=2,
            )
            if groups:
                print(f"[ExecutionEngine] Auto-created {len(groups)} tool groups")
        except Exception as e:
            print(f"[ExecutionEngine] Failed to auto-group tools: {e}")

    # =========================================================================
    # MAIN EXECUTION LOOP
    # =========================================================================

    async def execute(
        self,
        query: str,
        session_id: str,
        max_iterations: int = 15,
        ctx: "ExecutionContext | None" = None,
        get_ctx: bool = False,
    ) -> "tuple[str, ExecutionContext] | str":
        """
        Main execution loop.

        Flow:
        1. Initialize context and match skills
        2. Calculate tool relevance (once at start)
        3. Preload relevant tools from matched skills
        4. Build initial prompt with skills section
        5. Execute loop with tool calls
        6. On final_answer: Compress working → summary → permanent
        7. Learn from successful runs

        Args:
            query: User query
            session_id: Session identifier
            max_iterations: Max iterations (default 15)
            ctx: Existing ExecutionContext (for resume)
            get_ctx: If True, return (result, ctx) tuple

        Returns:
            str: Final response
            tuple[str, ExecutionContext]: If get_ctx=True
        """
        session = await self.agent.session_manager.get_or_create(session_id)

        # Use existing context or create new one
        is_resume = ctx is not None
        if ctx is None:
            ctx = ExecutionContext()

        # Track active execution
        self._active_executions[ctx.run_id] = ctx
        ctx.session_id = session_id
        ctx.query = query

        # Initialize SubAgentManager (only if NOT a sub-agent)
        if not self.is_sub_agent:
            self._sub_agent_manager = SubAgentManager(
                parent_engine=self, parent_session=session, is_sub_agent=False
            )
        else:
            # Sub-agent: Apply VFS write restriction
            if self.sub_agent_output_dir:
                session.vfs = RestrictedVFSWrapper(session.vfs, self.sub_agent_output_dir)
            # Limit iterations for sub-agents
            max_iterations = min(max_iterations, 10)

        # Store session reference for sub-agent spawning
        self._current_session = session

        # Only initialize if not resuming
        if not is_resume:
            # 1. Match skills (hybrid: keyword + embedding)
            try:
                ctx.matched_skills = await self.skills_manager.match_skills_async(
                    query, max_results=2
                )
            except:
                ctx.matched_skills = self.skills_manager.match_skills(
                    query, max_results=2
                )

            # 2. Calculate tool relevance scores (once at start, cached)
            self._calculate_tool_relevance(ctx, query)

            # 3. Preload relevant tools from matched skills
            self._preload_skill_tools(ctx, query)

            # 4. Build initial messages
            system_prompt = self._build_system_prompt(ctx, session)

            # Get compressed permanent history (last N turns)
            # Sub-agents get minimal history (isolated context)
            history_depth = 2 if self.is_sub_agent else 6
            permanent_history = session.get_history_for_llm(last_n=history_depth)

            # Initial working history
            ctx.working_history = [
                {"role": "system", "content": system_prompt},
                *permanent_history,
                {"role": "user", "content": query},
            ]

        agent_type = "SUB-AGENT" if self.is_sub_agent else "MAIN"
        action = "Resuming" if is_resume else "Start"
        print(f"🚀 {action} Execution [{agent_type}] [{ctx.run_id}]: {query[:50]}...")
        if ctx.matched_skills:
            print(f"📚 Matched Skills: {[s.name for s in ctx.matched_skills]}")
        if ctx.dynamic_tools:
            print(f"📦 Preloaded Tools: {ctx.get_dynamic_tool_names()}")

        final_response = None
        success = True

        # 5. Main loop
        while ctx.current_iteration < max_iterations:
            ctx.current_iteration += 1

            # Check for loop and inject warning if needed
            if self._should_warn_loop(ctx):
                ctx.working_history.append(
                    {
                        "role": "system",
                        "content": ctx.loop_detector.get_intervention_message(),
                    }
                )
                ctx.loop_warning_given = True

            # Build current tool list
            current_tools = self._get_tool_definitions(ctx)

            # Inject AutoFocus before LLM call
            messages = self._inject_auto_focus(ctx)

            # LLM Call
            try:
                response = await self.agent.a_run_llm_completion(
                    messages=messages,
                    tools=current_tools,
                    tool_choice="auto",
                    model_preference="fast",
                    stream=False,
                    get_response_message=True,
                    with_context=False,  # We built context manually
                )
            except Exception as e:
                print(f"❌ LLM Error: {e}")
                final_response = f"Es ist ein Fehler aufgetreten: {str(e)}\n\nIch konnte die Aufgabe leider nicht abschließen."
                success = False
                break

            # Add assistant message to working history
            if response:
                msg_dict = {"role": "assistant", "content": response.content}
                if hasattr(response, "tool_calls") and response.tool_calls:
                    msg_dict["tool_calls"] = response.tool_calls
                ctx.working_history.append(msg_dict)

            # Process tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    result, is_final = await self._execute_tool_call(
                        ctx, tool_call
                    )

                    # Check if final_answer was called
                    if is_final:
                        try:
                            args = json.loads(tool_call.function.arguments)
                            final_response = args.get("answer", result)
                            success = args.get("success", True)
                        except:
                            final_response = result
                            success = True
                        break

                # Exit loop if final_answer was called
                if final_response is not None:
                    break
            else:
                # No tool calls - text response (accept as final)
                if response and response.content:
                    final_response = response.content
                    break

        # 6. Handle max iterations reached
        if final_response is None:
            final_response = self._handle_max_iterations(ctx, query)
            success = False

        # 7. Compress and commit to permanent history
        await self._commit_run(ctx, session, query, final_response, success)

        # 8. Learn from successful runs
        if success and len(ctx.tools_used) >= 2:
            try:
                await self.skills_manager.learn_from_run(
                    query=query,
                    tools_used=ctx.tools_used,
                    final_answer=final_response,
                    success=success,
                    llm_completion_func=self.agent.a_run_llm_completion,
                )
            except Exception as e:
                print(f"[ExecutionEngine] Skill learning failed: {e}")

        # 9. Record skill usage
        for skill in ctx.matched_skills:
            self.skills_manager.record_skill_usage(skill.id, success)

        # Remove from active executions
        self._active_executions.pop(ctx.run_id, None)

        print(
            f"✅ Execution [{ctx.run_id}] complete (success={success}, iterations={ctx.current_iteration})"
        )

        if get_ctx:
            return final_response, ctx
        return final_response

    def _should_warn_loop(self, ctx: ExecutionContext) -> bool:
        """Check if we should inject a loop warning"""
        if ctx.loop_warning_given:
            return False

        if len(ctx.loop_detector.history) >= 3:
            # Check if last 3 are the same
            last3 = ctx.loop_detector.history[-3:]
            if len(set(last3)) == 1:
                return True

        return False

    # =========================================================================
    # TOOL EXECUTION
    # =========================================================================

    async def _execute_tool_call(
        self, ctx: ExecutionContext, tool_call
    ) -> Tuple[str, bool]:
        """
        Execute a single tool call and update context.

        Returns: (result_string, is_final_answer)
        """
        f_name = tool_call.function.name
        f_id = tool_call.id

        try:
            f_args = json.loads(tool_call.function.arguments)
        except:
            f_args = {}

        print(f"  🔧 Tool: {f_name}")

        # Track tool usage
        ctx.tools_used.append(f_name)

        # Loop detection (record before execution)
        loop_detected = ctx.loop_detector.record(f_name, f_args)

        result = ""
        is_final = False

        # === STATIC TOOLS ===
        if f_name == "think":
            thought = f_args.get("thought", "")
            result = f"Thought recorded."
            # Record in AutoFocus
            ctx.auto_focus.record(f_name, f_args, thought[:100])

        elif f_name == "final_answer":
            answer = f_args.get("answer", "")
            success = f_args.get("success", True)
            result = answer
            is_final = True
            # Don't record final_answer in AutoFocus

        # === DISCOVERY TOOLS ===
        elif f_name == "list_tools":
            result = self._tool_list_tools(f_args.get("category"))
            ctx.auto_focus.record(f_name, f_args, result[:100])

        elif f_name == "load_tools":
            tools_input = f_args.get("tools") or f_args.get("names")
            result = await self._tool_load_tools(ctx, tools_input)
            ctx.auto_focus.record(f_name, f_args, result[:100])

        # === SUB-AGENT TOOLS ===
        elif f_name == "spawn_sub_agent":
            if self.is_sub_agent:
                result = (
                    "ERROR: Sub-agents cannot spawn other sub-agents! Max depth is 1."
                )
            elif not self._sub_agent_manager:
                result = "ERROR: SubAgentManager not initialized."
            else:
                try:
                    task = f_args.get("task", "")
                    output_dir = f_args.get("output_dir", f"task_{uuid.uuid4().hex[:6]}")
                    wait = f_args.get("wait", True)
                    budget = f_args.get("budget", 5000)

                    spawn_result = await self._sub_agent_manager.spawn(
                        task=task, output_dir=output_dir, wait=wait, budget=budget
                    )

                    if wait:
                        # spawn_result is SubAgentResult
                        sub_result: SubAgentResult = spawn_result
                        if sub_result.success:
                            result = (
                                f"✅ Sub-Agent completed successfully.\n"
                                f"Output: {sub_result.output_dir}\n"
                                f"Files: {', '.join(sub_result.files_written)}\n"
                                f"Result: {sub_result.result[:500] if sub_result.result else 'No result'}"
                            )
                        else:
                            result = (
                                f"❌ Sub-Agent failed: {sub_result.error}\n"
                                f"Status: {sub_result.status.value}\n"
                                f"Output dir: {sub_result.output_dir}"
                            )
                        # Inject into AutoFocus
                        focus_text = (
                            self._sub_agent_manager.format_results_for_auto_focus(
                                {sub_result.id: sub_result}
                            )
                        )
                        ctx.auto_focus.actions.append(focus_text)
                    else:
                        # spawn_result is sub_agent_id string
                        result = f"🚀 Sub-Agent gestartet: {spawn_result}\nOutput dir: /sub/{output_dir}\nNutze wait_for('{spawn_result}') um auf das Ergebnis zu warten."

                except Exception as e:
                    result = f"ERROR spawning sub-agent: {str(e)}"

            ctx.auto_focus.record(f_name, f_args, result[:200])

        elif f_name == "wait_for":
            if not self._sub_agent_manager:
                result = "ERROR: SubAgentManager not initialized."
            else:
                try:
                    sub_agent_ids = f_args.get("sub_agent_ids", [])
                    timeout = f_args.get("timeout", 300)

                    results = await self._sub_agent_manager.wait_for(
                        sub_agent_ids=sub_agent_ids, timeout=timeout
                    )

                    # Format results
                    result_lines = ["Sub-Agent Ergebnisse:"]
                    for sub_id, sub_result in results.items():
                        status = "✅" if sub_result.success else "❌"
                        result_lines.append(
                            f"\n{status} [{sub_id}]:\n"
                            f"  Status: {sub_result.status.value}\n"
                            f"  Output: {sub_result.output_dir}\n"
                            f"  Files: {', '.join(sub_result.files_written[:5])}"
                        )
                        if sub_result.error:
                            result_lines.append(f"  Error: {sub_result.error[:100]}")

                    result = "\n".join(result_lines)

                    # Inject into AutoFocus
                    focus_text = self._sub_agent_manager.format_results_for_auto_focus(
                        results
                    )
                    ctx.auto_focus.actions.append(focus_text)

                except Exception as e:
                    result = f"ERROR waiting for sub-agents: {str(e)}"

            ctx.auto_focus.record(f_name, f_args, result[:200])

        # === VFS & DYNAMIC TOOLS ===
        elif f_name:
            is_vfs = f_name in VFS_TOOL_NAMES
            is_loaded = f_name in ctx.get_dynamic_tool_names()

            if is_vfs or is_loaded:
                try:
                    result = await self.agent.arun_function(f_name, **f_args)
                    result = str(result) if result is not None else "Success (no output)"
                except Exception as e:
                    result = f"Error executing {f_name}: {str(e)}"
            else:
                try:
                    result = await self.agent.arun_function(f_name, **f_args)
                    result = str(result) if result is not None else "Success (no output)"
                except Exception as e:
                    ctx.add_tool(f_name, 1, "auto-detect-use")
                    result = f"Error: Tool '{f_name}' war noch nicht geladen (auto lodet). Nutze list_tools() und load_tools  () um tools dynamisch zu aktivieren. damit du es fehlerfrei verwenden kannst! Aufgetretener fehler : {e}"

            ctx.auto_focus.record(f_name, f_args, result[:200])

        # Add tool result to working history (if not final_answer)
        if not is_final:
            ctx.working_history.append(
                {
                    "role": "tool",
                    "tool_call_id": f_id,
                    "name": f_name,
                    "content": str(result)[:4000],  # Truncate long outputs
                }
            )

        return result, is_final

    # =========================================================================
    # TOOL MANAGEMENT
    # =========================================================================

    def _calculate_tool_relevance(self, ctx: ExecutionContext, query: str):
        """Calculate relevance scores for all tools at query start (cached)"""
        all_tools = self.agent.tool_manager.get_all()

        for tool in all_tools:
            # Calculate relevance
            score = self.skills_manager.score_tool_relevance(
                query=query, tool_name=tool.name, tool_description=tool.description or ""
            )
            ctx.tool_relevance_cache[tool.name] = score

            # Cache categories
            categories = (
                tool.category if isinstance(tool.category, list) else [tool.category]
            )
            ctx.tool_category_cache[tool.name] = set(c for c in categories if c)

    def _preload_skill_tools(self, ctx: ExecutionContext, query: str):
        """Preload relevant tools from matched skills"""
        if not ctx.matched_skills:
            return

        tools_to_load = []

        for skill in ctx.matched_skills:
            # Add tools directly from skill
            for tool_name in skill.tools_used:
                if tool_name not in tools_to_load and tool_name not in [
                    "think",
                    "final_answer",
                    "list_tools",
                    "load_tools",
                ]:
                    tools_to_load.append(tool_name)

            # Add tools from skill's tool_groups
            for group_name in skill.tool_groups:
                relevant = self.skills_manager.get_relevant_tools_from_groups(
                    query=query,
                    tool_groups=[group_name],
                    tool_manager=self.agent.tool_manager,
                    max_tools=2,
                )
                for tool_name, score in relevant:
                    if tool_name not in tools_to_load:
                        tools_to_load.append(tool_name)

        # Load tools sorted by relevance, up to limit
        scored = [(t, ctx.tool_relevance_cache.get(t, 0.5)) for t in tools_to_load]
        scored.sort(key=lambda x: x[1], reverse=True)

        for tool_name, score in scored[: ctx.max_dynamic_tools]:
            tool_entry = self.agent.tool_manager.get(tool_name)
            if tool_entry:
                categories = (
                    tool_entry.category
                    if isinstance(tool_entry.category, list)
                    else [tool_entry.category]
                )
                category = categories[0] if categories else "unknown"
                ctx.add_tool(tool_name, score, category)

    async def _tool_load_tools(
        self, ctx: ExecutionContext, tools_input: Union[str, List[str]]
    ) -> str:
        """
        Load tools with intelligent slot management.

        - Auto-removes lowest relevance tool when limit reached
        - Triggers partial compression on category change + len > 3
        """
        if isinstance(tools_input, str):
            names = [tools_input]
        else:
            names = list(tools_input) if tools_input else []

        loaded = []
        failed = []
        removed = []

        all_tool_names = self.agent.tool_manager.list_names()

        for name in names:
            name = name.strip()

            # Already loaded?
            if name in ctx.get_dynamic_tool_names():
                loaded.append(f"{name} (bereits geladen)")
                continue

            # Exists?
            if name not in all_tool_names:
                failed.append(f"{name} (nicht gefunden)")
                continue

            # Get tool info
            tool_entry = self.agent.tool_manager.get(name)
            relevance = ctx.tool_relevance_cache.get(name, 0.5)
            categories = (
                tool_entry.category
                if tool_entry and isinstance(tool_entry.category, list)
                else [tool_entry.category if tool_entry else "unknown"]
            )
            new_category = categories[0] if categories else "unknown"

            # Check if we need to make room
            if len(ctx.dynamic_tools) >= ctx.max_dynamic_tools:
                # Check for category change (TRIGGER 2)
                majority_category = ctx.get_majority_category()

                if majority_category and new_category != majority_category:
                    # Category change detected - trigger partial compression if len > 3
                    if len(ctx.working_history) > 4:  # system + at least 3 messages
                        summary, new_history = HistoryCompressor.compress_partial(
                            ctx.working_history, keep_last_n=3
                        )
                        if summary:
                            ctx.working_history = new_history
                            print(
                                f"  📦 Partial compression triggered (category change: {majority_category} → {new_category})"
                            )

                # Remove least relevant tool
                least_relevant = ctx.get_least_relevant_tool()
                if least_relevant:
                    ctx.remove_tool(least_relevant)
                    removed.append(least_relevant)

            # Add new tool
            if ctx.add_tool(name, relevance, new_category):
                loaded.append(name)
            else:
                failed.append(f"{name} (limit erreicht)")

        # Build response message
        msg_parts = []
        if loaded:
            msg_parts.append(f"Geladen: {', '.join(loaded)}")
        if removed:
            msg_parts.append(f"Auto-entfernt (niedrige Relevanz): {', '.join(removed)}")
        if failed:
            msg_parts.append(f"Fehlgeschlagen: {', '.join(failed)}")

        return "\n".join(msg_parts) if msg_parts else "Keine Änderungen"

    def _tool_list_tools(self, category: Optional[str] = None) -> str:
        """List available tools with optional category filter"""
        all_tools = self.agent.tool_manager.get_all()
        lines = []

        for t in all_tools:
            # Filter by category
            match = True
            if category:
                t_cats = t.category if isinstance(t.category, list) else [t.category]
                if not any(category.lower() in str(c).lower() for c in t_cats if c):
                    match = False

            if match:
                desc = t.description.split("\n")[0] if t.description else "No description"
                desc = desc[:80]
                cats = ", ".join(
                    str(c)
                    for c in (
                        t.category[:2] if isinstance(t.category, list) else [t.category]
                    )
                    if c
                )
                lines.append(f"- {t.name}: {desc} [{cats}]")

        if not lines:
            return "Keine Tools für diese Kategorie gefunden."

        # List available categories at the end
        categories = set()
        for t in all_tools:
            if t.category:
                if isinstance(t.category, list):
                    categories.update(c for c in t.category if c)
                else:
                    categories.add(t.category)

        result = "\n".join(lines[:30])
        if len(lines) > 30:
            result += f"\n... und {len(lines) - 30} weitere"

        result += f"\n\nKategorien: {', '.join(sorted(categories))}"

        return result

    def _get_tool_definitions(self, ctx: ExecutionContext) -> List[dict]:
        """Build tool definitions for LLM (static + VFS + sub-agent + dynamic)"""
        definitions = []

        # 1. Static tools (always available, not counted in limit)
        definitions.extend(STATIC_TOOLS)

        # 2. Discovery tools
        definitions.extend(DISCOVERY_TOOLS)

        # 3. Sub-agent tools (ONLY for main agent, not for sub-agents)
        if not self.is_sub_agent:
            definitions.extend(SUB_AGENT_TOOLS)

        # 4. VFS tools (always available, not counted in limit)
        all_tools = self.agent.tool_manager.get_all_litellm()
        for tool in all_tools:
            t_name = tool["function"]["name"]
            if t_name in VFS_TOOL_NAMES:
                definitions.append(tool)

        # 5. Dynamic tools (from slots)
        dynamic_names = ctx.get_dynamic_tool_names()
        for tool in all_tools:
            t_name = tool["function"]["name"]
            if t_name in dynamic_names and t_name not in VFS_TOOL_NAMES:
                definitions.append(tool)

        return definitions

    # =========================================================================
    # PROMPT BUILDING
    # =========================================================================

    def _build_system_prompt(self, ctx: ExecutionContext, session) -> str:
        """Build system prompt with skills, status, and rules"""

        # Get categories from tool manager
        all_tools = self.agent.tool_manager.get_all()
        categories = set()
        for t in all_tools:
            if t.category:
                if isinstance(t.category, list):
                    categories.update(c for c in t.category if c)
                else:
                    categories.add(t.category)

        cat_list = ", ".join(sorted(categories)) if categories else "keine"
        active_str = (
            ", ".join(ctx.get_dynamic_tool_names()) if ctx.dynamic_tools else "keine"
        )

        # Base prompt - different for sub-agents
        if self.is_sub_agent:
            prompt_parts = [
                "You are a focused SUB-AGENT for a specific task.",
                "",
                "⚠️ SUB-AGENT CONSTRAINTS:",
                f"- You can ONLY write to {self.sub_agent_output_dir}/",
                "- You can read the entire VFS",
                "- You CANNOT spawn any additional sub-agents",
                "- You CANNOT ask follow-up questions — work with the given information",
                f"- Token budget: {self.sub_agent_budget}",
                "",
                "TASK: Execute the given task in a focused manner.",
                "Write your result to result.md in your output directory.",
                "",
                "STATUS:",
                f"- Loaded tools ({len(ctx.dynamic_tools)}/{ctx.max_dynamic_tools}): [{active_str}]",
                "",
                "RULES:",
                "1. Focus ONLY on the given task",
                "2. Write results to your output directory",
                "3. Use final_answer when finished",
                "4. If something is unclear: make the best assumption and document it",
            ]

        else:
            prompt_parts = [
                "IDENTITY: You are FlowAgent, an autonomous execution unit capable of file operations, code execution, and data processing.",
                "",
                "OPERATING PROTOCOL:",
                "1. INITIATIVE: Do not complain about missing tools. If a task requires file access, USE `vfs_list` or `vfs_read`. If you need to search, USE the search tools.",
                "2. FORMAT: When asked for data, output ONLY data (JSON/Markdown). Do not use conversational filler ('Here is the data').",
                "3. HONESTY: Differentiate between 'Information missing in context' (Unknown) and 'Factually non-existent' (False). Never apologize.",
                "4. ITERATION: If a step fails, analyze the error in `think()`, then try a different approach. Do not give up immediately.",
                "",
                "CAPABILITIES:",
                f"- Loaded Tools: ({len(ctx.dynamic_tools)}/{ctx.max_dynamic_tools}): [{active_str}]",
                f"- Context Access: {cat_list}",
                "",
                "MANDATORY WORKFLOW:",
                "A. PLAN: Use `think()` to decompose the request.",
                "B. ACT: Use tools (`load_tools`, `vfs_*`, etc.) to gather info or execute changes.",
                "C. VERIFY: Check if the tool output matches expectations.",
                "D. REPORT: Use `final_answer()` only when the objective is met or definitively impossible.",
            ]

        # Add skills section if matched
        if ctx.matched_skills:
            skill_section = self.skills_manager.build_skill_prompt_section(
                ctx.matched_skills
            )
            prompt_parts.append(skill_section)

        # Add VFS context if available
        try:
            vfs_context = session.build_vfs_context()
            if vfs_context and len(vfs_context) > 10:
                prompt_parts.append("")
                prompt_parts.append(vfs_context)
        except:
            pass

        return "\n".join(prompt_parts)

    def _inject_auto_focus(self, ctx: ExecutionContext) -> List[dict]:
        """Inject AutoFocus message before last user query"""
        messages = ctx.working_history.copy()

        focus_msg = ctx.auto_focus.get_focus_message()
        if focus_msg:
            # Find last user message and insert before it
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages.insert(i, focus_msg)
                    break

        return messages

    # =========================================================================
    # RUN COMPLETION
    # =========================================================================

    def _handle_max_iterations(self, ctx: ExecutionContext, query: str) -> str:
        """Generate graceful, honest response when max iterations reached"""

        # Summarize what was done
        summary = HistoryCompressor.compress_to_summary(ctx.working_history, ctx.run_id)
        summary_text = summary["content"] if summary else "Keine Aktionen durchgeführt."

        return f"""Ich konnte die Aufgabe leider nicht vollständig abschließen.

{summary_text}

**Warum?**
Die Aufgabe war möglicherweise zu komplex oder ich bin in einer Schleife gelandet.

**Mögliche nächste Schritte:**
1. Die Aufgabe in kleinere Teile aufteilen
2. Mir mehr Kontext oder Details geben
3. Eine spezifischere Frage stellen

Ich bin ehrlich mit dir: Ich weiß nicht genau, was schief gelaufen ist.
Wenn du mir mehr Informationen gibst, versuche ich es gerne erneut.

*Ursprüngliche Anfrage: {query[:100]}{"..." if len(query) > 100 else ""}*"""

    async def _commit_run(
        self,
        ctx: ExecutionContext,
        session,
        query: str,
        final_response: str,
        success: bool,
    ):
        """
        Compress working history and commit to permanent storage.

        Order in permanent history: User → Summary → Assistant
        Summary is stored in RAG for long-term retrieval.
        """

        # 1. Create summary from working history (TRIGGER 1: final_answer)
        summary = HistoryCompressor.compress_to_summary(ctx.working_history, ctx.run_id)

        # 2. Add to permanent history in correct order
        # User message
        await session.add_message({"role": "user", "content": query})

        # Summary (stored in RAG too)
        if summary:
            await session.add_message(
                summary,
                direct=True,
                type="action_summary",
                run_id=ctx.run_id,
                success=success,
            )

        # Final response
        await session.add_message({"role": "assistant", "content": final_response})

        print(f"  💾 Run {ctx.run_id} committed to permanent history")

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> dict:
        """Get engine statistics"""
        return {
            "skills_stats": self.skills_manager.get_stats(),
            "human_online": self.human_online,
            "agent_name": self.agent.amd.name,
            "active_executions": len(self._active_executions),
        }

    # =========================================================================
    # PAUSE / CANCEL / LIST EXECUTIONS
    # =========================================================================

    async def pause(self, execution_id: str) -> ExecutionContext | None:
        """
        Pause a running execution.

        Args:
            execution_id: The run_id of the execution to pause

        Returns:
            ExecutionContext if found and paused, None otherwise
        """
        ctx = self._active_executions.get(execution_id)
        if ctx is None:
            return None

        ctx.status = "paused"
        print(f"⏸️ Execution [{execution_id}] paused at iteration {ctx.current_iteration}")
        return ctx

    async def cancel(self, execution_id: str) -> bool:
        """
        Cancel an execution and clean up.

        Args:
            execution_id: The run_id of the execution to cancel

        Returns:
            True if cancelled, False if not found
        """
        ctx = self._active_executions.pop(execution_id, None)
        if ctx is None:
            return False

        ctx.status = "cancelled"
        print(f"❌ Execution [{execution_id}] cancelled")

        # Clean up sub-agents if any
        if self._sub_agent_manager:
            for sub_id in list(self._sub_agent_manager._sub_agents.keys()):
                try:
                    state = self._sub_agent_manager._sub_agents[sub_id]
                    if state._task and not state._task.done():
                        state._task.cancel()
                except Exception:
                    pass

        return True

    def list_executions(self) -> list[dict]:
        """
        List all active/paused executions.

        Returns:
            List of execution summaries
        """
        executions = []
        for run_id, ctx in self._active_executions.items():
            executions.append(
                {
                    "run_id": run_id,
                    "session_id": ctx.session_id,
                    "query": ctx.query[:50] + "..." if len(ctx.query) > 50 else ctx.query,
                    "status": ctx.status,
                    "iteration": ctx.current_iteration,
                    "tools_used": len(ctx.tools_used),
                    "dynamic_tools": ctx.get_dynamic_tool_names(),
                }
            )
        return executions

    def get_execution(self, execution_id: str) -> ExecutionContext | None:
        """
        Get an execution by ID.

        Args:
            execution_id: The run_id

        Returns:
            ExecutionContext or None
        """
        return self._active_executions.get(execution_id)

    async def resume(self, execution_id: str, max_iterations: int = 15) -> str:
        """
        Resume a paused execution.

        Args:
            execution_id: The run_id of the execution to resume
            max_iterations: Max additional iterations

        Returns:
            Final response string
        """
        ctx = self._active_executions.get(execution_id)
        if ctx is None:
            return f"Error: Execution {execution_id} not found"

        if ctx.status != "paused":
            return f"Error: Execution {execution_id} is not paused (status: {ctx.status})"

        ctx.status = "running"

        # Resume execution
        return await self.execute(
            query=ctx.query,
            session_id=ctx.session_id,
            max_iterations=max_iterations,
            ctx=ctx,
        )

    async def execute_stream(
        self,
        query: str,
        session_id: str,
        max_iterations: int = 15,
        ctx: "ExecutionContext | None" = None,
    ) -> tuple[Callable, ExecutionContext]:
        """
        Initialize execution and return stream generator + context.

        Returns:
            tuple[stream_generator_func, ExecutionContext]
        """
        session = await self.agent.session_manager.get_or_create(session_id)

        is_resume = ctx is not None
        if ctx is None:
            ctx = ExecutionContext()

        self._active_executions[ctx.run_id] = ctx
        ctx.session_id = session_id
        ctx.query = query

        # Initialize SubAgentManager
        if not self.is_sub_agent:
            self._sub_agent_manager = SubAgentManager(
                parent_engine=self, parent_session=session, is_sub_agent=False
            )
        else:
            if self.sub_agent_output_dir:
                session.vfs = RestrictedVFSWrapper(session.vfs, self.sub_agent_output_dir)
            max_iterations = min(max_iterations, 10)

        self._current_session = session

        if not is_resume:
            try:
                ctx.matched_skills = await self.skills_manager.match_skills_async(
                    query, max_results=2
                )
            except:
                ctx.matched_skills = self.skills_manager.match_skills(
                    query, max_results=2
                )

            self._calculate_tool_relevance(ctx, query)
            self._preload_skill_tools(ctx, query)

            system_prompt = self._build_system_prompt(ctx, session)
            history_depth = 2 if self.is_sub_agent else 6
            permanent_history = session.get_history_for_llm(last_n=history_depth)

            ctx.working_history = [
                {"role": "system", "content": system_prompt},
                *permanent_history,
                {"role": "user", "content": query},
            ]

        # Return the generator function and context
        async def stream_generator(ctx: ExecutionContext):
            """Generator that yields chunks during execution"""
            nonlocal session, max_iterations

            final_response = None
            success = True

            while ctx.current_iteration < max_iterations:
                ctx.current_iteration += 1

                # Check pause
                if ctx.status == "paused":
                    yield {"type": "paused", "run_id": ctx.run_id}
                    return

                # Build messages
                messages = list(ctx.working_history)
                focus_msg = ctx.auto_focus.get_focus_message()
                if focus_msg:
                    messages.insert(1, focus_msg)

                # Get tool definitions
                tool_definitions = self._get_tool_definitions(ctx)

                # Stream LLM response
                collected_content = ""
                tool_calls_buffer = {}

                stream_response = await self.agent.a_run_llm_completion(
                    messages=messages,
                    tools=tool_definitions if tool_definitions else None,
                    stream=True,
                    true_stream=True,
                )

                if asyncio.iscoroutine(stream_response):
                    stream_response = await stream_response

                async for chunk in stream_response:
                    delta = chunk.choices[0].delta if hasattr(chunk, "choices") and chunk.choices else None
                    # 1. Content sammeln
                    if delta and hasattr(delta, "content") and delta.content:
                        collected_content += delta.content
                        yield {"type": "content", "chunk": delta.content}

                    # 2. Reasoning sammeln (falls vorhanden)
                    if delta and hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        yield {"type": "reasoning", "chunk": delta.reasoning_content}

                    # 3. Tool Calls SAMMELN (nicht überschreiben!)
                    if delta and hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc_chunk in delta.tool_calls:
                            idx = tc_chunk.index

                            # Neuen Eintrag anlegen, falls Index noch nicht existiert
                            if idx not in tool_calls_buffer:
                                tool_calls_buffer[idx] = ChatCompletionMessageToolCall(
                                        id=tc_chunk.id,
                                        type="function",
                                        function=Function(name=tc_chunk.function.name or "", arguments=tc_chunk.function.arguments or "")
                                    )
                            else:
                                # Bestehenden Eintrag erweitern (Name und Argumente anhängen)
                                if tc_chunk.function.name:
                                    tool_calls_buffer[idx]["function"]["name"] += tc_chunk.function.name
                                if tc_chunk.function.arguments:
                                    tool_calls_buffer[idx]["function"]["arguments"] += tc_chunk.function.arguments

                # Nach dem Loop: Das Dictionary in eine Liste umwandeln
                tool_calls = list(tool_calls_buffer.values())

                # Process tool calls
                if tool_calls:
                    assistant_msg = {
                        "role": "assistant",
                        "content": collected_content,
                        "tool_calls": tool_calls,
                    }
                    ctx.working_history.append(assistant_msg)

                    for tc in tool_calls:
                        # Achtung: tc ist jetzt ein Dict, kein Objekt mehr, da wir es manuell gebaut haben
                        # Zugriff daher via ['key'] oder .get()
                        func_obj = tc.get("function", {})
                        f_name = func_obj.get("name", "")
                        f_args = func_obj.get("arguments", "{}")

                        yield {"type": "tool_start", "name": f_name}

                        # Check final_answer
                        if f_name == "final_answer":
                            try:
                                args = json.loads(f_args) if isinstance(f_args, str) else f_args
                                final_response = args.get("answer", collected_content)
                            except:
                                final_response = collected_content

                            yield {"type": "final_answer", "answer": final_response}
                            break

                        print(tc)
                        result = await self._execute_tool_call(ctx, tc)

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc.id
                            if hasattr(tc, "id")
                            else tc.get("id", ""),
                            "name": f_name,
                            "content": str(result)[:2000],
                        }
                        ctx.working_history.append(tool_msg)

                        yield {
                            "type": "tool_result",
                            "name": f_name,
                            "result": str(result)[:500],
                        }

                    if final_response:
                        break
                else:
                    # No tool calls - treat content as final
                    if collected_content:
                        final_response = collected_content
                        yield {"type": "final_answer", "answer": final_response}
                        break

            # Handle max iterations
            if final_response is None:
                final_response = self._handle_max_iterations(ctx, query)
                success = False
                yield {"type": "max_iterations", "answer": final_response}

            # Commit
            await self._commit_run(ctx, session, query, final_response, success)

            # Learn
            if success and len(ctx.tools_used) >= 2:
                try:
                    await self.skills_manager.learn_from_run(
                        query=query,
                        tools_used=ctx.tools_used,
                        final_answer=final_response,
                        success=success,
                        llm_completion_func=self.agent.a_run_llm_completion,
                    )
                except:
                    pass

            for skill in ctx.matched_skills:
                self.skills_manager.record_skill_usage(skill.id, success)

            self._active_executions.pop(ctx.run_id, None)

            yield {"type": "done", "success": success, "final_answer": final_response}

        return stream_generator, ctx

