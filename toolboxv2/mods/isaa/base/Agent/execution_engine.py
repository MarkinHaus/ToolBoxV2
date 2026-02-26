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
import logging
import os
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Coroutine

from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Function,
    ModelResponseStream,
)

from toolboxv2 import get_app

# Import Live State Management
from toolboxv2.mods.isaa.base.Agent.agent_live_state import (
    AgentLiveState,
    AgentPhase,
    TokenStream,
    ToolExecution,
)

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
    SubAgentStatus,
)

# Import Resume Extension
from toolboxv2.mods.isaa.base.Agent.sub_agent_resume_extension import (
    RESUME_SUB_AGENT_TOOL,
    SubAgentResumeExtension,
)

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
    {
        "type": "function",
        "function": {
            "name": "shift_focus",
            "description": "Archiviert die aktuelle Working History und setzt den Fokus auf ein neues Ziel. Nutze dies, wenn ein Teilabschnitt erledigt ist, um Kontext-Rauschen zu vermeiden. Alle bisherigen Ergebnisse werden permanent gespeichert.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_of_achievements": {
                        "type": "string",
                        "description": "Eine detaillierte Zusammenfassung dessen, was im letzten Abschnitt erreicht wurde (Pfade, Ergebnisse, Status).",
                    },
                    "next_objective": {
                        "type": "string",
                        "description": "Was ist das unmittelbare nächste Ziel?",
                    },
                },
                "required": ["summary_of_achievements", "next_objective"],
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
class ContextBudgetConfig:
    """Konfiguration für dynamisches Context-Budget-Management."""
    max_context_ratio: float = 0.85       # Wie viel % des Model-Kontexts genutzt werden dürfen (0.7 - 0.95)
    immediate_offload_ratio: float = 0.7  # Ab diesem Anteil am Gesamt-Kontext → sofort offloaden (Szenario C)
    displacement_threshold: float = 0.4   # Max Größe für Displacement-Strategie (Szenario B)
    safety_margin_tokens: int = 500       # Reserve
    heavy_hitter_min_tokens: int = 1000    # Min Größe für Offload-Kandidaten

@dataclass
class ToolSlot:
    """Represents a dynamically loaded tool slot with relevance tracking"""

    name: str
    relevance_score: float
    category: str = "unknown"
    loaded_at: datetime = field(default_factory=datetime.now)

# =========================================================================
# PERSONA ROUTING
# =========================================================================

@dataclass
class PersonaProfile:
    """Runtime persona applied to an execution."""
    name: str = "default"
    prompt_modifier: str = ""           # injected into system prompt
    model_preference: str = "fast"      # "fast" | "complex"
    temperature: float | None = None    # None = use model default
    max_iterations_factor: float = 1.0  # multiplied with base max_iterations
    verification_level: str = "basic"   # "none" | "basic" | "strict"
    source: str = "default"             # "default" | "matched" | "dreamer"

    def apply_max_iterations(self, base: int) -> int:
        return int(base * self.max_iterations_factor)


# Built-in personas — task-type → profile
_BUILTIN_PERSONAS: dict[str, PersonaProfile] = {
    "debugger": PersonaProfile(
        name="methodical_debugger",
        prompt_modifier=(
            "\nPERSONA: Methodical Debugger\n"
            "- Reproduce the error FIRST, then hypothesize\n"
            "- Check logs/stacktraces before guessing\n"
            "- Verify the fix by running the code again\n"
            "- One change at a time, test after each"
        ),
        model_preference="complex",
        temperature=0.1,
        max_iterations_factor=1.4,
        verification_level="strict",
    ),
    "creative": PersonaProfile(
        name="creative_writer",
        prompt_modifier=(
            "\nPERSONA: Creative Writer\n"
            "- Prioritize originality and style over structure\n"
            "- Explore multiple angles before committing\n"
            "- Use metaphors and vivid language"
        ),
        model_preference="complex",
        temperature=0.8,
        max_iterations_factor=0.6,
        verification_level="none",
    ),
    "devops": PersonaProfile(
        name="devops_operator",
        prompt_modifier=(
            "\nPERSONA: DevOps Operator\n"
            "- ALWAYS verify state before AND after changes\n"
            "- Use dry-run/preview when available\n"
            "- Rollback plan for every destructive operation\n"
            "- Log every action with timestamps"
        ),
        model_preference="fast",
        temperature=0.1,
        max_iterations_factor=1.2,
        verification_level="strict",
    ),
    "research": PersonaProfile(
        name="research_explorer",
        prompt_modifier=(
            "\nPERSONA: Research Explorer\n"
            "- Breadth first: survey multiple sources before deep-diving\n"
            "- Cite sources and note confidence levels\n"
            "- Distinguish facts from speculation\n"
            "- Summarize findings incrementally"
        ),
        model_preference="complex",
        temperature=0.4,
        max_iterations_factor=1.3,
        verification_level="basic",
    ),
    "data": PersonaProfile(
        name="data_analyst",
        prompt_modifier=(
            "\nPERSONA: Data Analyst\n"
            "- Start with data shape/schema before analysis\n"
            "- Validate assumptions with sample queries\n"
            "- Present numbers with context (%, deltas, trends)\n"
            "- Visualize when possible"
        ),
        model_preference="fast",
        temperature=0.2,
        max_iterations_factor=1.0,
        verification_level="basic",
    ),
}

# Keyword → persona name mapping for fast classification
_PERSONA_KEYWORDS: dict[str, list[str]] = {
    "debugger": ["debug", "fix", "error", "bug", "traceback", "exception", "stacktrace",
                 "crash", "broken", "fehler", "reparier", "fixen"],
    "creative": ["schreib", "write", "story", "gedicht", "poem", "blog", "creative",
                 "text", "artikel", "essay", "draft"],
    "devops":   ["deploy", "docker", "kubernetes", "nginx", "server", "container",
                 "pipeline", "ci/cd", "ssh", "systemctl", "service"],
    "research": ["research", "recherch", "find out", "herausfinden", "compare",
                 "vergleich", "analyze", "analys", "investigate", "untersu"],
    "data":     ["csv", "dataframe", "pandas", "sql", "chart", "graph", "statistik",
                 "statistics", "tabelle", "xlsx", "daten", "dataset"],
}


class PersonaRouter:
    """Selects the best persona based on query, matched skills, and dreamer insights."""

    def __init__(self, custom_personas: dict[str, PersonaProfile] | None = None):
        self.personas = dict(_BUILTIN_PERSONAS)
        if custom_personas:
            self.personas.update(custom_personas)

    def route(
        self,
        query: str,
        matched_skills: list | None = None,
        dreamer_insights: dict | None = None,
    ) -> PersonaProfile:
        """
        Classify task and return matching persona.

        Priority:
        1. Dreamer insights (if available and confident)
        2. Skill-based classification
        3. Keyword-based classification
        4. Default persona
        """
        query_lower = query.lower()

        # 1. Dreamer insights (from persona._dream_insights)
        if dreamer_insights:
            best_key, best_ratio = None, 0.0
            for intent_key, insight in dreamer_insights.items():
                ratio = insight.get("success_ratio", 0)
                if ratio > best_ratio and ratio > 0.6:
                    # Check if this intent matches our query
                    if any(w in query_lower for w in intent_key.lower().split()[:2]):
                        best_key = intent_key
                        best_ratio = ratio
            if best_key:
                # Map intent to persona via keyword overlap
                persona = self._match_intent_to_persona(best_key)
                if persona:
                    persona.source = "dreamer"
                    return persona

        # 2. Skill-based: use skill tool_groups / names as hints
        if matched_skills:
            for skill in matched_skills:
                name_lower = skill.name.lower()
                for persona_key, keywords in _PERSONA_KEYWORDS.items():
                    if any(kw in name_lower for kw in keywords):
                        p = PersonaProfile(**{
                            k: getattr(self.personas[persona_key], k)
                            for k in PersonaProfile.__dataclass_fields__
                        })
                        p.source = "matched"
                        return p

        # 3. Keyword classification
        scores: dict[str, int] = {}
        for persona_key, keywords in _PERSONA_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[persona_key] = score

        if scores:
            best = max(scores, key=scores.get)
            if scores[best] >= 1:
                p = PersonaProfile(**{
                    k: getattr(self.personas[best], k)
                    for k in PersonaProfile.__dataclass_fields__
                })
                p.source = "matched"
                return p

        # 4. Default
        return PersonaProfile()

    def _match_intent_to_persona(self, intent_key: str) -> PersonaProfile | None:
        intent_lower = intent_key.lower()
        for persona_key, keywords in _PERSONA_KEYWORDS.items():
            if any(kw in intent_lower for kw in keywords[:3]):
                return PersonaProfile(**{
                    k: getattr(self.personas[persona_key], k)
                    for k in PersonaProfile.__dataclass_fields__
                })
        return None

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
    max_dynamic_tools: int = os.getenv("MAX_DYNAMIC_TOOLS", 5)
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
    active_persona: PersonaProfile = field(default_factory=PersonaProfile)
    loop_warning_given: bool = False

    # Context Budget
    context_config: ContextBudgetConfig = field(default_factory=ContextBudgetConfig)
    offload_hashes: Dict[str, str] = field(default_factory=dict)  # content_hash -> vfs_path

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
            "offload_hashes": self.offload_hashes,
            "context_config": {
                "max_context_ratio": self.context_config.max_context_ratio,
                "immediate_offload_ratio": self.context_config.immediate_offload_ratio,
            },
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
        ctx.offload_hashes = data.get("offload_hashes", {})
        if "context_config" in data:
            ctx.context_config = ContextBudgetConfig(**data["context_config"])
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
        Semantic Ledger: Komprimiert History nach Kategorien der Veränderung.
        Trennt Erkenntnisse (Read), Veränderungen (Write), Fehler (Error).
        """
        if not working_history:
            return None

        # Kategorisierte Sammlung
        reads = []  # Erkenntnisse (Read-Only)
        writes = []  # State Changes (Write/Create/Edit)
        errors = []  # Fehler & Blocker
        findings = []  # Aus think-Blöcken extrahierte Erkenntnisse
        tools_used = set()

        for i, msg in enumerate(working_history):
            role = msg.get("role", "")

            if role == "assistant":
                # Think-Ergebnisse aus dem nächsten tool-result extrahieren
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    if hasattr(tc, "function"):
                        name = tc.function.name
                        tools_used.add(name)
                        if name == "think":
                            try:
                                args = json.loads(tc.function.arguments)
                                thought = args.get("thought", "")[:200]
                                if thought:
                                    findings.append(thought)
                            except:
                                pass
                    elif isinstance(tc, dict) and "function" in tc:
                        name = tc["function"].get("name", "")
                        tools_used.add(name)
                        if name == "think":
                            try:
                                args = json.loads(tc["function"].get("arguments", "{}"))
                                thought = args.get("thought", "")[:200]
                                if thought:
                                    findings.append(thought)
                            except:
                                pass

            elif role == "tool":
                name = msg.get("name", "")
                content = msg.get("content", "")
                tools_used.add(name)
                content_lower = content.lower()
                name_lower = name.lower()

                # Fehler-Erkennung
                if "error" in content_lower or "failed" in content_lower or "traceback" in content_lower:
                    errors.append(f"{name}: {content[:120]}")
                # Write-Ops
                elif any(kw in name_lower for kw in ("write", "create", "edit", "append", "delete", "mv")):
                    # Behalte den Pfad/Status, nicht den Inhalt
                    first_line = content.split("\n")[0][:100] if content else "ok"
                    writes.append(f"{name} → {first_line}")
                # Read-Ops (inkl. offloaded)
                elif any(
                    kw in name_lower for kw in ("read", "list", "navigate", "search", "query", "view", "open", "grep")):
                    if "[DATA OFFLOADED" in content:
                        reads.append(f"{name}: [offloaded]")
                    else:
                        size = len(content)
                        reads.append(f"{name} ({size} chars)")
                elif name == "think":
                    pass  # Schon oben verarbeitet
                else:
                    reads.append(f"{name}")

        # Build Semantic Ledger
        lines = [f"📋 SEMANTIC LEDGER [Run {run_id}]:"]

        if findings:
            lines.append(f"\n🔍 KEY FINDINGS ({len(findings)}):")
            for f in findings[-3:]:  # Letzte 3 Erkenntnisse
                lines.append(f"  • {f}")

        if writes:
            lines.append(f"\n✏️ STATE CHANGES ({len(writes)}):")
            for w in writes:
                lines.append(f"  • {w}")

        if reads:
            lines.append(f"\n📖 INFORMATION GATHERED: {len(reads)} operations")
            # Nur zusammenfassen, nicht jeden einzelnen auflisten
            unique_tools = list(set(r.split("(")[0].split(":")[0].strip() for r in reads))
            lines.append(f"  Tools: {', '.join(unique_tools[:5])}")

        if errors:
            lines.append(f"\n⚠️ BLOCKERS ({len(errors)}):")
            for e in errors[:3]:
                lines.append(f"  • {e}")

        meaningful = tools_used - {"think", "final_answer", "list_tools", "load_tools", "shift_focus"}
        lines.append(f"\n🔧 Total: {len(tools_used)} tool calls, {len(meaningful)} unique tools")

        return {
            "role": "system",
            "content": "\n".join(lines),
            "metadata": {"type": "semantic_ledger", "run_id": run_id},
        }

    @staticmethod
    def compress_partial(
        working_history: List[dict], keep_last_n: int = 3
    ) -> Tuple[Optional[dict], List[dict]]:
        """
        Relevance Filter: Priorisierte Verdichtung der History.
        P1 (Kritisch): User Messages, final_answer, shift_focus, Errors → BEHALTEN
        P2 (Wichtig): think, write (nur Pfad), exec (nur Status) → 1 ZEILE
        P3 (Flüchtig): read content, list output, cat → LÖSCHEN/POINTER
        """
        if len(working_history) <= keep_last_n + 1:
            return None, working_history

        system_msg = (
            working_history[0] if working_history[0].get("role") == "system" else None
        )

        if system_msg:
            to_process = working_history[1:-keep_last_n]
            to_keep = working_history[-keep_last_n:]
        else:
            to_process = working_history[:-keep_last_n]
            to_keep = working_history[-keep_last_n:]

        compressed_msgs = []
        stats = {"kept": 0, "summarized": 0, "dropped": 0}

        for msg in to_process:
            role = msg.get("role", "")
            name = msg.get("name", "")
            content = msg.get("content", "") or ""
            name_lower = name.lower()

            # P1: BEHALTEN (User, Errors, shift_focus)
            if role == "user":
                compressed_msgs.append(msg)
                stats["kept"] += 1

            elif role == "system" and "error" in content.lower():
                compressed_msgs.append(msg)
                stats["kept"] += 1

            elif role == "assistant":
                # Behalte assistant messages aber kürze content
                new_msg = {"role": "assistant"}
                if "tool_calls" in msg:
                    new_msg["tool_calls"] = msg["tool_calls"]
                    # Content kürzen
                    new_msg["content"] = (content[:80] + "...") if len(content) > 80 else content
                else:
                    new_msg["content"] = (content[:160] + "...") if len(content) > 160 else content
                compressed_msgs.append(new_msg)
                stats["summarized"] += 1

            elif role == "tool":
                # P2: ZUSAMMENFASSEN (think, write, exec)
                if name == "think":
                    compressed_msgs.append({
                        "role": "tool", "tool_call_id": msg.get("tool_call_id", ""),
                        "name": name,
                        "content": f"Think: {content[:100]}..." if len(content) > 100 else content,
                    })
                    stats["summarized"] += 1

                elif any(kw in name_lower for kw in ("write", "create", "edit", "exec", "shell", "run")):
                    first_line = content.split("\n")[0][:80]
                    compressed_msgs.append({
                        "role": "tool", "tool_call_id": msg.get("tool_call_id", ""),
                        "name": name,
                        "content": first_line,
                    })
                    stats["summarized"] += 1

                elif name in ("shift_focus", "final_answer"):
                    compressed_msgs.append(msg)
                    stats["kept"] += 1

                # P3: LÖSCHEN (read, list, navigate, search outputs)
                elif any(
                    kw in name_lower for kw in ("read", "list", "navigate", "search", "query", "view", "open", "cat")):
                    compressed_msgs.append({
                        "role": "tool", "tool_call_id": msg.get("tool_call_id", ""),
                        "name": name,
                        "content": f"[Viewed: {name}]",
                    })
                    stats["dropped"] += 1

                else:
                    # Unbekanntes Tool: Zusammenfassen
                    compressed_msgs.append({
                        "role": "tool", "tool_call_id": msg.get("tool_call_id", ""),
                        "name": name,
                        "content": content[:80] if content else "ok",
                    })
                    stats["summarized"] += 1
            else:
                # system messages etc. → behalten wenn kurz
                if len(content) < 200:
                    compressed_msgs.append(msg)
                stats["summarized"] += 1

        summary = {
            "role": "system",
            "content": (
                f"[CONTEXT COMPRESSED: {stats['kept']} kept, "
                f"{stats['summarized']} summarized, {stats['dropped']} dropped]"
            ),
        }

        if system_msg:
            return summary, [system_msg, summary] + compressed_msgs + to_keep
        else:
            return summary, [summary] + compressed_msgs + to_keep


# =============================================================================
# EXECUTION ENGINE V3
# =============================================================================


class ExecutionEngine(SubAgentResumeExtension):
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
    - TRIGGER 3: shift_focus → Compress working history to summary and archive
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
        self.live = AgentLiveState(
            agent_name=getattr(agent, "amd", None) and agent.amd.name or "?",
            is_sub=is_sub_agent,
        )

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
        if (
            hasattr(agent.session_manager, "skills_manager")
            and agent.session_manager.skills_manager
        ):
            self.skills_manager = agent.session_manager.skills_manager
        else:
            self.skills_manager = SkillsManager(
                agent_name=agent.amd.name, memory_instance=self._get_memory_instance()
            )
            # Store back on agent
            agent.session_manager.skills_manager = self.skills_manager
        self._persona_router = PersonaRouter()
        # Add parallel_subtasks skill if not present
        if "parallel_subtasks" not in self.skills_manager.skills:
            from toolboxv2.mods.isaa.base.Agent.skills import Skill

            self.skills_manager.skills["parallel_subtasks"] = Skill(
                **PARALLEL_SUBTASKS_SKILL
            )

        # Add job_management skill if not present (LÖSUNG 3: Agent-Verhalten verbessern)
        # Verwende das korrekte Dict-Format wie PARALLEL_SUBTASKS_SKILL
        if "job_management" not in self.skills_manager.skills:
            JOB_MANAGEMENT_SKILL = {
                "id": "job_management",
                "name": "Job Management Best Practices",
                "triggers": [
                    "create job", "scheduled job", "cron job", "interval job",
                    "job erstellen", "geplanter job", "automatisierung", "schedule",
                    "timer", "periodisch", "wöchentlich", "täglich", "stündlich"
                ],
                "instruction": """FÜR GEPLANTE JOBS (SCHEDULED TASKS):

1. JOB ERSTELLEN mit createJob():
   - name: Klare Bezeichnung (z.B. "daily-backup")
   - trigger_type: "on_cron", "on_interval", "on_time", etc.
   - Trigger-Parameter DIREKT übergeben (nicht in trigger_config):
     * cron_expression="0 2 * * 0" (für cron)
     * interval_seconds=300 (für interval)
     * at_datetime="2025-01-01T10:00:00Z" (für einmalig)
   - agent_name: "self" oder registrierter Agent
   - query: Die Aufgabe/Auftrag für den Agent

2. VERIFIZIERUNG mit listJobs():
   - IMMER nach createJob() listJobs() aufrufen!
   - Prüfen dass der Job in der Liste erscheint
   - Parameter überprüfen

3. FEHLERBEHANDLUNG:
   - Wenn createJob() mit "✗" antwortet: Fehler melden
   - NICHT erfolgreich behaupten ohne listJobs() Bestätigung
   - Fehlermeldung lesen und Parameter korrigieren

BEISPIELE:
   - Wöchentlich Sonntag 02:00:
     createJob(name="weekly-update", trigger_type="on_cron", cron_expression="0 2 * * 0", agent_name="self", query="run updates")
   - Alle 5 Minuten:
     createJob(name="heartbeat", trigger_type="on_interval", interval_seconds=300, agent_name="self", query="ping server")""",
                "tools_used": ["createJob", "listJobs", "deleteJob", "think", "final_answer"],
                "tool_groups": ["job_management"],
                "source": "predefined"
            }

            from toolboxv2.mods.isaa.base.Agent.skills import Skill
            self.skills_manager.skills["job_management"] = Skill(**JOB_MANAGEMENT_SKILL)

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
                self.live.log(f"Auto-created {len(groups)} tool groups")
        except Exception as e:
            self.live.log(f"Failed to auto-group tools: {e}", logging.WARNING)

    # =========================================================================
    # MAIN EXECUTION LOOP
    # =========================================================================

    async def execute(
        self,
        query: str,
        session_id: str,
        max_iterations: int = 25,
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
        # ── Dream intercept ──
        if query == "__dream__":
            from toolboxv2.mods.isaa.base.Agent.dreamer import DreamConfig
            cfg = DreamConfig()  # or parse from job.trigger.extra
            report = await self.agent.a_dream(cfg)
            return report.model_dump_json() if get_ctx else str(report)
        session = await self.agent.session_manager.get_or_create(session_id)

        # Use existing context or create new one
        is_resume = ctx is not None
        if ctx is None:
            ctx = ExecutionContext()

        if hasattr(self.agent, 'amd') and hasattr(self.agent.amd, 'context_config'):
            ctx.context_config = self.agent.amd.context_config

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
            max_iterations = min(max_iterations, 15)

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

            # 3b. Route persona based on query + skills + dreamer insights
            dreamer_insights = getattr(
                      getattr(self.agent.amd, 'persona', None), '_dream_insights', None)
            ctx.active_persona = self._persona_router.route(
                      query, ctx.matched_skills, dreamer_insights)

            # Apply persona overrides

            if ctx.active_persona.name != "default":
                max_iterations = ctx.active_persona.apply_max_iterations(max_iterations)
                self.live.log(
                    f"Persona: {ctx.active_persona.name} "
                    f"(model={ctx.active_persona.model_preference}, "
                    f"temp={ctx.active_persona.temperature}, "
                    f"max_iter={max_iterations})"
                )

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
        self.live.run_id = ctx.run_id
        self.live.max_iterations = max_iterations
        self.live.t_start = time.time()
        self.live.skills = [s.name for s in ctx.matched_skills] if ctx.matched_skills else []
        self.live.tools_loaded = ctx.get_dynamic_tool_names() if ctx.dynamic_tools else []
        self.live.persona = ctx.active_persona.name
        self.live.enter(AgentPhase.INIT, f"{action} [{agent_type}] {ctx.run_id}: {query[:80]}")

        final_response = None
        success = True

        # 5. Main loop
        while ctx.current_iteration < max_iterations:
            ctx.current_iteration += 1

            self.live.iteration = ctx.current_iteration
            self.live.enter(AgentPhase.LLM_CALL, f"iter {ctx.current_iteration}/{max_iterations}")

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
                llm_kwargs = {
                    "model_preference": ctx.active_persona.model_preference,
                    "stream": False,
                    "get_response_message": True,
                    "with_context": False,
                    }
                if ctx.active_persona.temperature is not None:
                    llm_kwargs["temperature"] = ctx.active_persona.temperature
                response = await self.agent.a_run_llm_completion(
                    messages=messages,
                    tools=current_tools,
                    tool_choice="auto",
                    **llm_kwargs,
                )
            except Exception as e:
                self.live.error = str(e)
                self.live.enter(AgentPhase.ERROR, f"LLM Error: {e}")
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
                    result, is_final = await self._execute_tool_call(ctx, tool_call)

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
            if self.is_sub_agent:
                # Sub-agent: Special handling mit Resume-Option
                (
                    final_response,
                    should_mark_resumable,
                ) = await self._handle_sub_agent_max_iterations(
                    ctx, query, max_iterations
                )
                success = False

                # Mark as resumable if progress was made
                if should_mark_resumable:
                    ctx.status = "max_iterations"  # Instead of "completed"
            else:
                # Main agent: Existing handling
                final_response = self._handle_max_iterations(ctx, query)
                success = False

        # 7. Compress and commit to permanent history
        await self._commit_run(ctx, session, query, final_response, success)

        # 8. Learn from successful runs
        if success:
            app = get_app()
            app.run_bg_task_advanced(
                self._background_learning_task,
                query=query,
                tools_used=ctx.tools_used,
                final_response=final_response,
                success=success,
                matched_skills=ctx.matched_skills,
            )

        # Remove from active executions
        self._active_executions.pop(ctx.run_id, None)

        self.live.status_msg = f"done (ok={success}, iters={ctx.current_iteration})"
        self.live.enter(AgentPhase.DONE, f"Execution [{ctx.run_id}] complete (success={success}, iters={ctx.current_iteration})")

        if get_ctx:
            return final_response, ctx
        return final_response


    async def execute_stream(
        self,
        query: str,
        session_id: str,
        max_iterations: int = 25,
        ctx: "ExecutionContext | None" = None,
        model=None,
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

            dreamer_insights = getattr(
                getattr(self.agent.amd, 'persona', None), '_dream_insights', None)
            ctx.active_persona = self._persona_router.route(
                query, ctx.matched_skills, dreamer_insights)

            # Apply persona overrides

            if ctx.active_persona.name != "default":
                max_iterations = ctx.active_persona.apply_max_iterations(max_iterations)
                self.live.log(
                    f"Persona: {ctx.active_persona.name} "
                    f"(model={ctx.active_persona.model_preference}, "
                    f"temp={ctx.active_persona.temperature}, "
                    f"max_iter={max_iterations})"
                )

            system_prompt = self._build_system_prompt(ctx, session)
            history_depth = 2 if self.is_sub_agent else 6
            permanent_history = session.get_history_for_llm(last_n=history_depth)

            ctx.working_history = [
                {"role": "system", "content": system_prompt},
                *permanent_history,
                {"role": "user", "content": query},
            ]

        # Init live state for stream path
        self.live.run_id = ctx.run_id
        self.live.max_iterations = max_iterations
        self.live.t_start = time.time()
        self.live.skills = [s.name for s in ctx.matched_skills] if ctx.matched_skills else []
        self.live.tools_loaded = ctx.get_dynamic_tool_names() if ctx.dynamic_tools else []
        self.live.persona = ctx.active_persona.name
        self.live.enter(AgentPhase.INIT, f"{'Resume' if is_resume else 'Start'} stream [{ctx.run_id}]")

        async def stream_generator(ctx: ExecutionContext):
            """Generator that yields chunks during execution"""
            nonlocal session, max_iterations

            final_response = None
            success = True

            # Context info for ZEN CLI
            agent_name = self.agent.amd.name
            # Determine depth/type (simple heuristic or passed param)
            is_sub = self.is_sub_agent

            # Helper to enrich chunks
            def enrich(chunk):
                chunk["agent"] = agent_name
                chunk["iter"] = ctx.current_iteration
                chunk["max_iter"] = max_iterations
                chunk["is_sub"] = is_sub
                # Token budget info
                chunk["tokens_used"] = self._calculate_context_load(ctx)
                chunk["tokens_max"] = self._get_max_context_tokens()
                # Skills & Persona
                chunk["skills"] = [s.name for s in ctx.matched_skills] if ctx.matched_skills else []
                chunk["persona"] = ctx.active_persona.name if ctx.active_persona else "default"
                chunk["persona_source"] = ctx.active_persona.source if ctx.active_persona else "default"
                chunk["persona_model"] = ctx.active_persona.model_preference if ctx.active_persona else "fast"
                chunk["persona_iterations_factor"] = ctx.active_persona.max_iterations_factor
                return chunk

            while ctx.current_iteration < max_iterations:
                ctx.current_iteration += 1
                self.live.iteration = ctx.current_iteration
                self.live.enter(AgentPhase.LLM_CALL, f"iter {ctx.current_iteration}/{max_iterations}")

                # Check pause
                if ctx.status == "paused":
                    yield enrich({"type": "paused", "run_id": ctx.run_id})
                    return

                # Check cancellation
                if ctx.status == "cancelled":
                    yield enrich({"type": "cancelled", "run_id": ctx.run_id})
                    return

                if self._should_warn_loop(ctx):
                    warning_msg = ctx.loop_detector.get_intervention_message()
                    # Inject into history so LLM sees it
                    ctx.working_history.append({"role": "system", "content": warning_msg})
                    ctx.loop_warning_given = True
                    # Optional: Inform UI about the warning
                    yield enrich(
                        {"type": "warning", "message": warning_msg.splitlines()[0]}
                    )

                # Build messages
                messages = list(ctx.working_history)
                focus_msg = ctx.auto_focus.get_focus_message()
                if focus_msg:
                    messages.insert(1, focus_msg)

                # Get tool definitions
                tool_definitions = self._get_tool_definitions(ctx)

                # Stream LLM response state
                collected_content = ""
                tool_calls_buffer = {}

                # --- AUTO-RESUME SCHLEIFE FÜR STREAMING ---
                MAX_CONTINUATIONS = 100
                continuation_count = 0

                current_messages = messages.copy()
                current_tools = tool_definitions if tool_definitions else None

                continuing_tool_idx = (
                    None  # Verfolgt, welches Tool gerade durch Text fortgesetzt wird
                )

                while continuation_count < MAX_CONTINUATIONS:
                    stream_kwargs = {"stream": True, "true_stream": True}
                    if model:
                        stream_kwargs["model"] = model
                    else:
                        stream_kwargs["model_preference"] = ctx.active_persona.model_preference
                    if ctx.active_persona.temperature is not None:
                        stream_kwargs["temperature"] = ctx.active_persona.temperature
                    stream_response = None
                    try:
                        stream_response = await self.agent.a_run_llm_completion(
                            messages=current_messages,
                            tools=current_tools,
                            **stream_kwargs,
                        )

                        if asyncio.iscoroutine(stream_response):
                            stream_response = await stream_response
                    except Exception as e:
                        err_msg = str(e)
                        if "Event loop is closed" in err_msg:
                            yield enrich({
                                "type": "error",
                                "error": f"LLM stream failed: {err_msg[:200]}"
                            })
                            break
                        raise

                    finish_reason = None

                    try:
                        async for chunk in stream_response:
                            delta = (
                                chunk.choices[0].delta
                                if hasattr(chunk, "choices") and chunk.choices
                                else None
                            )

                            if not delta:
                                continue

                            if (
                                hasattr(chunk.choices[0], "finish_reason")
                                and chunk.choices[0].finish_reason
                            ):
                                finish_reason = chunk.choices[0].finish_reason

                            # 1. Content sammeln (oder abgebrochenes Tool fortsetzen)
                            if delta and hasattr(delta, "content") and delta.content:
                                if continuing_tool_idx is not None:
                                    # Das LLM spuckt den Rest des JSON-Strings als reinen Text aus -> ins Tool umleiten!
                                    tool_calls_buffer[continuing_tool_idx]["function"][
                                        "arguments"
                                    ] += delta.content
                                else:
                                    collected_content += delta.content
                                    yield enrich({"type": "content", "chunk": delta.content})

                            # 2. Reasoning sammeln (falls vorhanden)
                            if (
                                delta
                                and hasattr(delta, "reasoning_content")
                                and delta.reasoning_content
                            ):
                                yield enrich(
                                    {"type": "reasoning", "chunk": delta.reasoning_content}
                                )

                            # 3. Tool Calls SAMMELN
                            if delta and hasattr(delta, "tool_calls") and delta.tool_calls:
                                for tc_chunk in delta.tool_calls:
                                    idx = tc_chunk.index

                                    # Falls das LLM während eines Resumes trotzdem native Tool_calls nutzt
                                    target_idx = (
                                        continuing_tool_idx
                                        if continuing_tool_idx is not None
                                        else idx
                                    )

                                    # Neuen Eintrag anlegen, falls Index noch nicht existiert
                                    if target_idx not in tool_calls_buffer:
                                        tool_calls_buffer[target_idx] = (
                                            ChatCompletionMessageToolCall(
                                                id=tc_chunk.id,
                                                type="function",
                                                function=Function(
                                                    name=tc_chunk.function.name or "",
                                                    arguments=tc_chunk.function.arguments
                                                    or "",
                                                ),
                                            )
                                        )
                                    else:
                                        # Bestehenden Eintrag erweitern
                                        if tc_chunk.function.name:
                                            tool_calls_buffer[target_idx]["function"][
                                                "name"
                                            ] += tc_chunk.function.name
                                        if tc_chunk.function.arguments:
                                            tool_calls_buffer[target_idx]["function"][
                                                "arguments"
                                            ] += tc_chunk.function.arguments
                    except RuntimeError as e:
                        if "Event loop is closed" in str(e):
                            # Partial result recovery
                            if collected_content:
                                final_response = collected_content
                                yield enrich({"type": "final_answer", "answer": final_response})
                            else:
                                yield enrich({
                                    "type": "error",
                                    "error": f"Stream aborted: {str(e)[:200]}"
                                })
                            break
                        raise
                    # --- Ende des Chunks. Prüfen auf Token Limit ---
                    if finish_reason not in ["length", "max_tokens"]:
                        break  # Generierung natürlich beendet

                    continuation_count += 1
                    self.live.status_msg = f"Auto-Resume ({continuation_count}/{MAX_CONTINUATIONS})"
                    self.live.log(f"Output limit reached. Auto-Resume ({continuation_count}/{MAX_CONTINUATIONS})")

                    # Analysieren, WAS genau abgebrochen ist (Text oder Tool-Call JSON?)
                    is_tool_cut_off = False
                    active_cut_off_tool_idx = None

                    if tool_calls_buffer:
                        last_idx = max(tool_calls_buffer.keys())
                        last_tc = tool_calls_buffer[last_idx]
                        try:
                            json.loads(last_tc["function"]["arguments"])
                            # Valide -> War nicht abgeschnitten
                        except json.JSONDecodeError:
                            is_tool_cut_off = True
                            active_cut_off_tool_idx = last_idx

                    if is_tool_cut_off:
                        # Ein Tool Call wurde mitten im JSON abgeschnitten
                        continuing_tool_idx = active_cut_off_tool_idx
                        last_tc = tool_calls_buffer[continuing_tool_idx]
                        t_name = last_tc["function"]["name"]
                        t_args = last_tc["function"]["arguments"]

                        resume_msg = (
                            f"Du hast das Output-Token-Limit erreicht, während du das Tool '{t_name}' ausgeführt hast. "
                            f"Hier ist der JSON-Argument-String, den du bisher geschrieben hast:\n`{t_args}`\n\n"
                            f"Bitte antworte AUSSCHLIESSLICH mit den fehlenden Zeichen, um das JSON zu vervollständigen. "
                            f"Gib keine Erklärungen und keinen Code-Block ab. Mache genau da weiter, wo der String abgerissen ist."
                        )
                        current_tools = (
                            None  # Zwinge das LLM, als reinen Text zu antworten!
                        )
                        current_messages = messages.copy() + [
                            {
                                "role": "assistant",
                                "content": collected_content
                                + f"\n[Starte Tool: {t_name}]",
                            },
                            {"role": "user", "content": resume_msg},
                        ]
                    else:
                        # Normaler Text wurde abgeschnitten
                        continuing_tool_idx = None
                        last_words = collected_content[-100:]
                        resume_msg = (
                            f"Du hast das maximale Output-Token-Limit deines Modells erreicht. "
                            f"Bitte fahre exakt an dem Punkt fort, an dem du aufgehört hast. "
                            f"Hier sind deine letzten Worte zur Orientierung:\n'...{last_words}'\n\n"
                            f"Setze den Text/Code lückenlos fort. Bitte benutze keine Floskeln."
                        )
                        current_tools = tool_definitions if tool_definitions else None
                        current_messages = messages.copy() + [
                            {"role": "assistant", "content": collected_content},
                            {"role": "user", "content": resume_msg},
                        ]

                # --- NACH DER SCHLEIFE ---
                # Das Dictionary in eine Liste umwandeln

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
                        func_obj = tc.get("function", {})
                        f_name = func_obj.get("name", "")
                        f_args = func_obj.get("arguments", "{}")

                        yield enrich(
                            {"type": "tool_start", "name": f_name, "args": f_args}
                        )

                        # Check final_answer
                        if f_name == "final_answer":
                            try:
                                args = (
                                    json.loads(f_args)
                                    if isinstance(f_args, str)
                                    else f_args
                                )
                                final_response = args.get("answer", collected_content)
                                success_status = args.get("success", True)
                            except:
                                final_response = collected_content
                                success_status = True

                            yield enrich(
                                {
                                    "type": "final_answer",
                                    "answer": final_response,
                                    "success": success_status,
                                }
                            )
                            break

                        # --- Sub-agent tools: forward chunks WHILE executing ---
                        if (
                            f_name in ("spawn_sub_agent", "wait_for", "resume_sub_agent")
                            and self._sub_agent_manager
                        ):
                            tool_task = asyncio.create_task(
                                self._execute_tool_call(ctx, tc)
                            )
                            # Drain sub-agent chunk queue while tool runs
                            while not tool_task.done():
                                try:
                                    sub_chunk = await asyncio.wait_for(
                                        self._sub_agent_manager._chunk_queue.get(),
                                        timeout=0.05,
                                    )
                                    if sub_chunk.get("type") == "_sub_done":
                                        continue
                                    yield sub_chunk
                                except asyncio.TimeoutError:
                                    continue
                                except Exception:
                                    break

                            result, is_final = await tool_task

                            # Drain remaining queued chunks
                            while not self._sub_agent_manager._chunk_queue.empty():
                                try:
                                    sub_chunk = self._sub_agent_manager._chunk_queue.get_nowait()
                                    if sub_chunk.get("type") != "_sub_done":
                                        yield sub_chunk
                                except asyncio.QueueEmpty:
                                    break
                        else:
                            # Normal tools: direct await
                            result, is_final = await self._execute_tool_call(ctx, tc)

                        yield enrich(
                            {
                                "type": "tool_result",
                                "name": f_name,
                                "is_final": is_final,
                                "result": str(result),
                            }
                        )

                    if final_response:
                        break
                else:
                    # No tool calls - treat content as final
                    if collected_content:
                        final_response = collected_content
                        yield enrich({"type": "final_answer", "answer": final_response})
                        break

            # Handle max iterations
            if final_response is None:
                final_response = self._handle_max_iterations(ctx, query)
                success = False
                yield enrich({"type": "max_iterations", "answer": final_response})

                # Commit
            await self._commit_run(ctx, session, query, final_response, success)

            # Learn
            if success:
                app = get_app()
                app.run_bg_task_advanced(
                    self._background_learning_task,
                    query=query,
                    tools_used=ctx.tools_used,
                    final_response=final_response,
                    success=success,
                    matched_skills=ctx.matched_skills,
                )

            self._active_executions.pop(ctx.run_id, None)

            yield enrich(
                {"type": "done", "success": success, "final_answer": final_response}
            )

        return stream_generator, ctx

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

    async def _background_learning_task(
        self,
        query: str,
        tools_used: list,
        final_response: str,
        success: bool,
        matched_skills: list,
    ):
        """Runs learning and recording in background to not block the UI"""
        try:
            # 1. Record usage stats (Fast)
            for skill in matched_skills:
                self.skills_manager.record_skill_usage(skill.id, success)

            # 2. Learn new skills (Slow - involves LLM)
            if success and len(tools_used) >= 2:
                await self.skills_manager.learn_from_run(
                    query=query,
                    tools_used=tools_used,
                    final_answer=final_response,
                    success=success,
                    llm_completion_func=self.agent.a_run_llm_completion,
                )
        except Exception as e:
            self.live.log(f"Background Learning Error: {e}", logging.WARNING)

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

        # Track tool usage
        ctx.tools_used.append(f_name)

        # Loop detection (record before execution)
        loop_detected = ctx.loop_detector.record(f_name, f_args)

        result = ""
        is_final = False
        self.live.tool = ToolExecution(name=f_name, args_summary=str(f_args)[:120], t_start=time.time())
        self.live.enter(AgentPhase.TOOL_EXEC)

        # === STATIC TOOLS ===
        if f_name == "think":
            thought = f_args.get("thought", "")
            result = f"Thought recorded."
            self.live.thought = thought  # visible to renderer
            # Record in AutoFocus
            ctx.auto_focus.record(f_name, f_args, thought[:200])
            ctx.current_iteration -= 1

        elif f_name == "final_answer":
            answer = f_args.get("answer", "")
            success = f_args.get("success", True)
            result = answer
            is_final = True
            # Don't record final_answer in AutoFocus

        # === DISCOVERY TOOLS ===
        elif f_name == "list_tools":
            result = self._tool_list_tools(f_args.get("category"))
            ctx.auto_focus.record(f_name, f_args, result[:200])
            ctx.current_iteration -= 1

        elif f_name == "load_tools":
            tools_input = f_args.get("tools") or f_args.get("names")
            result = await self._tool_load_tools(ctx, tools_input)
            ctx.auto_focus.record(f_name, f_args, result[:200])
            ctx.current_iteration -= 1

        elif f_name == "shift_focus":
            result = await self._tool_shift_focus(
                ctx,
                f_args.get("summary_of_achievements", ""),
                f_args.get("next_objective", ""),
            )
            ctx.current_iteration -= 5

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
                                f"Result: {sub_result.result if sub_result.result else 'result in files'}"
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

            ctx.auto_focus.record(f_name, f_args, result[:400])

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
                            result_lines.append(f"  Error: {sub_result.error[:200]}")

                    result = "\n".join(result_lines)

                    # Inject into AutoFocus
                    focus_text = self._sub_agent_manager.format_results_for_auto_focus(
                        results
                    )
                    ctx.auto_focus.actions.append(focus_text)

                except Exception as e:
                    result = f"ERROR waiting for sub-agents: {str(e)}"

            ctx.auto_focus.record(f_name, f_args, result[:400])

        elif f_name == "resume_sub_agent":
            if self.is_sub_agent:
                result = "ERROR: Sub-agents cannot resume other sub-agents!"
            elif not self._sub_agent_manager:
                result = "ERROR: SubAgentManager not initialized."
            else:
                try:
                    sub_agent_id = f_args.get("sub_agent_id")
                    additional_iterations = f_args.get("additional_iterations", 10)
                    additional_budget = f_args.get("additional_budget", 3000)
                    wait = f_args.get("wait", True)
                    context = f_args.get("context")  # NEW: Optional additional context

                    result = await self._tool_resume_sub_agent(
                        sub_agent_id=sub_agent_id,
                        additional_iterations=additional_iterations,
                        additional_budget=additional_budget,
                        wait=wait,
                        context=context,  # Pass context parameter
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    result = f"ERROR resuming sub-agent: {str(e)}"

            ctx.auto_focus.record(f_name, f_args, result[:400])

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
            # Context Budget Management: Szenario A/B/C
            managed_msg = self._manage_context_budget(
                ctx, f_name, str(result), f_id
            )
            ctx.working_history.append(managed_msg)

        return result, is_final

    # =========================================================================
    # CONTEXT BUDGET MANAGEMENT
    # =========================================================================

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Schnelle Token-Schätzung (~4 chars/token)."""
        if not text:
            return 0
        return len(text) // 4

    def _get_max_context_tokens(self) -> int:
        """Hole max context window des aktuellen Models."""
        try:
            import litellm
            model = getattr(self.agent, 'amd', None)
            if model and hasattr(model, 'model_name'):
                info = litellm.get_model_info(model.model_name)
                return info.get("max_input_tokens", 128000)
        except Exception:
            pass
        return 128000  # Fallback

    def _calculate_context_load(self, ctx: ExecutionContext) -> int:
        """Berechne aktuelle Context-Größe in Tokens."""
        total = 0
        for msg in ctx.working_history:
            content = msg.get("content", "") or ""
            total += self._estimate_tokens(content)
            # Tool calls in assistant messages
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict):
                    total += self._estimate_tokens(tc.get("function", {}).get("arguments", ""))
                elif hasattr(tc, "function"):
                    total += self._estimate_tokens(tc.function.arguments or "")
        return total

    def _content_hash(self, content: str) -> str:
        """Erzeuge Hash für Dedup."""
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]

    def _manage_context_budget(
        self, ctx: ExecutionContext, tool_name: str, raw_result: str, tool_call_id: str
    ) -> dict:
        """
        Zentrale Budget-Verwaltung. Gibt die finale tool-message zurück
        die in working_history eingefügt werden soll.

        Implementiert Szenario A/B/C aus der Dynamic Displacement Strategy.
        """
        session = self._current_session
        cfg = ctx.context_config
        max_tokens = self._get_max_context_tokens()
        budget_limit = int(max_tokens * cfg.max_context_ratio)

        new_result_tokens = self._estimate_tokens(raw_result)
        current_load = self._calculate_context_load(ctx)
        remaining = budget_limit - current_load - cfg.safety_margin_tokens

        # --- Hash-Dedup Check ---
        content_hash = self._content_hash(raw_result)
        if content_hash in ctx.offload_hashes:
            existing_path = ctx.offload_hashes[content_hash]
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": f"[DUPLICATE: Tool result already offloaded to {existing_path}. Use vfs_read to access if needed.]",
            }

        # --- Szenario C: Sofort-Offload (Result > immediate_offload_ratio des Gesamtkontexts) ---
        if new_result_tokens > int(max_tokens * cfg.immediate_offload_ratio):
            return self._offload_immediate(ctx, session, tool_name, raw_result, tool_call_id, content_hash)

        # --- Szenario A: Happy Path (passt rein) ---
        if new_result_tokens <= remaining:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": raw_result,
            }

        # --- Szenario B: Displacement (zu groß, aber < displacement_threshold) ---
        if new_result_tokens < int(max_tokens * cfg.displacement_threshold):
            freed = self._retroactive_offload(ctx, session, new_result_tokens - remaining)
            new_remaining = remaining + freed
            if new_result_tokens <= new_remaining:
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": raw_result,
                }

        # Fallback: Konnte nicht genug Platz schaffen → Offload wie Szenario C
        return self._offload_immediate(ctx, session, tool_name, raw_result, tool_call_id, content_hash)

    def _offload_immediate(
        self, ctx: ExecutionContext, session, tool_name: str,
        raw_result: str, tool_call_id: str, content_hash: str
    ) -> dict:
        """Szenario C: Sofort ins VFS schreiben, nur Pointer + Preview in History."""
        path = f"/.memory/overflow/{ctx.run_id}_{ctx.current_iteration}_{tool_name}.txt"
        try:
            session.vfs.mkdir("/.memory/overflow", parents=True)
            session.vfs.write(path, raw_result)
        except Exception as e:
            # Fallback: Truncate wenn VFS fehlschlägt
            truncated = raw_result[:2000] + f"\n\n... [TRUNCATED, {len(raw_result)} chars total, VFS write failed: {e}]"
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": truncated,
            }

        ctx.offload_hashes[content_hash] = path

        # Head + Tail Preview
        lines = raw_result.split("\n")
        head = "\n".join(lines[:5])
        tail = "\n".join(lines[-3:]) if len(lines) > 8 else ""
        preview = head
        if tail:
            preview += f"\n...\n{tail}"

        pointer_msg = (
            f"[DATA OFFLOADED to {path}]\n"
            f"Output too large ({self._estimate_tokens(raw_result)} tokens). Saved to VFS.\n"
            f"Preview:\n{preview}\n\n"
            f"Use `vfs_read` on '{path}' to access full content, or use grep to search specific sections."
        )
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": pointer_msg,
        }

    def _retroactive_offload(
        self, ctx: ExecutionContext, session, tokens_needed: int
    ) -> int:
        """
        Szenario B: Alte Tool-Outputs aus working_history ins VFS verschieben.
        Gibt die Anzahl freigegebener Tokens zurück.
        Mit Hash-Dedup: Prüft ob Content bereits offloaded wurde.
        """
        cfg = ctx.context_config
        freed = 0

        # Sammle Kandidaten: (index, token_count, tool_name)
        candidates = []
        for i, msg in enumerate(ctx.working_history):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            tokens = self._estimate_tokens(content)
            if tokens >= cfg.heavy_hitter_min_tokens:
                # Skip bereits offloadete Pointer
                if "[DATA OFFLOADED" in content or "[DUPLICATE:" in content:
                    continue
                candidates.append((i, tokens, msg.get("name", "unknown"), content))

        # Sortiere nach Größe (größte zuerst)
        candidates.sort(key=lambda x: x[1], reverse=True)

        for idx, tokens, name, content in candidates:
            if freed >= tokens_needed:
                break

            content_hash = self._content_hash(content)

            # Hash-Dedup: Bereits offloaded?
            if content_hash in ctx.offload_hashes:
                existing_path = ctx.offload_hashes[content_hash]
                ctx.working_history[idx]["content"] = (
                    f"[DATA OFFLOADED: Previous output from '{name}' already at {existing_path}]"
                )
                freed += tokens
                continue

            # Neu offloaden
            step = ctx.working_history[idx].get("tool_call_id", str(idx))
            path = f"/.memory/archive/{ctx.run_id}_{step}_{name}.txt"
            try:
                session.vfs.mkdir("/.memory/archive", parents=True)
                session.vfs.write(path, content)
                ctx.offload_hashes[content_hash] = path
                ctx.working_history[idx]["content"] = (
                    f"[DATA OFFLOADED: Output from '{name}' moved to {path} to free context space.]"
                )
                freed += tokens
            except Exception:
                continue  # Skip bei Fehler, nächsten Kandidaten versuchen

        return freed

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

    async def _tool_shift_focus(
        self, ctx: ExecutionContext, summary_of_achievements: str, next_objective: str
    ) -> str:
        """
        Verschiebt den aktuellen Arbeitsfortschritt in die Permanent History
        und leert die Working History für einen frischen Start.
        """
        session = await self.agent.session_manager.get_or_create(ctx.session_id)

        # 1. Erzeuge automatische Zusammenfassung der bisherigen Tool-Calls
        auto_summary = HistoryCompressor.compress_to_summary(
            ctx.working_history, ctx.run_id
        )

        # 2. Kombiniere automatische Summary mit der manuellen des Agenten
        combined_content = (
            f"--- FOKUS-WECHSEL / MEILENSTEIN ---\n"
            f"ERGEBNISSE: {summary_of_achievements}\n\n"
            f"TECHNISCHES PROTOKOLL:\n{auto_summary['content'] if auto_summary else 'Keine Tools genutzt.'}"
        )

        # 3. Permanent Speichern (RAG & History)
        await session.add_message(
            {"role": "system", "content": combined_content},
            direct=True,
            type="milestone_summary",
            run_id=ctx.run_id,
        )

        # 4. Working History RESET
        # Wir behalten den ursprünglichen System-Prompt (Index 0)
        system_prompt = ctx.working_history[0]
        ctx.working_history = [
            system_prompt,
            {
                "role": "system",
                "content": f"Vorheriger Abschnitt abgeschlossen. Stand: {summary_of_achievements}",
            },
            {"role": "user", "content": f"Neues Ziel: {next_objective}. Fahre fort."},
        ]

        # 5. Trackers zurücksetzen für neue Phase
        ctx.auto_focus.clear()
        ctx.loop_detector.reset()
        ctx.loop_warning_given = False
        # 1. Begrenze, wie oft ein Agent den Fokus shiften darf (Sicherung gegen Loops)
        if not hasattr(ctx, "focus_shifts_count"):
            ctx.focus_shifts_count = 0

        if ctx.focus_shifts_count >= 3:  # Maximal 3 Resets pro Run
            return "Fehler: Maximale Anzahl an Fokus-Wechseln erreicht. Bitte schließe die Aufgabe jetzt ab."

        ctx.focus_shifts_count += 1

        # 2. Iterations-Bonus statt komplettem Reset
        # Wir setzen nicht auf 1, sondern geben ihm z.B. 10 neue Versuche,
        # aber überschreiten niemals das ursprüngliche Limit.
        ctx.current_iteration = max(1, ctx.current_iteration - 10)

        # Optional: Tool-Relevanz für neues Ziel neu berechnen
        self._calculate_tool_relevance(ctx, next_objective)

        return f"Fokus erfolgreich gewechselt. Dein Gedächtnis wurde bereinigt. Nächstes Ziel: {next_objective}"

    async def _tool_load_tools(
        self, ctx: ExecutionContext, tools_input: Union[str, List[str]]
    ) -> str:
        """
        Load tools with intelligent slot management.

        - Auto-removes lowest relevance tool when limit reached
        - Triggers partial compression on category change + len > 3
        """
        self.live.status_msg = f"Loading tools: {tools_input}"
        self.live.log(f"Loading tools: {tools_input}")
        if (
            isinstance(tools_input, str)
            and tools_input.startswith("[")
            and tools_input.endswith("]")
        ):
            names = tools_input[1:-1].replace(" ", "").replace('"', "").replace("'", "")
            if "," in names:
                names = names.split(",")
            else:
                names = [names]
        elif isinstance(tools_input, str):
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
                            self.live.enter(AgentPhase.COMPRESSING, f"Partial compression ({majority_category} -> {new_category})")

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

        result = "\n".join(lines[:150])
        if len(lines) > 150:
            result += f"\n... und {len(lines) - 150} weitere"

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
            definitions.append(RESUME_SUB_AGENT_TOOL)  # NEW: Resume capability

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
                "(if no fitting tool is loaded, discover it with list_tools and load it with load_tools!)",
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
                "- Sub-Agent Management: spawn_sub_agent, wait_for, resume_sub_agent",
                "  → If a sub-agent hits max_iterations but made progress, resume it with more iterations",
                "",
                "MANDATORY WORKFLOW:",
                "A. PLAN: Use `think()` to decompose the request.",
                "B. ACT: Use tools (`load_tools`, `vfs_*`, etc.) to gather info or execute changes. (if no fitting tool is loaded, discover it with list_tools and load it with load_tools!)",
                "C. VERIFY: Check if the tool output matches expectations.",
                "   → AFTER state-changing tools (createJob, deleteJob, spawn_sub_agent, etc.), ALWAYS call the corresponding list tool (listJobs, list_agents, etc.) to verify!",
                "D. REPORT: Use `final_answer()` only when the objective is met or definitively impossible.",
            ]

        # Add skills section if matched
        if ctx.matched_skills:
            skill_section = self.skills_manager.build_skill_prompt_section(
                ctx.matched_skills
            )
            prompt_parts.append(skill_section)
       # Add persona modifier if active

        if ctx.active_persona.prompt_modifier:
            prompt_parts.append(ctx.active_persona.prompt_modifier)

        # Add verification directive based on persona

        if ctx.active_persona.verification_level == "strict":
            prompt_parts.append(
                       "\n⚠️ VERIFICATION REQUIRED: After EVERY state-changing action, "
            "verify the result with a read/list/status tool before proceeding."
            )

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
        Archiviert den Lauf im VFS und generiert eine LLM-Zusammenfassung.
        """

        # 1. Archivierung: Speichere den vollen Verlauf im VFS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "/.memory/logs"
        log_file = f"{log_dir}/{timestamp}_{ctx.run_id}.md"

        summary = HistoryCompressor.compress_to_summary(ctx.working_history, ctx.run_id)

        try:
            # Sicherstellen, dass Verzeichnis existiert
            session.vfs.mkdir(log_dir, parents=True)

            # Log formatieren
            full_log = [f"# Execution Log: {ctx.run_id}", f"Query: {query}", "-" * 40]
            for msg in ctx.working_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                full_log.append(f"\n### {role.upper()}")
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        fn = (
                            tc.get("function", {})
                            if isinstance(tc, dict)
                            else tc.function
                        )
                        name = fn.get("name", "") if isinstance(fn, dict) else fn.name
                        full_log.append(f"`Tool Call: {name}`")
                full_log.append(content)
            if ctx.active_persona.name != "default":
                full_log.append(f"Persona: {ctx.active_persona.name} (source: {ctx.active_persona.source})")
            session.vfs.write(log_file, "\n".join(full_log))
        except Exception as e:
            self.live.log(f"Failed to write execution log: {e}", logging.WARNING)

        # 2. LLM Summarization (Dynamisch statt Regelbasiert)
        summary_text = "Keine Zusammenfassung."
        try:
            # Nutze schnelles Modell für Zusammenfassung
            summary_prompt = (
                f"Analysiere den folgenden Verlauf eines Agenten-Laufs.\n"
                f"Aufgabe: {query}\n"
                f"Status: {'Erfolg' if success else 'Fehlschlag'}\n"
                f"Tools genutzt: {', '.join(ctx.tools_used)}\n\n"
                f"compressed summary : {summary}"
                f"Erstelle eine prägnante Zusammenfassung mit wichtigen details (max 2-3 Sätze) der durchgeführten Aktionen und des Ergebnisses."
                f"Erwähne erstellte/bearbeitete Dateien."
            )

            # Wir nutzen working_history als Kontext, aber limitiert
            summary_text = await self.agent.a_run_llm_completion(
                messages=ctx.working_history[-10:]
                + [{"role": "user", "content": summary_prompt}],
                model_preference="fast",
                max_tokens=400,
                stream=False,
            )
        except Exception as e:
            # Fallback auf Rule-Based Compressor bei Fehler
            auto_sum = HistoryCompressor.compress_to_summary(
                ctx.working_history, ctx.run_id
            )
            summary_text = (
                auto_sum["content"] if auto_sum else "Fehler bei Zusammenfassung."
            )

        # 3. Permanent Speichern
        # User message
        await session.add_message({"role": "user", "content": query})

        # System Summary mit Referenz auf Log-Datei
        summary_msg = {
            "role": "system",
            "content": f"⚡ RUN SUMMARY [{ctx.run_id}]: {summary_text}\n(Full Log: {log_file})",
            "metadata": {"type": "run_summary", "log_file": log_file},
        }

        await session.add_message(
            summary_msg,
            direct=True,
            type="action_summary",
            run_id=ctx.run_id,
            success=success,
        )

        # Final response
        await session.add_message({"role": "assistant", "content": final_response})

        self.live.log(f"Run {ctx.run_id} archived to {log_file}")

        # ── Notify idle tracker ──
        try:
            from toolboxv2 import get_app
            sched = get_app().get_mod("isaa").job_scheduler
            if hasattr(sched, '_agent_idle_eval'):
                sched._agent_idle_eval.record_activity(self.agent.amd.name)
        except Exception:
            pass

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
        self.live.enter(AgentPhase.PAUSED, f"Paused [{execution_id}] at iter {ctx.current_iteration}")
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
        self.live.enter(AgentPhase.ERROR, f"Cancelled [{execution_id}]")

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

    async def resume(self, execution_id: str, max_iterations: int = 15, content="", stream=False) -> str | tuple[
        Callable[[...], Any], ExecutionContext] | tuple[str, ExecutionContext]:
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
        if content:
            ctx.working_history.append({"role": "system", "content": "Continue with old task using new user information's"})
            ctx.working_history.append({"role": "user", "content": content})

        if stream:
            # Resume execution
            return await self.execute_stream(
                query=ctx.query,
                session_id=ctx.session_id,
                max_iterations=max_iterations,
                ctx=ctx,
            )
        # Resume execution
        return await self.execute(
            query=ctx.query,
            session_id=ctx.session_id,
            max_iterations=max_iterations,
            ctx=ctx,
        )



if __name__ == "__main__":
    print("ExecutionEngine loaded")
    ctx = ExecutionContext()
    print(ctx.run_id)
    from toolboxv2.tests.test_mods.test_isaa.test_base.test_agent.test_execution_engine import (
        MockAgent,
    )

    execution_engine = ExecutionEngine(MockAgent(), None, None)
    print(
        asyncio.run(
            execution_engine._tool_load_tools(
                ctx, '["scout_interface", "execute_action"]'
            )
        )
    )
