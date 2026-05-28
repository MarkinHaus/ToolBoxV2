"""
ExecutionEngine V3 - Intelligent Agent Orchestration

Features:
- Dynamic Tool Loading with keyword-based relevance scoring
- Working/Permanent History separation with rule-based compression
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
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union


from toolboxv2.mods.isaa.base.Agent.ctx_cleaner import clean_messages
from toolboxv2 import get_app, get_logger

# Import Live State Management
from toolboxv2.mods.isaa.base.Agent.agent_live_state import (
    AgentLiveState,
    AgentPhase,
    TokenStream,
    ToolExecution,
)
from toolboxv2.mods.isaa.base.Agent.narrator import AgentLiveNarrator

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

MAX_CONTINUATIONS = os.environ.get("AGENT_INTERN_MAX_CONTINUATIONS", 5)
MAX_PARALLEL_SKILLS = os.environ.get("AGENT_INTERN_MAX_PARALLEL_SKILLS", 3)
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
                    },
                    "effort": {
                        "type": "string",
                        "description": "the effort for the situation assessment [fast|complex]",
                    },
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
                        "type": ["string", "null"],
                        "description": "Optional category filter (e.g., 'discord', 'vfs', 'memory')",
                    },
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

SKILL_DISCOVERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List all available skills with their triggers, confidence, and status. Use this to discover learned behaviors and workflows you can activate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": ["string", "null"],
                        "description": "Optional: filter skills by keyword relevance to this query",
                    },
                    "include_inactive": {
                        "type": "boolean",
                        "description": "Include skills below activation threshold (default: false)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_skill",
            "description": "Activate a skill by name or ID. This injects the skill's instruction into your current context and preloads its recommended tools. Use after list_skills to apply a specific workflow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "The skill ID or name to activate",
                    },
                },
                "required": ["skill_id"],
            },
        },
    },
]

CODING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_patch",
            "description": (
                "Apply a patch to an existing VFS file. Internally: "
                "1) Reads the file + collects context you provide, "
                "2) Uses a complex LLM to generate a precise patch in one shot, "
                "3) Validates the result (syntax, structure). "
                "You provide the file path and a clear description of WHAT to change and WHY. "
                "The tool handles the actual code generation. One file per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "VFS path of the file to patch (e.g. /project/src/main.py)",
                    },
                    "task": {
                        "type": "string",
                        "description": "Detailed description of what to change, why, and any constraints. Include relevant context (function signatures, error messages, test expectations).",
                    },
                    "context_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: additional VFS file paths to read as context for the patch (imports, related modules, tests)",
                    },
                },
                "required": ["file_path", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new file in VFS. Internally: "
                "1) Collects context from your description + optional reference files, "
                "2) Uses a complex LLM to generate the complete file in one shot, "
                "3) Validates syntax. "
                "You provide the target path, purpose, and detailed spec. One file per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "VFS path for the new file (e.g. /project/src/utils.py)",
                    },
                    "task": {
                        "type": "string",
                        "description": "Detailed spec: what the file should contain, its purpose, interfaces, constraints. The more precise, the better the output.",
                    },
                    "context_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: VFS file paths to read as reference (existing modules, interfaces, types)",
                    },
                },
                "required": ["file_path", "task"],
            },
        },
    },
]

# VFS Tools - always available for file navigation (part of static)
VFS_TOOL_NAMES = ["vfs_read", "vfs_write", "vfs_list", "vfs_navigate", "vfs_control"]
_VFS_PERSONAS = "/global/.memory/dreamer/personas.json"

# =============================================================================
# HELPER COMPONENTS
# =============================================================================


class ToolValidationError(Exception):
    """Raised when the LLM provider rejects a tool call as invalid
    (unknown tool name, schema violation). Permanent per-request error —
    must be handled at the agent loop level, not retried mid-stream."""

    pass

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

    max_context_ratio: float = (
        0.85  # Wie viel % des Model-Kontexts genutzt werden dürfen (0.7 - 0.95)
    )
    immediate_offload_ratio: float = (
        0.7  # Ab diesem Anteil am Gesamt-Kontext → sofort offloaden (Szenario C)
    )
    displacement_threshold: float = (
        0.4  # Max Größe für Displacement-Strategie (Szenario B)
    )
    safety_margin_tokens: int = 500  # Reserve
    heavy_hitter_min_tokens: int = 1000  # Min Größe für Offload-Kandidaten


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
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PersonaStats:
    """Tracks usage, routing reasons, and effectiveness of a PersonaProfile."""

    # Usage counters
    total_uses: int = 0
    uses_by_source: dict[str, int] = field(
        default_factory=lambda: {
            "default": 0,
            "matched": 0,
            "dreamer": 0,
            "dreamer_learned": 0,
        }
    )

    # Effectiveness
    successful_runs: int = 0
    failed_runs: int = 0
    total_iterations_used: int = 0
    total_iterations_budget: int = 0  # sum of apply_max_iterations outputs

    # Temporal
    first_used_at: datetime | None = None
    last_used_at: datetime | None = None
    last_success_at: datetime | None = None

    # Routing reasons — which keywords/skills triggered this persona
    trigger_keywords: dict[str, int] = field(default_factory=dict)  # keyword -> hit count
    trigger_skills: dict[str, int] = field(
        default_factory=dict
    )  # skill name -> hit count

    # Per-query log (capped to last N entries)
    recent_queries: list[dict] = field(default_factory=list)
    _recent_queries_max: int = field(default=20, repr=False, compare=False)

    # ── derived ──────────────────────────────────────────────────────────────

    @property
    def success_ratio(self) -> float:
        total = self.successful_runs + self.failed_runs
        return self.successful_runs / total if total else 0.0

    @property
    def avg_iterations(self) -> float:
        return self.total_iterations_used / self.total_uses if self.total_uses else 0.0

    @property
    def budget_efficiency(self) -> float:
        """How much of the iteration budget was actually consumed (lower = leaner)."""
        return (
            self.total_iterations_used / self.total_iterations_budget
            if self.total_iterations_budget
            else 0.0
        )

    # ── mutation helpers ──────────────────────────────────────────────────────

    def record_use(
        self,
        *,
        source: str,
        query: str,
        success: bool,
        iterations_used: int,
        iterations_budget: int,
        trigger_keyword: str | None = None,
        trigger_skill: str | None = None,
    ) -> None:
        now = datetime.utcnow()

        self.total_uses += 1
        self.uses_by_source[source] = self.uses_by_source.get(source, 0) + 1

        if success:
            self.successful_runs += 1
            self.last_success_at = now
        else:
            self.failed_runs += 1

        self.total_iterations_used += iterations_used
        self.total_iterations_budget += iterations_budget

        if self.first_used_at is None:
            self.first_used_at = now
        self.last_used_at = now

        if trigger_keyword:
            self.trigger_keywords[trigger_keyword] = (
                self.trigger_keywords.get(trigger_keyword, 0) + 1
            )
        if trigger_skill:
            self.trigger_skills[trigger_skill] = (
                self.trigger_skills.get(trigger_skill, 0) + 1
            )

        entry = {
            "ts": now.isoformat(),
            "query": query[:120],
            "source": source,
            "success": success,
            "iters": iterations_used,
            "budget": iterations_budget,
        }
        self.recent_queries.append(entry)
        if len(self.recent_queries) > self._recent_queries_max:
            self.recent_queries.pop(0)

    def to_dict(self) -> dict:
        return {
            "total_uses": self.total_uses,
            "uses_by_source": self.uses_by_source,
            "success_ratio": round(self.success_ratio, 3),
            "avg_iterations": round(self.avg_iterations, 2),
            "budget_efficiency": round(self.budget_efficiency, 3),
            "first_used_at": self.first_used_at.isoformat()
            if self.first_used_at
            else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "last_success_at": self.last_success_at.isoformat()
            if self.last_success_at
            else None,
            "top_trigger_keywords": sorted(
                self.trigger_keywords.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "top_trigger_skills": sorted(
                self.trigger_skills.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "recent_queries": self.recent_queries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaStats":
        obj = cls()
        obj.total_uses = data.get("total_uses", 0)
        obj.uses_by_source = data.get("uses_by_source", {})
        obj.successful_runs = data.get("successful_runs", 0)
        obj.failed_runs = data.get("failed_runs", 0)
        obj.total_iterations_used = data.get("total_iterations_used", 0)
        obj.total_iterations_budget = data.get("total_iterations_budget", 0)
        obj.trigger_keywords = dict(data.get("trigger_keywords", {}))
        obj.trigger_skills = dict(data.get("trigger_skills", {}))
        obj.recent_queries = data.get("recent_queries", [])
        for attr in ("first_used_at", "last_used_at", "last_success_at"):
            raw = data.get(attr)
            setattr(obj, attr, datetime.fromisoformat(raw) if raw else None)
        return obj


@dataclass
class PersonaProfile:
    """Runtime persona applied to an execution."""

    name: str = "default"
    prompt_modifier: str = ""  # injected into system prompt
    model_preference: str = "fast"  # "fast" | "complex"
    temperature: float | None = None  # None = use model default
    max_iterations_factor: float = 1.0  # multiplied with base max_iterations
    verification_level: str = "basic"  # "none" | "basic" | "strict"
    source: str = "default"  # "default" | "matched" | "dreamer"
    stats: PersonaStats = field(default_factory=PersonaStats)

    def apply_max_iterations(self, base: int) -> int:
        return int(min(base * 5, int(base * self.max_iterations_factor)))


from toolboxv2.mods.isaa.base.Agent.default_personas import (
    _BUILTIN_PERSONAS,
    _PERSONA_KEYWORDS,
)


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
    ) -> tuple[PersonaProfile, str | None, str | None]:
        """
        Returns:
            (persona, trigger_keyword, trigger_skill)
        """
        query_lower = query.lower()

        # 1. Dreamer insights
        if dreamer_insights:
            best_key, best_ratio = None, 0.0
            for intent_key, insight in dreamer_insights.items():
                ratio = insight.get("success_ratio", 0)
                if ratio > best_ratio and ratio > 0.6:
                    if any(w in query_lower for w in intent_key.lower().split()[:2]):
                        best_key = intent_key
                        best_ratio = ratio
            if best_key:
                persona = self._match_intent_to_persona(best_key)
                if persona:
                    persona.source = "dreamer"
                    return persona, best_key, None

        # 1.5 Learned personas
        for _pk, _pp in list(self.personas.items()):
            if _pp.source != "dreamer_learned":
                continue
            _words = [w for w in _pk.replace("learned_", "").split("_") if len(w) > 3]
            if _words and sum(1 for w in _words if w in query_lower) >= min(
                2, len(_words)
            ):
                import dataclasses as _dc

                matched_word = next((w for w in _words if w in query_lower), None)
                return _dc.replace(_pp), matched_word, None

        # 2. Skill-based
        if matched_skills:
            for skill in matched_skills:
                name_lower = skill.name.lower()
                for persona_key, keywords in _PERSONA_KEYWORDS.items():
                    hit_kw = next((kw for kw in keywords if kw in name_lower), None)
                    if hit_kw:
                        return (
                            dataclasses.replace(
                                self.personas[persona_key], source="matched"
                            ),
                            hit_kw,
                            skill.name,
                        )

        # 3. Keyword classification
        scores: dict[str, int] = {}
        best_kw_per_persona: dict[str, str] = {}
        for persona_key, keywords in _PERSONA_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    scores[persona_key] = scores.get(persona_key, 0) + 1
                    if persona_key not in best_kw_per_persona:
                        best_kw_per_persona[persona_key] = kw

        if scores:
            best = max(scores, key=scores.get)
            if scores[best] >= 1 and best in self.personas:
                return (
                    dataclasses.replace(self.personas[best], source="matched"),
                    best_kw_per_persona.get(best),
                    None,
                )

        # 4. Default
        return PersonaProfile(), None, None

    def _match_intent_to_persona(self, intent_key: str) -> PersonaProfile | None:
        intent_lower = intent_key.lower()
        for persona_key, keywords in _PERSONA_KEYWORDS.items():
            if any(kw in intent_lower for kw in keywords[:3]):
                return PersonaProfile(
                    **{
                        k: getattr(self.personas[persona_key], k)
                        for k in PersonaProfile.__dataclass_fields__
                    }
                )
        return None

    def load_learned_personas(self, session) -> None:
        """
        Inject VFS-persisted learned personas into this router.
        Called once per ExecutionEngine lifecycle (lazy, on first execute).
        Only loads entries with confidence >= 0.30.
        """
        _VFS_PERSONAS = "/global/.memory/dreamer/personas.json"
        try:
            result = session.vfs_read(_VFS_PERSONAS)
            if not result.get("success"):
                return
            store: dict = json.loads(result["content"])
        except Exception:
            return
        loaded = 0
        for key, entry in store.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("confidence", 0.0) < 0.30:
                continue
            pd_ = entry.get("profile", {})
            try:
                p = PersonaProfile(
                    name=pd_.get("name", key),
                    prompt_modifier=pd_.get("prompt_modifier", ""),
                    model_preference=pd_.get("model_preference", "fast"),
                    temperature=pd_.get("temperature"),
                    max_iterations_factor=float(pd_.get("max_iterations_factor", 1.0)),
                    verification_level=pd_.get("verification_level", "basic"),
                    source="dreamer_learned",
                )
                # ── Stats restore ──
                if "stats" in entry:
                    p.stats = PersonaStats.from_dict(entry["stats"])

                self.personas[key] = p
                loaded += 1
            except Exception:
                pass
        if loaded:
            get_logger().debug(
                f"[PersonaRouter] Loaded {loaded} learned persona(s) from VFS"
            )


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
    max_dynamic_tools: int = int(os.getenv("MAX_DYNAMIC_TOOLS", 5))
    tool_relevance_cache: Dict[str, float] = field(default_factory=dict)
    tool_category_cache: Dict[str, Set[str]] = field(default_factory=dict)

    # History Management
    working_history: List[dict] = field(default_factory=list)

    # Tracker
    loop_detector: LoopDetector = field(default_factory=LoopDetector)

    # Run State
    tools_used: List[str] = field(default_factory=list)
    tools_dict: List[dict] = field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 0
    matched_skills: List[Skill] = field(default_factory=list)
    active_persona: PersonaProfile = field(default_factory=PersonaProfile)
    loop_warning_given: bool = False

    # Context Budget
    context_config: ContextBudgetConfig = field(default_factory=ContextBudgetConfig)
    offload_hashes: Dict[str, str] = field(
        default_factory=dict
    )  # content_hash -> vfs_path

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

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
        # Evict the slot with the lowest relevance score,
        # excluding the newly added tool (never self-evict).
        if len(self.dynamic_tools) > self.max_dynamic_tools:
            evict = min(
                (s for s in self.dynamic_tools if s.name != name),
                key=lambda s: s.relevance_score,
                default=None,
            )
            if evict:
                self.remove_tool(evict.name)
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
            "loop_detector_history": self.loop_detector.history,
            "max_iterations": self.max_iterations,

        "matched_skill_ids": [s.id for s in self.matched_skills] if self.matched_skills else [],
        "active_persona_name": self.active_persona.name if self.active_persona else "default",
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
        ctx.loop_detector.history = data.get("loop_detector_history", [])
        ctx.max_iterations = data.get("max_iterations", os.getenv("DEFAULT_MAX_ITERATIONS", 30))

        ctx._cold_skill_ids = data.get("matched_skill_ids", [])
        ctx._cold_persona_name = data.get("active_persona_name", "default")
        return ctx

    @property
    def lock(self):
        return self._lock


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
        last_think_result = (
            None  # Nur das letzte think-Tool-Result (informationsdichteste Stelle)
        )
        tools_used = set()

        for i, msg in enumerate(working_history):
            role = msg.get("role", "")

            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    if hasattr(tc, "function"):
                        tools_used.add(tc.function.name)
                    elif isinstance(tc, dict) and "function" in tc:
                        tools_used.add(tc["function"].get("name", ""))

            elif role == "tool":
                name = msg.get("name", "")
                content = msg.get("content", "")
                tools_used.add(name)
                content_lower = content.lower()
                name_lower = name.lower()

                # Fehler-Erkennung
                if (
                    "error" in content_lower
                    or "failed" in content_lower
                    or "traceback" in content_lower
                ):
                    errors.append(f"{name}: {content[:120]}")
                # Write-Ops
                elif any(
                    kw in name_lower
                    for kw in ("write", "create", "edit", "append", "delete", "mv")
                ):
                    # Behalte den Pfad/Status, nicht den Inhalt
                    first_line = content.split("\n")[0][:100] if content else "ok"
                    writes.append(f"{name} → {first_line}")
                # Read-Ops (inkl. offloaded)
                elif any(
                    kw in name_lower
                    for kw in (
                        "read",
                        "list",
                        "navigate",
                        "search",
                        "query",
                        "view",
                        "open",
                        "grep",
                    )
                ):
                    if "[DATA OFFLOADED" in content:
                        reads.append(f"{name}: [offloaded]")
                    else:
                        size = len(content)
                        reads.append(f"{name} ({size} chars)")
                elif name == "think":
                    last_think_result = content
                else:
                    reads.append(f"{name}")

        # Build Semantic Ledger
        lines = [f"📋 SEMANTIC LEDGER [Run {run_id}]:"]

        if last_think_result:
            lines.append(f"\n🔍 LAST THINK RESULT:")
            lines.append(last_think_result)

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

        meaningful = tools_used - {
            "think",
            "final_answer",
            "list_tools",
            "load_tools",
            "shift_focus",
        }
        lines.append(
            f"\n🔧 Total: {len(tools_used)} tool calls, {len(meaningful)} unique tools"
        )

        return {
            "role": "system",
            "content": "\n".join(lines),
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

        # Robusterer Split: Verhindert das Abschneiden von Tool-Ketten
        split_idx = len(working_history) - keep_last_n

        # Sicherheits-Check: Wenn die erste zu behaltende Nachricht ein Tool ist,
        # müssen wir den Split-Punkt nach vorne verschieben, um den Assistant-Call einzuschließen.
        max_itter = 1000000000
        i = 0
        while (
            split_idx > 1
            and working_history[split_idx].get("role") == "tool"
            and i < max_itter
        ):
            i += 1
            split_idx -= 1

        # Jetzt zeigt split_idx entweder auf den Assistant oder eine User-Nachricht
        if system_msg:
            to_process = working_history[1:split_idx]
            to_keep = working_history[split_idx:]
        else:
            to_process = working_history[:split_idx]
            to_keep = working_history[split_idx:]

        # Finde Index des letzten think-Tool-Results in to_process → vor Kompression schützen
        last_think_idx = -1
        for idx in range(len(to_process) - 1, -1, -1):
            m = to_process[idx]
            if m.get("role") == "tool" and m.get("name") == "think":
                last_think_idx = idx
                break

        compressed_msgs = []
        stats = {"kept": 0, "summarized": 0, "dropped": 0}

        for idx, msg in enumerate(to_process):
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
                    new_msg["content"] = (
                        (content[:80] + "...") if len(content) > 80 else content
                    )
                else:
                    new_msg["content"] = (
                        (content[:160] + "...") if len(content) > 160 else content
                    )
                compressed_msgs.append(new_msg)
                stats["summarized"] += 1

            elif role == "tool":
                # P2: ZUSAMMENFASSEN (think, write, exec)
                if name == "think":
                    if idx == last_think_idx:
                        # Letztes think vollständig behalten
                        compressed_msgs.append(msg)
                        stats["kept"] += 1
                    else:
                        compressed_msgs.append(
                            {
                                "role": "tool",
                                "tool_call_id": msg.get("tool_call_id", ""),
                                "name": name,
                                "content": f"Think: {content[:100]}..."
                                if len(content) > 100
                                else content,
                            }
                        )
                        stats["summarized"] += 1

                elif any(
                    kw in name_lower
                    for kw in ("write", "create", "edit", "exec", "shell", "run")
                ):
                    first_line = content.split("\n")[0][:80]
                    compressed_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": msg.get("tool_call_id", ""),
                            "name": name,
                            "content": first_line,
                        }
                    )
                    stats["summarized"] += 1

                elif name in ("shift_focus", "final_answer"):
                    compressed_msgs.append(msg)
                    stats["kept"] += 1

                # P3: LÖSCHEN (read, list, navigate, search outputs)
                elif any(
                    kw in name_lower
                    for kw in (
                        "read",
                        "list",
                        "navigate",
                        "search",
                        "query",
                        "view",
                        "open",
                        "cat",
                    )
                ):
                    compressed_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": msg.get("tool_call_id", ""),
                            "name": name,
                            "content": f"[Viewed: {name}]",
                        }
                    )
                    stats["dropped"] += 1

                else:
                    # Unbekanntes Tool: Zusammenfassen
                    compressed_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": msg.get("tool_call_id", ""),
                            "name": name,
                            "content": content[:80] if content else "ok",
                        }
                    )
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


def _parse_tool_arguments(args_str: str) -> dict:
    """
    Robust parser für LLM-tool-call arguments.

    Handles in order:
      1. Strict json.loads (fast path, normalerweise 99% der Fälle).
      2. Strip markdown code fences (``` / ```json) falls das Modell sie mitschickt.
      3. Common escape mistakes: \\' (unnötig escaped single quote).
      4. Raw control chars inside strings (newlines/tabs die nicht escaped wurden)
         — häufig bei Modellen die Markdown/Code als Tool-Arg schicken.
      5. Python-literal fallback (ast.literal_eval) für single-quoted dicts.

    Raises ValueError if nothing works. Never returns partial results.
    """
    if not isinstance(args_str, str):
        raise ValueError(f"Expected str, got {type(args_str).__name__}")

    s = args_str.strip()
    if not s:
        return {}

    # --- Pass 1: strict ---
    try:
        result = json.loads(s)
        if isinstance(result, dict):
            return result
        raise ValueError(f"Expected JSON object, got {type(result).__name__}")
    except json.JSONDecodeError:
        pass

    # --- Pass 2: strip ``` / ```json fences ---
    stripped = s
    if stripped.startswith("```"):
        # remove leading fence line
        nl = stripped.find("\n")
        if nl != -1:
            stripped = stripped[nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
        try:
            result = json.loads(stripped)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # --- Pass 3: unescape \' (invalid per JSON spec, common LLM bug) ---
    if r"\'" in stripped:
        fixed = stripped.replace(r"\'", "'")
        try:
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # --- Pass 4: escape raw control chars inside string literals ---
    # Models frequently dump markdown/code into string values with literal \n,
    # \t, and even unescaped \r. JSON forbids these raw; escape them only
    # when they appear INSIDE a "..." string (outside strings they're fine).
    repaired = _escape_raw_controls_in_json_strings(stripped)
    if repaired != stripped:
        try:
            result = json.loads(repaired)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # --- Pass 5: Python literal fallback (single-quoted dicts from small models) ---
    try:
        import ast

        result = ast.literal_eval(stripped)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # --- Give up ---
    raise ValueError(
        f"Could not parse tool arguments as JSON. "
        f"Length={len(args_str)}, first 200 chars: {args_str[:200]!r}"
    )


def _escape_raw_controls_in_json_strings(s: str) -> str:
    """
    Walk the JSON source char-by-char. Inside "..." string literals,
    replace raw control chars (\\n, \\r, \\t, \\b, \\f) with their escaped
    equivalents. Outside strings, leave everything alone.

    Respects backslash escapes so that already-valid \\" is not miscounted.
    Does NOT attempt to fix unescaped internal quotes — that's ambiguous
    and would need a proper tokenizer. For tool calls, unescaped " inside
    strings is rare; unescaped newlines in markdown content is common.
    """
    out = []
    in_string = False
    escape_next = False
    replacements = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\b": "\\b",
        "\f": "\\f",
    }
    for ch in s:
        if escape_next:
            out.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch in replacements:
            out.append(replacements[ch])
            continue
        out.append(ch)
    return "".join(out)

def _is_code_truncated(text: str) -> bool:
    if not text:
        return False
    if text.count("```") % 2 != 0:
        return True
    if text.count("{") - text.count("}") > 2:
        return True
    if text.count("[") - text.count("]") > 2:
        return True
    if text.count("(") - text.count(")") > 2:
        return True
    for tag in ["<script>", "<style>", "<html>", "<body>"]:
        close = tag.replace("<", "</")
        if text.count(tag) > text.count(close):
            return True
    return False

import re
import json
import ast

# ── Shared core ───────────────────────────────────────────────────
_JUNK_RE = re.compile(r"<[^>]*>")  # XML-tag artifacts like </arg_value>


def _clean_tool_name(raw_name: str, existing_args: dict) -> tuple[str, dict]:
    """Return (clean_name, merged_args). Mutates nothing."""
    # 1. strip XML junk
    name = _JUNK_RE.sub("", raw_name).strip()

    # 2. split off call-parens:  load_tools(tools=["vfs"]) → name, argstr
    paren = name.find("(")
    if paren != -1:
        argstr = name[paren + 1:].rstrip(")").strip()
        name = name[:paren].strip()
        if argstr:
            existing_args = _merge_inline_args(argstr, existing_args)

    # 3. remove any remaining non-identifier chars
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)

    return name, existing_args


def _merge_inline_args(argstr: str, existing: dict) -> dict:
    """Parse Python-style kwargs from the paren content, merge into existing."""
    merged = dict(existing)
    # try wrapping in dict() so ast can parse it
    try:
        parsed = ast.literal_eval(f"dict({argstr})")
        for k, v in parsed.items():
            if k not in merged:
                merged[k] = v
        return merged
    except Exception:
        pass
    # fallback: naive key=value split
    for part in argstr.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            if k and k not in merged:
                try:
                    merged[k] = ast.literal_eval(v.strip())
                except Exception:
                    merged[k] = v.strip()
    return merged


# ── Wrapper 1: OpenAI-compatible object ──────────────────────────
def clean_tc_object(tc):
    """In-place clean on a LiteLLM/OpenAI tool_call object."""
    raw = tc.function.name or ""
    try:
        existing = json.loads(tc.function.arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        existing = {}

    name, args = _clean_tool_name(raw, existing)
    tc.function.name = name
    tc.function.arguments = json.dumps(args, ensure_ascii=False)
    return tc


# ── Wrapper 2: dict equivalent ───────────────────────────────────
def clean_tc_dict(tc: dict):
    """In-place clean on a tool_call dict."""
    func = tc.get("function", {})
    raw = func.get("name", "")
    existing = func.get("arguments", {})
    if isinstance(existing, str):
        try:
            existing = json.loads(existing)
        except (json.JSONDecodeError, TypeError):
            existing = {}

    name, args = _clean_tool_name(raw, existing)
    func["name"] = name
    func["arguments"] = json.dumps(args, ensure_ascii=False)
    return tc

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
        do_narrator: bool = True,
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
        self.agent: "FlowAgent" = agent
        self.human_online = human_online
        self.live = AgentLiveState(
            agent_name=getattr(agent, "amd", None) and agent.amd.name or "?",
            is_sub=is_sub_agent,
        )

        self._narrator = AgentLiveNarrator(
            live=self.live, agent=agent, do_narator=do_narrator
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
        self._session_last_run: dict[str, str] = {}  # session_id → run_id
        self._current_session: "AgentSessionV2" = None

        # Finalization locks (for background commit after final_answer)
        self._finalize_locks: Dict[str, asyncio.Lock] = {}  # per session
        self._skills_stats_lock = asyncio.Lock()  # global (skills shared file)
        self._persona_stats_lock = asyncio.Lock()  # global (persona shared file)
        self._pending_finalize_tasks: Dict[str, asyncio.Task] = {}  # run_id → task


        _obs = getattr(agent, 'obs', None)
        self.max_resumable: int = _obs.max_runs if _obs else 3

        # Get or create SkillsManager
        if (
            hasattr(agent.session_manager, "skills_manager")
            and agent.session_manager.skills_manager
        ):
            self.skills_manager = agent.session_manager.skills_manager
        else:
            self.skills_manager = SkillsManager(
                agent_name=agent.amd.name, memory_instance=self._get_memory_instance(), match_embedding=False
            )
            # Store back on agent
            agent.session_manager.skills_manager = self.skills_manager
        self._persona_router = PersonaRouter()
        self._personas_loaded = False  # lazy VFS load on first execute
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
                    "create job",
                    "scheduled job",
                    "cron job",
                    "interval job",
                    "job erstellen",
                    "geplanter job",
                    "automatisierung",
                    "schedule",
                    "timer",
                    "periodisch",
                    "wöchentlich",
                    "täglich",
                    "stündlich",
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
                "tools_used": [
                    "createJob",
                    "listJobs",
                    "deleteJob",
                    "think",
                    "final_answer",
                ],
                "tool_groups": ["job_management"],
                "source": "predefined",
            }

            from toolboxv2.mods.isaa.base.Agent.skills import Skill

            self.skills_manager.skills["job_management"] = Skill(**JOB_MANAGEMENT_SKILL)

        # Add coding_workflow skill
        if "coding_workflow" not in self.skills_manager.skills:
            CODING_WORKFLOW_SKILL = {
                "id": "coding_workflow",
                "name": "OODA Coding Workflow",
                "triggers": [
                    "code", "coding", "programmieren", "implement", "implementieren",
                    "fix bug", "bugfix", "refactor", "patch", "write code", "code schreiben",
                    "function", "funktion", "class", "klasse", "method", "methode",
                    "file erstellen", "datei erstellen", "code ändern", "modify code",
                    "feature implementieren", "test schreiben", "write test",
                ],
                "instruction": (
                    "CODING WORKFLOW — OODA LOOP (mandatory for every code task):\n\n"
                    "O — OBSERVE: Gather raw data.\n"
                    "  • Use vfs_read / vfs_list / vfs_navigate to read existing code.\n"
                    "  • If no fitting tool loaded → list_tools + load_tools.\n"
                    "  • Read errors, stacktraces, test outputs — raw facts only.\n"
                    "  • NEVER hypothesize without code evidence.\n\n"
                    "O — ORIENT: Use think() to analyze.\n"
                    "  • Extract facts that follow directly from the code.\n"
                    "  • List what is NOT known and cannot be determined without more code.\n"
                    "  • If multiple root causes possible → list all, pick most likely.\n"
                    "  • NO fixes in this phase.\n\n"
                    "D — DECIDE: Pick exactly ONE next action.\n"
                    "  • For new code/files → use write_file tool.\n"
                    "  • For patches to existing files → use write_patch tool.\n"
                    "  • Both tools handle LLM generation + validation internally.\n"
                    "  • If objective met → final_answer.\n\n"
                    "A — ACT: Execute the decision.\n"
                    "  • After state-changing actions → OBSERVE again to verify.\n"
                    "  • Only exact fixes needed, no refactorings, no bonus features.\n\n"
                    "RULES:\n"
                    "  • Concise. 50 lines when 500 would be possible.\n"
                    "  • No 'probably', 'might be', 'could be' without code evidence.\n"
                    "  • Python 3.10+, async-first where framework requires it.\n"
                    "  • unittest exclusively — never pytest.\n"
                    "  • Error handling: never swallow errors (except: pass).\n"
                    "  • Deliver ONLY what is requested. No extras."
                ),
                "tools_used": [
                    "vfs_read", "vfs_write", "vfs_list", "vfs_navigate",
                    "write_patch", "write_file",
                    "think", "final_answer", "load_tools", "list_tools",
                ],
                "tool_groups": ["vfs"],
                "source": "predefined",
            }

            from toolboxv2.mods.isaa.base.Agent.skills import Skill as _Skill
            self.skills_manager.skills["coding_workflow"] = _Skill(**CODING_WORKFLOW_SKILL)

        # Auto-group tools if not done yet
        if not self.skills_manager.tool_groups:
            self._auto_setup_tool_groups()

    @contextlib.asynccontextmanager
    async def track_phase(self, name):
        start = time.perf_counter()
        yield
        duration = time.perf_counter() - start
        if duration > 0.5:  # Alles über 500ms loggen
            get_logger().warning(f"⚠️ PHASE LONG DURATION: {name} took {duration:.2f}s")

    async def wait_all_pending_finalizes(self) -> None:
        """Wait for all pending finalize tasks (e.g. on shutdown)."""
        tasks = list(self._pending_finalize_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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

    def _get_finalize_lock(self, session_id: str) -> asyncio.Lock:
        """Lock pro Session für add_message-Ordering."""
        if session_id not in self._finalize_locks:
            self._finalize_locks[session_id] = asyncio.Lock()
        return self._finalize_locks[session_id]

    async def _wait_for_pending_finalize(self, session_id: str) -> None:
        """Warte auf pending finalize der vorherigen Query auf dieser Session.

        Wird am Start von execute/execute_stream aufgerufen, damit
        Folge-Queries den commit der vorherigen sehen.
        """
        lock = self._finalize_locks.get(session_id)
        if lock is not None and lock.locked():
            self.live.status_msg = "Waiting for previous finalization"
            async with lock:
                pass  # just wait

    async def _finalize_run(
        self,
        ctx: ExecutionContext,
        session,
        query: str,
        final_response: str,
        success: bool,
        trigger_kw: str | None,
        trigger_skill: str | None,
    ) -> None:
        """Background finalization: log + summary + commit + stats.

        Runs under session-lock so concurrent queries on same session
        wait for completion.
        """
        session_id = ctx.session_id or "default"
        _fin_start = time.time()
        session_lock = self._get_finalize_lock(session_id)

        async with session_lock:

            try:
                await self._commit_run_slow(ctx, session, query, final_response, success)
            except Exception as e:
                self.live.log(f"[Finalize] commit_run_slow failed: {e}", logging.ERROR)

            self.live.log(f"Skills stats", logging.INFO)
            # 2. Skills stats (global shared file → global lock)
            async with self._skills_stats_lock:
                try:
                    self.skills_manager.record_matched_skills_usage(
                        matched_skills=ctx.matched_skills,
                        success=success,
                        query=query,
                        iterations_used=ctx.current_iteration,
                    )
                except Exception as e:
                    self.live.log(f"[Finalize] skills record failed: {e}", logging.ERROR)

            self.live.log(f"Persona stats", logging.INFO)
            # 3. Persona stats persist (global shared file → global lock)
            if ctx.active_persona and ctx.active_persona.name != "default":
                async with self._persona_stats_lock:
                    try:
                        await self._persist_persona_stats(session, ctx.active_persona)
                    except Exception as e:
                        self.live.log(
                            f"[Finalize] persona persist failed: {e}", logging.ERROR
                        )

            self.live.log(f"{success=}", logging.INFO)
            # 4. Learning task (already internally bg, schedule after commit)
            if success:
                try:
                    app = get_app()
                    app.run_bg_task_advanced(
                        self._background_learning_task,
                        query=query,
                        tools_used=ctx.tools_used,
                        final_response=final_response,
                        success=success,
                        matched_skills=ctx.matched_skills,
                        iterations_used=ctx.current_iteration,
                    )
                except Exception as e:
                    self.live.log(
                        f"[Finalize] learning schedule failed: {e}", logging.ERROR
                    )
            # 1. Commit run (log + LLM summary + session.add_message + obs close)
            try:
                _obs = getattr(self.agent, 'obs', None)
                if _obs:
                    # Always snapshot final ctx → cold resume reliable after eviction
                    _obs.end_step(ctx_checkpoint=ctx.to_checkpoint())
                    _obs.end_run(success=success, final_answer=final_response or "")
            except Exception as e:
                self.live.log(f"[Finalize] OBS end_step failed: {e}", logging.ERROR)

        _obs = getattr(self.agent, 'obs', None)
        if _obs:
            _obs._audit("FINALIZE", ctx.run_id, details={
                "duration_s": round(time.time() - _fin_start, 3),
                "session_id": ctx.session_id,
                "final_status": ctx.status,
            })
        # Outside lock: cleanup
        self._pending_finalize_tasks.pop(ctx.run_id, None)
        if ctx.status == "cancelled":
            self._active_executions.pop(ctx.run_id, None)
            if self._session_last_run.get(ctx.session_id) == ctx.run_id:
                self._session_last_run.pop(ctx.session_id, None)
        elif ctx.status == "running":
            # Normal completion (final_answer) — mark completed, eviction reclaims
            ctx.status = "completed"
        # paused / max_iterations / completed: stay in _active_executions
        # until _evict_if_over_limit reclaims slots.

    # =========================================================================
    # MAIN EXECUTION LOOP
    # =========================================================================

    async def execute(
        self,
        query: str,
        session_id: str,
        max_iterations: int = os.getenv("DEFAULT_MAX_ITERATIONS", 60),
        ctx: "ExecutionContext | None" = None,
        get_ctx: bool = False,
        persist_blocking: bool = False,
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
            max_iterations: Max iterations (default 30)
            ctx: Existing ExecutionContext (for resume)
            get_ctx: If True, return (result, ctx) tuple

        Returns:
            str: Final response
            tuple[str, ExecutionContext]: If get_ctx=True
        """
        # ── Dream intercept ──
        if query == "__dream__":
            # V3: Dream runs as its own agent stream, not inside execute()
            report = await self.agent.a_dream()
            result = (
                report.get("report", str(report))
                if isinstance(report, dict)
                else str(report)
            )
            return (result, None) if get_ctx else result

        ctx, session, is_resume = await self._setup_context(query, session_id, max_iterations, ctx)

        max_iterations, trigger_kw, trigger_skill = await self._setup_init(
            ctx, session, query, max_iterations, is_resume
        )

        final_response = None
        success = True
        log = get_logger()
        # 5. Main loop
        try:
            while ctx.current_iteration < ctx.max_iterations:
                if ctx.status == "paused":
                    final_response = (
                        f"Run paused at iteration {ctx.current_iteration}. "
                        f"Use resume_execution() to continue."
                    )
                    success = False
                    break
                if ctx.status == "cancelled":
                    final_response = "Execution cancelled."
                    success = False
                    break
                warning = self._loop_preamble(ctx)
                # Build current tool list
                current_tools = self._get_tool_definitions(ctx)

                messages = self._sanitize_history_for_api(ctx.working_history.copy())

                log.info(f"messages AutoFocus {len(messages)} done")

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
                        max_tokens=16384,
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
                    # ── Loop 1: Classify ──────────────────────────────────────────
                    final_tc = None
                    final_tc, normal_tcs, sub_agent_tcs, think_tcs = self._classify_tool_calls(response.tool_calls, dict_mode=False)
                    normal_tcs.extend(think_tcs)
                    if final_tc and len(response.tool_calls) > 1:
                        ctx.working_history.append(
                            {
                                "role": "system",
                                "content": "Action pattern not valid !!! NO tools called. use final_answer ALONE !",
                            }
                        )
                        continue
                    if final_tc:
                        _obs = getattr(self.agent, 'obs', None)
                        raw = final_tc.function.arguments
                        if _obs:
                            _obs.record_tool_start("final_answer", str(raw))
                        try:
                            args = json.loads(raw) if isinstance(raw, str) else raw
                        except Exception as parse_err:
                            log.error(f"[Engine] final_answer JSON invalid: {parse_err} | raw={str(raw)[:300]}")
                            ctx.working_history.append({
                                "role": "system",
                                "content": f"final_answer JSON invalid ({parse_err}). Retry with: final_answer({{\"answer\":\"...\",\"success\":true}})",
                            })
                            if _obs:
                                _obs.record_tool_end("final_answer", result_summary=f"parse: {parse_err}",
                                                     status="error")
                            continue
                        final_response = args.get("answer", "")
                        success = args.get("success", True)
                        if not final_response:
                            log.warning(f"[Engine] final_answer empty answer field, raw={str(raw)[:200]}")
                            ctx.working_history.append({
                                "role": "system",
                                "content": "final_answer missing 'answer' field. Retry with the actual answer text.",
                            })
                            if _obs:
                                _obs.record_tool_end("final_answer", result_summary="empty answer", status="error")
                            final_response = None
                            success = False
                            continue
                        if _obs:
                            _obs.record_tool_end("final_answer", result_summary="break loop", status="ok")
                        self._narrator.on_summarise()
                        break

                    for tc in normal_tcs + sub_agent_tcs:
                        tc_clean = clean_tc_object(tc)
                        args_preview = ""
                        try:
                            _parsed = json.loads(tc_clean.function.arguments)
                            args_preview = str(_parsed.get(list(_parsed.keys())[0], ""))[:80] if _parsed else ""
                        except Exception:
                            args_preview = str(tc_clean.function.arguments)[:80]
                        self._narrator.on_tool_start(f"{tc_clean.function.name} {args_preview}")

                    # ── Loop 2: Execute in parallel ───────────────────────────────
                    all_tcs = normal_tcs + sub_agent_tcs
                    results = await asyncio.gather(
                        *[self._execute_tool_call(ctx, tc) for tc in all_tcs]
                    )

                    final_in_batch = False
                    for tc, (result, is_final) in zip(all_tcs, results):
                        if is_final:
                            raw = tc.function.arguments
                            try:
                                args = json.loads(raw) if isinstance(raw, str) else raw
                                answer = args.get("answer")
                                if not answer:
                                    raise ValueError(f"missing 'answer' field, raw={str(raw)[:200]}")
                                final_response = answer
                                success = args.get("success", True)
                                final_in_batch = True
                            except Exception as parse_err:
                                log.error(f"[Engine] parallel final_answer failed: {parse_err}")
                                ctx.working_history.append({
                                    "role": "system",
                                    "content": f"final_answer failed ({parse_err}). Retry with valid args.",
                                })
                                success = False
                            break

                    if final_in_batch:
                        self._narrator.on_summarise()
                        break
                    # else: loop continues, model gets correction system-message
                else:
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
                    ctx.status = "max_iterations"
            await self._record_and_finalize(ctx, session, query, final_response, success, trigger_kw, trigger_skill, persist_blocking)
        except Exception as e:
            log.exception("[Engine] CRASH")
            self.live.log(f"[CRASH] {type(e).__name__}: {e}", logging.ERROR)
            success = False
            crash_msg = f"\n\n⚠️ INTERNAL ERROR: {type(e).__name__}: {e}"
            final_response = (
                    final_response + crash_msg) if final_response else f"Agent crashed: {type(e).__name__}: {e}"
        finally:
            # OBS: IMMER end_run, auch bei crash
            _obs = getattr(self.agent, 'obs', None)
            if _obs and _obs._current_run:
                _obs.end_run(success=success, final_answer=(final_response if isinstance(final_response, str) else final_response.final_text_content) if final_response is not None else "CRASH")
        if get_ctx:
            return final_response, ctx
        return final_response

    async def execute_stream(
        self,
        query: str,
        session_id: str,
        max_iterations: int = os.getenv("DEFAULT_MAX_ITERATIONS", 60),
        ctx: "ExecutionContext | None" = None,
        model=None,
        persist_blocking: bool = False,
    ) -> tuple[Callable, ExecutionContext]:
        """
        Initialize execution and return stream generator + context.

        Returns:
            tuple[stream_generator_func, ExecutionContext]
        """

        log = get_logger()
        if query == "__dream__":
            async def _dream_stream_wrapper(ctx):
                async for chunk in self.agent.a_dream_stream():
                    yield chunk
                yield {
                    "type": "done",
                    "success": True,
                    "final_answer": "Dream cycle complete",
                }

            return _dream_stream_wrapper, ctx

        ctx, session, is_resume = await self._setup_context(query, session_id, max_iterations, ctx)

        async def stream_generator(ctx: ExecutionContext):
            """Generator that yields chunks during execution"""
            nonlocal session, max_iterations

            final_response = None
            success = True

            # Context info for ZEN CLI
            agent_name = self.agent.amd.name
            # Determine depth/type (simple heuristic or passed param)
            is_sub = self.is_sub_agent

            if not is_resume:
                # Status sofort raus (vor gather)
                self.live.status_msg = "Initializing (skills + tools + personas)"
                yield {
                    "type": "status",
                    "status_msg": "Initializing (skills + tools + personas)",
                    "agent": agent_name,
                    "iter": 0,
                }


            max_iterations, trigger_kw, trigger_skill = await self._setup_init(
                ctx, session, query, max_iterations, is_resume
            )
            # Helper to enrich chunks
            def enrich(chunk):
                chunk.setdefault("agent", agent_name)
                chunk.setdefault("iter", ctx.current_iteration)
                chunk.setdefault("max_iter", ctx.max_iterations)
                chunk.setdefault("is_sub", is_sub)
                try:
                    chunk.setdefault("tokens_used", self._calculate_context_load(ctx))
                except Exception:
                    chunk.setdefault("tokens_used", 0)
                chunk.setdefault("tokens_max", self._get_max_context_tokens())
                chunk.setdefault("narrator_msg", self.live.narrator_msg)
                chunk.setdefault("narrator_mini_plan", self._narrator._mini.plan_summary)
                chunk.setdefault("status_msg", self.live.status_msg)
                chunk.setdefault(
                    "skills",
                    [s.name for s in ctx.matched_skills] if ctx.matched_skills else [],
                )
                persona = ctx.active_persona
                chunk.setdefault("persona", persona.name if persona else "default")
                chunk.setdefault(
                    "persona_source", persona.source if persona else "default"
                )
                chunk.setdefault(
                    "persona_model", persona.model_preference if persona else "fast"
                )
                chunk.setdefault(
                    "persona_iterations_factor",
                    persona.max_iterations_factor if persona else 1.0,
                )
                return chunk

            try:
                while ctx.current_iteration < ctx.max_iterations:

                    # Check pause — fall through to finalize so system learns
                    if ctx.status == "paused":
                        final_response = (
                            f"Run paused at iteration {ctx.current_iteration}. "
                            f"Use resume_execution() to continue."
                        )
                        success = False
                        yield enrich({"type": "paused", "run_id": ctx.run_id, "answer": final_response})
                        break

                    # Check cancellation — also finalize, ctx will be popped in _finalize_run
                    if ctx.status == "cancelled":
                        final_response = "Execution cancelled."
                        success = False
                        yield enrich({"type": "cancelled", "run_id": ctx.run_id, "answer": final_response})
                        break

                    warning = self._loop_preamble(ctx)
                    # Sofortiges Iteration-Start-Signal (vor LLM-Latenz)
                    yield enrich(
                        {
                            "type": "iteration_start",
                            "iteration": ctx.current_iteration,
                        }
                    )
                    if warning is not None:
                        yield enrich(
                            {"type": "warning", "message": warning.splitlines()[0]}
                        )

                    # Build messages
                    messages = self._sanitize_history_for_api(ctx.working_history.copy())

                    # Get tool definitions
                    tool_definitions = self._get_tool_definitions(ctx)

                    _last_progress = 0
                    current_tools = tool_definitions if tool_definitions else None
                    # ── LLM Call via a_run_llm_completion stream ──
                    collected_content = ""
                    result_message = None

                    multiplex_queue = asyncio.Queue()

                    def narrator_stream_cb(msg):
                        try:
                            multiplex_queue.put_nowait({"_type": "narrator", "msg": msg})
                        except Exception:
                            pass

                    self._narrator.on_live_update_callback = narrator_stream_cb

                    stream_kwargs = {}
                    if model:
                        stream_kwargs["model"] = model
                    else:
                        stream_kwargs["model_preference"] = ctx.active_persona.model_preference
                    if ctx.active_persona.temperature is not None:
                        stream_kwargs["temperature"] = ctx.active_persona.temperature

                    stream_fn = await self.agent.a_run_llm_completion(
                        messages=messages,
                        tools=current_tools,
                        max_tokens=16384,
                        stream=True,
                        with_context=False,
                        get_response_message=True,
                        **stream_kwargs,
                    )

                    async def pump_stream():
                        try:
                            last_tc = None
                            async for item in stream_fn():
                                await multiplex_queue.put({"_type": "chunk", "data": item})
                                if hasattr(item, "tool_calls"):
                                    last_tc = item.tool_calls
                            await multiplex_queue.put({"_type": "done", "tool_calls": last_tc})
                        except ToolValidationError as err:
                            await multiplex_queue.put({"_type": "tool_error", "error": err})
                        except Exception as err:
                            await multiplex_queue.put({"_type": "error", "error": err})

                    pump_task = asyncio.create_task(pump_stream())
                    tool_validation_fail_msg = None

                    try:
                        tool_calls = None
                        while True:
                            item = await multiplex_queue.get()

                            if item["_type"] == "narrator":
                                yield enrich({"type": "narrator", "narrator_msg": item["msg"]})
                                continue

                            if item["_type"] == "done":
                                tool_calls = item["tool_calls"]
                                break

                            if item["_type"] == "tool_error":
                                tool_validation_fail_msg = str(item["error"])
                                self.live.log(f"[Engine] Provider rejected tool call: {tool_validation_fail_msg}",
                                              logging.WARNING)
                                break

                            if item["_type"] == "error":
                                err = item["error"]
                                if "Event loop is closed" in str(err):
                                    yield enrich({"type": "error", "error": str(err)})
                                    final_response = "Stream aborted: event loop closed"
                                    success = False
                                    return
                                raise err

                            data = item["data"]

                            if hasattr(data, 'choices') and data.choices:
                                delta = data.choices[0].delta if data.choices else None
                                if not delta:
                                    continue
                                if hasattr(delta, 'content') and delta.content:
                                    collected_content += delta.content
                                    self._narrator._inspier += delta.content
                                    yield enrich({"type": "content", "chunk": delta.content})
                                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                                    yield enrich({"type": "reasoning", "chunk": delta.reasoning_content})

                            # Content chunk (string)
                            if isinstance(data, str):
                                collected_content += data
                                self._narrator._inspier += data
                                yield enrich({"type": "content", "chunk": data})

                    finally:
                        self._narrator._set_thought(collected_content[:250], moc=False)
                        if not pump_task.done():
                            pump_task.cancel()

                    # ── Handle tool validation failure ──
                    if tool_validation_fail_msg is not None:
                        correction = (
                            "System: your previous tool call was rejected by the provider. "
                            f"Reason: {tool_validation_fail_msg[:400]}\n\n"
                            "Load the missing tool first via list_tools + load_tools, then retry."
                        )
                        ctx.working_history.append({"role": "system", "content": correction})
                        yield enrich({"type": "warning", "message": "Tool rejected — correction injected."})
                        continue

                    # Process tool calls
                    if tool_calls:
                        tool_calls_dicts = []
                        for tc in tool_calls:
                            _f_name = tc.function.name
                            dict_data = tc.to_dict() if hasattr(tc, "to_dict") else {
                                "id": tc.id,
                                "function": {"name":tc.function.name,"arguments": tc.function.arguments},
                                "type": tc.type
                            }
                            tool_calls_dicts.append(dict_data)
                        assistant_msg = {
                            "role": "assistant",
                            "content": collected_content,
                            "tool_calls": tool_calls_dicts,
                        }
                        ctx.working_history.append(assistant_msg)

                        # ── Loop 1: Classify ──────────────────────────────────────────
                        final_tc, normal_tcs, sub_agent_tcs, think_tcs = self._classify_tool_calls(tool_calls, dict_mode=False)

                        # Emit tool_start for all
                        for tc in (
                            normal_tcs
                            + think_tcs
                            + sub_agent_tcs
                            + ([final_tc] if final_tc else [])
                        ):
                            f_name = tc.function.name
                            f_args = tc.function.arguments
                            f_id = tc.id
                            # Notify narrator so external systems see tool context
                            args_preview = ""
                            try:
                                _parsed = json.loads(f_args) if isinstance(f_args, str) else f_args
                                args_preview = str(_parsed.get(list(_parsed.keys())[0], ""))[:80] if _parsed else ""
                            except Exception:
                                args_preview = str(f_args)[:80]
                            self._narrator.on_tool_start(f"{f_name} {args_preview}")
                            yield enrich(
                                {"type": "tool_start", "name": f_name, "args": f_args, "id": f_id}
                            )
                        if final_tc:
                            _obs = getattr(self.agent, 'obs', None)
                            raw = final_tc.function.arguments
                            if _obs:
                                _obs.record_tool_start("final_answer", str(raw))
                            try:
                                args = json.loads(raw) if isinstance(raw, str) else raw
                            except Exception as parse_err:
                                log.error(f"[Engine] final_answer JSON invalid: {parse_err} | raw={str(raw)[:300]}")
                                ctx.working_history.append({
                                    "role": "system",
                                    "content": f"final_answer JSON invalid ({parse_err}). Retry with: final_answer({{\"answer\":\"...\",\"success\":true}})",
                                })
                                if _obs:
                                    _obs.record_tool_end("final_answer", result_summary=f"parse: {parse_err}",
                                                         status="error")
                                continue
                            final_response = args.get("answer", "")
                            success = args.get("success", True)
                            if not final_response:
                                log.warning(f"[Engine] final_answer empty answer field, raw={str(raw)[:200]}")
                                ctx.working_history.append({
                                    "role": "system",
                                    "content": "final_answer missing 'answer' field. Retry with the actual answer text.",
                                })
                                if _obs:
                                    _obs.record_tool_end("final_answer", result_summary="empty answer", status="error")
                                final_response = None
                                success = False
                                continue
                            if _obs:
                                _obs.record_tool_end("final_answer", result_summary="break loop", status="ok")
                            self._narrator.on_summarise()
                            yield enrich(
                                {
                                    "type": "final_answer",
                                    "answer": final_response,
                                    "success": success,
                                }
                            )
                            break


                        # ── Loop 2: Execute in parallel ───────────────────────────────

                        # 2a) Normal tools: gather, no streaming needed
                        async def _run_normal(tc):
                            result, is_final = await self._execute_tool_call(ctx, tc)
                            return tc, result, is_final

                        normal_results = await asyncio.gather(
                            *[_run_normal(tc) for tc in normal_tcs]
                        )

                        for tc, result, is_final in normal_results:
                            f_name = tc.get("function", {}).get("name", "")
                            f_id = tc.get("id", tc.get("function", {}).get("id", f_name))
                            yield enrich(
                                {
                                    "type": "tool_result",
                                    "name": f_name,
                                    "is_final": is_final,
                                    "result": str(result),
                                    "id": f_id
                                }
                            )

                        # 2a- Think tools: stream chunks as events
                        for tc in think_tcs:
                            f_name = tc.get("function", {}).get("name", "")
                            f_args_raw = tc.get("function", {}).get("arguments", "{}")
                            try:
                                f_args = (
                                    json.loads(f_args_raw)
                                    if isinstance(f_args_raw, str)
                                    else f_args_raw
                                )
                            except Exception:
                                f_args = {}

                            async for chunk_event in self._execute_think_streaming(
                                ctx, tc.get("id"), f_args
                            ):
                                yield chunk_event
                        # 2b) Sub-agent tools: parallel tasks + merged chunk draining
                        if sub_agent_tcs:
                            sub_tasks = {
                                asyncio.create_task(self._execute_tool_call(ctx, tc)): tc
                                for tc in sub_agent_tcs
                            }

                            pending = set(sub_tasks.keys())
                            while pending:
                                # Drain any queued chunks
                                while not self._sub_agent_manager._chunk_queue.empty():
                                    try:
                                        sub_chunk = (
                                            self._sub_agent_manager._chunk_queue.get_nowait()
                                        )
                                        if sub_chunk.get("type") != "_sub_done":
                                            yield sub_chunk
                                    except asyncio.QueueEmpty:
                                        break

                                # Check for completed tasks
                                done = {t for t in pending if t.done()}
                                for task in done:
                                    tc = sub_tasks[task]
                                    f_name = tc.get("function", {}).get("name", "")
                                    result, is_final = await task
                                    yield enrich(
                                        {
                                            "type": "tool_result",
                                            "name": f_name,
                                            "is_final": is_final,
                                            "result": str(result),
                                        }
                                    )
                                pending -= done

                                if pending:
                                    # Yield control briefly so other tasks can progress
                                    try:
                                        sub_chunk = await asyncio.wait_for(
                                            self._sub_agent_manager._chunk_queue.get(),
                                            timeout=0.05,
                                        )
                                        if sub_chunk.get("type") != "_sub_done":
                                            yield sub_chunk
                                    except asyncio.TimeoutError:
                                        pass

                            # Final drain
                            while not self._sub_agent_manager._chunk_queue.empty():
                                try:
                                    sub_chunk = (
                                        self._sub_agent_manager._chunk_queue.get_nowait()
                                    )
                                    if sub_chunk.get("type") != "_sub_done":
                                        yield sub_chunk
                                except asyncio.QueueEmpty:
                                    break

                    else:
                        if collected_content:
                            final_response = collected_content
                            yield enrich({"type": "final_answer", "answer": final_response})
                            break

                # Handle max iterations
                if final_response is None:
                    if self.is_sub_agent:
                        (
                            final_response,
                            should_mark_resumable,
                        ) = await self._handle_sub_agent_max_iterations(
                            ctx, query, max_iterations
                        )
                        success = False
                        if should_mark_resumable:
                            ctx.status = "max_iterations"
                    else:
                        final_response = self._handle_max_iterations(ctx, query)
                        success = False
                        ctx.status = "max_iterations"
                    yield enrich({"type": "max_iterations", "answer": final_response})

                # Narrator cleanup (billig, muss jetzt)
                self._narrator.on_live_update_callback = None

                # Dezenter Hinweis dass Post-Processing läuft
                yield enrich(
                    {
                        "type": "post_processing",
                        "status_msg": "Saving context",
                    }
                )

                # Finalize: commit + skills + persona + learning.
                # Default: background (user sieht done SOFORT).
                # persist_blocking=True: await für crash-safety.
                await self._record_and_finalize(ctx, session, query, final_response, success, trigger_kw, trigger_skill, persist_blocking)
            except Exception as e:
                log.exception("[Engine] CRASH")
                self.live.log(f"[CRASH] {type(e).__name__}: {e}", logging.ERROR)
                success = False
                crash_msg = f"\n\n⚠️ INTERNAL ERROR: {type(e).__name__}: {e}"
                final_response = (
                        final_response + crash_msg) if final_response else f"Agent crashed: {type(e).__name__}: {e}"
            finally:
                # OBS: IMMER end_run, auch bei crash
                _obs = getattr(self.agent, 'obs', None)
                if _obs and _obs._current_run:
                    _obs.end_run(success=success, final_answer=final_response or "CRASH")
            yield enrich(
                {"type": "done", "success": success, "final_answer": final_response}
            )

        _model_for_call = (
            getattr(ctx.active_persona, "model_preference", "fast") == "fast"
            and self.agent.amd.fast_llm_model
            or self.agent.amd.complex_llm_model
        )
        _is_ollama = _model_for_call.startswith("ollama") or _model_for_call.startswith(
            "cerebras"
        )
        if _is_ollama:

            async def _non_stream_fallback(ctx: ExecutionContext):
                async for chunk in self._execute_generator(
                    ctx,
                    session,
                    query,
                    max_iterations,
                    is_resume,
                    None,
                    None,
                    persist_blocking,
                    model,
                ):
                    yield chunk

            return _non_stream_fallback, ctx
        return stream_generator, ctx

    async def _execute_generator(
        self,
        ctx: "ExecutionContext",
        session,
        query: str,
        max_iterations: int,
        is_resume: bool,
        trigger_kw,
        trigger_skill,
        persist_blocking: bool = False,
        model=None,
    ) -> "AsyncGenerator[dict, None]":
        """
        Non-streaming execution that yields chunks compatible with stream_generator.
        LLM is called with stream=False, but output is chunked for consumer compatibility.
        """

        log = get_logger()
        final_response = None
        success = True
        agent_name = self.agent.amd.name
        is_sub = self.is_sub_agent

        # --- Init (nur wenn nicht resumed) ---
        if not is_resume:
            yield {
                "type": "status",
                "status_msg": "Initializing (skills + tools + personas)",
                "agent": agent_name,
                "iter": 0,
            }
        max_iterations, trigger_kw, trigger_skill = await self._setup_init(
            ctx, session, query, max_iterations, is_resume
        )

        # --- enrich (same as stream_generator) ---
        def enrich(chunk):
            chunk.setdefault("agent", agent_name)
            chunk.setdefault("iter", ctx.current_iteration)
            chunk.setdefault("max_iter", ctx.max_iterations)
            chunk.setdefault("is_sub", is_sub)
            try:
                chunk.setdefault("tokens_used", self._calculate_context_load(ctx))
            except Exception:
                chunk.setdefault("tokens_used", 0)
            chunk.setdefault("tokens_max", self._get_max_context_tokens())
            chunk.setdefault("narrator_msg", self.live.narrator_msg)
            chunk.setdefault("narrator_mini_plan", self._narrator._mini.plan_summary)
            chunk.setdefault("status_msg", self.live.status_msg)
            chunk.setdefault(
                "skills",
                [s.name for s in ctx.matched_skills] if ctx.matched_skills else [],
            )
            persona = ctx.active_persona
            chunk.setdefault("persona", persona.name if persona else "default")
            chunk.setdefault("persona_source", persona.source if persona else "default")
            chunk.setdefault(
                "persona_model", persona.model_preference if persona else "fast"
            )
            chunk.setdefault(
                "persona_iterations_factor",
                persona.max_iterations_factor if persona else 1.0,
            )
            return chunk

        # --- Narrator callback (fire-and-forget via yield queue) ---
        narrator_pending = []

        def narrator_cb(msg):
            narrator_pending.append(msg)

        self._narrator.on_live_update_callback = narrator_cb

        def _flush_narrator():
            """Yield-ready list of narrator chunks, then clear."""
            chunks = [
                enrich({"type": "narrator", "narrator_msg": m}) for m in narrator_pending
            ]
            narrator_pending.clear()
            return chunks

        try:
            try:
                # --- Main loop ---
                while ctx.current_iteration < ctx.max_iterations:
                    # Flush narrator updates that accumulated
                    for nc in _flush_narrator():
                        yield nc

                    # Pause / Cancel — fall through to finalize
                    if ctx.status == "paused":
                        final_response = (
                            f"Run paused at iteration {ctx.current_iteration}. "
                            f"Use resume_execution() to continue."
                        )
                        success = False
                        yield enrich({"type": "paused", "run_id": ctx.run_id, "answer": final_response})
                        break
                    if ctx.status == "cancelled":
                        final_response = "Execution cancelled."
                        success = False
                        yield enrich({"type": "cancelled", "run_id": ctx.run_id, "answer": final_response})
                        break

                    warning = self._loop_preamble(ctx)

                    yield enrich(
                        {"type": "iteration_start", "iteration": ctx.current_iteration}
                    )

                    current_tools = self._get_tool_definitions(ctx)
                    messages = self._sanitize_history_for_api(ctx.working_history.copy())

                    # --- LLM Call (non-streaming) ---
                    try:
                        llm_kwargs = {
                            "stream": False,
                            "get_response_message": True,
                            "with_context": False,
                        }
                        if model:
                            llm_kwargs["model"] = model
                        else:
                            llm_kwargs["model_preference"] = (
                                ctx.active_persona.model_preference
                            )
                        if ctx.active_persona.temperature is not None:
                            llm_kwargs["temperature"] = ctx.active_persona.temperature

                        response = await self.agent.a_run_llm_completion(
                            messages=messages,
                            tools=current_tools,
                            **llm_kwargs,
                        )
                    except Exception as e:
                        self.live.error = str(e)
                        self.live.enter(AgentPhase.ERROR, f"LLM Error: {e}")
                        final_response = f"Es ist ein Fehler aufgetreten: {str(e)}\n\nIch konnte die Aufgabe leider nicht abschließen."
                        success = False
                        yield enrich({"type": "error", "error": str(e)})
                        break

                    # Flush narrator after LLM call
                    for nc in _flush_narrator():
                        yield nc

                    if not response:
                        self._empty_streak = getattr(self, "_empty_streak", 0) + 1
                        log.warning(f"[Engine] empty LLM response (streak={self._empty_streak})")
                        if self._empty_streak >= 3:
                            final_response = "LLM lieferte 3x in Folge leere Response. Abbruch."
                            success = False
                            yield enrich({"type": "error", "error": final_response})
                            break
                        continue
                    self._empty_streak = 0

                    # --- Extract content + reasoning ---
                    content = response.content or ""
                    reasoning = getattr(response, "reasoning_content", None) or ""

                    # Emit reasoning as block
                    if reasoning:
                        yield enrich({"type": "reasoning", "chunk": reasoning})

                    # Add assistant message to history
                    msg_dict = {"role": "assistant", "content": content}
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        msg_dict["tool_calls"] = response.tool_calls

                    ctx.working_history.append(msg_dict)

                    self._narrator._set_thought(content[:250], moc=False)
                    self._narrator._inspier = content

                    # --- Process tool calls ---
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        # Classify
                        final_tc, normal_tcs, sub_agent_tcs, think_tcs = self._classify_tool_calls(response.tool_calls, dict_mode=False)

                        # Emit tool_start for all
                        for tc in (
                            normal_tcs
                            + think_tcs
                            + sub_agent_tcs
                            + ([final_tc] if final_tc else [])
                        ):
                            f_name = tc.function.name
                            try:
                                f_args = tc.function.arguments
                            except Exception:
                                f_args = "{}"
                            # Notify narrator so external systems see tool context
                            args_preview = ""
                            try:
                                _parsed = json.loads(f_args) if isinstance(f_args, str) else f_args
                                args_preview = str(_parsed.get(list(_parsed.keys())[0], ""))[:80] if _parsed else ""
                            except Exception:
                                args_preview = str(f_args)[:80]
                            self._narrator.on_tool_start(f"{f_name} {args_preview}")
                            yield enrich(
                                {"type": "tool_start", "name": f_name, "args": f_args}
                            )

                        # Solo tool check
                        if final_tc and len(response.tool_calls) > 1:
                            ctx.working_history.append(
                                {
                                    "role": "system",
                                    "content": "Action pattern not valid !!! NO tools called. use final_answer ALONE !",
                                }
                            )
                            continue
                        if final_tc:
                            _obs = getattr(self.agent, 'obs', None)
                            raw = final_tc.function.arguments
                            if _obs:
                                _obs.record_tool_start("final_answer", str(raw))
                            try:
                                args = json.loads(raw) if isinstance(raw, str) else raw
                            except Exception as parse_err:
                                log.error(f"[Engine] final_answer JSON invalid: {parse_err} | raw={str(raw)[:300]}")
                                ctx.working_history.append({
                                    "role": "system",
                                    "content": f"final_answer JSON invalid ({parse_err}). Retry with: final_answer({{\"answer\":\"...\",\"success\":true}})",
                                })
                                if _obs:
                                    _obs.record_tool_end("final_answer", result_summary=f"parse: {parse_err}",
                                                         status="error")
                                continue
                            final_response = args.get("answer", "")
                            success = args.get("success", True)
                            if not final_response:
                                log.warning(f"[Engine] final_answer empty answer field, raw={str(raw)[:200]}")
                                ctx.working_history.append({
                                    "role": "system",
                                    "content": "final_answer missing 'answer' field. Retry with the actual answer text.",
                                })
                                if _obs:
                                    _obs.record_tool_end("final_answer", result_summary="empty answer", status="error")
                                final_response = None
                                success = False
                                continue
                            if _obs:
                                _obs.record_tool_end("final_answer", result_summary="break loop", status="ok")
                            self._narrator.on_summarise()
                            yield enrich(
                                {
                                    "type": "final_answer",
                                    "answer": final_response,
                                    "success": success,
                                }
                            )
                            break
                        # Execute normal tools in parallel
                        all_tcs = normal_tcs + sub_agent_tcs
                        results = await asyncio.gather(
                            *[self._execute_tool_call(ctx, tc) for tc in all_tcs]
                        )

                        final_in_batch = False
                        for tc, (result, is_final) in zip(all_tcs, results):
                            f_name = tc.function.name
                            yield enrich(
                                {
                                    "type": "tool_result",
                                    "name": f_name,
                                    "is_final": is_final,
                                    "result": str(result),
                                }
                            )
                            if is_final:
                                raw = tc.function.arguments
                                try:
                                    args = json.loads(raw) if isinstance(raw, str) else raw
                                    answer = args.get("answer")
                                    if not answer:
                                        raise ValueError(f"missing 'answer' field, raw={str(raw)[:200]}")
                                    final_response = answer
                                    success = args.get("success", True)
                                    final_in_batch = True
                                except Exception as parse_err:
                                    log.error(f"[Engine] parallel final_answer failed: {parse_err}")
                                    ctx.working_history.append({
                                        "role": "system",
                                        "content": f"final_answer failed ({parse_err}). Retry with valid args.",
                                    })
                                    success = False
                                break

                        # Think tools
                        for tc in think_tcs:
                            f_args_raw = tc.function.arguments
                            try:
                                f_args = (
                                    json.loads(f_args_raw)
                                    if isinstance(f_args_raw, str)
                                    else f_args_raw
                                )
                            except Exception:
                                f_args = {}
                            async for chunk_event in self._execute_think_streaming(
                                ctx, tc.id, f_args
                            ):
                                yield chunk_event

                        # Flush narrator after tool execution
                        for nc in _flush_narrator():
                            yield nc

                        if final_response is not None:
                            self._narrator.on_summarise()
                            break
                    else:
                        # No tool calls — content IS the final answer
                        if content:
                            final_response = content
                            yield enrich({"type": "content", "chunk": content})
                            yield enrich({"type": "final_answer", "answer": final_response})
                            break

                # --- Max iterations ---
                if final_response is None:
                    if self.is_sub_agent:
                        (
                            final_response,
                            should_mark_resumable,
                        ) = await self._handle_sub_agent_max_iterations(
                            ctx, query, max_iterations
                        )
                        success = False
                        if should_mark_resumable:
                            ctx.status = "max_iterations"
                    else:
                        final_response = self._handle_max_iterations(ctx, query)
                        success = False
                        ctx.status = "max_iterations"
                    yield enrich({"type": "max_iterations", "answer": final_response})

                # --- Post-processing ---
                yield enrich({"type": "post_processing", "status_msg": "Saving context"})

                await self._record_and_finalize(ctx, session, query, final_response, success, trigger_kw, trigger_skill, persist_blocking)
            except Exception as e:
                log.exception("[Engine] CRASH")
                self.live.log(f"[CRASH] {type(e).__name__}: {e}", logging.ERROR)
                success = False
                crash_msg = f"\n\n⚠️ INTERNAL ERROR: {type(e).__name__}: {e}"
                final_response = (
                        final_response + crash_msg) if final_response else f"Agent crashed: {type(e).__name__}: {e}"
            finally:
                # OBS: IMMER end_run, auch bei crash
                _obs = getattr(self.agent, 'obs', None)
                if _obs and _obs._current_run:
                    _obs.end_run(success=success, final_answer=final_response or "CRASH")

            ctx.status = "paused"
            yield enrich(
                {"type": "done", "success": success, "final_answer": final_response}
            )

        finally:
            self._narrator.on_live_update_callback = None

    async def _setup_context(
        self,
        query: str,
        session_id: str,
        max_iterations: int,
        ctx: "ExecutionContext | None",
    ) -> "tuple[ExecutionContext, Session, bool]":
        """
        Phase 1: Session, context object, tracking, sub-agent manager.
        Cheap — no LLM, no disk. Safe to call before first yield.
        """
        session = await self.agent.session_manager.get_or_create(session_id)
        is_resume = ctx is not None
        if ctx is None:
            ctx = ExecutionContext()

        if (
            not is_resume
            and hasattr(self.agent, "amd")
            and hasattr(self.agent.amd, "context_config")
        ):
            ctx.context_config = self.agent.amd.context_config

        self._active_executions[ctx.run_id] = ctx
        self._session_last_run[session_id] = ctx.run_id
        ctx.session_id = session_id
        ctx.query = query

        if not is_resume:
            self._evict_if_over_limit(exclude_run_id=ctx.run_id)

        await self._wait_for_pending_finalize(session_id)

        if not self.is_sub_agent and not is_resume:
            self._sub_agent_manager = SubAgentManager(
                parent_engine=self, parent_session=session, is_sub_agent=False
            )
        else:
            if self.sub_agent_output_dir:
                session.vfs = RestrictedVFSWrapper(
                    session.vfs, self.sub_agent_output_dir
                )

        self._current_session = session
        # Wire VFS → Obs delta tracking
        _obs = getattr(self.agent, 'obs', None)
        if _obs is not None:
            _obs.hook_vfs(session.vfs)
        return ctx, session, is_resume

    async def _setup_init(
        self,
        ctx: "ExecutionContext",
        session: "Session",
        query: str,
        max_iterations: int,
        is_resume: bool,
    ) -> "tuple[int, str | None, str | None]":
        """
        Phase 2: parallel_init, history, live state, narrator.
        Expensive — skill matching, tool loading, embeddings.
        Returns (max_iterations, trigger_kw, trigger_skill).
        """
        trigger_kw, trigger_skill = None, None
        _setup_t0 = time.time()
        if not is_resume:
            max_iterations, trigger_kw, trigger_skill = await self._parallel_init(
                ctx, session, query, max_iterations
            )
            system_prompt = self._build_system_prompt(ctx, session)
            history_depth = 2 if self.is_sub_agent else 6
            permanent_history = session.get_history_for_llm(last_n=history_depth)
            ctx.working_history = [
                {"role": "system", "content": system_prompt},
                *permanent_history,
                {"role": "user", "content": query},
            ]
            ctx.max_iterations = max_iterations
        # Live state
        agent_type = "SUB-AGENT" if self.is_sub_agent else "MAIN"
        action = "Resuming" if is_resume else "Start"
        self.live.run_id = ctx.run_id
        self.live.max_iterations = ctx.max_iterations
        self.live.t_start = time.time()
        self.live.skills = (
            [s.name for s in ctx.matched_skills] if ctx.matched_skills else []
        )
        self.live.tools_loaded = (
            ctx.get_dynamic_tool_names() if ctx.dynamic_tools else []
        )
        self.live.persona = ctx.active_persona.name
        self.live.enter(
            AgentPhase.INIT,
            f"{action} [{agent_type}] {ctx.run_id}: {query[:80]}",
        )

        if self.agent.obs is None:
            await self.agent.post_init()
        if self.agent.obs:
            self.agent.obs.begin_run(
                ctx.run_id, query, session.session_id,
                persona=ctx.active_persona.name,
                skills=[s.name for s in ctx.matched_skills] if ctx.matched_skills else [],
            )

        # Narrator
        self._narrator.reset(query)
        self._narrator.on_init(query)
        self._narrator.schedule_skills_update(
            query, ctx.working_history, self.skills_manager, ctx=ctx
        )
        self._narrator.schedule_ruleset_update(ctx.working_history, session, ctx)
        if self.agent.obs:
            self.agent.obs._audit("INIT", ctx.run_id, details={
                "duration_s": round(time.time() - _setup_t0, 3) if '_setup_t0' in dir() else 0,
                "persona": ctx.active_persona.name if ctx.active_persona else "default",
                "skills_matched": len(ctx.matched_skills) if ctx.matched_skills else 0,
            })
        return max_iterations, trigger_kw, trigger_skill

    async def _parallel_init(
        self,
        ctx: "ExecutionContext",
        session: "AgentSessionV2",
        query: str,
        max_iterations: int,
    ) -> tuple[int, str | None, str | None]:
        """
        Run heavy init ops in parallel: skill matching, tool relevance,
        persona loading. Followed by sequential preload + persona routing.

        Returns:
            (updated_max_iterations, trigger_kw, trigger_skill)
        """

        async def _match_skills():
            try:
                return await self.skills_manager.match_skills_async(
                    query, max_results=MAX_PARALLEL_SKILLS
                )
            except Exception as _skill_err:
                self.live.log(
                    f"[Skills] async match failed "
                    f"({type(_skill_err).__name__}), fallback keyword: "
                    f"{_skill_err}",
                    logging.WARNING,
                )
                return self.skills_manager.match_skills(
                    query, max_results=MAX_PARALLEL_SKILLS
                )

        async def _calc_tool_relevance():
            await asyncio.to_thread(self._calculate_tool_relevance, ctx, query)

        async def _load_personas():
            if not self._personas_loaded:
                await asyncio.to_thread(
                    self._persona_router.load_learned_personas, session
                )
                self._personas_loaded = True

        matched_skills_result, _, _ = await asyncio.gather(
            _match_skills(),
            _calc_tool_relevance(),
            _load_personas(),
            return_exceptions=False,
        )
        ctx.matched_skills = matched_skills_result

        self._preload_skill_tools(ctx, query)

        dreamer_insights = getattr(
            getattr(self.agent.amd, "persona", None), "_dream_insights", None
        )
        ctx.active_persona, trigger_kw, trigger_skill = self._persona_router.route(
            query, ctx.matched_skills, dreamer_insights
        )

        if ctx.active_persona.name != "default":
            max_iterations = ctx.active_persona.apply_max_iterations(max_iterations)
            self.live.log(
                f"Persona: {ctx.active_persona.name} "
                f"(model={ctx.active_persona.model_preference}, "
                f"temp={ctx.active_persona.temperature}, "
                f"max_iter={max_iterations})"
            )

        return max_iterations, trigger_kw, trigger_skill

    async def _record_and_finalize(
        self,
        ctx: ExecutionContext,
        session,
        query: str,
        final_response: str,
        success: bool,
        trigger_kw: str | None,
        trigger_skill: str | None,
        persist_blocking: bool = False,
    ):
        """Common post-loop: stats + finalize."""
        if not hasattr(ctx.active_persona, "stats"):
            ctx.active_persona.stats = PersonaStats()

        ctx.active_persona.stats.record_use(
            source=ctx.active_persona.source,
            query=query,
            success=success,
            iterations_used=ctx.current_iteration,
            iterations_budget=ctx.max_iterations,
            trigger_keyword=trigger_kw,
            trigger_skill=trigger_skill,
        )

        self.live.status_msg = f"done (ok={success}, iters={ctx.current_iteration})"
        self.live.enter(
            AgentPhase.DONE,
            f"Execution [{ctx.run_id}] complete (success={success}, iters={ctx.current_iteration})",
        )

        finalize_coro = self._finalize_run(
            ctx, session, query, final_response, success, trigger_kw, trigger_skill
        )
        if persist_blocking:
            await finalize_coro
        else:
            task = asyncio.create_task(finalize_coro)
            self._pending_finalize_tasks[ctx.run_id] = task

    def _loop_preamble(self, ctx: ExecutionContext) -> str | None:
        """Iteration bookkeeping + loop warning. Returns warning msg or None."""
        _obs = getattr(self.agent, 'obs', None)
        if _obs:
            snapshot = ctx.to_checkpoint() if (ctx.current_iteration % self.agent.obs.snapshot_interval == 0) else None
            _obs.end_step(ctx_checkpoint=snapshot)
        ctx.current_iteration += 1
        if _obs:
            _obs.begin_step(ctx.current_iteration)
        self.live.max_iterations = ctx.max_iterations
        self.live.iteration = ctx.current_iteration
        self._narrator.on_llm_pre_call(len(ctx.working_history))
        self.live.status_msg = (
            f"Thinking (iter {ctx.current_iteration}/{ctx.max_iterations})"
        )
        self.live.enter(
            AgentPhase.LLM_CALL,
            f"iter {ctx.current_iteration}/{ctx.max_iterations}",
        )

        warning = None
        if self._should_warn_loop(ctx):
            warning = ctx.loop_detector.get_intervention_message()
            if ctx.current_iteration >= ctx.max_iterations - 1:
                warning += (
                    "\nThis is the last iteration! must finalize task "
                    "immediately and return a final answer with the current status!"
                )
            ctx.working_history.append({"role": "system", "content": warning})
            ctx.loop_warning_given = True
        ctx.working_history = clean_messages(ctx.working_history)
        return warning

    def _classify_tool_calls(
        self, tool_calls: list, dict_mode: bool = False,
    ) -> tuple[dict | None, list, list, list]:
        """
        Returns (final_tc, normal_tcs, sub_agent_tcs, think_tcs).
        dict_mode=True for execute_stream (dicts), False for execute (objects).
        """
        SOLO_TOOLS = {"final_answer", "shift_focus"}
        final_tc = None
        normal_tcs = []
        sub_agent_tcs = []
        think_tcs = []

        clean = clean_tc_dict if dict_mode else clean_tc_object

        for tc in tool_calls:
            tc = clean(tc)
            f_name = (
                tc.get("function", {}).get("name", "")
                if dict_mode
                else tc.function.name
            )
            if f_name in SOLO_TOOLS:
                final_tc = tc
                break
            elif (
                f_name in ("spawn_sub_agent", "wait_for", "resume_sub_agent")
                and self._sub_agent_manager
            ):
                sub_agent_tcs.append(tc)
            elif f_name == "think":
                think_tcs.append(tc)
            else:
                normal_tcs.append(tc)

        return final_tc, normal_tcs, sub_agent_tcs, think_tcs

    def _extract_final_answer(
        self, tc, dict_mode: bool = False, fallback: str = "",
    ) -> tuple[str, bool]:
        """Returns (answer, success)."""
        try:
            raw = (
                tc.get("function", {}).get("arguments", "{}")
                if dict_mode
                else tc.function.arguments
            )
            args = json.loads(raw) if isinstance(raw, str) else raw
            return args.get("answer", fallback), args.get("success", True)
        except Exception:
            return fallback, True

    def _should_warn_loop(self, ctx: ExecutionContext) -> bool:
        """Check if we should inject a loop warning"""
        if ctx.loop_warning_given:
            return False

        if len(ctx.loop_detector.history) >= 3:
            # Check if last 3 are the same
            last3 = ctx.loop_detector.history[-3:]
            if len(set([str(x)+str(y)for x,y in last3])) == 1:
                return True

        return False

    async def _background_learning_task(
        self,
        query: str,
        tools_used: list,
        final_response: str,
        success: bool,
        matched_skills: list,
        iterations_used: int = 0,
    ):
        """Runs learning and recording in background to not block the UI"""
        try:
            # 1. Record usage stats (Fast)
            self.skills_manager.record_matched_skills_usage(
                matched_skills=matched_skills,
                success=success,
                query=query,
                iterations_used=iterations_used,
            )

            # 2. Learn new skills (Slow - involves LLM)
            if success and len(tools_used) >= 2:
                await self.skills_manager.learn_from_run(
                    query=query,
                    tools_used=tools_used,
                    final_answer=final_response,
                    success=success,
                    llm_completion_func=self.agent.a_run_llm_completion,
                )
            # 3. clasify task für memory system
            # 4. colectct programict data on how the gola was acaeved. and waht didaet worked. ( consomed unessasary time )
            # 5. inital build or extend golbal vfs task map. mit global index
            # datt structure for the task map  first layer classes name like [codeing, conversational, brainstoming, homwork, freelancing, ... mor spesifc]
            # jede kategorie hat dann seinen eigenen ordern. beispel für codeing. diser order hat wider classen spezifische unter order [genral, toolbox, isaa, etc]
            # jeder dieser unter order hat dann die filgenden unter order. history. experianace. so wie die datei _index.json
            # in hostry werden die informationen aus step 4 gespeichert und aktumulirt.
            # um einen ord zu ershaffen der genau sagt das gibt es das habe ich gemacht. so hat es fuctoniert. und was nicht fuktoniert hatte.
            # zusammen brainstomen wie entwder direkt hier mit minimaler latens sinvolle experianaces erstellt werden können. und wie der dreamer dann dise informationen verwendet um experianxes zu erstellen.
            # und dann nicht vergessen in dem setup gucken ob dieser task typneu ist oder existirt. dann ob der type |sein untertype neu ist oder exsitert.
            # und wenn er neu ist. schonmal den neuen order anlagen. wenn er exitiert die experiances "für die task" kopiliren. so wie den order
            # dann als pre context mit dem aent gaben. damit isaa wirklich aitomatisch adaptive lernt.
        except Exception as e:
            self.live.log(f"Background Learning Error: {e}", logging.WARNING)

    # =========================================================================
    # TOOL EXECUTION
    # =========================================================================
    async def _execute_think_streaming(self, ctx: ExecutionContext, f_id, f_args: dict):
        """Execute think tool with streaming — yields chunk events, then final tool_result."""
        f_name = "think"
        # ── Pre-call bookkeeping (same as _execute_tool_call) ──
        async with ctx.lock:
            ctx.tools_used.append(f_name)
            loop_detected = ctx.loop_detector.record(f_name, f_args)

        if loop_detected:
            _obs = getattr(self.agent, 'obs', None)
            if _obs:
                _obs.record_tool_start(f_name, "loop_detected")
                _obs.record_tool_end(f_name, result_summary="loop_intervention", status="ok")
            yield {
                "type": "tool_result",
                "name": f_name,
                "is_final": False,
                "result": ctx.loop_detector.get_intervention_message(),
            }
            return

        self.live.tools[f_id] = ToolExecution(
            name=f_name, args_summary=str(f_args)[:120], t_start=time.time()
        )
        self.live.enter(AgentPhase.TOOL_EXEC)
        self.live.status_msg = f"Calling tool {f_name}"
        self._narrator.on_tool_start(f_name + " " + str(f_args.get("thought", "")[:80]))
        if self.agent.obs: self.agent.obs.record_tool_start(f_name, str(f_args), call_id=f_id)

        thought = f_args.get("thought", "")
        effort = f_args.get("effort", "fast")
        working_history = str(ctx.working_history[-25:])
        vfs_content = ""
        if self._current_session is not None:
            vfs_content = self._current_session.vfs.build_context_string()
        current_user_task = ctx.query

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strategic reasoning assistant embedded inside an AI agent execution loop.\n"
                    "Your role is NOT to execute tasks — you analyze the current situation and guide the agent.\n\n"
                    "## Your output must include:\n"
                    "1. **Situation Assessment** — What has happened so far? Where is the agent stuck or making progress?\n"
                    "2. **Key Insights** — Patterns, risks, or opportunities visible in the history.\n"
                    "3. **Concrete Tips** — Actionable next steps the agent can directly attempt.\n"
                    "4. **Partial Solutions / Hints** — Sketch approaches, pseudo-steps, or known-good patterns relevant to the task.\n"
                    "5. **Pitfalls to Avoid** — Common mistakes or dead ends visible from the current trajectory.\n\n"
                    "Be direct and dense. No filler. The agent will act on your output.\n\n"
                    f"All available tools : {await self._tool_list_tools()}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Original User Task:\n{current_user_task}\n---\n\n"
                    f"## Vfs Content:\n{vfs_content}\n---\n\n"
                    f"## Agent's Working History:\n{working_history}\n---\n\n"
                    f"## Agent's Current Thought:\n{thought}\n---\n\n"
                    "Based on the above, provide your situation assessment, tips, hints, and partial solutions."
                ),
            },
        ]

        # Schedule background tasks
        def _():
            try:


                self._narrator.schedule_skills_update(
                    ctx.query, ctx.working_history, self.skills_manager, ctx=ctx
                )
                self._narrator.schedule_memory_extraction(
                    query=ctx.query,
                    history=ctx.working_history,
                    ctx=ctx,
                    session=self._current_session,
                )
                self._narrator.schedule_ruleset_update(
                    history=ctx.working_history, session=self._current_session, ctx=ctx
                )

            except Exception as e:
                if self.agent.obs: self.agent.obs.record_tool_end(f_name, error=str(e), status="error", call_id=f_id)
        threading.Thread(target=_).start()

        thought_acc = ""
        try:
            stream_response = await self.agent.a_run_llm_completion(
                messages=messages,
                model= os.getenv("BLITZMODEL", os.getenv("LIGHNIGMODEL", self.agent.amd.fast_llm_model)) if effort == "fast" else os.getenv("LIGHNIGMODEL", self.agent.amd.fast_llm_model),
                max_tokens=2048,
                stream=True,
                true_stream=True,
                with_context=False,
                tool_choice="none",
            )

            chunk_buffer = ""
            pause_chars = {".", "\n", ":", ";", "?"}

            async for chunk in stream_response():
                delta = (
                    chunk.choices[0].delta
                    if hasattr(chunk, "choices") and chunk.choices
                    else None
                )
                if delta and hasattr(delta, "content") and delta.content:
                    content = delta.content
                    thought_acc += content
                    chunk_buffer += content
                    self.live.thought = thought_acc[-200:]

                    # ── YIELD CHUNK EVENT ──
                    yield {"type": "reasoning", "content": content, "chunk": content}

                    if (
                        any(pc in content for pc in pause_chars)
                        and len(chunk_buffer) > 40
                    ):
                        clean_sentence = chunk_buffer.strip().replace("\n", " ")
                        self._narrator._inspier += " " + clean_sentence
                        self._narrator.mock("llm_pre")
                        chunk_buffer = ""

            result = thought_acc
        except Exception as e:
            result = thought_acc + str(e) if thought_acc else str(e)
            if self.agent.obs: self.agent.obs.record_tool_end(f_name, error=result, status="error", call_id=f_id)

        self.live.thought = result[-200:] if result else ""

        # ── Post-call: ctx writes unter Lock ──────────────────────────
        async with ctx.lock:
            # Add tool result to working history (if not final_answer)
            # Context Budget Management: Szenario A/B/C
            managed_msg = self._manage_context_budget(ctx, f_name, str(result), f_id)
            ctx.working_history.append(managed_msg)
            ctx.tools_dict.append({"name": f_name, "args": f_args, "result": result})
            if f_name in ["think", "load_tools", "list_tools"]:
                ctx.max_iterations += 1
        try:
            if f_name == "think":
                # think result ist typischerweise der komplette gedankengang
                thinking_content = str(result)
                self._narrator.schedule_think_result(
                    thinking_content=thinking_content,
                    history=ctx.working_history,
                )
            else:
                self._narrator.schedule_tool_end(
                    tool_name=f_name,
                    result_snippet=str(result)[:80],
                    history=ctx.working_history,
                )

        except Exception as e:
            self.live.narrator_msg = "Failed to execute narrator tool post processing"
            if self.agent.obs: self.agent.obs.record_tool_end(f_name, error="Failed to execute narrator tool post processing", status="error", call_id=f_id)
            get_app().debug_rains(e)
            get_app().print(e)

        # Final tool_result with complete thought
        if self.agent.obs: self.agent.obs.record_tool_end(f_name, str(result), status="ok" if not isinstance(result, dict) else ( "ok" if  result.get("success", True) else "error"), call_id=f_id)
        yield {"type": "tool_result", "name": f_name, "is_final": False, "result": result}

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
            args_str = tool_call.function.arguments or "{}"
            f_args = _parse_tool_arguments(args_str)
        except Exception as e:
            return (
                f"System-Error: Invalid JSON in tool arguments: {e}\n"
                r"Your tool arguments must be a single valid JSON object. "
                r"Inside string values, write long text / code / markdown naturally — "
                r"only \" and \\ and control chars (\n, \t) need escaping. "
                r"Do NOT escape single quotes (\'). Do NOT wrap the JSON in ``` fences. "
                f"Raw args received (first 500 chars): {str(tool_call.function.arguments)[:500]}"
            ), False
        if self.agent.obs: self.agent.obs.record_tool_start(f_name, args_str, call_id=f_id)
        self._narrator.on_tool_start(
            f_name + " " + str(f_args.get(list(f_args.keys())[0], "") if f_args else "")
        )

        # ── Pre-call: ctx reads/writes unter Lock ─────────────────────
        async with ctx.lock:
            # Track tool usage
            ctx.tools_used.append(f_name)
            # Loop detection (record before execution)
            loop_detected = ctx.loop_detector.record(f_name, f_args)

        if loop_detected:
            _obs = getattr(self.agent, 'obs', None)
            if _obs:
                _obs.record_tool_start(f_name, "loop_detected")
                _obs.record_tool_end(f_name, result_summary="loop_intervention", status="ok")
            return ctx.loop_detector.get_intervention_message(), False

        result = ""
        is_final = False
        self.live.tools[f_id] = ToolExecution(
            name=f_name, args_summary=str(f_args)[:120], t_start=time.time()
        )
        self.live.enter(AgentPhase.TOOL_EXEC)
        self.live.status_msg = f"Calling tool {f_name}"

        # === STATIC TOOLS ===
        if f_name == "think":

            thought_acc = ""
            try:
                async for chunk in self._execute_think_streaming(ctx, f_id, f_args):
                    # {"type": "tool_result", "name": f_name, "is_final": False, "result": result}
                    # {"type": "reasoning", "content": content, "chunk": content}
                    if not chunk:
                        continue
                    c_type = chunk.get("type", "")
                    is_final = chunk.get("is_final", "")
                    thought_acc = chunk.get("result", "")
                    if c_type == "tool_result":
                        break

                thought = thought_acc
                result = thought
            except Exception as e:
                result = (
                    thought_acc if "thought_acc" in locals() and thought_acc else str(e)
                )
                result += str(e)
                thought = result
            self.live.thought = thought if thought else str(result)
            # Record in AutoFocus

        elif f_name == "final_answer":
            answer = f_args.get("answer", "")
            success = f_args.get("success", True)
            result = answer
            is_final = True
            # Don't record final_answer in AutoFocus

        elif f_name == "shift_focus":
            try:
                result = await self._tool_shift_focus(
                    ctx,
                    f_args.get("summary_of_achievements", ""),
                    f_args.get("next_objective", ""),
                )
            except Exception as e:
                print(e)
                import traceback
                result = traceback.format_exc()

        # === DISCOVERY TOOLS ===
        elif f_name == "list_tools":
            result = await self._tool_list_tools(
                f_args.get("category"),
            )

        elif f_name == "load_tools":
            tools_input: str | list[str] = f_args.get("tools", [])
            result = await self._tool_load_tools(ctx, tools_input)

        # === CODING TOOLS ===
        elif f_name == "write_patch":
            result = await self._tool_write_patch(
                ctx,
                file_path=f_args.get("file_path", ""),
                task=f_args.get("task", ""),
                context_files=f_args.get("context_files", []),
            )

        elif f_name == "write_file":
            result = await self._tool_write_new_file(
                ctx,
                file_path=f_args.get("file_path", ""),
                task=f_args.get("task", ""),
                context_files=f_args.get("context_files", []),
            )

        # === SKILL DISCOVERY TOOLS ===
        elif f_name == "list_skills":
            result = await self._tool_list_skills(
                query=f_args.get("query"),
                include_inactive=f_args.get("include_inactive", False),
            )

        elif f_name == "activate_skill":
            result = await self._tool_activate_skill(
                ctx, f_args.get("skill_id", "")
            )


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
                                f"Result: {sub_result.result if sub_result.result else 'result in files'}\n"
                            )
                        else:
                            result = (
                                f"❌ Sub-Agent failed: {sub_result.error}\n"
                                f"Status: {sub_result.status.value}\n"
                                f"Output dir: {sub_result.output_dir}\n"
                            )
                        # Inject into AutoFocus
                        focus_text = (
                            self._sub_agent_manager.format_results_for_auto_focus(
                                {sub_result.id: sub_result}
                            )
                        )
                        result += focus_text
                    else:
                        # spawn_result is sub_agent_id string
                        result = f"🚀 Sub-Agent gestartet: {spawn_result}\nOutput dir: /sub/{output_dir}\nNutze wait_for('{spawn_result}') um auf das Ergebnis zu warten."

                except Exception as e:
                    result = f"ERROR spawning sub-agent: {str(e)}"

        elif f_name == "sub_agents_status":
            pass

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
                            result_lines.append(f"  Error: {sub_result.error}")

                    result = "\n".join(result_lines)

                    # Inject into AutoFocus
                    focus_text = self._sub_agent_manager.format_results_for_auto_focus(
                        results
                    )
                    result += focus_text

                except Exception as e:
                    result = f"ERROR waiting for sub-agents: {str(e)}"

        elif f_name == "sub_agents_status":
            if not self._sub_agent_manager:
                result = "ERROR: SubAgentManager not initialized."

            else:
                sub_agent_id = f_args.get("sub_agent_id", f_args.get("id", f_args.get("agent")))

                if sub_agent_id:
                    status_data = self._sub_agent_manager.get_status(sub_agent_id)

                    if status_data:
                        lines = [f"📊 Status für Sub-Agent [{sub_agent_id}]:"]

                        for k, v in status_data.items():
                            lines.append(f"  - {k}: {v}")

                        result = "\n".join(lines)

                    else:
                        result = f"❌ Kein Sub-Agent mit ID '{sub_agent_id}' gefunden."

                else:
                    all_status = self._sub_agent_manager.get_all_status()
                    lines = ["📊 Übersicht aller Sub-Agents:"]

                    for category in ["running", "completed"]:
                        lines.append(f"\n--- {category.upper()} ---")

                        if not all_status[category]:
                            lines.append("  (Keine)")

                        for sid, data in all_status[category].items():
                            t_used = data.get('tokens_used', 0)
                            t_budget = data.get('token_budget', 'N/A')
                            status_val = data.get('status', 'unknown')
                            n_msg = data.get('narrator_msg', '')
                            n_str = f"\n    ↳ Aktuell: {n_msg}" if n_msg else ""
                            lines.append(f"  • [{sid}] Status: {status_val} | Tokens: {t_used}/{t_budget}{n_str}")
                    result = "\n".join(lines)

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

        # ── Post-call: ctx writes unter Lock ──────────────────────────
        async with ctx.lock:
            # Add tool result to working history (if not final_answer)
            if not is_final:
                # Context Budget Management: Szenario A/B/C
                managed_msg = self._manage_context_budget(ctx, f_name, str(result), f_id)
                ctx.working_history.append(managed_msg)

                if f_name == "vfs_shell":
                    f_name = f_name+f_args.get("command", " _").split(" ")[0]
                    f_name = f_name.strip()
                ctx.tools_dict.append({"name": f_name, "args": f_args, "result": result})
            if f_name in ["think", "load_tools", "list_tools"]:
                ctx.max_iterations += 1
        try:
            if f_name == "think":
                # think result ist typischerweise der komplette gedankengang
                thinking_content = str(result)
                self._narrator.schedule_think_result(
                    thinking_content=thinking_content,
                    history=ctx.working_history,
                )
                if self.agent.obs: self.agent.obs.record_tool_end(f_name, str(result),
                                                                  status=(( "ok" if result.get("success", True) else "error")
                                                                          if isinstance(result, dict) else "ok"), call_id=f_id)

            else:
                self._narrator.schedule_tool_end(
                    tool_name=f_name,
                    result_snippet=str(result)[:80],
                    history=ctx.working_history,
                )
                if self.agent.obs: self.agent.obs.record_tool_end(f_name, str(result),
                                                                  status=(
                                                                      "error" if result.startswith("Error") else "ok") if not isinstance(
                                                                      result, dict) else (
                                                                      "ok" if result.get("success", True) else "error"), call_id=f_id)

        except Exception as e:
            self.live.narrator_msg = "Failed to execute narrator tool post processing"
            get_app().debug_rains(e)
            get_app().print(e)


        self.live.tools.pop(f_id, None)
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
            model = getattr(self.agent, "amd", None)
            if model and hasattr(model, "model_name"):
                from toolboxv2.mods.isaa.base.llm_router.model_info import ctx_limit
                return ctx_limit(model.model_name)
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
                    total += self._estimate_tokens(
                        tc.get("function", {}).get("arguments", "")
                    )
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
            return self._offload_immediate(
                ctx, session, tool_name, raw_result, tool_call_id, content_hash
            )

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
        return self._offload_immediate(
            ctx, session, tool_name, raw_result, tool_call_id, content_hash
        )

    def _offload_immediate(
        self,
        ctx: ExecutionContext,
        session,
        tool_name: str,
        raw_result: str,
        tool_call_id: str,
        content_hash: str,
    ) -> dict:
        """Szenario C: Sofort ins VFS schreiben, nur Pointer + Preview in History."""
        path = f"/.overflow/{ctx.run_id}_{ctx.current_iteration}_{tool_name}.txt"
        try:
            session.vfs.mkdir("/.overflow", parents=True)
            session.vfs.write(path, raw_result)
        except Exception as e:
            # Fallback: Truncate wenn VFS fehlschlägt
            truncated = (
                raw_result[:2000]
                + f"\n\n... [TRUNCATED, {len(raw_result)} chars total, VFS write failed: {e}]"
            )
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
            path = f"/global/.memory/archive/{ctx.run_id}_{step}_{name}.txt"
            try:
                session.vfs.mkdir("/global/.memory/archive", parents=True)
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
        """Calculate relevance scores for all tools at query start (cached).

        Early-exit falls Cache bereits gefüllt (Resume-Szenario).
        """
        if ctx.tool_relevance_cache:
            return

        all_tools = self.agent.tool_manager.get_all()

        # Batch: vermeide wiederholte isinstance-Checks
        for tool in all_tools:
            name = tool.name
            desc = tool.description or ""
            ctx.tool_relevance_cache[name] = self.skills_manager.score_tool_relevance(
                query=query, tool_name=name, tool_description=desc
            )

            raw_cat = tool.category
            if isinstance(raw_cat, list):
                ctx.tool_category_cache[name] = {c for c in raw_cat if c}
            elif raw_cat:
                ctx.tool_category_cache[name] = {raw_cat}
            else:
                ctx.tool_category_cache[name] = set()

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
        try:
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
            system_prompt = (
                ctx.working_history[0]
                if ctx.working_history
                else {"role": "system", "content": self.agent.amd.system_message}
            )
            ctx.working_history = [
                system_prompt,
                {
                    "role": "system",
                    "content": f"Vorheriger Abschnitt abgeschlossen. Stand: {summary_of_achievements}",
                },
                {"role": "user", "content": f"Neues Ziel: {next_objective}. Fahre fort."},
            ]

            # 5. Trackers zurücksetzen für neue Phase
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
            ctx.max_iterations += 10

            # Optional: Tool-Relevanz für neues Ziel neu berechnen
            self._calculate_tool_relevance(ctx, next_objective)

            return f"Fokus erfolgreich gewechselt. Dein Gedächtnis wurde bereinigt. Nächstes Ziel: {next_objective}"
        except Exception as e:
            import traceback
            return f"Fehler: {e} + {traceback.format_exc()}"

    async def _tool_load_tools(
        self, ctx: ExecutionContext, tools_input: Union[str, List[str]]
    ) -> str:
        """
        Load tools with intelligent slot management.

        - Auto-removes lowest relevance tool when limit reached
        - Triggers partial compression on category change + len > 3
        """
        self.live.status_msg = f"Loading {tools_input}"
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
        # --- Category expansion: name that isn't an exact tool → treat as category ---
        all_tool_entries = self.agent.tool_manager.get_all()
        category_map: dict[str, list[str]] = {}
        for t in all_tool_entries:
            cats = t.category if isinstance(t.category, list) else [t.category]
            for c in cats:
                if c:
                    category_map.setdefault(c.lower(), []).append(t.name)

        expanded: list[str] = []
        for name in names:
            name = name.strip()
            if name in self.agent.tool_manager.list_names():
                expanded.append(name)
            elif name.lower() in category_map:
                expanded.extend(category_map[name.lower()])
            else:
                expanded.append(name)  # keep original → "not found" error below
            names = expanded
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
                            self.live.enter(
                                AgentPhase.COMPRESSING,
                                f"Partial compression ({majority_category} -> {new_category})",
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

        # Build response message with slot usage
        msg_parts = []
        if loaded:
            msg_parts.append(f"Geladen: {', '.join(loaded)}")
        if removed:
            msg_parts.append(f"Auto-entfernt (niedrige Relevanz): {', '.join(removed)}")
        if failed:
            msg_parts.append(f"Fehlgeschlagen: {', '.join(failed)}")
        msg_parts.append(f"Slots: {len(ctx.dynamic_tools)}/{ctx.max_dynamic_tools} used")

        return "\n".join(msg_parts) if msg_parts else "Keine Änderungen"

    async def _tool_list_tools(
        self,
        category: Optional[str] = None,
    ) -> str:
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

    # =========================================================================
    # CODING TOOLS: write_patch / write_file
    # 3-step loop: 1) collect data  2) one-shot LLM generation  3) validate
    # =========================================================================

    _CODING_MODEL = os.getenv("CODING_MODEL", None)  # explicit override

    def _get_coding_model_kwargs(self) -> dict:
        """Get model kwargs for coding LLM calls."""
        if self._CODING_MODEL:
            return {"model": self._CODING_MODEL}
        return {"model_preference": "complex"}

    async def _collect_vfs_context(
        self, ctx: ExecutionContext, file_paths: list[str]
    ) -> str:
        """Read multiple VFS files and return concatenated context string."""
        if not self._current_session:
            return ""
        parts = []
        for path in file_paths:
            path = path.strip()
            if not path:
                continue
            try:
                result = self._current_session.vfs_read(path)
                if isinstance(result, dict) and result.get("success"):
                    content = result.get("content", "")
                    parts.append(f"=== FILE: {path} ===\n{content}\n=== END: {path} ===")
                elif isinstance(result, str):
                    parts.append(f"=== FILE: {path} ===\n{result}\n=== END: {path} ===")
            except Exception as e:
                parts.append(f"=== FILE: {path} === ERROR: {e} ===")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_code_from_md(md_text: str) -> list[tuple[str, str]]:
        """
        Parse markdown code blocks. Returns list of (file_path, code_content).
        Supports:
          ```python:path/to/file.py
          ```path/to/file.py
          ```python
          (no path → returned as ("", content))
        """
        import re
        blocks = []
        # Match ``` with optional lang:path or just path
        pattern = re.compile(
            r'```(?:[a-zA-Z]*:?)?([^\n`]*)\n(.*?)```',
            re.DOTALL,
        )
        for match in pattern.finditer(md_text):
            path_hint = match.group(1).strip().strip(":").strip()
            content = match.group(2)
            # Clean trailing whitespace but preserve internal structure
            content = content.rstrip() + "\n"
            blocks.append((path_hint, content))

        # Fallback: if no code blocks found, treat entire text as code
        if not blocks and md_text.strip():
            blocks.append(("", md_text.strip() + "\n"))

        return blocks

    @staticmethod
    def _validate_python_syntax(code: str, file_path: str = "") -> tuple[bool, str]:
        """Validate Python syntax. Returns (ok, error_message)."""
        if not file_path.endswith(".py"):
            return True, ""  # Skip non-Python files
        try:
            import ast
            ast.parse(code, filename=file_path or "<string>")
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    @staticmethod
    def _validate_patch_applies(original: str, patch_content: str) -> tuple[bool, str]:
        """
        Validate that a patch (full file replacement) is structurally sound.
        Checks: not empty, has content, preserves key structures.
        """
        if not patch_content.strip():
            return False, "Patch is empty"
        if len(patch_content.strip()) < 10:
            return False, "Patch suspiciously short"

        # Check that imports from original are roughly preserved
        import re
        orig_imports = set(re.findall(r'^(?:from|import)\s+\S+', original, re.MULTILINE))
        patch_imports = set(re.findall(r'^(?:from|import)\s+\S+', patch_content, re.MULTILINE))
        missing_imports = orig_imports - patch_imports
        if missing_imports and len(missing_imports) > len(orig_imports) * 0.5:
            return False, f"Patch drops >50% of imports: {', '.join(list(missing_imports)[:5])}"

        return True, ""

    async def _tool_write_new_file(
        self,
        ctx: ExecutionContext,
        file_path: str,
        task: str,
        context_files: list[str] | None = None,
    ) -> str:
        """
        3-step tool: collect context → one-shot generate → validate + write.
        """
        if not file_path or not task:
            return "Error: file_path and task are required."

        if not self._current_session:
            return "Error: No active session."

        self.live.status_msg = f"write_file: generating {file_path}"
        max_retries = 2

        # ── STEP 1: Data Collection ──
        context_str = ""
        if context_files:
            context_str = await self._collect_vfs_context(ctx, context_files)

        for attempt in range(max_retries):
            self.live.status_msg = f"write_file: generating (attempt {attempt + 1}/{max_retries})"

            # ── STEP 2: One-shot LLM generation ──
            system_prompt = (
                "You are a precise code generation engine. Generate a complete file based on the spec.\n"
                f"Output the file inside a single markdown code block:\n"
                f"```python:{file_path}\n<complete file content>\n```\n\n"
                "RULES:\n"
                "- Generate ONLY what the spec requires. No extras.\n"
                "- Follow existing patterns from context files if provided.\n"
                "- Output ONLY the code block, no explanation.\n"
                "- Python 3.10+, type hints, docstrings for public APIs."
            )

            user_prompt = f"## New File: {file_path}\n\n## Spec:\n{task}"
            if context_str:
                user_prompt += f"\n\n## Reference Files:\n\n{context_str}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                response = await self.agent.a_run_llm_completion(
                    messages=messages,
                    stream=False,
                    with_context=False,
                    max_tokens=16384,
                    **self._get_coding_model_kwargs(),
                )

                if not response or not response:
                    if attempt < max_retries - 1:
                        continue
                    return "Error: LLM returned empty response."

                llm_output = response
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                return f"Error: LLM call failed: {e}"

            # ── STEP 3: Parse + Validate ──
            blocks = self._parse_code_from_md(llm_output)
            if not blocks:
                if attempt < max_retries - 1:
                    continue
                return "Error: Could not parse code from LLM output."

            _, file_content = blocks[0]

            # Validate syntax
            syntax_ok, syntax_err = self._validate_python_syntax(file_content, file_path)
            if not syntax_ok:
                if attempt < max_retries - 1:
                    task = f"{task}\n\nPREVIOUS ATTEMPT HAD SYNTAX ERROR: {syntax_err}\nFix this."
                    continue
                return f"Error: Syntax error after {max_retries} attempts: {syntax_err}"

            # ── Write to VFS ──
            try:
                self._current_session.vfs_write(file_path, file_content)
            except Exception as e:
                return f"Error: VFS write failed: {e}"

            lines = len(file_content.splitlines())
            self.live.status_msg = f"write_file: done ({file_path})"

            return (
                f"✅ File created: {file_path}\n"
                f"Lines: {lines}\n"
                f"Attempt: {attempt + 1}/{max_retries}"
            )

        return "Error: All generation attempts failed."

    async def _tool_write_patch(
        self,
        ctx: ExecutionContext,
        file_path: str,
        task: str,
        context_files: list[str] | None = None,
    ) -> str:
        """
        3-step tool: collect → generate unified diff patch → validate via git apply --check → apply.
        NEVER writes a full file replacement. Always produces a .patch (unified diff).
        """
        if not file_path or not task:
            return "Error: file_path and task are required."

        if not self._current_session:
            return "Error: No active session."

        self.live.status_msg = f"write_patch: reading {file_path}"
        max_retries = 3

        # ── STEP 1: Data Collection (programmatic, no agent loop) ──
        try:
            target_result = self._current_session.vfs_read(file_path)
            if isinstance(target_result, dict):
                if not target_result.get("success"):
                    return f"Error: Cannot read {file_path}: {target_result.get('error', 'unknown')}"
                original_content = target_result.get("content", "")
            else:
                original_content = str(target_result)
        except Exception as e:
            return f"Error: Cannot read {file_path}: {e}"

        context_str = ""
        if context_files:
            context_str = await self._collect_vfs_context(ctx, context_files)

        # Resolve real disk path for git apply (VFS may shadow a real file)
        disk_path = None
        try:
            vfs = self._current_session.vfs
            f_entry = vfs.files.get(file_path)
            if f_entry and hasattr(f_entry, "real_path") and f_entry.real_path:
                disk_path = str(f_entry.real_path)
        except Exception:
            pass

        for attempt in range(max_retries):
            self.live.status_msg = f"write_patch: generating diff (attempt {attempt + 1}/{max_retries})"

            # ── STEP 2: One-shot LLM → unified diff ──
            system_prompt = (
                "You are a precise patch generation engine.\n"
                "You receive a source file and a task. Generate a UNIFIED DIFF PATCH.\n\n"
                "OUTPUT FORMAT — strictly:\n"
                "```diff\n"
                f"--- a/{file_path}\n"
                f"+++ b/{file_path}\n"
                "@@ -<start>,<count> +<start>,<count> @@\n"
                " <context line>\n"
                "-<removed line>\n"
                "+<added line>\n"
                " <context line>\n"
                "```\n\n"
                "RULES:\n"
                "- Output a valid unified diff (as produced by `git diff`), nothing else.\n"
                "- Include 3 lines of context around each hunk.\n"
                "- Use multiple hunks if changes are in separate regions.\n"
                "- Line counts in @@ headers MUST be correct.\n"
                "- Do NOT output the full file. Only the diff.\n"
                "- Do NOT add explanations before or after the diff block.\n"
                "- Preserve exact whitespace and indentation of unchanged lines.\n"
                "- If the task requires changes in multiple places, use multiple @@ hunks.\n"
            )

            user_prompt = (
                f"## Target File: {file_path}\n"
                f"## Lines: {len(original_content.splitlines())}\n\n"
                f"```\n{original_content}\n```\n\n"
            )
            if context_str:
                user_prompt += f"## Context Files:\n\n{context_str}\n\n"
            user_prompt += f"## Task:\n{task}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                response = await self.agent.a_run_llm_completion(
                    messages=messages,
                    stream=False,
                    with_context=False,
                    max_tokens=16384,
                    **self._get_coding_model_kwargs(),
                )
                if not response:
                    if attempt < max_retries - 1:
                        continue
                    return "Error: LLM returned empty response for patch generation."
                llm_output = response
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                return f"Error: LLM call failed: {e}"

            # ── Parse diff block from markdown ──
            patch_text = self._extract_diff_from_md(llm_output, file_path)
            if not patch_text:
                if attempt < max_retries - 1:
                    task = f"{task}\n\nPREVIOUS ATTEMPT: output was not a valid unified diff block. Output ONLY a ```diff block."
                    continue
                return "Error: Could not extract unified diff from LLM output."

            # ── STEP 3: Validate via git apply --check ──
            import tempfile
            import subprocess

            # Write original to temp for validation
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_file = os.path.join(tmpdir, os.path.basename(file_path))
                tmp_patch = os.path.join(tmpdir, "change.patch")

                with open(tmp_file, "w", encoding="utf-8") as f:
                    f.write(original_content)

                # Rewrite patch header to use temp-local basename
                basename = os.path.basename(file_path)
                localized_patch = patch_text.replace(
                    f"--- a/{file_path}", f"--- a/{basename}"
                ).replace(
                    f"+++ b/{file_path}", f"+++ b/{basename}"
                )

                with open(tmp_patch, "w", encoding="utf-8") as f:
                    f.write(localized_patch)

                # git init + add so git apply works
                subprocess.run(
                    ["git", "init"], cwd=tmpdir,
                    capture_output=True, timeout=5,
                )
                subprocess.run(
                    ["git", "add", basename], cwd=tmpdir,
                    capture_output=True, timeout=5,
                )

                # --check: dry-run validation
                check_result = subprocess.run(
                    ["git", "apply", "--check", "--verbose", tmp_patch],
                    cwd=tmpdir, capture_output=True, text=True, timeout=10,
                )

                if check_result.returncode != 0:
                    err = (check_result.stderr or check_result.stdout or "unknown error").strip()
                    if attempt < max_retries - 1:
                        task = (
                            f"{task}\n\n"
                            f"PREVIOUS PATCH FAILED `git apply --check`:\n{err}\n"
                            f"Fix the hunk headers and context lines."
                        )
                        continue
                    return f"Error: Patch failed git apply --check after {max_retries} attempts:\n{err}"

                # Apply for real (in temp) to get the result
                apply_result = subprocess.run(
                    ["git", "apply", tmp_patch],
                    cwd=tmpdir, capture_output=True, text=True, timeout=10,
                )

                if apply_result.returncode != 0:
                    err = (apply_result.stderr or "unknown").strip()
                    if attempt < max_retries - 1:
                        task = f"{task}\n\nPATCH APPLY FAILED: {err}"
                        continue
                    return f"Error: git apply failed: {err}"

                # Read patched result
                with open(tmp_file, "r", encoding="utf-8") as f:
                    patched_content = f.read()

            # ── Syntax check on result ──
            syntax_ok, syntax_err = self._validate_python_syntax(patched_content, file_path)
            if not syntax_ok:
                if attempt < max_retries - 1:
                    task = f"{task}\n\nPATCH PRODUCES SYNTAX ERROR: {syntax_err}\nFix the patch."
                    continue
                return f"Error: Patched file has syntax error: {syntax_err}"

            # ── Write patched content to VFS ──
            try:
                self._current_session.vfs_write(file_path, patched_content)
            except Exception as e:
                return f"Error: VFS write failed: {e}"

            # ── Also store the .patch file for reference ──
            patch_vfs_path = file_path + ".patch"
            try:
                self._current_session.vfs_write(patch_vfs_path, patch_text)
            except Exception:
                pass  # non-critical

            lines_before = len(original_content.splitlines())
            lines_after = len(patched_content.splitlines())
            hunks = patch_text.count("@@") // 2
            self.live.status_msg = f"write_patch: done ({file_path})"

            return (
                f"✅ Patch applied to {file_path}\n"
                f"Hunks: {hunks} | Lines: {lines_before} → {lines_after} (Δ{lines_after - lines_before:+d})\n"
                f"Patch saved: {patch_vfs_path}\n"
                f"Validated: git apply --check ✓ | syntax ✓\n"
                f"Attempt: {attempt + 1}/{max_retries}"
            )

        return "Error: All patch attempts failed."

    @staticmethod
    def _extract_diff_from_md(md_text: str, file_path: str) -> str | None:
        """
        Extract a unified diff from markdown output.
        Looks for ```diff ... ``` blocks. Falls back to scanning for --- a/ headers.
        Returns the raw patch text or None.
        """
        import re

        # Try ```diff block first
        match = re.search(r'```diff\n(.*?)```', md_text, re.DOTALL)
        if match:
            patch = match.group(1).strip() + "\n"
            # Sanity: must contain --- and +++
            if "---" in patch and "+++" in patch and "@@" in patch:
                return patch

        # Fallback: find raw unified diff (--- a/... +++ b/... @@)
        match = re.search(
            r'(---\s+a/.*?\n\+\+\+\s+b/.*?\n(?:@@.*?\n(?:[ +\-].*?\n|.*?\n))*)',
            md_text, re.DOTALL,
        )
        if match:
            patch = match.group(1).strip() + "\n"
            if "@@" in patch:
                return patch

        return None

    async def _tool_list_skills(
        self,
        query: Optional[str] = None,
        include_inactive: bool = False,
    ) -> str:
        """List available skills, optionally filtered by query relevance."""
        skills = self.skills_manager.skills
        if not skills:
            return "Keine Skills verfügbar."

        lines = []
        for sid, skill in skills.items():
            if not include_inactive and not skill.is_active():
                continue

            # Query-filter: keyword match
            if query:
                query_lower = query.lower()
                name_match = query_lower in skill.name.lower()
                trigger_match = any(t.lower() in query_lower or query_lower in t.lower()
                                    for t in skill.triggers)
                instr_match = query_lower in skill.instruction.lower()[:200]
                if not (name_match or trigger_match or instr_match):
                    continue

            status = "✅" if skill.is_active() else "⚠️"
            effectiveness = f"{skill.effectiveness:.0%}" if skill.total_uses > 0 else "n/a"
            triggers_str = ", ".join(skill.triggers[:5])
            tools_str = ", ".join(skill.tools_used[:5]) if skill.tools_used else "none"

            lines.append(
                f"{status} [{sid}] {skill.name}\n"
                f"   Triggers: {triggers_str}\n"
                f"   Tools: {tools_str}\n"
                f"   Confidence: {skill.confidence:.2f} | Uses: {skill.total_uses} | Effectiveness: {effectiveness}\n"
                f"   Source: {skill.source}"
            )

        if not lines:
            if query:
                return f"Keine Skills gefunden für '{query}'. Nutze list_skills ohne query um alle zu sehen."
            return "Keine aktiven Skills verfügbar. Nutze include_inactive=true um alle zu sehen."

        return f"=== {len(lines)} Skills ===\n\n" + "\n\n".join(lines)

    async def _tool_activate_skill(
        self, ctx: ExecutionContext, skill_id: str
    ) -> str:
        """Activate a skill: inject instruction into context, preload its tools."""
        skill_id = skill_id.strip()
        skill = self.skills_manager.skills.get(skill_id)

        # Fallback: search by name
        if skill is None:
            for sid, s in self.skills_manager.skills.items():
                if s.name.lower() == skill_id.lower():
                    skill = s
                    skill_id = sid
                    break

        if skill is None:
            return f"Skill '{skill_id}' nicht gefunden. Nutze list_skills um verfügbare Skills zu sehen."

        if not skill.is_active():
            return (
                f"Skill '{skill.name}' ist inaktiv (confidence: {skill.confidence:.2f}, "
                f"threshold: {skill.activation_threshold}). Kann nicht aktiviert werden."
            )

        # Check if already in matched_skills
        if any(s.id == skill_id for s in ctx.matched_skills):
            return f"Skill '{skill.name}' ist bereits aktiv in dieser Execution."

        # Add to matched skills
        ctx.matched_skills.append(skill)

        # Inject skill instruction into working history
        ctx.working_history.append({
            "role": "system",
            "content": (
                f"=== SKILL ACTIVATED: {skill.name} ===\n"
                f"{skill.instruction}\n"
                f"=== END SKILL ==="
            ),
        })

        # Preload recommended tools
        loaded_tools = []
        if skill.tools_used:
            all_tool_names = self.agent.tool_manager.list_names()
            for tool_name in skill.tools_used:
                if tool_name in all_tool_names and tool_name not in ctx.get_dynamic_tool_names():
                    relevance = 0.8  # skill-recommended tools get high relevance
                    ctx.add_tool(tool_name, relevance, "skill")
                    loaded_tools.append(tool_name)

        # Preload tool groups
        if skill.tool_groups:
            for group_name in skill.tool_groups:
                group = self.skills_manager.tool_groups.get(group_name)
                if group:
                    for tool_name in group.tool_names:
                        if tool_name not in ctx.get_dynamic_tool_names():
                            all_tool_names = self.agent.tool_manager.list_names()
                            if tool_name in all_tool_names:
                                ctx.add_tool(tool_name, 0.75, group_name)
                                loaded_tools.append(tool_name)

        msg = f"Skill '{skill.name}' aktiviert. Instruction injected."
        if loaded_tools:
            msg += f"\nTools geladen: {', '.join(loaded_tools)}"
        msg += f"\nSlots: {len(ctx.dynamic_tools)}/{ctx.max_dynamic_tools}"

        self.live.log(f"Skill activated: {skill.name} (+{len(loaded_tools)} tools)")
        return msg

    def _get_tool_definitions(self, ctx: ExecutionContext) -> List[dict]:
        """Build tool definitions for LLM (static + VFS + sub-agent + dynamic)"""
        definitions = []

        # 1. Static tools (always available, not counted in limit)
        definitions.extend(STATIC_TOOLS)

        # 2. Discovery tools
        definitions.extend(DISCOVERY_TOOLS)

        # 2b. Skill discovery tools (always available, not counted in limit)
        definitions.extend(SKILL_DISCOVERY_TOOLS)

        # 2c. Coding tools (always available)
        definitions.extend(CODING_TOOLS)

        # 3. Sub-agent tools (ONLY for main agent, not for sub-agents)
        if not self.is_sub_agent:
            definitions.extend(SUB_AGENT_TOOLS)
            definitions.append(RESUME_SUB_AGENT_TOOL)  # NEW: Resume capability

        # 4. VFS tools (always available, not counted in limit)
        all_tools = self.agent.tool_manager.get_all_litellm()

        # 5. Dynamic tools (from slots)
        dynamic_names = ctx.get_dynamic_tool_names()

        # 5. Filter für VFS + SYSTEM_TOOL_BY_NAME + DYNAMIC SLOTS
        for tool_def in all_tools:
            t_name = tool_def["function"]["name"]
            t_entry = self.agent.tool_manager.get(t_name)

            # Check flags
            is_system = (
                t_entry.flags.get("system_tool_by_name", False)
                if t_entry and t_entry.flags
                else False
            )
            is_vfs = t_name in VFS_TOOL_NAMES
            is_dynamic = t_name in dynamic_names

            if is_system or is_vfs or is_dynamic:
                # Duplikate vermeiden (falls ein Tool in mehreren Listen ist)
                if not any(d["function"]["name"] == t_name for d in definitions):
                    definitions.append(tool_def)

        return definitions

    # =========================================================================
    # PROMPT BUILDING
    # =========================================================================

    def _build_system_prompt(self, ctx: ExecutionContext, session) -> str:
        """Order: STATIC prefix → marker → DYNAMIC suffix.
        Static prefix is byte-stable across runs → provider caching works.
        """
        static_parts: list[str] = []
        dynamic_parts: list[str] = []

        # ═══ STATIC PREFIX (cacheable across runs) ═══
        static_parts.append(self.agent.amd.system_message)

        # Identity — pure static, dynamic_slots_prompt moved out
        if self.is_sub_agent:
            static_parts.append("\n".join([
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
                "Write your result to result.md in your output directory. If result.md already exists, write to _result.md instead.",
                "",
                "RULES:",
                "1. Focus ONLY on the given task",
                "2. Write results to your output directory",
                "3. Use final_answer when finished",
                "4. If something is unclear: make the best assumption and document it",
            ]))
        else:
            static_parts.append("\n".join([
                "IDENTITY: You are FlowAgent, an autonomous execution unit capable of file operations, code execution, and data processing.",
                "",
                "OPERATING PROTOCOL:",
                "1. INITIATIVE: Do not complain about missing tools. If a task requires file access, USE `vfs_view` or `vfs_shell`. If you need to search, USE the memory tools.",
                "2. FORMAT: When asked for data, output ONLY data (JSON/Markdown). Do not use conversational filler ('Here is the data').",
                "3. HONESTY: Differentiate between 'Information missing in context' (Unknown) and 'Factually non-existent' (False). Never apologize.",
                "4. ITERATION: If a step fails, analyze the error in `think()`, then try a different approach. Do not give up immediately.",
                "",
                "- Sub-Agent Management: spawn_sub_agent, wait_for, resume_sub_agent",
                "  → If a sub-agent hits max_iterations but made progress, resume it with more iterations",
            ]))

        static_parts.append(
            "\nIf the task has exceeded 10 iterations and prior summaries exist in the history, "
            "use your second-to-last tool call to invoke `think` — assess your progress so far, "
            "identify what remains, and leave a clear handoff note so work can seamlessly continue "
            "if the run is resumed later."
        )

        static_parts.append("\n--- RUNTIME CONTEXT (varies per run, not cached) ---\n")

        # ═══ DYNAMIC SUFFIX (changes per run/query) ═══
        # Tool slots
        loaded_tool_names = ctx.get_dynamic_tool_names()
        slots_lines = [f"--- DYNAMIC TOOL SLOTS ({len(loaded_tool_names)}/{ctx.max_dynamic_tools} used) ---"]
        if not loaded_tool_names:
            slots_lines.append("No dynamic tools currently loaded.")
        else:
            for i, t_name in enumerate(loaded_tool_names, 1):
                t_entry = self.agent.tool_manager.get(t_name)
                if t_entry:
                    slots_lines.append(f"[{i}] {t_name}{t_entry.args_schema}:")
                    for line in t_entry.description.strip().split("\n"):
                        slots_lines.append(f"    {line}")
                    slots_lines.append(" ---\n")
                else:
                    slots_lines.append(f"[{i}] {t_name}()\n    (Description unavailable)")
        empty_slots = ctx.max_dynamic_tools - len(loaded_tool_names)
        if empty_slots > 0:
            slots_lines.append(f"[+] {empty_slots} empty slots available. Use load_tools() to equip more.")
        slots_lines.append("(if no fitting tool is loaded, discover it with list_tools and load it with load_tools!)")
        dynamic_parts.append("\n".join(slots_lines))

        # Categories
        all_tools = self.agent.tool_manager.get_all()
        categories = set()
        for t in all_tools:
            if t.category:
                if isinstance(t.category, list):
                    categories.update(c for c in t.category if c)
                else:
                    categories.add(t.category)
        cat_list = ", ".join(sorted(categories)) if categories else "keine"
        dynamic_parts.append(f"- Context Access: {cat_list}")

        if ctx.matched_skills:
            dynamic_parts.append(self.skills_manager.build_skill_prompt_section(ctx.matched_skills))

        if ctx.active_persona.prompt_modifier:
            dynamic_parts.append(ctx.active_persona.prompt_modifier)

        if ctx.active_persona.verification_level == "strict":
            dynamic_parts.append(
                "\n⚠️ VERIFICATION REQUIRED: After EVERY state-changing action, "
                "verify the result with a read/list/status tool before proceeding."
            )

        return "\n".join(static_parts + dynamic_parts)

    def _sanitize_history_for_api(self, messages: List[dict]) -> List[dict]:
        """Fixes invalid JSON in assistant tool_calls to prevent API 400 errors."""
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    args = tc["function"].get("arguments", "")
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        # ADMIN DEBUG: Hier siehst du im Terminal, was das Modell vermurkst hat

                        print(f"⚠️ ADMIN WARNUNG: Modell hat ungültiges JSON generiert!")
                        print(f"   Tool: {tc['function'].get('name')}")
                        print(f"   Kaputter String: {args}")

                        # Die "Safe-Box": Kaputtes JSON als String kapseln, um API 400 zu verhindern
                        tc["function"]["arguments"] = json.dumps({"_raw_error": args})
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

    async def _commit_run_slow(
        self,
        ctx: ExecutionContext,
        session,
        query: str,
        final_response: str,
        success: bool,
    ):
        """
        Background-phase commit: log-write + LLM summary + session messages.

        WICHTIG: persona stats + skills stats sind jetzt in _finalize_run
        zentralisiert (unter globalen Locks). Hier NICHT mehr doppelt.
        """

        # 1. Archivierung: Speichere den vollen Verlauf im VFS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "/global/.memory/logs"
        log_file = f"{log_dir}/{timestamp}_{ctx.run_id}.md"

        summary = HistoryCompressor.compress_to_summary(ctx.working_history, ctx.run_id)

        try:
            # Sicherstellen, dass Verzeichnis existiert
            session.vfs.mkdir(log_dir, parents=True)

            # Log formatieren
            full_log = [f"# Execution Log: {ctx.run_id}", f"Query: {query}", "-" * 40]

            for msg in ctx.working_history[1:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                content = f"{content}"
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
                full_log.append(
                    f"Persona: {ctx.active_persona.name} (source: {ctx.active_persona.source})"
                )

            def helper(__full_log):
                session.vfs.write(log_file, "\n".join(__full_log))

            helper(full_log)
        except Exception as e:
            import traceback

            self.live.log(
                f"Failed to write execution log: {e} {traceback.format_exc()}",
                logging.WARNING,
            )

        # 2. LLM Summarization (Dynamisch statt Regelbasiert)
        summary_text = "Keine Zusammenfassung."
        try:
            # Nutze schnelles Modell für Zusammenfassung
            narrator_hint = ""
            if self._narrator and self._narrator._mini.plan_summary:
                flags = []
                if self._narrator._mini.repeat:
                    lang = self._narrator._lang
                    flags.append(
                        "Wiederholungen erkannt – fasse diese zusammen"
                        if lang == "de"
                        else "repetitions detected – consolidate these"
                    )
                if self._narrator._mini.drift:
                    lang = self._narrator._lang
                    flags.append(
                        "Plan-Abweichungen erkannt – betone was vom Originalziel abwich"
                        if lang == "de"
                        else "drift detected – highlight deviations from original goal"
                    )
                narrator_hint = (
                    f"Agent-Intent (auto-erkannt): {self._narrator._mini.plan_summary}\n"
                    + ("\n".join(flags) + "\n" if flags else "")
                )
            summary_prompt = (
                f"Analysiere den folgenden Verlauf eines Agenten-Laufs.\n"
                f"Aufgabe: {query}\n"
                f"Status: {'Erfolg' if success else 'Fehlschlag'}\n"
                f"Tools genutzt: {', '.join(ctx.tools_used)}\n\n"
                f"{narrator_hint}"
                f"compressed summary : {summary}"
                f"Erstelle eine prägnante Zusammenfassung mit wichtigen details (max 2-3 Sätze) der durchgeführten Aktionen und des Ergebnisses."
                f"Erwähne erstellte/bearbeitete Dateien."
            )

            # Wir nutzen working_history als Kontext, aber limitiert
            summary_text = await self._narrator.blitz(
                system=summary_prompt, messages=ctx.working_history[-20:]
            )

            if not summary_text:
                summary_text = await self.agent.a_run_llm_completion(
                    messages=ctx.working_history[-20:]
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

        # Persist persona stats
        if ctx.active_persona:
            await self._persist_persona_stats(session, ctx.active_persona)

        # 3. Permanent Speichern
        # User message
        await session.add_message({"role": "user", "content": query})

        # System Summary mit Referenz auf Log-Datei
        summary_msg = {
            "role": "system",
            "content": f"⚡ RUN SUMMARY [{ctx.run_id}]: {summary_text}\n(Full Log: {log_file})",
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
            if hasattr(sched, "_agent_idle_eval"):
                sched._agent_idle_eval.record_activity(self.agent.amd.name)
        except Exception:
            pass

    async def _persist_persona_stats(self, session, persona: PersonaProfile) -> None:
        """Merge updated stats back into the VFS personas store."""
        if persona.name == "default":
            return

        try:
            # Load existing store
            result = session.vfs_read(_VFS_PERSONAS)
            store: dict = json.loads(result["content"]) if result.get("success") else {}
        except Exception:
            store = {}

        key = next(
            (
                k
                for k, p in self._persona_router.personas.items()
                if p.name == persona.name
            ),
            persona.name,
        )

        entry = store.setdefault(key, {})
        entry["profile"] = {
            "name": persona.name,
            "prompt_modifier": persona.prompt_modifier,
            "model_preference": persona.model_preference,
            "temperature": persona.temperature,
            "max_iterations_factor": persona.max_iterations_factor,
            "verification_level": persona.verification_level,
        }
        entry["confidence"] = entry.get("confidence", 1.0)  # builtin personas = 1.0
        entry["stats"] = persona.stats.to_dict()

        try:
            session.vfs_write(_VFS_PERSONAS, json.dumps(store, indent=2))
        except Exception as e:
            get_logger().warning(f"[PersonaStats] persist failed: {e}")

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
        self.live.enter(
            AgentPhase.PAUSED, f"Paused [{execution_id}] at iter {ctx.current_iteration}"
        )
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
        if ctx.session_id in self._session_last_run:
            if self._session_last_run[ctx.session_id] == execution_id:
                del self._session_last_run[ctx.session_id]
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

    def _evict_if_over_limit(self, exclude_run_id: str = ""):
        """
        Two-stage eviction:
        1. Drop completed runs immediately (no resume value).
        2. Cap paused/max_iterations runs at max_resumable, oldest progress first.
        Running contexts are never evicted.
        """
        # Stage 1: completed runs out (no resume needed)
        completed_ids = [
            rid for rid, ctx in self._active_executions.items()
            if ctx.status == "completed" and rid != exclude_run_id
        ]
        for rid in completed_ids:
            ctx = self._active_executions.pop(rid, None)
            if ctx and self._session_last_run.get(ctx.session_id) == rid:
                self._session_last_run.pop(ctx.session_id, None)

        # Stage 2: cap resumable runs
        resumable = [
            (rid, ctx) for rid, ctx in self._active_executions.items()
            if ctx.status in ("paused", "max_iterations") and rid != exclude_run_id
        ]
        if len(resumable) < self.max_resumable:
            return

        resumable.sort(key=lambda x: x[1].current_iteration)
        evict_count = len(resumable) - self.max_resumable + 1
        for i in range(evict_count):
            rid, ctx = resumable[i]
            self._active_executions.pop(rid, None)
            if self._session_last_run.get(ctx.session_id) == rid:
                self._session_last_run.pop(ctx.session_id, None)
            self.live.log(
                f"[MemCap] Evicted paused run {rid} (iter={ctx.current_iteration}) "
                f"from memory. Cold resume still available via obs.",
                logging.DEBUG,
            )

    async def _restore_cold_ctx(self, ctx, session):
        """
        Resolve cold-resume stubs: re-attach matched_skills objects
        and active_persona from their stored IDs/names.

        Called ONLY on cold resume (from disk), not hot resume.
        """
        # 1. Resolve matched_skills from IDs
        if hasattr(ctx, '_cold_skill_ids') and ctx._cold_skill_ids:
            resolved = []
            for sid in ctx._cold_skill_ids:
                skill = self.skills_manager.skills.get(sid)
                if skill:
                    resolved.append(skill)
            ctx.matched_skills = resolved
            del ctx._cold_skill_ids

        # 2. Resolve active_persona from name
        if hasattr(ctx, '_cold_persona_name') and ctx._cold_persona_name:
            # Load learned personas if not done yet
            if not self._personas_loaded:
                self._persona_router.load_learned_personas(session)
                self._personas_loaded = True

            persona_name = ctx._cold_persona_name
            if persona_name in self._persona_router.personas:
                import dataclasses
                ctx.active_persona = dataclasses.replace(
                    self._persona_router.personas[persona_name],
                    source="cold_resume",
                )
            else:
                # Fallback: default persona with the stored name
                ctx.active_persona = PersonaProfile(name=persona_name, source="cold_resume")
            del ctx._cold_persona_name

        # 3. Ensure status is correct
        ctx.status = "running"

    async def resume_from_disk(
        self,
        run_id: str,
        max_iterations: int = 30,
        content: str = "",
        stream: bool = False,
    ):
        """
        Resume an interrupted execution from disk (cold resume).

        Flow:
        1. Load ctx_snapshot from obs layer's live JSONL
        2. Reconstruct ExecutionContext
        3. Validate session exists
        4. Resolve cold stubs (skills, persona)
        5. Re-enter execute() loop

        Args:
            run_id: The interrupted run's ID
            max_iterations: Additional iterations to allow
            content: Optional new user message to inject
            stream: If True, return (stream_func, ctx)

        Returns:
            Final response string or (stream_func, ctx)
        """
        obs = getattr(self.agent, 'obs', None)
        if obs is None:
            return f"Error: ObservabilityLayer not configured on agent"

        # 1. Load from disk
        result = obs.get_resumable_run(run_id)
        if result is None:
            return f"Error: No resumable data found for run_id={run_id}"

        run_record, ctx_snapshot = result

        if ctx_snapshot is None:
            return (
                f"Error: Run {run_id} found ({len(run_record.steps)} steps recorded) "
                f"but no ctx_snapshot available. Cannot resume without execution state. "
                f"Increase obs.snapshot_interval or ensure snapshots are written."
            )

        # 2. Reconstruct ExecutionContext
        ctx = ExecutionContext.from_checkpoint(ctx_snapshot)

        # 3. Validate session
        session_id = ctx.session_id or "default"
        try:
            session = await self.agent.session_manager.get_or_create(session_id)
        except Exception as e:
            return f"Error: Session '{session_id}' restore failed: {e}"

        # 4. Resolve cold stubs
        await self._restore_cold_ctx(ctx, session)

        # 5. Set iteration ceiling
        ctx.max_iterations = ctx.current_iteration + max_iterations

        # 6. Inject new user content if provided
        if content:
            ctx.working_history.append({
                "role": "system",
                "content": "[COLD RESUME] Agent restarted from disk checkpoint. Continuing previous task.",
            })
            ctx.working_history.append({"role": "user", "content": content})
        else:
            ctx.working_history.append({
                "role": "system",
                "content": (
                    "[COLD RESUME] Agent restarted from disk checkpoint at "
                    f"iteration {ctx.current_iteration}. Continue where you left off."
                ),
            })

        # 7. Register in active executions
        self._active_executions[ctx.run_id] = ctx

        # 8. Clean up live file (obs will create a new one on begin_run)
        # Keep the old live file as backup until new run completes
        live_backup = None
        import os
        live_path = os.path.join(obs.obs_dir, f"live_{run_id}.jsonl")
        if os.path.exists(live_path):
            live_backup = live_path + ".bak"
            try:
                os.rename(live_path, live_backup)
            except OSError:
                pass

        self.live.log(
            f"[COLD RESUME] Restored run {run_id} at iter {ctx.current_iteration}, "
            f"session={session_id}, skills={len(ctx.matched_skills)}, "
            f"persona={ctx.active_persona.name}",
            logging.INFO,
        )

        # 9. Enter execution
        if stream:
            return await self.execute_stream(
                query=ctx.query,
                session_id=session_id,
                max_iterations=max_iterations,
                ctx=ctx,
            )

        return await self.execute(
            query=ctx.query,
            session_id=session_id,
            max_iterations=max_iterations,
            ctx=ctx,
        )

    def get_interrupted_runs(self) -> list[dict]:
        """
        List runs that can be cold-resumed from disk.

        Returns:
            List of {run_id, step_count, last_step_id, has_snapshot}
        """
        obs = getattr(self.agent, 'obs', None)
        if obs is None:
            return []
        return obs.get_interrupted_runs()



    async def resume(
        self,
        execution_id: str,
        max_iterations: int = os.getenv("DEFAULT_MAX_ITERATIONS", 30),
        content="",
        stream=False,
    ) -> str | tuple[Callable[[...], Any], ExecutionContext] | tuple[str, ExecutionContext]:
        """
           Resume a paused execution (hot) or interrupted execution (cold from disk).

           Hot: run_id found in _active_executions (in-memory, process still alive)
           Cold: run_id NOT in memory → fallback to resume_from_disk via obs layer

           Args:
               execution_id: The run_id of the execution to resume
               max_iterations: Max additional iterations
               content: Additional user input
               stream: If True, return streaming interface

           Returns:
               Final response string or (stream_func, ctx)
           """
        # --- HOT PATH: in-memory resume ---
        ctx = self._active_executions.get(execution_id)
        if ctx is not None:
            if ctx.status not in ("paused", "max_iterations"):
                return f"Error: Execution {execution_id} is not resumable (status: {ctx.status})"

            ctx.status = "running"
            ctx.max_iterations = ctx.current_iteration + max_iterations
            if content:
                ctx.working_history.append({
                    "role": "system",
                    "content": "Continue with old task using new user information's",
                })
                ctx.working_history.append({"role": "user", "content": content})

            if stream:
                return await self.execute_stream(
                    query=ctx.query,
                    session_id=ctx.session_id,
                    max_iterations=max_iterations,
                    ctx=ctx,
                )
            return await self.execute(
                query=ctx.query,
                session_id=ctx.session_id,
                max_iterations=max_iterations,
                ctx=ctx,
            )

        # --- COLD PATH: disk resume ---
        return await self.resume_from_disk(
            run_id=execution_id,
            max_iterations=max_iterations,
            content=content,
            stream=stream,
        )


if __name__ == "__main__":
    print("ExecutionEngine loaded")
    ctx = ExecutionContext()
    print(ctx.run_id)
    from toolboxv2.tests.test_isaa.test_base.test_agent.test_execution_engine import (
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
