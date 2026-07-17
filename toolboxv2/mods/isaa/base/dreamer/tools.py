"""
Dreamer V3 — Tool Definitions

All dream_* tool schemas for the DreamerAgent.
Grouped by phase for clarity.

Author: FlowAgent V3
"""

from typing import List


# ═══════════════════════════════════════════════════════════════════
# BUDGET CALCULATION
# ═══════════════════════════════════════════════════════════════════

_SUB_AGENT_BASE_BUDGET = 800
_SUB_AGENT_PER_RECORD = 800
_SUB_AGENT_MAX_BUDGET = 8000



def calculate_sub_agent_budget(cluster_size: int) -> int:
    """
    Auto-budget for cluster analysis sub-agents.

    Formula: base + cluster_size * per_record, capped at max.
    """
    budget = _SUB_AGENT_BASE_BUDGET + cluster_size * _SUB_AGENT_PER_RECORD
    return min(budget, _SUB_AGENT_MAX_BUDGET)


# ═══════════════════════════════════════════════════════════════════
# SINGLE DREAMER TOOL (replaces 22 dream_* tools)
# ═══════════════════════════════════════════════════════════════════
# The Dreamer does not need 20+ dream_* tools. The Tool-Slot-Manager
# evicts most of them, which breaks the meta-learning cycle.
#
# One master tool:
#   dream_act(action, payload)
#
# The handler routes actions to the existing handle_* methods, validates
# payloads, and keeps old action names available for compatibility.
# ═══════════════════════════════════════════════════════════════════

DREAM_ACT_TOOL = {
    "type": "function",
    "function": {
        "name": "dream_act",
        "description": (
            "Master tool for the Dreamer meta-learning cycle. Use this for all Dreamer actions: "
            "data access, migration, skill/persona/rule creation or rewrite, evolution, cleanup, "
            "taskmap guide writing, persistence. The Dreamer should prefer this single tool instead "
            "of many dream_* tools to avoid Tool-Slot-Manager eviction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "Action to execute. Available actions: get_taskmap, get_all_state, "
                        "migrate_logs, create_skill, create_rule, create_persona, evolve_skill, "
                        "merge_skills, split_skill, compress_skill, cleanup, delete_skill, "
                        "delete_rule, extract_rules, learn_pattern, write_taskmap_guide, add_task_class, update_classify_guide,"
                        "persist_checkpoint."
                    ),
                    "enum": [
                        "get_taskmap",
                        "get_all_state",
                        "migrate_logs",
                        "create_skill",
                        "create_rule",
                        "create_persona",
                        "evolve_skill",
                        "merge_skills",
                        "split_skill",
                        "compress_skill",
                        "cleanup",
                        "delete_skill",
                        "delete_rule",
                        "extract_rules",
                        "learn_pattern",
                        "write_taskmap_guide",
                        "update_classify_guide"
                        "add_task_class"
                        "persist_checkpoint",
                    ],
                },
                "payload": {
                    "type": "object",
                    "description": (
                        "Action-specific payload. Examples: "
                        "get_taskmap={\"task_type\":\"coding\",\"subtype\":\"toolbox\",\"limit\":20}; "
                        "create_skill={\"name\":\"...\",\"triggers\":[\"...\"],\"instruction\":\"...\",\"tools_used\":[...]}; "
                        "create_rule={\"situation\":\"...\",\"intent\":\"...\",\"instructions\":[\"...\"],\"confidence\":0.5}; "
                        "create_persona={\"name\":\"...\",\"prompt_modifier\":\"...\",\"model_preference\":\"fast\",\"temperature\":0.3}; "
                        "write_taskmap_guide={\"task_type\":\"coding\",\"subtype\":\"toolbox\",\"content\":\"...\"}; "
                        "write_taskmap_guide={\"task_type\":\"coding\",\"subtype\":\"toolbox\",\"content\":\"...\"}; "
                        "update_classify_guide={\"additions\":[\"neuer_typ/subtyp: keyword1 keyword2\"]}; " #
                        "add_task_class={\"task_type\":\"coding\",\"subtype\":\"deployment\",\"classification_keywords\":[\"server\",\"docker\"]}; "
                        "persist_checkpoint={}."
                    ),
                },
            },
            "required": ["action"],
        },
    },
}


# Backward-compatible registry helpers.
# External code can still call these; they now expose only the single master tool.
def get_all_dream_tool_definitions() -> list[dict]:
    """Get all dream tool definitions as flat list."""
    return [DREAM_ACT_TOOL]


def get_dream_tool_names() -> list[str]:
    """Get all dream tool names."""
    return [t["function"]["name"] for t in get_all_dream_tool_definitions()]


# ÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍ
# COMPAT SHIM — legacy multi-tool constants for test imports.
# Production uses the single dream_act master tool. These exist
# so test_dreamer_phase_a.py (written pre-migration) can import them.
# ÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍ

def _legacy_tool(name: str, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Legacy shim for {name} — use dream_act master tool instead.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": required or [],
            },
        },
    }


DREAM_DATA_TOOLS    = [_legacy_tool("dream_get_records")]
DREAM_CLUSTER_TOOLS = [_legacy_tool("dream_cluster_records")]
DREAM_SKILL_TOOLS   = [
    _legacy_tool("dream_get_skills"),
    _legacy_tool("dream_create_skill"),
    _legacy_tool("dream_evolve_skill", ["skill_id", "cluster_size", "success_ratio"]),
    _legacy_tool("dream_merge_skills"),
    _legacy_tool("dream_split_skill"),
    _legacy_tool("dream_compress_skill"),
]
DREAM_RULE_TOOLS    = [
    _legacy_tool("dream_get_rules"),
    _legacy_tool("dream_create_rule"),
    _legacy_tool("dream_extract_rules"),
    _legacy_tool("dream_learn_pattern"),
]
DREAM_PERSONA_TOOLS = [
    _legacy_tool("dream_get_personas"),
    _legacy_tool("dream_create_persona"),
    _legacy_tool("dream_evolve_persona"),
    _legacy_tool("dream_prune_personas"),
]
DREAM_CLEANUP_TOOLS = [
    _legacy_tool("dream_cleanup_skills"),
    _legacy_tool("dream_cleanup_rules"),
    _legacy_tool("dream_prune_personas"),
    _legacy_tool("dream_delete_skill"),
    _legacy_tool("dream_delete_rule"),
]
DREAM_PERSIST_TOOLS = [_legacy_tool("dream_persist_checkpoint")]
