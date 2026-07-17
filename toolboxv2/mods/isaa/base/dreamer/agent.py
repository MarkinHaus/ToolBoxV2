"""
Dreamer V3 — Agent Module

Creates and configures the DreamerAgent (a standalone FlowAgent).

Author: FlowAgent V3
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from toolboxv2.mods.isaa.base.dreamer.harvest import RunRecord

_log = logging.getLogger("isaa.dreamer_v3.agent")

# ═══════════════════════════════════════════════════════════════════
# DEFAULT MODEL
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_DREAMER_MODEL = "groq/llama-3.1-70b-versatile"

# Path to the skill-creator SKILL.md used as reference for sub-agents
_SKILLS_GUIDE_PATHS = [
    "/mnt/skills/examples/skill-creator/SKILL.md",
    # Fallback paths for different environments
    os.path.join(os.path.dirname(__file__), "fallback_skills_guide.md"),
]


def _load_skills_guide() -> str:
    """Load the dreamer_skills_guide content from disk."""
    for path in _SKILLS_GUIDE_PATHS:
        try:
            return Path(path).read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError):
            continue

    # Minimal fallback if file not found
    return (
        "# Skill Writing Guide (Fallback)\n\n"
        "## Anatomy of a Skill\n"
        "- SKILL.md with YAML frontmatter (name, description) + Markdown body\n"
        "- Keep instructions clear, numbered, 4-6 steps\n"
        "- Include triggers (keywords that activate the skill)\n"
        "- List tools_used for context\n"
        "- Add failure patterns as warnings\n\n"
        "## Writing Style\n"
        "- Use imperative form\n"
        "- Be specific, not generic\n"
        "- Include examples where helpful\n"
    )


# ═══════════════════════════════════════════════════════════════════
# DREAMER PERSONA
# ═══════════════════════════════════════════════════════════════════

# Keywords that the PersonaRouter uses to match this persona.
# These match the dream_* tool names and dreamer-specific terms
# so the router always picks this persona for dream queries.
DREAMER_PERSONA_KEYWORDS = [
    "dream", "dreamer", "meta-learning", "skill evolution",
    "cleanup", "pruning", "bloat", "cluster", "harvest", "migrate",
    "dream_act", "taskmap", "reconcile", "checkpoint",
]

# The persona profile applied to the DreamerAgent.
# temperature=0.2 → deterministic analytical reasoning
# model_preference="fast" → Dreamer uses cheaper model (lots of tool calls)
# max_iterations_factor=1.5 → Dreamer needs more iterations (10 phases)
# verification_level="basic" → trust tool outputs, don't double-check
DREAMER_PERSONA_PROFILE = {
    "name": "dreamer_analyst",
    "prompt_modifier": (
        "\nPERSONA: Dreamer Analyst\n"
        "SPRACHE: Antworte in der Sprache des Parent-Agents (DE/EN gemischt OK).\n"
        "ROLLE: Offline Meta-Learning Engine. Analysiert Logs, optimiert Skills, "
        "räumt auf, extrahiert Patterns. Kein User wartet — Gründlichkeit vor Geschwindigkeit.\n"
        "KERNVERHALTEN:\n"
        "- Denke in DATEN: Confidence-Scores, Bloat-Prozente, Usage-Counts — keine Bauchgefühle\n"
        "- Lösche KONSEQUENT: Schlechte Skills/Rules aktiv entfernen ist genauso wichtig wie Neue erstellen\n"
        "- Merge VOR Evolve: Erst Duplikate aufräumen, dann verbessern\n"
        "- Evidence-Gate: Mature Skills (predefined, conf≥0.7) NUR mit ≥3 Records ändern\n"
        "- Parallel denken: Cluster-Analysen via Sub-Agents parallel spawnen, nicht sequentiell\n"
        "- Immer Vorher/Nachher: Jede Aktion dokumentieren mit konkreten Zahlen-Deltas\n"
        "- Checkpoint PFLICHT: dream_act({\"action\":\"persist_checkpoint\"}) VOR final_answer()\n"
        "- Schreibe KEINEN Code im Parent-VFS — du arbeitest NUR mit dream_act() und direktem TaskMap-Lesen\n"
        "OUTPUT-FORMAT: Phase → Aktionen mit Zahlen → System Health Delta → Empfehlungen"
    ),
    "model_preference": "fast",
    "temperature": 0.2,
    "max_iterations_factor": 1.5,
    "verification_level": "basic",
}


# ═══════════════════════════════════════════════════════════════════
# AGENT CONFIG
# ═══════════════════════════════════════════════════════════════════

def create_dreamer_agent_config(
    parent_name: str,
    parent_fast_model: str = "",
) -> dict:
    """
    Build config dict for DreamerAgent creation.

    Returns a dict that can be used to construct an AMD / FlowAgent.
    Does NOT import FlowAgent — that's the caller's job.
    """
    fast_model = os.getenv("DREAMER_FAST_MODEL", "")
    if not fast_model:
        fast_model = parent_fast_model or _DEFAULT_DREAMER_MODEL

    complex_model = os.getenv("DREAMER_COMPLEX_MODEL", "")
    if not complex_model:
        complex_model = fast_model  # Dreamer uses fast for everything by default

    return {
        "name": f"dreamer_{parent_name}",
        "fast_llm_model": fast_model,
        "complex_llm_model": complex_model,
        "stream": True,
        "persona": DREAMER_PERSONA_PROFILE,
        "persona_keywords": DREAMER_PERSONA_KEYWORDS,
    }


# ═══════════════════════════════════════════════════════════════════
# DREAM QUERY BUILDING
# ═══════════════════════════════════════════════════════════════════

def build_dream_query(
    config,
    record_count: int,
    skill_count: int,
    rule_count: int,
) -> str:
    """
    Build the query string that the DreamerAgent receives.

    This is what kicks off the dream cycle. It tells the agent
    what data is available and what config flags are set.
    """
    lines = [
        "Starte Meta-Learning Cycle.",
        "",
        f"TaskMap: prim\u00e4re Datenquelle in /global/.memory/taskmap/ (direkt im VFS lesen).",
        f"Records seit letztem Cutoff: {record_count}.",
        f"Aktuell: {skill_count} Skills, {rule_count} Regeln.",
        f"Budget: {config.max_budget} tokens.",
        "",
        "Aktivierte Features:",
    ]

    features = []
    if config.do_skill_evolve:
        features.append("  ✓ Skill-Evolution")
    else:
        features.append("  ✗ Skill-Evolution DEAKTIVIERT — keine Skills evolven/erstellen")

    if config.do_skill_split:
        features.append("  ✓ Skill-Split (bloated Skills aufteilen)")
    else:
        features.append("  ✗ Skill-Split DEAKTIVIERT — keine Skills splitten")

    if config.do_persona_evolve:
        features.append("  ✓ Persona-Evolution")
    else:
        features.append("  ✗ Persona-Evolution DEAKTIVIERT — keine Personas ändern")

    if config.do_create_new:
        features.append("  ✓ Neue Skills/Personas erstellen erlaubt")
    else:
        features.append("  ✗ Keine neuen Skills/Personas erstellen")

    lines.extend(features)
    lines.append("")
    lines.append("Beginne mit Phase 0: dream_act({\"action\":\"get_session_histories\",\"payload\":{\"max_per_session\":100}}) - lies die vollen User-Chatverlaeufe aller Sessions, halte deren Aussagen fuer die Cross-Referenz bereit.")
    lines.append("Dann Phase 1: lies /global/.memory/taskmap/_index.json direkt im VFS. Nutze dream_act({\"action\":\"get_all_state\"}) für Skills+Rules+Personas.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# VFS PREPARATION
# ═══════════════════════════════════════════════════════════════════

def prepare_dreamer_vfs(vfs, harvest_data: dict) -> None:
    """
    Write harvest data and reference files into the dreamer session VFS.

    Args:
        vfs: VFS instance (must support write + mkdir)
        harvest_data: Dict with records, skill_snapshot, rule_snapshot, persona_snapshot
    """
    # Ensure directories exist
    for d in ["/harvest", "/reference"]:
        try:
            if hasattr(vfs, "mkdir"):
                vfs.mkdir(d, parents=True)
        except Exception:
            pass

    # Serialize records
    records = harvest_data.get("records", [])
    records_data = []
    for r in records:
        if hasattr(r, "to_dict"):
            records_data.append(r.to_dict())
        elif isinstance(r, dict):
            records_data.append(r)
        else:
            # Dataclass fallback
            from dataclasses import asdict
            records_data.append(asdict(r))

    vfs.write("/harvest/records.json", json.dumps(records_data, indent=2, default=str))

    # Write snapshots
    vfs.write(
        "/harvest/skills_snapshot.json",
        json.dumps(harvest_data.get("skill_snapshot", {}), indent=2, default=str),
    )
    vfs.write(
        "/harvest/rules_snapshot.json",
        json.dumps(harvest_data.get("rule_snapshot", {}), indent=2, default=str),
    )
    vfs.write(
        "/harvest/personas_snapshot.json",
        json.dumps(harvest_data.get("persona_snapshot", {}), indent=2, default=str),
    )

    # Write skill creator guide as reference
    guide_content = _load_skills_guide()
    vfs.write("/reference/dreamer_skills_guide.md", guide_content)
