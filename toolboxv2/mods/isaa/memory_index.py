"""
Memory Index — persistent, BLITZMODEL-updated index of what's stored where in memory.

Integration point: module.py get_agent_builder() closures.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from pydantic import BaseModel, Field

from toolboxv2 import get_logger

logger = get_logger()


# ── Schema ──────────────────────────────────────────────────────────────

class MemoryIndexEntry(BaseModel):
    key_concepts: list[str] = Field(default_factory=list)
    summary: str = ""  # mini-summary over related entries


class MemoryIndexEdits(BaseModel):
    edits: list[MemoryIndexEntry] = Field(default_factory=list)

class MemoryIndexEdit(BaseModel):
    space: str
    concept_cluster: str
    new_information: str


class MemoryIndex(BaseModel):
    entries: dict[str, list[MemoryIndexEntry]] = Field(default_factory=dict)


# ── Persistence ─────────────────────────────────────────────────────────

def _index_path(data_dir: str, agent_name: str) -> Path:
    return Path(data_dir) / "Agents" / agent_name / "memory_index.json"


def load_index(data_dir: str, agent_name: str) -> MemoryIndex:
    p = _index_path(data_dir, agent_name)
    if p.exists():
        try:
            return MemoryIndex.model_validate_json(p.read_text())
        except Exception as e:
            logger.warning(f"Failed to load memory index from {p}: {e}")
    return MemoryIndex()


def save_index(data_dir: str, agent_name: str, index: MemoryIndex):
    p = _index_path(data_dir, agent_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(index.model_dump_json(indent=2))


# ── Apply edit from BLITZMODEL ──────────────────────────────────────────

def apply_edit(index: MemoryIndex, edit: MemoryIndexEdit) -> MemoryIndex:
    """Patch a single concept_cluster in a space. Upsert logic."""
    space = edit.space
    if space not in index.entries:
        index.entries[space] = []

    # find existing cluster by concept overlap
    for entry in index.entries[space]:
        if edit.concept_cluster.lower() in [c.lower() for c in entry.key_concepts]:
            # update existing
            entry.summary = edit.new_information
            return index

    # new cluster
    index.entries[space].append(MemoryIndexEntry(
        key_concepts=[edit.concept_cluster],
        summary=edit.new_information,
    ))
    return index


# ── Render to VFS string ────────────────────────────────────────────────

def render_index(index: MemoryIndex) -> str:
    """Render index as compact markdown for the VFS system file."""
    if not index.entries:
        return "# Memory Index\n\n_Empty — no entries yet._"

    lines = ["# Memory Index", ""]
    for space in sorted(index.entries.keys()):
        entries = index.entries[space]
        if not entries:
            continue
        lines.append(f"## {space}")
        for e in entries:
            concepts = ", ".join(e.key_concepts)
            lines.append(f"- **[{concepts}]** {e.summary}")
        lines.append("")

    return "\n".join(lines)


# ── Recall filter — local keyword match against index ───────────────────

def filter_spaces_by_query(index: MemoryIndex, query: str) -> list[str]:
    """Return space names whose concept clusters match query keywords. No LLM call."""
    if not index.entries:
        return []

    query_tokens = set(query.lower().split())
    scored: list[tuple[str, int]] = []

    for space, entries in index.entries.items():
        hits = 0
        for entry in entries:
            for concept in entry.key_concepts:
                if concept.lower() in query_tokens or any(t in concept.lower() for t in query_tokens):
                    hits += 1
            # also check summary words
            summary_tokens = set(entry.summary.lower().split())
            hits += len(query_tokens & summary_tokens)
        if hits > 0:
            scored.append((space, hits))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored]


# ── Initial build — scan all spaces, build index via BLITZMODEL ─────────

INITIAL_BUILD_PROMPT = """You receive a list of memory spaces with their entry counts and sample concepts.
Create a structured index. For each NON-EMPTY space, produce a JSON array of MemoryIndexEdit objects.

Each MemoryIndexEdit has:
- space: the space name
- concept_cluster: a short label grouping related concepts
- new_information: a 1-2 sentence summary of what this cluster contains

Rules:
- Skip spaces with 0 entries
- Group related concepts into clusters (max 5 clusters per space)
- Keep summaries concise — this is an overview, not documentation
- Output ONLY valid JSON array, no markdown, no explanation

Input:
{space_data}"""


async def build_initial_index(isaa_ref, agent_name: str, data_dir: str) -> MemoryIndex:
    """Scan all non-empty spaces and ask BLITZMODEL to produce the initial index."""
    mem = isaa_ref.get_memory()
    spaces = list(mem.memories.keys())

    # collect metadata per space
    space_data = []
    for space_name in spaces:
        try:
            mka_temp = None
            # import here to avoid circular at module level
            from toolboxv2.mods.isaa.base.MemoryKnowledgeActor import MemoryKnowledgeActor
            mka_temp = MemoryKnowledgeActor(memory=mem, space_name=space_name)
            stats = await mka_temp.get_stats() if hasattr(mka_temp, 'get_stats') else {}
            entry_count = stats.get('total_entries', 0) if isinstance(stats, dict) else 0

            # try to get concepts
            concepts = []
            if hasattr(mka_temp, 'list_concepts'):
                concepts = await mka_temp.list_concepts()
            elif hasattr(mka_temp, 'get_all_concepts'):
                concepts = await mka_temp.get_all_concepts()

            if entry_count == 0 and not concepts:
                continue  # skip empty spaces

            space_data.append({
                "space": space_name,
                "entry_count": entry_count,
                "sample_concepts": concepts[:20] if concepts else [],
            })
        except Exception as e:
            logger.debug(f"Skipping space {space_name} during index build: {e}")
            continue

    if not space_data:
        return MemoryIndex()

    # call BLITZMODEL via format_class
    prompt = INITIAL_BUILD_PROMPT.format(space_data=json.dumps(space_data, indent=2))

    try:
        result = await isaa_ref.format_class(
            format_schema=MemoryIndexEdits,
            task=prompt,
            agent_name=agent_name,
        )
    except Exception as e:
        logger.warning(f"BLITZMODEL initial index build failed: {e}")
        return MemoryIndex()

    if not result:
        return MemoryIndex()

    # result should be a list of dicts matching MemoryIndexEdit
    index = MemoryIndex()
    edits = result if isinstance(result, list) else result.get("items", [])
    for edit_data in edits:
        try:
            if isinstance(edit_data, dict):
                edit = MemoryIndexEdit(**edit_data)
            elif isinstance(edit_data, MemoryIndexEdit):
                edit = edit_data
            else:
                continue
            index = apply_edit(index, edit)
        except Exception as e:
            logger.debug(f"Skipping malformed edit: {e}")
            continue

    save_index(data_dir, agent_name, index)
    return index


# ── Post-save update — ask BLITZMODEL to patch index ────────────────────

POST_SAVE_PROMPT = """A new fact was saved to memory space "{space}".

New content: "{content}"
Concepts: {concepts}

Current index for this space:
{current_space_index}

Produce a single MemoryIndexEdit to update the index:
- space: "{space}"
- concept_cluster: which cluster this belongs to (existing or new)
- new_information: updated summary incorporating the new fact

Output ONLY valid JSON object, no markdown."""


async def update_index_after_save(
    isaa_ref,
    agent_name: str,
    data_dir: str,
    index: MemoryIndex,
    space: str,
    content: str,
    concepts: list[str] | None,
) -> MemoryIndex:
    """Ask BLITZMODEL to produce a partial edit after a memory_save."""
    current_entries = index.entries.get(space, [])
    current_str = json.dumps([e.model_dump() for e in current_entries], indent=1) if current_entries else "[]"

    prompt = POST_SAVE_PROMPT.format(
        space=space,
        content=content[:500],  # cap to keep prompt small
        concepts=json.dumps(concepts or []),
        current_space_index=current_str,
    )

    try:
        result = await isaa_ref.format_class(
            format_schema=MemoryIndexEdit,
            task=prompt,
            agent_name=agent_name,
        )
    except Exception as e:
        logger.warning(f"BLITZMODEL index update failed: {e}")
        return index

    if not result:
        return index

    try:
        if isinstance(result, dict):
            edit = MemoryIndexEdit(**result)
        elif isinstance(result, MemoryIndexEdit):
            edit = result
        else:
            return index
        index = apply_edit(index, edit)
        save_index(data_dir, agent_name, index)
    except Exception as e:
        logger.debug(f"Failed to apply post-save edit: {e}")

    return index
