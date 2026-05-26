"""
Memory Index — derived from the graph (entities + relations + concepts).
Zero LLM calls. Rebuilds passively on every memory_save via SQL queries.
"""
from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field

from toolboxv2 import get_logger
from toolboxv2.mods.isaa.base.memory_graph_visualizer import MemoryGraphVisualizer

logger = get_logger()


# ── Schema ──────────────────────────────────────────────────────────────

class SpaceSnapshot(BaseModel):
    nodes: list[dict] = Field(default_factory=list)   # from to_json()
    edges: list[dict] = Field(default_factory=list)
    concepts: dict[str, int] = Field(default_factory=dict)  # concept -> count
    entry_count: int = 0


class MemoryIndex(BaseModel):
    spaces: dict[str, SpaceSnapshot] = Field(default_factory=dict)

    # back-compat: module.py checks `len(idx.entries) > 0`
    @property
    def entries(self) -> dict[str, SpaceSnapshot]:
        return self.spaces


# ── Persistence ─────────────────────────────────────────────────────────

def _index_path(data_dir: str, agent_name: str) -> Path:
    return Path(data_dir) / "Agents" / agent_name / "memory_index.json"


def load_index(data_dir: str, agent_name: str) -> MemoryIndex:
    p = _index_path(data_dir, agent_name)
    if p.exists():
        try:
            return MemoryIndex.model_validate_json(p.read_text())
        except Exception as e:
            logger.warning(f"memory_index load failed ({p}): {e}")
    return MemoryIndex()


def save_index(data_dir: str, agent_name: str, index: MemoryIndex) -> None:
    p = _index_path(data_dir, agent_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(index.model_dump_json(indent=2))


# ── Per-space snapshot (no LLM) ─────────────────────────────────────────

def _top_concepts(store, limit: int = 25) -> dict[str, int]:
    try:
        with store._tx() as conn:
            cur = conn.execute(
                """SELECT c.concept, COUNT(*) AS cnt
                   FROM concept_index c
                   JOIN entries e ON c.entry_id = e.id
                   WHERE e.space = ? AND e.is_active = 1
                   GROUP BY c.concept
                   ORDER BY cnt DESC
                   LIMIT ?""",
                (store.space, limit),
            )
            return {row[0]: int(row[1]) for row in cur}
    except Exception as e:
        logger.debug(f"_top_concepts({store.space}) failed: {e}")
        return {}


def _entry_count(store) -> int:
    try:
        with store._tx() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM entries WHERE space = ? AND is_active = 1",
                (store.space,),
            )
            return int(cur.fetchone()[0])
    except Exception:
        return 0


def build_snapshot(store) -> SpaceSnapshot:
    """One space → snapshot. Pure SQL via MemoryGraphVisualizer + concept_index."""
    graph = MemoryGraphVisualizer(store).to_json()
    return SpaceSnapshot(
        nodes=graph.get("nodes", []),
        edges=graph.get("edges", []),
        concepts=_top_concepts(store),
        entry_count=_entry_count(store),
    )


def build_index_from_memory(mem) -> MemoryIndex:
    """All non-empty spaces. Sync, fast, no LLM."""
    idx = MemoryIndex()
    for space_name, store in mem.memories.items():
        try:
            snap = build_snapshot(store)
            if snap.entry_count > 0 or snap.nodes:
                idx.spaces[space_name] = snap
        except Exception as e:
            logger.warning(f"snapshot {space_name} failed: {e}")
    return idx


# ── Compat wrappers (same signatures module.py already calls) ───────────

async def build_initial_index(isaa_ref, agent_name: str, data_dir: str) -> MemoryIndex:
    idx = build_index_from_memory(isaa_ref.get_memory())
    save_index(data_dir, agent_name, idx)
    return idx


async def update_index_after_save(
    isaa_ref,
    agent_name: str,
    data_dir: str,
    index: MemoryIndex,
    space: str,
    content: str = "",          # ignored — graph is the source of truth now
    concepts: list[str] | None = None,  # ignored
) -> MemoryIndex:
    """Refresh just the affected space — other snapshots stay cached."""
    mem = isaa_ref.get_memory()
    store = mem.memories.get(space)
    if store is None:
        return index
    try:
        snap = build_snapshot(store)
        if snap.entry_count > 0 or snap.nodes:
            index.spaces[space] = snap
        else:
            index.spaces.pop(space, None)
        save_index(data_dir, agent_name, index)
    except Exception as e:
        logger.warning(f"update_index_after_save({space}) failed: {e}")
    return index


# ── Markdown render (VFS file) ──────────────────────────────────────────

def render_index(index: MemoryIndex) -> str:
    if not index.spaces:
        return "# Memory Index\n\n_Empty — no entries yet._"

    out: list[str] = ["# Memory Index", ""]
    for space in sorted(index.spaces.keys()):
        snap = index.spaces[space]
        if snap.entry_count == 0 and not snap.nodes:
            continue

        out.append(
            f"## {space}  ·  {snap.entry_count} entries  ·  "
            f"{len(snap.nodes)} entities  ·  {len(snap.edges)} relations"
        )
        out.append("")

        if snap.nodes:
            id_to_label = {n["id"]: (n.get("label") or n["id"]) for n in snap.nodes}
            outgoing: dict[str, list[tuple[str, str]]] = {}
            for e in snap.edges:
                outgoing.setdefault(e["source"], []).append(
                    (e["target"], e.get("type") or "rel")
                )

            out.append("### Entities & Relations")
            for n in sorted(snap.nodes, key=lambda x: (x.get("label") or x["id"]).lower()):
                label = n.get("label") or n["id"]
                ntype = n.get("type", "?")
                outs = outgoing.get(n["id"], [])
                if outs:
                    rels = ", ".join(
                        f"{rtype}→{id_to_label.get(tgt, tgt)}" for tgt, rtype in outs
                    )
                    out.append(f"- **{label}** *({ntype})* — {rels}")
                else:
                    out.append(f"- **{label}** *({ntype})*")
            out.append("")

        if snap.concepts:
            top = ", ".join(f"`{c}`({n})" for c, n in list(snap.concepts.items())[:20])
            out.append("### Concepts")
            out.append(top)
            out.append("")

    return "\n".join(out)


# ── Recall filter (no LLM) ──────────────────────────────────────────────

def filter_spaces_by_query(index: MemoryIndex, query: str) -> list[str]:
    if not index.spaces:
        return []

    tokens = {t for t in query.lower().split() if len(t) > 2}
    if not tokens:
        return []

    scored: list[tuple[str, int]] = []
    for space, snap in index.spaces.items():
        hits = 0
        for n in snap.nodes:
            label = (n.get("label") or "").lower()
            ntype = (n.get("type") or "").lower()
            if any(t in label or t in ntype for t in tokens):
                hits += 2
        for e in snap.edges:
            if any(t in (e.get("type") or "").lower() for t in tokens):
                hits += 1
        for concept, cnt in snap.concepts.items():
            if any(t in concept.lower() for t in tokens):
                hits += min(cnt, 3)
        if hits > 0:
            scored.append((space, hits))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored]
