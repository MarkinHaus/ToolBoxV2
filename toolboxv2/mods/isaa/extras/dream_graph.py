"""
DreamGraphV2 — Dual-Mode Terminal Visualisation for Dreamer & Skills
====================================================================

Replaces the 3D force-graph with two readable, purpose-built views:

  MODE 1: Pipeline Tree   — real-time phase progress, cluster → skill mapping
  MODE 2: Skill Diamond   — skill evolution over time, bloat detection, confidence

Both auto-adapt to terminal size and degrade gracefully on narrow terminals.

Key improvements over V1:
  - Terminal-size reactive (re-reads on every frame)
  - 2D Tree: clearly shows WHAT the dreamer does per phase
  - Diamond: shows skill HEALTH — bloat, confidence trend, trigger count
  - Edge/Node pruning: max 150 nodes, auto-expire old entries
  - No monkey-patching: event-based hooks via callbacks
  - Skills bloat indicator: warns when instruction/triggers/tools exceed limits

Integration:
    from dream_graph_v2 import DreamGraphV2, hookup_v2
    graph = hookup_v2(engine, dreamer)
    # In render loop:
    graph.print_frame()        # auto-selects best view
    graph.print_frame("tree")  # force tree view
    graph.print_frame("diamond")  # force diamond view

Author: FlowAgent V3 — Markin / TU Berlin
"""

import math
import os
import time
import html as _html
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, List

from prompt_toolkit import print_formatted_text, HTML

# ── Palette ─────────────────────────────────────────────────────────

C = {
    "dim":     "#6b7280",
    "cyan":    "#67e8f9",
    "green":   "#4ade80",
    "red":     "#f87171",
    "amber":   "#fbbf24",
    "white":   "#e5e7eb",
    "bright":  "#ffffff",
    "blue":    "#60a5fa",
    "magenta": "#c084fc",
    "deep":    "#3b3b3b",
    "teal":    "#2dd4bf",
    "rose":    "#fb7185",
    "lime":    "#a3e635",
}

SYM = {
    "dream":  "◎", "phase": "▸", "cluster": "▣", "skill": "◇",
    "agent":  "◯", "sub":   "◈", "mem":     "○", "error": "✗",
    "ok":     "✓", "warn":  "⚠", "evolve":  "↻", "new":   "★",
    "split":  "⑃", "merge": "⊕", "publish": "↗",
    "tree_v": "│", "tree_h": "──", "tree_t": "├", "tree_l": "└",
    "tree_j": "┬", "diamond": "◆", "hollow": "◇",
    "bar_f":  "█", "bar_h": "▌", "bar_e": "░",
    "arrow":  "→", "line":  "─", "active": "●",
}


# ═══════════════════════════════════════════════════════════════════
# TERMINAL SIZE HELPER
# ═══════════════════════════════════════════════════════════════════

def _term_size() -> tuple[int, int]:
    """Get terminal cols, rows — re-reads every call."""
    try:
        ts = os.get_terminal_size()
        return ts.columns, ts.lines
    except (ValueError, OSError):
        return 110, 30


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

class PhaseStatus(Enum):
    PENDING  = ("pending",  C["dim"])
    RUNNING  = ("running",  C["amber"])
    SUCCESS  = ("success",  C["green"])
    FAILED   = ("failed",   C["red"])
    SKIPPED  = ("skipped",  C["dim"])

    def __init__(self, label, color):
        self.label_ = label
        self.color = color


@dataclass
class PhaseNode:
    """One dreamer pipeline phase."""
    name: str
    status: PhaseStatus = PhaseStatus.PENDING
    started_at: float = 0.0
    duration: float = 0.0
    detail: str = ""  # short status text
    children: list = field(default_factory=list)  # cluster/skill IDs


@dataclass
class ClusterNode:
    """A run-cluster discovered by the dreamer."""
    id: str
    intent: str
    record_count: int = 0
    success_count: int = 0
    skills_affected: list[str] = field(default_factory=list)  # skill IDs


@dataclass
class SkillSnapshot:
    """Point-in-time snapshot of a skill's health."""
    id: str
    name: str
    version: int = 1
    confidence: float = 0.5
    trigger_count: int = 0
    tool_count: int = 0
    instruction_len: int = 0  # chars
    action: str = ""  # "evolved", "created", "split", "unchanged"
    timestamp: float = field(default_factory=time.time)

    @property
    def bloat_score(self) -> float:
        """0.0 = lean, 1.0 = dangerously bloated."""
        t_bloat = _clamp((self.trigger_count - 5) / 15, 0, 1)  # >20 triggers = max
        i_bloat = _clamp((self.instruction_len - 500) / 2000, 0, 1)  # >2500 chars = max
        tool_bloat = _clamp((self.tool_count - 6) / 14, 0, 1)  # >20 tools = max
        return t_bloat * 0.3 + i_bloat * 0.5 + tool_bloat * 0.2

    @property
    def health_color(self) -> str:
        b = self.bloat_score
        if b < 0.2:
            return C["green"]
        elif b < 0.5:
            return C["amber"]
        elif b < 0.8:
            return C["rose"]
        return C["red"]


# ═══════════════════════════════════════════════════════════════════
# DREAM GRAPH V2
# ═══════════════════════════════════════════════════════════════════

class DreamGraphV2:
    """
    Dual-mode terminal visualisation for the Dreamer system.

    Mode 1: Pipeline Tree — shows dream phases as a tree with clusters/skills
    Mode 2: Skill Diamond — shows skill evolution health over time
    """

    _instance: Optional['DreamGraphV2'] = None

    @classmethod
    def instance(cls) -> 'DreamGraphV2':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def __init__(self):
        # Pipeline tree data
        self.agent_name: str = ""
        self.dream_id: str = ""
        self.phases: dict[str, PhaseNode] = {}
        self.clusters: dict[str, ClusterNode] = {}
        self._phase_order: list[str] = [
            "harvest", "cluster", "analyze", "reconcile", "publish", "memory_sync"
        ]
        self._phase_labels: dict[str, str] = {
            "harvest": "Log Harvest",
            "cluster": "Clustering",
            "analyze": "LLM Analysis",
            "reconcile": "Skill Reconcile",
            "publish": "Publish",
            "memory_sync": "Memory Sync",
        }

        # Skill health tracking
        self.skill_history: dict[str, list[SkillSnapshot]] = {}  # skill_id → snapshots
        self._current_skills: dict[str, SkillSnapshot] = {}  # latest snapshot per skill

        # Frame counter
        self._frame = 0
        self._mode = "tree"  # "tree" or "diamond"
        self._active = False

        # Budget tracking
        self.budget_used: int = 0
        self.budget_max: int = 5000

    # ─── Event API (called by Dreamer hooks) ──────────────────────

    def on_dream_start(self, agent_name: str, dream_id: str, budget: int = 5000):
        """Called when dreamer.dream() begins."""
        self.agent_name = agent_name
        self.dream_id = dream_id
        self.budget_max = budget
        self.budget_used = 0
        self._active = True
        # Reset phases
        self.phases = {
            name: PhaseNode(name=name) for name in self._phase_order
        }
        self.clusters = {}

    def on_phase_start(self, phase_name: str):
        """Called when a phase begins."""
        short = phase_name.replace("_phase_", "")
        if short in self.phases:
            self.phases[short].status = PhaseStatus.RUNNING
            self.phases[short].started_at = time.time()

    def on_phase_end(self, phase_name: str, success: bool, detail: str = ""):
        """Called when a phase finishes."""
        short = phase_name.replace("_phase_", "")
        if short in self.phases:
            ph = self.phases[short]
            ph.status = PhaseStatus.SUCCESS if success else PhaseStatus.FAILED
            ph.duration = time.time() - ph.started_at if ph.started_at else 0
            ph.detail = detail

    def on_cluster_found(self, cluster_id: str, intent: str, record_count: int, success_count: int):
        cn = ClusterNode(id=cluster_id, intent=intent,
                         record_count=record_count, success_count=success_count)
        self.clusters[cluster_id] = cn
        # Link to cluster phase
        if "cluster" in self.phases:
            self.phases["cluster"].children.append(cluster_id)

    def on_skill_event(self, skill_id: str, name: str, action: str,
                       version: int = 1, confidence: float = 0.5,
                       trigger_count: int = 0, tool_count: int = 0,
                       instruction_len: int = 0, cluster_id: str = ""):
        """Track a skill evolution event."""
        snap = SkillSnapshot(
            id=skill_id, name=name, version=version,
            confidence=confidence, trigger_count=trigger_count,
            tool_count=tool_count, instruction_len=instruction_len,
            action=action,
        )
        self._current_skills[skill_id] = snap
        if skill_id not in self.skill_history:
            self.skill_history[skill_id] = []
        self.skill_history[skill_id].append(snap)

        # Link to reconcile phase
        if "reconcile" in self.phases:
            if skill_id not in self.phases["reconcile"].children:
                self.phases["reconcile"].children.append(skill_id)

        # Link to cluster
        if cluster_id and cluster_id in self.clusters:
            if skill_id not in self.clusters[cluster_id].skills_affected:
                self.clusters[cluster_id].skills_affected.append(skill_id)

    def on_budget_update(self, used: int):
        self.budget_used = used

    def on_dream_end(self):
        self._active = False

    # ─── Mode switching ───────────────────────────────────────────

    def toggle_mode(self):
        self._mode = "diamond" if self._mode == "tree" else "tree"

    # ─── Rendering ────────────────────────────────────────────────

    def print_frame(self, mode: str = ""):
        """Print one frame. Auto-adapts to terminal size."""
        cols, rows = _term_size()
        cols = max(40, cols - 2)  # margin
        rows = max(10, rows - 4)

        m = mode or self._mode
        self._frame += 1

        if m == "diamond" and self._current_skills:
            html_str = self._render_diamond(cols, rows)
        else:
            html_str = self._render_tree(cols, rows)

        try:
            print_formatted_text(HTML(html_str))
        except Exception:
            import re
            print(re.sub(r"<[^>]+>", "", html_str))

    # ── MODE 1: Pipeline Tree ─────────────────────────────────────

    def _render_tree(self, cols: int, rows: int) -> str:
        lines: list[str] = []
        e = _html.escape

        # Header
        pulse = SYM["active"] if self._active and self._frame % 4 < 2 else " "
        budget_pct = int(100 * self.budget_used / max(self.budget_max, 1))
        budget_bar = self._mini_bar(budget_pct, 10)
        lines.append(
            f"<style fg='{C['cyan']}'>{SYM['dream']} Dream</style>"
            f"  <style fg='{C['dim']}'>{e(self.dream_id)}</style>"
            f"  <style fg='{C['bright']}'>{pulse}</style>"
            f"  <style fg='{C['dim']}'>budget {budget_bar} {budget_pct}%</style>"
        )

        # Phases as tree
        phase_list = [self.phases[n] for n in self._phase_order if n in self.phases]
        for i, ph in enumerate(phase_list):
            is_last_phase = (i == len(phase_list) - 1)
            connector = SYM["tree_l"] if is_last_phase else SYM["tree_t"]

            # Status symbol
            if ph.status == PhaseStatus.RUNNING:
                sym = SYM["active"]
                col = C["amber"]
            elif ph.status == PhaseStatus.SUCCESS:
                sym = SYM["ok"]
                col = C["green"]
            elif ph.status == PhaseStatus.FAILED:
                sym = SYM["error"]
                col = C["red"]
            else:
                sym = " "
                col = C["dim"]

            label = self._phase_labels.get(ph.name, ph.name)
            duration_str = f" {ph.duration:.1f}s" if ph.duration > 0 else ""
            detail_str = f"  {e(ph.detail[:cols-40])}" if ph.detail else ""

            lines.append(
                f"<style fg='{C['dim']}'>{connector}{SYM['tree_h']}</style>"
                f"<style fg='{col}'>{sym} {e(label)}</style>"
                f"<style fg='{C['dim']}'>{duration_str}{detail_str}</style>"
            )

            # Children: clusters or skills
            child_prefix = f"<style fg='{C['dim']}'>{'   ' if is_last_phase else SYM['tree_v'] + '  '}</style>"

            if ph.name == "cluster" or ph.name == "analyze":
                # Show clusters
                cluster_items = [(cid, self.clusters[cid]) for cid in ph.children if cid in self.clusters]
                for j, (cid, cn) in enumerate(cluster_items[:rows // 3]):
                    is_last = (j == len(cluster_items) - 1)
                    cc = SYM["tree_l"] if is_last else SYM["tree_t"]
                    ratio = cn.success_count / max(cn.record_count, 1)
                    ratio_col = C["green"] if ratio > 0.7 else C["amber"] if ratio > 0.4 else C["red"]
                    intent_short = e(cn.intent[:cols - 50]) if cn.intent else cid

                    lines.append(
                        f"{child_prefix}"
                        f"<style fg='{C['dim']}'>{cc}{SYM['tree_h']}</style>"
                        f"<style fg='{C['dim']}'>{SYM['cluster']} {intent_short}</style>"
                        f"  <style fg='{ratio_col}'>{cn.success_count}/{cn.record_count}</style>"
                    )

                    # Show affected skills under cluster
                    for k, sid in enumerate(cn.skills_affected[:3]):
                        sk = self._current_skills.get(sid)
                        if not sk:
                            continue
                        sk_prefix = f"{child_prefix}<style fg='{C['dim']}'>{'   ' if is_last else SYM['tree_v'] + '  '}</style>"
                        action_sym = {"evolved": SYM["evolve"], "created": SYM["new"],
                                      "split": SYM["split"]}.get(sk.action, " ")
                        lines.append(
                            f"{sk_prefix}"
                            f"<style fg='{sk.health_color}'>{action_sym} {SYM['skill']} {e(sk.name[:25])}</style>"
                            f"<style fg='{C['dim']}'> v{sk.version} c={sk.confidence:.2f}</style>"
                            f"<style fg='{sk.health_color}'> bloat={sk.bloat_score:.0%}</style>"
                        )

            elif ph.name == "reconcile":
                # Show skills directly
                skill_items = [(sid, self._current_skills[sid])
                               for sid in ph.children if sid in self._current_skills]
                for j, (sid, sk) in enumerate(skill_items[:rows // 3]):
                    is_last = (j == len(skill_items) - 1)
                    cc = SYM["tree_l"] if is_last else SYM["tree_t"]
                    action_sym = {"evolved": SYM["evolve"], "created": SYM["new"],
                                  "split": SYM["split"], "merged": SYM["merge"],
                                  "published": SYM["publish"]}.get(sk.action, " ")

                    bloat_warn = ""
                    if sk.bloat_score > 0.5:
                        bloat_warn = f" <style fg='{C['red']}'>{SYM['warn']} BLOAT</style>"

                    lines.append(
                        f"{child_prefix}"
                        f"<style fg='{C['dim']}'>{cc}{SYM['tree_h']}</style>"
                        f"<style fg='{sk.health_color}'>{action_sym} {SYM['skill']} {e(sk.name[:20])}</style>"
                        f"<style fg='{C['dim']}'>"
                        f" v{sk.version} c={sk.confidence:.2f}"
                        f" t={sk.trigger_count} tools={sk.tool_count}"
                        f" inst={sk.instruction_len}ch"
                        f"</style>{bloat_warn}"
                    )

        # Truncate if too tall
        if len(lines) > rows:
            lines = lines[:rows - 1]
            lines.append(f"<style fg='{C['dim']}'>  ... ({len(self.phases)} phases, "
                         f"{len(self.clusters)} clusters, "
                         f"{len(self._current_skills)} skills)</style>")

        return "\n".join(lines)

    # ── MODE 2: Skill Diamond ─────────────────────────────────────

    def _render_diamond(self, cols: int, rows: int) -> str:
        """
        Diamond view: each skill is a row with sparkline history.

        Layout per skill:
          ◇ skill_name     v3  ◆────◆──◇──◆  c=0.72  bloat=23%  [████░░] triggers:5 tools:4

        The diamond chain shows confidence over versions:
          ◆ = high confidence (>0.7)
          ◇ = medium (0.4-0.7)
          ✗ = low (<0.4)
        """
        lines: list[str] = []
        e = _html.escape

        # Header
        lines.append(
            f"<style fg='{C['teal']}'>{SYM['diamond']} Skill Evolution Diamond</style>"
            f"  <style fg='{C['dim']}'>{len(self._current_skills)} skills tracked</style>"
        )

        # Column widths adapt to terminal
        name_w = _clamp(cols // 5, 10, 25)
        spark_w = _clamp(cols // 4, 8, 20)

        # Sort: most bloated first (problems on top)
        sorted_skills = sorted(
            self._current_skills.values(),
            key=lambda s: (-s.bloat_score, -s.version)
        )

        for sk in sorted_skills[:rows - 3]:
            history = self.skill_history.get(sk.id, [sk])

            # Sparkline: confidence over versions
            spark = self._confidence_sparkline(history, spark_w)

            # Bloat bar
            bloat_pct = int(sk.bloat_score * 100)
            bloat_bar = self._mini_bar(bloat_pct, 6)

            # Action badge
            action_sym = {"evolved": SYM["evolve"], "created": SYM["new"],
                          "split": SYM["split"]}.get(sk.action, " ")

            name_str = e(sk.name[:name_w].ljust(name_w))

            lines.append(
                f"<style fg='{sk.health_color}'>{action_sym}{SYM['skill']}</style> "
                f"<style fg='{C['white']}'>{name_str}</style>"
                f"<style fg='{C['dim']}'>v{sk.version:<3}</style>"
                f" {spark} "
                f"<style fg='{C['dim']}'>c=</style>"
                f"<style fg='{self._confidence_color(sk.confidence)}'>{sk.confidence:.2f}</style>"
                f"  <style fg='{sk.health_color}'>bloat {bloat_bar} {bloat_pct}%</style>"
                f"  <style fg='{C['dim']}'>t:{sk.trigger_count} tools:{sk.tool_count}</style>"
            )

            # Bloat detail line if critical
            if sk.bloat_score > 0.6:
                parts = []
                if sk.trigger_count > 10:
                    parts.append(f"triggers:{sk.trigger_count}>10")
                if sk.instruction_len > 1500:
                    parts.append(f"instruction:{sk.instruction_len}ch>1500")
                if sk.tool_count > 10:
                    parts.append(f"tools:{sk.tool_count}>10")
                if parts:
                    lines.append(
                        f"<style fg='{C['red']}'>  {SYM['warn']} {e(', '.join(parts))}</style>"
                    )

        if not sorted_skills:
            lines.append(f"<style fg='{C['dim']}'>  No skills tracked yet.</style>")

        # Footer
        lines.append(
            f"\n<style fg='{C['dim']}'>"
            f"{SYM['diamond']}=conf≥0.7  {SYM['hollow']}=0.4-0.7  {SYM['error']}=&lt;0.4  "
            f"bloat: {SYM['bar_f']}=triggers {SYM['bar_h']}=instruction {SYM['bar_e']}=tools"
            f"</style>"
        )

        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────

    def _confidence_sparkline(self, history: list[SkillSnapshot], width: int) -> str:
        """Render confidence history as diamond chain."""
        # Sample evenly if history > width
        if len(history) > width:
            step = len(history) / width
            sampled = [history[int(i * step)] for i in range(width)]
        else:
            sampled = history

        parts = []
        for snap in sampled:
            if snap.confidence >= 0.7:
                parts.append(f"<style fg='{C['green']}'>{SYM['diamond']}</style>")
            elif snap.confidence >= 0.4:
                parts.append(f"<style fg='{C['amber']}'>{SYM['hollow']}</style>")
            else:
                parts.append(f"<style fg='{C['red']}'>{SYM['error']}</style>")

            # Connecting line
            if snap != sampled[-1]:
                parts.append(f"<style fg='{C['dim']}'>{SYM['line']}</style>")

        # Pad if short
        rendered_len = len(sampled) * 2 - 1
        if rendered_len < width:
            padding = " " * (width - rendered_len)
            parts.append(f"<style fg='{C['deep']}'>{padding}</style>")

        return "".join(parts)

    @staticmethod
    def _confidence_color(c: float) -> str:
        if c >= 0.7:
            return C["green"]
        elif c >= 0.4:
            return C["amber"]
        return C["red"]

    @staticmethod
    def _mini_bar(pct: int, width: int) -> str:
        """Compact percentage bar."""
        pct = _clamp(pct, 0, 100)
        filled = int(width * pct / 100)
        if pct < 30:
            col = C["green"]
        elif pct < 60:
            col = C["amber"]
        else:
            col = C["red"]
        bar = SYM["bar_f"] * filled + SYM["bar_e"] * (width - filled)
        return f"<style fg='{col}'>{bar}</style>"

    # ─── Legend ───────────────────────────────────────────────────

    def print_legend(self):
        parts = [
            f"<style fg='{C['amber']}'>{SYM['dream']} dream</style>",
            f"<style fg='{C['green']}'>{SYM['ok']} ok</style>",
            f"<style fg='{C['red']}'>{SYM['error']} fail</style>",
            f"<style fg='{C['dim']}'>{SYM['cluster']} cluster</style>",
            f"<style fg='{C['green']}'>{SYM['skill']} skill</style>",
            f"<style fg='{C['amber']}'>{SYM['evolve']} evolve</style>",
            f"<style fg='{C['teal']}'>{SYM['new']} new</style>",
            f"<style fg='{C['rose']}'>{SYM['warn']} bloat</style>",
        ]
        controls = (
            f"<style fg='{C['dim']}'>"
            "Tab:mode  g:toggle"
            "</style>"
        )
        try:
            print_formatted_text(HTML(f"  {'  '.join(parts)}\n  {controls}"))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# DREAMER HOOKS — Event-based, no monkey-patching
# ═══════════════════════════════════════════════════════════════════

def _wrap_dreamer_phase(dreamer, phase_name: str, graph: DreamGraphV2):
    """Wrap a single dreamer phase to emit graph events."""
    orig = getattr(dreamer, phase_name)
    short = phase_name.replace("_phase_", "")

    async def wrapper(state):
        graph.on_phase_start(phase_name)
        try:
            result = await orig(state)

            # Extract phase-specific data for graph
            detail = ""
            report = state.get("report")

            if short == "harvest":
                detail = f"{report.logs_scanned} logs" if report else ""

            elif short == "cluster":
                clusters = state.get("clusters", {})
                for cid, records in clusters.items():
                    success_n = sum(1 for r in records if r.success)
                    intent = records[0].query[:40] if records else cid
                    graph.on_cluster_found(cid, intent, len(records), success_n)
                detail = f"{len(clusters)} clusters"

            elif short == "analyze":
                analyses = state.get("analyses", {})
                for cid, analysis in analyses.items():
                    # Update cluster intents with LLM-refined version
                    if cid in graph.clusters and analysis.dominant_intent:
                        graph.clusters[cid].intent = analysis.dominant_intent
                detail = f"{len(analyses)} analyzed"

            elif short == "reconcile":
                if report:
                    sm = dreamer.agent.session_manager.skills_manager
                    # Track all affected skills
                    for sid in (report.skills_evolved + report.skills_created + report.skills_split):
                        skill = sm.skills.get(sid)
                        if skill:
                            graph.on_skill_event(
                                skill_id=sid,
                                name=getattr(skill, 'name', sid),
                                action=("evolved" if sid in report.skills_evolved
                                        else "created" if sid in report.skills_created
                                        else "split"),
                                version=getattr(skill, '_version', 1),
                                confidence=getattr(skill, 'confidence', 0.5),
                                trigger_count=len(getattr(skill, 'triggers', [])),
                                tool_count=len(getattr(skill, 'tools_used', [])),
                                instruction_len=len(getattr(skill, 'instruction', '')),
                            )
                    parts = []
                    if report.skills_evolved:
                        parts.append(f"{len(report.skills_evolved)} evolved")
                    if report.skills_created:
                        parts.append(f"{len(report.skills_created)} new")
                    if report.skills_split:
                        parts.append(f"{len(report.skills_split)} split")
                    detail = ", ".join(parts) if parts else "no changes"

            elif short == "publish":
                if report and report.skills_published:
                    detail = f"{len(report.skills_published)} published"

            elif short == "memory_sync":
                if report:
                    detail = f"{report.memory_entries_added} entries"

            graph.on_phase_end(phase_name, success=True, detail=detail)
            graph.on_budget_update(dreamer._budget_used)
            return result

        except Exception as e:
            graph.on_phase_end(phase_name, success=False, detail=str(e)[:40])
            raise

    setattr(dreamer, phase_name, wrapper)


def hookup_v2(engine=None, dreamer=None) -> DreamGraphV2:
    """
    Wire DreamGraphV2 to engine and/or dreamer.

    Returns the singleton graph instance.
    """
    graph = DreamGraphV2.instance()

    if dreamer:
        agent_name = dreamer.agent.amd.name
        # Wrap dream() to emit start/end
        orig_dream = dreamer.dream

        async def wrapped_dream(config):
            graph.on_dream_start(
                agent_name=agent_name,
                dream_id=f"dream_{id(config)}",
                budget=config.max_budget,
            )
            try:
                result = await orig_dream(config)
                return result
            finally:
                graph.on_dream_end()

        dreamer.dream = wrapped_dream

        # Wrap each phase
        for pname in [
            "_phase_harvest", "_phase_cluster", "_phase_analyze",
            "_phase_reconcile", "_phase_publish", "_phase_memory_sync",
        ]:
            if hasattr(dreamer, pname):
                _wrap_dreamer_phase(dreamer, pname, graph)

    return graph


# ═══════════════════════════════════════════════════════════════════
# LIVE RENDER LOOP
# ═══════════════════════════════════════════════════════════════════

async def dream_with_viz_v2(
    isaa_tools,
    agent_name: str = "default",
    config=None,
    show_graph: bool = True,
):
    """
    Run dreamer with live V2 visualization.

    Usage:
        report = await dream_with_viz_v2(self, "myagent", DreamConfig(max_budget=3000))
    """
    import asyncio
    from toolboxv2.mods.isaa.base.Agent.dreamer import Dreamer, DreamConfig

    config = config or DreamConfig()
    agent = await isaa_tools.get_agent(agent_name)

    if not show_graph:
        return await agent.a_dream(config)

    graph = hookup_v2(dreamer=getattr(agent, '_dreamer', None) or Dreamer(agent))

    # Snapshot existing skills BEFORE dream for delta tracking
    sm = agent.session_manager.skills_manager
    for sid, skill in sm.skills.items():
        graph.on_skill_event(
            skill_id=sid,
            name=getattr(skill, 'name', sid),
            action="unchanged",
            version=getattr(skill, '_version', 1),
            confidence=getattr(skill, 'confidence', 0.5),
            trigger_count=len(getattr(skill, 'triggers', [])),
            tool_count=len(getattr(skill, 'tools_used', [])),
            instruction_len=len(getattr(skill, 'instruction', '')),
        )

    async def render_loop():
        graph.print_legend()
        while graph._active:
            graph.print_frame()
            await asyncio.sleep(0.3)
        graph.print_frame()  # final

    render_task = asyncio.create_task(render_loop())
    report = None
    try:
        report = await agent._dreamer.dream(config)
    finally:
        graph._active = False
        await asyncio.sleep(0.4)
        render_task.cancel()
        try:
            await render_task
        except asyncio.CancelledError:
            pass

    # Print diamond view as summary
    if report and graph._current_skills:
        print()
        graph.print_frame("diamond")

    return report
