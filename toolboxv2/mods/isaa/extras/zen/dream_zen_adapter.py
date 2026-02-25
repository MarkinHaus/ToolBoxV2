"""
DreamZenAdapter — Bridges DreamGraphV2 into ZenPlus TUI.

Renders dream pipeline tree / skill diamond as a Zen+ detail view,
centered (h/w) in the main pane. Accessible via 'd' key in Focus mode.

Updates dream_with_viz_v2 to run inside ZenPlus with a [d]ream menu entry.

Integration:
    from dream_zen_adapter import patch_zen_dream, dream_with_viz_v2

    # Once at startup:
    patch_zen_dream()

    # Run dream with Zen+ UI:
    report = await dream_with_viz_v2(isaa_tools, "myagent", config)

Author: FlowAgent V3 — Markin
"""

import html as _html
import math
import time
from typing import Optional, Any

from toolboxv2.mods.isaa.extras.dream_graph import (
    DreamGraphV2, PhaseNode, PhaseStatus, ClusterNode, SkillSnapshot,
    SYM as DSYM, C as DC, _clamp, hookup_v2,
)
from toolboxv2.mods.isaa.extras.zen.zen_plus import (
    ZenPlus, AgentPane, ViewMode, SYM, C, _short, _bar,
)


# ═══════════════════════════════════════════════════════════════
# FormattedText renderer for DreamGraphV2
# ═══════════════════════════════════════════════════════════════

class DreamRenderer:
    """
    Renders DreamGraphV2 state as FormattedText tuples (style, text),
    centered within the given w/h bounds. Zen+ compatible.
    """

    def __init__(self, graph: DreamGraphV2):
        self.graph = graph
        self._mode = "tree"  # "tree" or "diamond"

    def toggle_mode(self):
        self._mode = "diamond" if self._mode == "tree" else "tree"

    def render(self, w: int, h: int) -> list[tuple[str, str]]:
        """Produce FormattedText tuples for the current dream view, centered."""
        g = self.graph

        # Header
        out: list[tuple[str, str]] = []
        mode_label = "Pipeline Tree" if self._mode == "tree" else "Skill Diamond"
        out.append(("fg:#67e8f9 bold", f"  {DSYM['dream']} Dream Visualisation"))
        out.append(("fg:#6b7280", f"  [{mode_label}]"))
        out.append(("fg:#374151", f"  Tab:toggle mode\n"))
        out.append(("fg:#374151", f"  {'─' * (w - 4)}\n"))

        content_h = h - 4  # header + separator + footer

        if self._mode == "diamond" and g._current_skills:
            lines = self._render_diamond(w - 4, content_h)
        else:
            lines = self._render_tree(w - 4, content_h)

        # Vertical centering: pad top
        content_lines = _count_lines(lines)
        pad_top = max(0, (content_h - content_lines) // 2)
        for _ in range(pad_top):
            out.append(("", "\n"))

        out.extend(lines)

        # Pad bottom
        pad_bottom = max(0, content_h - content_lines - pad_top)
        for _ in range(pad_bottom):
            out.append(("", "\n"))

        # Legend footer
        out.extend(self._render_legend(w))
        return out

    # ── Pipeline Tree ─────────────────────────────────────────

    def _render_tree(self, w: int, max_h: int) -> list[tuple[str, str]]:
        g = self.graph
        out: list[tuple[str, str]] = []
        lines = 0

        # Dream header with budget
        pulse = DSYM["active"] if g._active and g._frame % 4 < 2 else " "
        budget_pct = int(100 * g.budget_used / max(g.budget_max, 1))
        budget_bar = _dream_bar(budget_pct, 10)

        # Center the header horizontally
        header_parts = [
            ("fg:#67e8f9", f"  {DSYM['dream']} Dream"),
            ("fg:#6b7280", f"  {g.dream_id[:20]}"),
            ("fg:#ffffff", f"  {pulse}"),
            ("fg:#6b7280", f"  budget "),
        ]
        header_parts.extend(budget_bar)
        header_parts.append(("fg:#6b7280", f" {budget_pct}%\n"))
        out.extend(header_parts)
        lines += 1

        # Phases as tree
        phase_list = [g.phases[n] for n in g._phase_order if n in g.phases]
        for i, ph in enumerate(phase_list):
            if lines >= max_h - 2:
                break
            is_last = (i == len(phase_list) - 1)
            connector = DSYM["tree_l"] if is_last else DSYM["tree_t"]

            # Status
            if ph.status == PhaseStatus.RUNNING:
                sym, col = DSYM["active"], DC["amber"]
            elif ph.status == PhaseStatus.SUCCESS:
                sym, col = DSYM["ok"], DC["green"]
            elif ph.status == PhaseStatus.FAILED:
                sym, col = DSYM["error"], DC["red"]
            else:
                sym, col = " ", DC["dim"]

            label = g._phase_labels.get(ph.name, ph.name)
            dur = f" {ph.duration:.1f}s" if ph.duration > 0 else ""
            detail = f"  {ph.detail[:w - 40]}" if ph.detail else ""

            out.append(("fg:#6b7280", f"  {connector}{DSYM['tree_h']}"))
            out.append((f"fg:{col}", f"{sym} {label}"))
            out.append(("fg:#6b7280", f"{dur}{detail}\n"))
            lines += 1

            # Children
            child_pfx_char = "   " if is_last else f"{DSYM['tree_v']}  "

            if ph.name in ("cluster", "analyze"):
                cluster_items = [(cid, g.clusters[cid])
                                 for cid in ph.children if cid in g.clusters]
                for j, (cid, cn) in enumerate(cluster_items[:max_h // 4]):
                    if lines >= max_h - 2:
                        break
                    is_last_c = (j == len(cluster_items) - 1)
                    cc = DSYM["tree_l"] if is_last_c else DSYM["tree_t"]
                    ratio = cn.success_count / max(cn.record_count, 1)
                    ratio_col = DC["green"] if ratio > 0.7 else DC["amber"] if ratio > 0.4 else DC["red"]
                    intent = (cn.intent[:w - 50]) if cn.intent else cid

                    out.append(("fg:#6b7280", f"  {child_pfx_char}{cc}{DSYM['tree_h']}"))
                    out.append(("fg:#6b7280", f"{DSYM['cluster']} {intent}"))
                    out.append((f"fg:{ratio_col}", f"  {cn.success_count}/{cn.record_count}\n"))
                    lines += 1

                    # Skills under cluster
                    sk_pfx = "   " if is_last_c else f"{DSYM['tree_v']}  "
                    for sid in cn.skills_affected[:3]:
                        if lines >= max_h - 2:
                            break
                        sk = g._current_skills.get(sid)
                        if not sk:
                            continue
                        a_sym = {"evolved": DSYM["evolve"], "created": DSYM["new"],
                                 "split": DSYM["split"]}.get(sk.action, " ")
                        out.append(("fg:#6b7280", f"  {child_pfx_char}{sk_pfx}"))
                        out.append((f"fg:{sk.health_color}", f"{a_sym} {DSYM['skill']} {sk.name[:25]}"))
                        out.append(("fg:#6b7280", f" v{sk.version} c={sk.confidence:.2f}"))
                        out.append((f"fg:{sk.health_color}", f" bloat={sk.bloat_score:.0%}\n"))
                        lines += 1

            elif ph.name == "reconcile":
                skill_items = [(sid, g._current_skills[sid])
                               for sid in ph.children if sid in g._current_skills]
                for j, (sid, sk) in enumerate(skill_items[:max_h // 4]):
                    if lines >= max_h - 2:
                        break
                    is_last_s = (j == len(skill_items) - 1)
                    cc = DSYM["tree_l"] if is_last_s else DSYM["tree_t"]
                    a_sym = {"evolved": DSYM["evolve"], "created": DSYM["new"],
                             "split": DSYM["split"], "merged": DSYM["merge"],
                             "published": DSYM["publish"]}.get(sk.action, " ")

                    out.append(("fg:#6b7280", f"  {child_pfx_char}{cc}{DSYM['tree_h']}"))
                    out.append((f"fg:{sk.health_color}", f"{a_sym} {DSYM['skill']} {sk.name[:20]}"))
                    out.append(("fg:#6b7280",
                                f" v{sk.version} c={sk.confidence:.2f}"
                                f" t={sk.trigger_count} tools={sk.tool_count}"
                                f" inst={sk.instruction_len}ch"))
                    if sk.bloat_score > 0.5:
                        out.append(("fg:#f87171", f" {DSYM['warn']} BLOAT"))
                    out.append(("", "\n"))
                    lines += 1

        return out

    # ── Skill Diamond ─────────────────────────────────────────

    def _render_diamond(self, w: int, max_h: int) -> list[tuple[str, str]]:
        g = self.graph
        out: list[tuple[str, str]] = []

        out.append(("fg:#2dd4bf", f"  {DSYM['diamond']} Skill Evolution Diamond"))
        out.append(("fg:#6b7280", f"  {len(g._current_skills)} skills tracked\n"))

        name_w = _clamp(w // 5, 10, 25)
        spark_w = _clamp(w // 4, 8, 20)

        sorted_skills = sorted(
            g._current_skills.values(),
            key=lambda s: (-s.bloat_score, -s.version)
        )

        lines = 1
        for sk in sorted_skills[:max_h - 4]:
            if lines >= max_h - 2:
                break
            history = g.skill_history.get(sk.id, [sk])
            spark = _confidence_sparkline_ft(history, spark_w)
            bloat_pct = int(sk.bloat_score * 100)
            bloat_bar = _dream_bar(bloat_pct, 6)

            a_sym = {"evolved": DSYM["evolve"], "created": DSYM["new"],
                     "split": DSYM["split"]}.get(sk.action, " ")

            name_str = sk.name[:name_w].ljust(name_w)

            out.append((f"fg:{sk.health_color}", f"  {a_sym}{DSYM['skill']} "))
            out.append(("fg:#e5e7eb", name_str))
            out.append(("fg:#6b7280", f"v{sk.version:<3} "))
            out.extend(spark)
            out.append(("fg:#6b7280", " c="))
            out.append((f"fg:{_confidence_color(sk.confidence)}", f"{sk.confidence:.2f}"))
            out.append(("  ", "  "))
            out.append((f"fg:{sk.health_color}", "bloat "))
            out.extend(bloat_bar)
            out.append((f"fg:{sk.health_color}", f" {bloat_pct}%"))
            out.append(("fg:#6b7280", f"  t:{sk.trigger_count} tools:{sk.tool_count}\n"))
            lines += 1

            # Bloat warning
            if sk.bloat_score > 0.6:
                parts = []
                if sk.trigger_count > 10:
                    parts.append(f"triggers:{sk.trigger_count}>10")
                if sk.instruction_len > 1500:
                    parts.append(f"instruction:{sk.instruction_len}ch>1500")
                if sk.tool_count > 10:
                    parts.append(f"tools:{sk.tool_count}>10")
                if parts:
                    out.append(("fg:#f87171", f"    {DSYM['warn']} {', '.join(parts)}\n"))
                    lines += 1

        if not sorted_skills:
            out.append(("fg:#6b7280", "    No skills tracked yet.\n"))

        return out

    # ── Legend ─────────────────────────────────────────────────

    def _render_legend(self, w: int) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        out.append(("fg:#374151", f"  {'─' * (w - 4)}\n"))
        out.append(("fg:#fbbf24", f"  {DSYM['dream']}dream "))
        out.append(("fg:#4ade80", f"{DSYM['ok']}ok "))
        out.append(("fg:#f87171", f"{DSYM['error']}fail "))
        out.append(("fg:#6b7280", f"{DSYM['cluster']}cluster "))
        out.append(("fg:#4ade80", f"{DSYM['skill']}skill "))
        out.append(("fg:#fbbf24", f"{DSYM['evolve']}evolve "))
        out.append(("fg:#2dd4bf", f"{DSYM['new']}new "))
        out.append(("fg:#fb7185", f"{DSYM['warn']}bloat"))
        out.append(("fg:#6b7280", "  │  d:toggle tree/diamond  Esc:back\n"))
        return out


# ═══════════════════════════════════════════════════════════════
# Helpers (FormattedText versions of DreamGraphV2 helpers)
# ═══════════════════════════════════════════════════════════════

def _count_lines(parts: list[tuple[str, str]]) -> int:
    return sum(t.count("\n") for _, t in parts)


def _dream_bar(pct: int, width: int) -> list[tuple[str, str]]:
    """Compact percentage bar as FormattedText tuples."""
    pct = _clamp(pct, 0, 100)
    filled = int(width * pct / 100)
    col = DC["green"] if pct < 30 else DC["amber"] if pct < 60 else DC["red"]
    return [
        (f"fg:{col}", DSYM["bar_f"] * filled + DSYM["bar_e"] * (width - filled)),
    ]


def _confidence_color(c: float) -> str:
    if c >= 0.7:
        return DC["green"]
    elif c >= 0.4:
        return DC["amber"]
    return DC["red"]


def _confidence_sparkline_ft(history: list[SkillSnapshot], width: int) -> list[tuple[str, str]]:
    """Confidence sparkline as FormattedText."""
    if len(history) > width:
        step = len(history) / width
        sampled = [history[int(i * step)] for i in range(width)]
    else:
        sampled = history

    out: list[tuple[str, str]] = []
    for i, snap in enumerate(sampled):
        if snap.confidence >= 0.7:
            out.append((f"fg:{DC['green']}", DSYM["diamond"]))
        elif snap.confidence >= 0.4:
            out.append((f"fg:{DC['amber']}", DSYM["hollow"]))
        else:
            out.append((f"fg:{DC['red']}", DSYM["error"]))
        if i < len(sampled) - 1:
            out.append(("fg:#6b7280", DSYM["line"]))

    rendered_len = len(sampled) * 2 - 1
    if rendered_len < width:
        out.append(("fg:#3b3b3b", " " * (width - rendered_len)))

    return out


# ═══════════════════════════════════════════════════════════════
# Patch ZenPlus: add "dream" detail type + 'd' keybinding
# ═══════════════════════════════════════════════════════════════

_dream_renderer: Optional[DreamRenderer] = None


def _get_dream_renderer() -> Optional[DreamRenderer]:
    return _dream_renderer


def set_dream_renderer(graph: DreamGraphV2):
    """Attach a DreamGraphV2 to the global Zen+ dream slot."""
    global _dream_renderer
    _dream_renderer = DreamRenderer(graph)

def patch_zen_dream(_originals):
    """
    Monkey-patch ZenPlus to support 'd' key → dream detail view.

    Call once at startup. Adds:
      - 'd' in Focus mode → opens dream tree/diamond as detail
      - 'd' in dream Detail → toggles tree/diamond mode
      - Dream rendering delegated to DreamRenderer
    """
    # Patch AgentPane.render_detail to dispatch "dream"
    if hasattr(ZenPlus, "__has_dream_patch") and ZenPlus.__has_dream_patch :
        return


    _originals["render_detail"] = AgentPane.render_detail
    _originals["_detail_item_count"] = AgentPane._detail_item_count
    _originals["render_focus"] = AgentPane.render_focus
    _originals["_build_keybindings"] = ZenPlus._build_keybindings
    _originals["_render_title"] = ZenPlus._render_title
    _orig_render_detail = AgentPane.render_detail

    def _patched_render_detail(self, w: int, h: int, dtype: str):
        if dtype == "dream":
            dr = _get_dream_renderer()
            if dr:
                return dr.render(w, h)
            return [("fg:#6b7280", "  dream graph not active\n")]
        return _orig_render_detail(self, w, h, dtype)

    AgentPane.render_detail = _patched_render_detail

    # Patch _detail_item_count
    _orig_item_count = AgentPane._detail_item_count

    def _patched_item_count(self, dtype: str):
        if dtype == "dream":
            return 1  # not navigable per-item
        return _orig_item_count(self, dtype)

    AgentPane._detail_item_count = _patched_item_count

    # Patch render_focus to show [d]ream hint
    _orig_render_focus = AgentPane.render_focus

    def _patched_render_focus(self, w: int, h: int):
        out = _orig_render_focus(self, w, h)
        dr = _get_dream_renderer()
        if dr and dr.graph._active:
            # Insert dream hint into the shortcuts line
            out.append(("fg:#67e8f9", f"  [d]ream\n"))
        return out

    AgentPane.render_focus = _patched_render_focus

    # Patch _build_keybindings to add 'd'
    _orig_build_kb = ZenPlus._build_keybindings

    def _patched_build_kb(self):
        kb = _orig_build_kb(self)
        zp = self

        @kb.add("d")
        def _dream(event):
            dr = _get_dream_renderer()
            if not dr:
                return
            if zp._view == ViewMode.FOCUS:
                zp._detail_type = "dream"
                zp._view = ViewMode.DETAIL
            elif zp._view == ViewMode.DETAIL and zp._detail_type == "dream":
                # Toggle tree/diamond
                dr.toggle_mode()

        return kb

    ZenPlus._build_keybindings = _patched_build_kb

    # Patch title hints
    _orig_render_title = ZenPlus._render_title

    def _patched_render_title(self):
        parts = _orig_render_title(self)
        if self._view == ViewMode.DETAIL and self._detail_type == "dream":
            # Replace hint with dream-specific hint
            parts.append(("fg:#67e8f9", "  d=toggle tree/diamond"))
        return parts

    ZenPlus._render_title = _patched_render_title

    ZenPlus.__has_dream_patch = True
    return _originals


def unpatch_zen_dream(_originals):
    """Reverse all monkey-patches applied by patch_zen_dream()."""

    if hasattr(ZenPlus, "__has_dream_patch") and not ZenPlus.__has_dream_patch:
        return

    if "render_detail" in _originals:
        AgentPane.render_detail = _originals["render_detail"]
    if "_detail_item_count" in _originals:
        AgentPane._detail_item_count = _originals["_detail_item_count"]
    if "render_focus" in _originals:
        AgentPane.render_focus = _originals["render_focus"]
    if "_build_keybindings" in _originals:
        ZenPlus._build_keybindings = _originals["_build_keybindings"]
    if "_render_title" in _originals:
        ZenPlus._render_title = _originals["_render_title"]
    ZenPlus.__has_dream_patch = False

    _originals.clear()

# ═══════════════════════════════════════════════════════════════
# Updated dream_with_viz_v2 — runs inside ZenPlus
# ═══════════════════════════════════════════════════════════════

async def dream_with_viz_v2(
    isaa_tools,
    agent_name: str = "default",
    config=None,
    show_graph: bool = True,
):
    """
    Run dreamer with live Zen+ visualization.

    The dream pipeline tree and skill diamond are available via 'd' key
    in ZenPlus Focus view. Agent streaming is shown in the normal grid/focus.

    Usage:
        from dream_zen_adapter import patch_zen_dream, dream_with_viz_v2

        patch_zen_dream()  # once at startup
        report = await dream_with_viz_v2(self, "myagent", DreamConfig(max_budget=3000))
    """
    import asyncio
    from toolboxv2.mods.isaa.base.Agent.dreamer import Dreamer, DreamConfig

    config = config or DreamConfig()
    agent = await isaa_tools.get_agent(agent_name)

    if not show_graph:
        return await agent.a_dream(config)

    # Wire dream graph
    dreamer = getattr(agent, '_dreamer', None) or Dreamer(agent)
    graph = hookup_v2(dreamer=dreamer)

    # Snapshot existing skills
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

    # Attach to Zen+
    set_dream_renderer(graph)
    zp = ZenPlus.get()

    # Run dream as background task, feed progress into Zen+
    report = None

    async def dream_task():
        nonlocal report
        try:
            report = await dreamer.dream(config)
        finally:
            graph._active = False
            zp.signal_stream_done()

    # Feed dream phase events as synthetic chunks so Zen+ shows activity
    _orig_on_phase_start = graph.on_phase_start
    _orig_on_phase_end = graph.on_phase_end

    def _hooked_phase_start(phase_name: str):
        _orig_on_phase_start(phase_name)
        short = phase_name.replace("_phase_", "")
        label = graph._phase_labels.get(short, short)
        zp.feed_chunk({
            "agent": agent_name,
            "type": "reasoning",
            "chunk": f"[Dream] {label} started...",
            "iter": 0,
            "max_iter": len(graph._phase_order),
        })

    def _hooked_phase_end(phase_name: str, success: bool, detail: str = ""):
        _orig_on_phase_end(phase_name, success, detail)
        short = phase_name.replace("_phase_", "")
        label = graph._phase_labels.get(short, short)
        status = "✓" if success else "✗"
        zp.feed_chunk({
            "agent": agent_name,
            "type": "content",
            "chunk": f"{status} {label}: {detail}\n",
            "iter": 0,
            "max_iter": len(graph._phase_order),
        })
        if zp._app:
            zp._app.invalidate()

    graph.on_phase_start = _hooked_phase_start
    graph.on_phase_end = _hooked_phase_end

    # Periodic invalidate for dream animation
    async def dream_refresh():
        while graph._active:
            if zp._app:
                zp._app.invalidate()
            await asyncio.sleep(0.3)

    task = asyncio.create_task(dream_task())
    refresh = asyncio.create_task(dream_refresh())

    try:
        await zp.start()
    finally:
        graph._active = False
        refresh.cancel()
        if not task.done():
            await task
        try:
            await refresh
        except asyncio.CancelledError:
            pass

    return report
