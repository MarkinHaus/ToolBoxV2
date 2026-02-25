"""
ZenPlus Phase 3 — Fullscreen TUI with 3D force-directed graph.

Features:
  - Always-moving 3D ASCII graph with force simulation + auto-rotation
  - 3 view layers: Grid → Focus → Detail (graph/tools/thoughts/iterations)
  - Full information on selection (no truncation)
  - Syntax highlighting for JSON/MD/config in tool results
  - Sub-agent nodes with twinkling star effect + distinct colors
  - Job/task injection for background processes

Navigation:
  Grid:   Tab/↑↓←→ select, Enter=focus, Esc=exit to Zen
  Focus:  ↑↓ scroll, g=3D graph, t=tools, i=iterations, h=thoughts, Esc=grid
  Detail: ↑↓ navigate, Enter=expand, Esc=back
"""

import asyncio
import html as _html
import json as _json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Callable, Any, Tuple

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

SYM = {
    "agent": "◯", "sub": "◈", "tool": "◇", "think": "◎",
    "ok": "✓", "fail": "✗", "done": "●", "warn": "⚠",
    "job": "⧫", "task": "▹", "bg": "▾", "arrow": "→",
    "bar_fill": "━", "bar_empty": "─", "vline": "│",
    "select": "▸", "star": "✦", "iter": "◆",
}

C = {
    "dim": "#6b7280", "cyan": "#67e8f9", "green": "#4ade80",
    "red": "#f87171", "amber": "#fbbf24", "white": "#e5e7eb",
    "bright": "#ffffff", "blue": "#60a5fa", "purple": "#a78bfa",
    "deep": "#374151", "pink": "#f472b6", "teal": "#2dd4bf",
    "orange": "#fb923c",
}

# Sub-agent color cycle (twinkling palette)
SUB_COLORS = ["#f472b6", "#a78bfa", "#2dd4bf", "#fb923c", "#818cf8", "#34d399"]

STATUS_COLOR = {
    "running": C["cyan"], "completed": C["green"], "done": C["green"],
    "failed": C["red"], "cancelled": C["amber"], "error": C["red"],
    "pending": C["dim"], "waiting": C["dim"],
}

GRAPH_NODE_SYM = {
    "agent": "◯", "tool": "◇", "thought": "◎", "content": "·",
    "sub_agent": "✦", "iteration": "◆", "job": "⧫",
}
GRAPH_NODE_COLOR = {
    "agent": C["cyan"], "tool": C["blue"], "thought": C["dim"],
    "content": C["white"], "sub_agent": C["pink"], "iteration": C["green"],
    "job": C["purple"],
}


def _short(s: str, n: int = 40) -> str:
    return s[:n] + ".." if len(s) > n + 2 else s


def _bar(cur: int, total: int, w: int = 15) -> str:
    if total <= 0:
        return SYM["bar_empty"] * w
    f = int(w * min(cur, total) / total)
    return SYM["bar_fill"] * f + SYM["bar_empty"] * (w - f)

def _fmt_elapsed(secs: float) -> str:
    s = int(secs)

    weeks, s = divmod(s, 604800)   # 7 * 24 * 3600
    days, s = divmod(s, 86400)

    if weeks > 0:
        return f"{weeks}w{days}d"

    hours, s = divmod(s, 3600)

    if days > 0:
        return f"{days}d{hours:02d}h"

    minutes, seconds = divmod(s, 60)

    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"

class ViewMode(Enum):
    GRID = "grid"
    FOCUS = "focus"
    DETAIL = "detail"


# ═══════════════════════════════════════════════════════════════
# Syntax highlighting (dependency-free)
# ═══════════════════════════════════════════════════════════════

def syntax_highlight(text: str, lang: str = "") -> list[tuple[str, str]]:
    """Return (style, text) tuples with syntax coloring. No dependencies."""
    if not text.strip():
        return [("fg:#6b7280", "(empty)\n")]

    if not lang:
        lang = _detect_lang(text)

    if lang == "json":
        return _hl_json(text)
    elif lang == "markdown":
        return _hl_markdown(text)
    else:
        return _hl_generic(text)


def _detect_lang(text: str) -> str:
    t = text.strip()
    if t.startswith("{") or t.startswith("["):
        try:
            _json.loads(t)
            return "json"
        except Exception:
            pass
    if t.startswith("#") or "```" in t or re.search(r'\*\*.*\*\*', t):
        return "markdown"
    return "text"


def _hl_json(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    try:
        obj = _json.loads(text)
        formatted = _json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        formatted = text

    for line in formatted.split("\n"):
        # Colorize JSON structure
        m = re.match(r'^(\s*)"([^"]+)"(\s*:\s*)(.*)', line)
        if m:
            out.append(("", m.group(1)))
            out.append(("fg:#67e8f9", f'"{m.group(2)}"'))  # key: cyan
            out.append(("fg:#6b7280", m.group(3)))          # colon: dim
            val = m.group(4)
            out.append((_json_val_color(val), val))
            out.append(("", "\n"))
        else:
            # Brackets, values
            color = "fg:#6b7280" if line.strip() in "{}[]," else "fg:#e5e7eb"
            out.append((color, f"{line}\n"))
    return out


def _json_val_color(val: str) -> str:
    v = val.strip().rstrip(",")
    if v.startswith('"'):
        return "fg:#4ade80"    # string: green
    if v in ("true", "false"):
        return "fg:#fbbf24"    # bool: amber
    if v == "null":
        return "fg:#f87171"    # null: red
    try:
        float(v)
        return "fg:#fb923c"    # number: orange
    except ValueError:
        return "fg:#e5e7eb"


def _hl_markdown(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    in_code = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(("fg:#6b7280", f"{line}\n"))
        elif in_code:
            out.append(("fg:#4ade80", f"{line}\n"))
        elif line.startswith("#"):
            out.append(("fg:#67e8f9 bold", f"{line}\n"))
        elif re.match(r'^\s*[-*]\s', line):
            out.append(("fg:#fbbf24", f"{line}\n"))
        elif "**" in line:
            parts = re.split(r'(\*\*.*?\*\*)', line)
            for p in parts:
                if p.startswith("**") and p.endswith("**"):
                    out.append(("fg:#ffffff bold", p))
                else:
                    out.append(("fg:#e5e7eb", p))
            out.append(("", "\n"))
        else:
            out.append(("fg:#e5e7eb", f"{line}\n"))
    return out


def _hl_generic(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i, line in enumerate(text.split("\n")):
        out.append(("fg:#6b7280", f" {i + 1:3d} "))
        # Highlight key=value patterns
        if "=" in line or ":" in line:
            m = re.match(r'^(\s*)(\S+?)(\s*[:=]\s*)(.*)', line)
            if m:
                out.append(("", m.group(1)))
                out.append(("fg:#67e8f9", m.group(2)))
                out.append(("fg:#6b7280", m.group(3)))
                out.append(("fg:#e5e7eb", m.group(4)))
                out.append(("", "\n"))
                continue
        out.append(("fg:#e5e7eb", f"{line}\n"))
    return out


# ═══════════════════════════════════════════════════════════════
# MiniGraph3D — Force-directed ASCII 3D graph
# ═══════════════════════════════════════════════════════════════

@dataclass
class GNode:
    id: str
    label: str
    kind: str = "agent"
    status: str = "active"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    is_sub: bool = False
    sub_color_idx: int = 0

# Dynamic label length per node kind
_LABEL_LEN = {
    "thought": 35, "iteration": 10, "tool": 14,
    "sub_agent": 18, "agent": 20, "content": 20, "job": 16,
}


@dataclass
class GEdge:
    src: str
    dst: str


class MiniGraph3D:
    """Embedded force-directed 3D graph for AgentPane detail view."""

    def __init__(self):
        self.nodes: Dict[str, GNode] = {}
        self.edges: List[GEdge] = []
        self.azimuth = 0.0
        self.elevation = 0.3
        self.zoom = 1.2
        self._frame = 0
        self._paused = False
        self._selected: Optional[str] = None
        self._current: Optional[str] = None   # latest active node (agent's head)
        self._detail_node: Optional[str] = None  # Enter-opened node (None=closed)
        # Physics params
        self._repulsion = 0.4
        self._attraction = 0.15
        self._damping = 0.88
        self._center_pull = 0.035
        self._last_interact = time.time()

    def add_node(self, id: str, label: str, kind: str = "agent",
                 is_sub: bool = False, sub_color_idx: int = 0):
        if id not in self.nodes:
            n = len(self.nodes)
            angle = n * 2.399
            r = 0.3 + (n % 5) * 0.15
            self.nodes[id] = GNode(
                id=id, label=label, kind=kind,
                x=r * math.cos(angle), y=r * math.sin(angle),
                z=0.2 * math.sin(n * 1.7),
                is_sub=is_sub, sub_color_idx=sub_color_idx,
            )
            self._current = id  # latest node = current

    def add_edge(self, src: str, dst: str):
        if src in self.nodes and dst in self.nodes:
            for e in self.edges:
                if e.src == src and e.dst == dst:
                    return
            self.edges.append(GEdge(src, dst))

    def update_status(self, id: str, status: str):
        if id in self.nodes:
            self.nodes[id].status = status

    def select(self, idx: int):
        """Select node by index."""
        keys = list(self.nodes.keys())
        if 0 <= idx < len(keys):
            self._selected = keys[idx]
            self._detail_node = None
            self._last_interact = time.time()

    def select_by_id(self, nid: str):
        if nid in self.nodes:
            self._selected = nid
            self._detail_node = None
            self._last_interact = time.time()

    @property
    def selected_node(self) -> Optional[GNode]:
        return self.nodes.get(self._selected) if self._selected else None

    def _selected_edges(self) -> set[str]:
        """IDs of all nodes connected to selected node."""
        if not self._selected:
            return set()
        connected = set()
        for e in self.edges:
            if e.src == self._selected:
                connected.add(e.dst)
            elif e.dst == self._selected:
                connected.add(e.src)
        return connected

    def navigate_edge(self, direction: int):
        """Navigate along edges: -1=parent/src, +1=child/dst."""
        if not self._selected:
            return
        targets = []
        for e in self.edges:
            if direction > 0 and e.src == self._selected:
                targets.append(e.dst)
            elif direction < 0 and e.dst == self._selected:
                targets.append(e.src)
        if not targets:
            # Fallback: any connected
            for e in self.edges:
                if e.src == self._selected:
                    targets.append(e.dst)
                elif e.dst == self._selected:
                    targets.append(e.src)
        if targets:
            self._selected = targets[0]
            self._detail_node = None
            self._last_interact = time.time()

    def toggle_detail(self):
        """Enter: open/close detail for selected node."""
        self._last_interact = time.time()
        if self._detail_node:
            self._detail_node = None
        elif self._selected:
            self._detail_node = self._selected

    def tick(self):
        """One physics + animation step."""
        if self._paused:
            self._frame += 1
            return

        nodes = list(self.nodes.values())
        n = len(nodes)

        # Repulsion (all pairs)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = nodes[i], nodes[j]
                dx = a.x - b.x
                dy = a.y - b.y
                dz = a.z - b.z
                d2 = dx * dx + dy * dy + dz * dz + 0.001
                f = self._repulsion / d2
                fx, fy, fz = f * dx, f * dy, f * dz
                a.vx += fx; a.vy += fy; a.vz += fz
                b.vx -= fx; b.vy -= fy; b.vz -= fz

        # Attraction (edges)
        for e in self.edges:
            a, b = self.nodes.get(e.src), self.nodes.get(e.dst)
            if not a or not b:
                continue
            dx = b.x - a.x
            dy = b.y - a.y
            dz = b.z - a.z
            d = math.sqrt(dx * dx + dy * dy + dz * dz) + 0.001
            f = self._attraction * (d - 0.4)  # rest length 0.4
            fx, fy, fz = f * dx / d, f * dy / d, f * dz / d
            a.vx += fx; a.vy += fy; a.vz += fz
            b.vx -= fx; b.vy -= fy; b.vz -= fz

        # Center pull + damping + position update
        for nd in nodes:
            nd.vx -= nd.x * self._center_pull
            nd.vy -= nd.y * self._center_pull
            nd.vz -= nd.z * self._center_pull
            nd.vx *= self._damping
            nd.vy *= self._damping
            nd.vz *= self._damping
            nd.x += nd.vx
            nd.y += nd.vy
            nd.z += nd.vz
            # Clamp
            nd.x = max(-1.2, min(1.2, nd.x))
            nd.y = max(-1.2, min(1.2, nd.y))
            nd.z = max(-1.0, min(1.0, nd.z))

        # Auto-rotate
        self.azimuth += 0.015
        self._frame += 1

        # Auto-reset selection to newest after 5s inactivity
        if (self._current and self._selected != self._current and
                time.time() - self._last_interact > 60.0):
            self._selected = self._current
            self._detail_node = None

    def render(self, cols: int, rows: int) -> list[tuple[str, str]]:
        """Render to FormattedText tuples."""
        self.tick()

        if not self.nodes:
            return [("fg:#6b7280", "  ◎ graph: no nodes\n")]

        # Reserve rows for stats/detail/help below graph
        rows = max(4, rows - 2 - (8 if self._detail_node else 0))

        # Character + color buffer
        buf = [[' '] * cols for _ in range(rows)]
        cbuf = [[C["deep"]] * cols for _ in range(rows)]
        zbuf = [[float('-inf')] * cols for _ in range(rows)]

        cos_a = math.cos(self.azimuth)
        sin_a = math.sin(self.azimuth)
        cos_e = math.cos(self.elevation)
        sin_e = math.sin(self.elevation)

        def project(x, y, z):
            rx = x * cos_a - z * sin_a
            rz = x * sin_a + z * cos_a
            ry2 = y * cos_e - rz * sin_e
            rz2 = y * sin_e + rz * cos_e
            d = 3.0 + rz2
            if d < 0.1:
                return -1, -1, -999
            sc = int(rx / d * self.zoom * cols * 0.4 + cols * 0.5)
            sr = int(ry2 / d * self.zoom * rows * 0.4 + rows * 0.5)
            return sc, sr, rz2

        # Draw edges (highlight connected to selected)
        sel_connected = self._selected_edges()
        for e in self.edges:
            a, b = self.nodes.get(e.src), self.nodes.get(e.dst)
            if not a or not b:
                continue
            ac, ar, az = project(a.x, a.y, a.z)
            bc, br, bz = project(b.x, b.y, b.z)
            steps = max(abs(bc - ac), abs(br - ar), 1)
            ez = (az + bz) / 2
            # Highlighted if connected to selected
            is_hl = (self._selected and
                     (e.src == self._selected or e.dst == self._selected))
            edge_ch = '─' if is_hl else '·'
            edge_col = C["cyan"] if is_hl else C["deep"]
            for s in range(1, steps):
                t = s / steps
                c = int(ac + (bc - ac) * t)
                r = int(ar + (br - ar) * t)
                if 0 <= c < cols and 0 <= r < rows and zbuf[r][c] < ez - 0.5:
                    buf[r][c] = edge_ch
                    cbuf[r][c] = edge_col
                    zbuf[r][c] = ez - 0.5

        # Draw nodes (z-sorted)
        projected = []
        for nd in self.nodes.values():
            sc, sr, sz = project(nd.x, nd.y, nd.z)
            if 0 <= sc < cols and 0 <= sr < rows:
                projected.append((sc, sr, sz, nd))
        projected.sort(key=lambda p: p[2])

        sel_connected = self._selected_edges()

        for sc, sr, sz, nd in projected:
            if zbuf[sr][sc] > sz:
                continue
            # Skip ghost nodes (no label)
            if not nd.label.strip():
                continue
            zbuf[sr][sc] = sz

            is_current = nd.id == self._current
            is_sel = nd.id == self._selected
            is_connected = nd.id in sel_connected

            # Node color
            if is_sel:
                col = C["bright"]
            elif is_current and not is_sel:
                # Current node: pulsing green
                col = C["green"] if self._frame % 4 < 3 else C["bright"]
            elif is_connected:
                # Connected to selected: soft highlight
                col = C["cyan"] if self._frame % 6 < 4 else C["blue"]
            elif nd.is_sub:
                base_col = SUB_COLORS[nd.sub_color_idx % len(SUB_COLORS)]
                if self._frame % 8 < 3:
                    col = C["bright"]
                elif self._frame % 8 < 5:
                    col = base_col
                else:
                    col = C["dim"]
            elif nd.status == "done":
                col = C["green"]
            elif nd.status == "failed":
                col = C["red"]
            elif nd.status == "active":
                col = GRAPH_NODE_COLOR.get(nd.kind, C["cyan"])
                if self._frame % 6 < 2:
                    col = C["bright"]
            else:
                col = GRAPH_NODE_COLOR.get(nd.kind, C["dim"])
                if sz < -0.5:
                    col = C["deep"]
                elif sz < 0:
                    col = C["dim"]

            sym = SYM["star"] if nd.is_sub else GRAPH_NODE_SYM.get(nd.kind, "?")
            if is_current and not is_sel:
                sym = "▶"
            buf[sr][sc] = sym
            cbuf[sr][sc] = col

            # Dynamic label length by kind
            max_lbl = _LABEL_LEN.get(nd.kind, 18)
            if is_sel:
                max_lbl = max(max_lbl, 30)  # selected always gets more
            if sz > -0.2:
                lbl = nd.label[:max_lbl]
                # Clamp label to screen bounds
                max_chars = max(0, cols - sc - 2)
                lbl = lbl[:max_chars]
                lbl_col = C["white"] if is_sel else (C["cyan"] if is_connected else C["dim"])
                for k, ch in enumerate(lbl):
                    c2 = sc + 1 + k
                    if 0 <= c2 < cols and zbuf[sr][c2] <= sz:
                        buf[sr][c2] = ch
                        cbuf[sr][c2] = lbl_col
                        zbuf[sr][c2] = sz

        # Build FormattedText with run-length encoding
        out: list[tuple[str, str]] = []
        for r in range(rows):
            cur_col = None
            seg: list[str] = []
            for c_ in range(cols):
                if cbuf[r][c_] != cur_col:
                    if seg:
                        out.append((f"fg:{cur_col}", _html.escape("".join(seg))))
                    seg = [buf[r][c_]]
                    cur_col = cbuf[r][c_]
                else:
                    seg.append(buf[r][c_])
            if seg:
                out.append((f"fg:{cur_col}", _html.escape("".join(seg))))
            out.append(("", "\n"))

        # Stats bar
        n_active = sum(1 for nd in self.nodes.values() if nd.status == "active")
        n_sub = sum(1 for nd in self.nodes.values() if nd.is_sub)
        out.append(("fg:#67e8f9", f" {len(self.nodes)}nodes "))
        out.append(("fg:#6b7280", f"{len(self.edges)}edges "))
        out.append(("fg:#4ade80", f"●{n_active} "))
        if n_sub:
            out.append(("fg:#f472b6", f"✦{n_sub} "))
        # Current node indicator
        if self._current and self._current in self.nodes:
            cur = self.nodes[self._current]
            out.append(("fg:#4ade80 bold", f"▶{cur.label[:14]} "))
        # Selection
        if self._selected and self._selected in self.nodes:
            sel = self.nodes[self._selected]
            out.append(("fg:#ffffff bold", f" sel:{sel.label[:20]}"))
        out.append(("", "\n"))

        # ── Node detail panel (Enter-opened) ──
        if self._detail_node and self._detail_node in self.nodes:
            dnd = self.nodes[self._detail_node]
            out.append(("fg:#374151", f" {'─' * (cols - 2)}\n"))
            kind_sym = GRAPH_NODE_SYM.get(dnd.kind, "?")
            status_col = {"active": C["cyan"], "done": C["green"],
                          "failed": C["red"]}.get(dnd.status, C["dim"])
            out.append(("fg:#ffffff bold", f" {kind_sym} {dnd.label}\n"))
            out.append(("fg:#6b7280", f"   id: {dnd.id}  kind: {dnd.kind}  "))
            out.append((f"fg:{status_col}", f"status: {dnd.status}"))
            if dnd.is_sub:
                out.append(("fg:#f472b6", f"  ✦sub"))
            out.append(("", "\n"))
            # Show edges
            parents = [e.src for e in self.edges if e.dst == dnd.id]
            children = [e.dst for e in self.edges if e.src == dnd.id]
            if parents:
                p_labels = [self.nodes[p].label[:15] for p in parents if p in self.nodes]
                out.append(("fg:#6b7280", f"   ← from: "))
                out.append(("fg:#67e8f9", f"{', '.join(p_labels)}\n"))
            if children:
                c_labels = [self.nodes[c].label[:15] for c in children if c in self.nodes]
                out.append(("fg:#6b7280", f"   → to:   "))
                out.append(("fg:#67e8f9", f"{', '.join(c_labels)}\n"))
            out.append(("fg:#6b7280", f"   Enter=close  ←→=follow edges\n"))

        out.append(("fg:#6b7280", " ↑↓=select  ←=parent  →=child  Enter=detail  Esc=back\n"))

        return out


# ═══════════════════════════════════════════════════════════════
# GlobalGraph — Cross-agent overview
# ═══════════════════════════════════════════════════════════════

# Edge relationship types + colors
REL_PARENT = "parent"       # agent → sub-agent
REL_RESOURCE = "resource"   # shared file/path
REL_TASK = "task"           # similar/same task
REL_SIMILAR = "similar"     # related actions

REL_COLOR = {
    REL_PARENT: C["cyan"],
    REL_RESOURCE: C["amber"],
    REL_TASK: C["green"],
    REL_SIMILAR: C["purple"],
}
REL_CHAR = {
    REL_PARENT: "─",
    REL_RESOURCE: "═",
    REL_TASK: "━",
    REL_SIMILAR: "·",
}

# Colors per agent slot
AGENT_COLORS = [C["cyan"], C["green"], C["blue"], C["purple"],
                C["pink"], C["teal"], C["orange"], C["amber"]]


@dataclass
class GlobalEdge:
    src: str
    dst: str
    rel: str = REL_PARENT  # parent, resource, task, similar


@dataclass
class GlobalNode:
    """A node in the global graph: represents an agent action."""
    id: str
    agent: str
    label: str
    kind: str = "action"     # action, agent_hub, sub_hub
    resource: str = ""       # file path / url if applicable
    task_hint: str = ""      # query / task description
    original_nid: str = ""   # id in agent's own graph (for jump)
    newest: bool = False
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


class GlobalGraph:
    """Cross-agent 3D graph. Shows last N actions per agent, shared resources,
    same-task connections, sub-agent relationships."""

    MAX_ACTIONS = 3  # last N actions per agent

    def __init__(self):
        self.nodes: Dict[str, GlobalNode] = {}
        self.edges: List[GlobalEdge] = []
        self._selected: Optional[str] = None
        self._detail_node: Optional[str] = None
        self._frame = 0
        self.azimuth = 0.0
        self.elevation = 0.25
        self.zoom = 1.1
        self._damping = 0.86
        self._repulsion = 0.5
        self._attraction = 0.12
        self._center_pull = 0.04
        self._last_interact = time.time()

    def rebuild(self, panes: Dict[str, "AgentPane"]):
        """Rebuild from all AgentPanes. Called periodically."""
        old_sel = self._selected
        old_detail = self._detail_node

        self.nodes.clear()
        self.edges.clear()

        agent_idx: Dict[str, int] = {}
        all_resources: Dict[str, list[str]] = {}   # path → [node_ids]
        all_tasks: Dict[str, list[str]] = {}        # agent_name → task_hint

        for ai, (agent_name, pane) in enumerate(panes.items()):
            agent_idx[agent_name] = ai
            # Agent hub node
            hub_id = f"hub_{agent_name}"
            angle = ai * (2 * math.pi / max(len(panes), 1))
            r = 0.8
            self.nodes[hub_id] = GlobalNode(
                id=hub_id, agent=agent_name, label=agent_name,
                kind="agent_hub", x=r * math.cos(angle),
                y=r * math.sin(angle), z=0.0,
            )

            # Sub-agent hubs
            for sub_name in pane.sub_agents:
                sub_hub = f"sub_{agent_name}_{sub_name}"
                sa = angle + 0.4
                self.nodes[sub_hub] = GlobalNode(
                    id=sub_hub, agent=agent_name, label=f"✦{sub_name}",
                    kind="sub_hub", x=r * 0.5 * math.cos(sa),
                    y=r * 0.5 * math.sin(sa), z=0.1,
                )
                self.edges.append(GlobalEdge(hub_id, sub_hub, REL_PARENT))

            # Last N tool events as action nodes
            recent_tools = pane.tool_history[-self.MAX_ACTIONS:]
            for ti, ev in enumerate(recent_tools):
                is_newest = (ti == len(recent_tools) - 1)
                nid = f"act_{agent_name}_{ev.name}_{ev.iteration}"
                spread = 0.3 + ti * 0.15
                self.nodes[nid] = GlobalNode(
                    id=nid, agent=agent_name,
                    label=f"{ev.name}" + (f"({ev.sub_agent})" if ev.is_sub else ""),
                    kind="action", newest=is_newest,
                    original_nid=f"tool_{pane.tool_history.index(ev) + 1}",
                    x=r * math.cos(angle) + spread * math.cos(angle + ti * 0.5),
                    y=r * math.sin(angle) + spread * math.sin(angle + ti * 0.5),
                    z=0.1 * ti,
                )

                # Connect to agent hub (or sub hub if sub-agent)
                if ev.is_sub and ev.sub_agent:
                    sub_hub = f"sub_{agent_name}_{ev.sub_agent}"
                    if sub_hub in self.nodes:
                        self.edges.append(GlobalEdge(sub_hub, nid, REL_PARENT))
                    else:
                        self.edges.append(GlobalEdge(hub_id, nid, REL_PARENT))
                else:
                    self.edges.append(GlobalEdge(hub_id, nid, REL_PARENT))

                # Extract resource paths for cross-agent linking
                resource = ""
                if ev.args_parsed and isinstance(ev.args_parsed, dict):
                    for k in ("path", "file", "url", "target"):
                        if k in ev.args_parsed:
                            resource = str(ev.args_parsed[k])
                            break
                if resource:
                    self.nodes[nid].resource = resource
                    all_resources.setdefault(resource, []).append(nid)

                # Extract task hint
                if ev.args_parsed and isinstance(ev.args_parsed, dict):
                    for k in ("query", "task", "command"):
                        if k in ev.args_parsed:
                            self.nodes[nid].task_hint = str(ev.args_parsed[k])[:50]
                            break

        # ── Cross-agent edges ──

        # Same resource (read/write same file) → amber
        for path, nids in all_resources.items():
            if len(nids) < 2:
                continue
            agents_seen = set()
            for nid in nids:
                if nid in self.nodes:
                    agents_seen.add(self.nodes[nid].agent)
            if len(agents_seen) >= 2:
                # Connect all pairs from different agents
                for i in range(len(nids)):
                    for j in range(i + 1, len(nids)):
                        a, b = nids[i], nids[j]
                        if a in self.nodes and b in self.nodes:
                            if self.nodes[a].agent != self.nodes[b].agent:
                                self.edges.append(GlobalEdge(a, b, REL_RESOURCE))

        # Same tool name across agents → similar (purple)
        tool_nodes: Dict[str, list[str]] = {}
        for nid, nd in self.nodes.items():
            if nd.kind == "action":
                base_name = nd.label.split("(")[0]  # strip sub-agent suffix
                tool_nodes.setdefault(base_name, []).append(nid)
        for tool_name, nids in tool_nodes.items():
            agents_seen = set()
            for nid in nids:
                if nid in self.nodes:
                    agents_seen.add(self.nodes[nid].agent)
            if len(agents_seen) >= 2:
                for i in range(len(nids)):
                    for j in range(i + 1, len(nids)):
                        a, b = nids[i], nids[j]
                        if a in self.nodes and b in self.nodes:
                            if self.nodes[a].agent != self.nodes[b].agent:
                                self.edges.append(GlobalEdge(a, b, REL_SIMILAR))

        # Sub-agent connections across agents
        for nid, nd in self.nodes.items():
            if nd.kind == "sub_hub":
                sub_name = nd.label.lstrip("✦")
                for nid2, nd2 in self.nodes.items():
                    if nid2 != nid and nd2.kind == "sub_hub" and sub_name in nd2.label:
                        self.edges.append(GlobalEdge(nid, nid2, REL_TASK))

        # Restore selection
        if old_sel and old_sel in self.nodes:
            self._selected = old_sel
        elif self.nodes:
            # Select newest action
            newest = [nid for nid, nd in self.nodes.items() if nd.newest]
            self._selected = newest[-1] if newest else list(self.nodes.keys())[-1]

        if old_detail and old_detail in self.nodes:
            self._detail_node = old_detail
        else:
            self._detail_node = None

    def select(self, idx: int):
        keys = list(self.nodes.keys())
        if 0 <= idx < len(keys):
            self._selected = keys[idx]
            self._detail_node = None
            self._last_interact = time.time()

    def select_by_id(self, nid: str):
        if nid in self.nodes:
            self._selected = nid
            self._detail_node = None
            self._last_interact = time.time()

    @property
    def selected_node(self) -> Optional[GlobalNode]:
        return self.nodes.get(self._selected) if self._selected else None

    def _connected(self) -> set[str]:
        if not self._selected:
            return set()
        s = set()
        for e in self.edges:
            if e.src == self._selected:
                s.add(e.dst)
            elif e.dst == self._selected:
                s.add(e.src)
        return s

    def navigate_edge(self, direction: int):
        if not self._selected:
            return
        targets = []
        for e in self.edges:
            if direction > 0 and e.src == self._selected:
                targets.append(e.dst)
            elif direction < 0 and e.dst == self._selected:
                targets.append(e.src)
        if not targets:
            for e in self.edges:
                if e.src == self._selected:
                    targets.append(e.dst)
                elif e.dst == self._selected:
                    targets.append(e.src)
        if targets:
            self._selected = targets[0]
            self._detail_node = None
            self._last_interact = time.time()

    def toggle_detail(self):
        self._last_interact = time.time()
        if self._detail_node:
            self._detail_node = None
        elif self._selected:
            self._detail_node = self._selected

    def tick(self):
        """Physics step."""
        nodes = list(self.nodes.values())
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = nodes[i], nodes[j]
                dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
                d2 = dx * dx + dy * dy + dz * dz + 0.001
                f = self._repulsion / d2
                a.vx += f * dx; a.vy += f * dy; a.vz += f * dz
                b.vx -= f * dx; b.vy -= f * dy; b.vz -= f * dz

        for e in self.edges:
            a, b = self.nodes.get(e.src), self.nodes.get(e.dst)
            if not a or not b:
                continue
            dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
            d = math.sqrt(dx * dx + dy * dy + dz * dz) + 0.001
            rest = 0.5 if e.rel == REL_PARENT else 0.7
            f = self._attraction * (d - rest)
            a.vx += f * dx / d; a.vy += f * dy / d; a.vz += f * dz / d
            b.vx -= f * dx / d; b.vy -= f * dy / d; b.vz -= f * dz / d

        for nd in nodes:
            nd.vx -= nd.x * self._center_pull
            nd.vy -= nd.y * self._center_pull
            nd.vz -= nd.z * self._center_pull
            nd.vx *= self._damping; nd.vy *= self._damping; nd.vz *= self._damping
            nd.x += nd.vx; nd.y += nd.vy; nd.z += nd.vz
            nd.x = max(-1.5, min(1.5, nd.x))
            nd.y = max(-1.5, min(1.5, nd.y))
            nd.z = max(-1.0, min(1.0, nd.z))

        self.azimuth += 0.012
        self._frame += 1

        # Auto-reset to newest after 5s inactivity
        newest = [nid for nid, nd in self.nodes.items() if nd.newest]
        if (newest and self._selected not in newest and
                time.time() - self._last_interact > 60.0):
            self._selected = newest[-1]
            self._detail_node = None

    def render(self, cols: int, rows: int) -> list[tuple[str, str]]:
        """Render global graph to FormattedText."""
        self.tick()

        if not self.nodes:
            return [("fg:#6b7280", "  ◎ global graph: no agents\n")]

        # Reserve rows for legend/detail/help below graph
        rows = max(4, rows - 2 - (8 if self._detail_node else 0))

        buf = [[' '] * cols for _ in range(rows)]
        color_buf = [[C["deep"]] * cols for _ in range(rows)]
        zbuf = [[float('-inf')] * cols for _ in range(rows)]

        cos_a, sin_a = math.cos(self.azimuth), math.sin(self.azimuth)
        cos_e, sin_e = math.cos(self.elevation), math.sin(self.elevation)

        def proj(x, y, z):
            rx = x * cos_a - z * sin_a
            rz = x * sin_a + z * cos_a
            ry2 = y * cos_e - rz * sin_e
            rz2 = y * sin_e + rz * cos_e
            d = 3.0 + rz2
            if d < 0.1:
                return -1, -1, -999
            return (int(rx / d * self.zoom * cols * 0.4 + cols * 0.5),
                    int(ry2 / d * self.zoom * rows * 0.4 + rows * 0.5), rz2)

        connected = self._connected()

        # Edges
        for e in self.edges:
            a, b = self.nodes.get(e.src), self.nodes.get(e.dst)
            if not a or not b:
                continue
            ac, ar, az = proj(a.x, a.y, a.z)
            bc, br, bz = proj(b.x, b.y, b.z)
            steps = max(abs(bc - ac), abs(br - ar), 1)
            ez = (az + bz) / 2
            is_hl = self._selected and (e.src == self._selected or e.dst == self._selected)
            ech = REL_CHAR.get(e.rel, '·') if is_hl else '·'
            ecol = REL_COLOR.get(e.rel, C["deep"]) if is_hl else C["deep"]
            for s in range(1, steps):
                t = s / steps
                c, r = int(ac + (bc - ac) * t), int(ar + (br - ar) * t)
                if 0 <= c < cols and 0 <= r < rows and zbuf[r][c] < ez - 0.5:
                    buf[r][c] = ech
                    color_buf[r][c] = ecol
                    zbuf[r][c] = ez - 0.5

        # Nodes
        projected = []
        for nd in self.nodes.values():
            sc, sr, sz = proj(nd.x, nd.y, nd.z)
            if 0 <= sc < cols and 0 <= sr < rows:
                projected.append((sc, sr, sz, nd))
        projected.sort(key=lambda p: p[2])

        for sc, sr, sz, nd in projected:
            if zbuf[sr][sc] > sz:
                continue
            # Skip ghost nodes (no label)
            if not nd.label.strip():
                continue
            zbuf[sr][sc] = sz

            is_sel = nd.id == self._selected
            is_conn = nd.id in connected
            agent_col = AGENT_COLORS[hash(nd.agent) % len(AGENT_COLORS)]

            # Color logic
            if is_sel:
                col = C["bright"]
            elif nd.newest:
                col = C["green"] if self._frame % 4 < 3 else C["bright"]
            elif is_conn:
                col = agent_col if self._frame % 5 < 3 else C["cyan"]
            elif nd.kind == "agent_hub":
                col = agent_col
            elif nd.kind == "sub_hub":
                col = C["pink"] if self._frame % 8 < 4 else C["dim"]
            else:
                col = agent_col if sz > -0.3 else C["dim"]

            # Symbol
            sym_map = {"agent_hub": "◯", "sub_hub": "✦", "action": "◇"}
            sym = sym_map.get(nd.kind, "·")
            if nd.newest:
                sym = "▶"
            buf[sr][sc] = sym
            color_buf[sr][sc] = col

            # Label
            max_lbl = 20 if nd.kind == "agent_hub" else (25 if is_sel else 14)
            if sz > -0.2:
                lbl = nd.label[:max_lbl]
                # Clamp label to screen bounds
                max_chars = max(0, cols - sc - 2)
                lbl = lbl[:max_chars]
                lbl_col = C["white"] if is_sel else (C["cyan"] if is_conn else C["dim"])
                for k, ch in enumerate(lbl):
                    c2 = sc + 1 + k
                    if 0 <= c2 < cols and zbuf[sr][c2] <= sz:
                        buf[sr][c2] = ch
                        color_buf[sr][c2] = lbl_col
                        zbuf[sr][c2] = sz

        # Build output
        out: list[tuple[str, str]] = []
        for r in range(rows):
            cur_col = None
            seg: list[str] = []
            for c_ in range(cols):
                if color_buf[r][c_] != cur_col:
                    if seg:
                        out.append((f"fg:{cur_col}", _html.escape("".join(seg))))
                    seg = [buf[r][c_]]
                    cur_col = color_buf[r][c_]
                else:
                    seg.append(buf[r][c_])
            if seg:
                out.append((f"fg:{cur_col}", _html.escape("".join(seg))))
            out.append(("", "\n"))

        # Legend
        out.append(("fg:#fbbf24", " ═resource "))
        out.append(("fg:#4ade80", "━task "))
        out.append(("fg:#a78bfa", "·similar "))
        out.append(("fg:#67e8f9", "─parent "))
        out.append(("fg:#6b7280", f" │ {len(self.nodes)}n {len(self.edges)}e"))
        if self._selected and self._selected in self.nodes:
            sn = self.nodes[self._selected]
            out.append(("fg:#ffffff bold", f" sel:{sn.label[:16]}"))
            out.append(("fg:#6b7280", f" @{sn.agent}"))
        out.append(("", "\n"))

        # Detail panel
        if self._detail_node and self._detail_node in self.nodes:
            dnd = self.nodes[self._detail_node]
            out.append(("fg:#374151", f" {'─' * (cols - 2)}\n"))
            agent_col = AGENT_COLORS[hash(dnd.agent) % len(AGENT_COLORS)]
            out.append((f"fg:{agent_col} bold", f" ◇ {dnd.label}"))
            out.append(("fg:#6b7280", f"  agent: {dnd.agent}  kind: {dnd.kind}\n"))
            if dnd.resource:
                out.append(("fg:#fbbf24", f"   resource: {dnd.resource}\n"))
            if dnd.task_hint:
                out.append(("fg:#6b7280", f"   task: {dnd.task_hint}\n"))
            # Edges with relationship types
            for e in self.edges:
                other = None
                if e.src == dnd.id:
                    other = e.dst
                elif e.dst == dnd.id:
                    other = e.src
                if other and other in self.nodes:
                    ecol = REL_COLOR.get(e.rel, C["dim"])
                    out.append((f"fg:{ecol}", f"   {e.rel}: {self.nodes[other].label} @{self.nodes[other].agent}\n"))
            if dnd.original_nid:
                out.append(("fg:#67e8f9", f"   Enter=jump to agent graph  "))
            out.append(("fg:#6b7280", "Esc=close\n"))

        out.append(("fg:#6b7280", " ↑↓=select  ←→=edges  Enter=detail/jump  G=close  Esc=back\n"))

        return out


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════

@dataclass
class ToolEvent:
    name: str
    args_raw: str = ""
    args_parsed: Optional[dict] = None
    result_raw: str = ""
    result_parsed: Any = None
    success: bool = True
    started: float = 0.0
    elapsed: float = 0.0
    iteration: int = 0
    is_sub: bool = False
    sub_agent: str = ""


@dataclass
class IterationInfo:
    """Everything that happened in one iteration."""
    number: int
    thoughts: list[str] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)
    content: list[str] = field(default_factory=list)
    tokens_used: int = 0
    tokens_max: int = 0


@dataclass
class JobInfo:
    task_id: str
    agent_name: str
    query: str
    status: str = "running"
    started_at: float = 0.0
    run_id: str = ""
    kind: str = "job"


# ═══════════════════════════════════════════════════════════════
# AgentPane — Full detail tracking + graph
# ═══════════════════════════════════════════════════════════════

class AgentPane:
    def __init__(self, name: str):
        self.name = name
        self.iteration = 0
        self.max_iter = 0
        self.phase = "running"
        self.last_tool = ""
        self.last_tool_status = ""
        self.last_thought = ""
        self.content_lines: list[str] = [""]
        self.persona = ""
        self.skills: list[str] = []
        self.tokens_used = 0
        self.tokens_max = 0
        self._chunks_seen = 0

        # Full history
        self.tool_history: list[ToolEvent] = []
        self.thoughts: list[str] = []       # complete, no truncation
        self.iterations: Dict[int, IterationInfo] = {}
        self._last_tool_event: Optional[ToolEvent] = None
        self.files_touched = set()

        # Sub-agents
        self.sub_agents: Dict[str, int] = {}  # name → color index
        self._sub_color_counter = 0

        # 3D Graph
        self.graph = MiniGraph3D()

        # Navigation
        self.scroll_offset = 0
        self.selected_item = 0

        self.started_at: float = time.time()
        self.collapsed = False
        self._last_chunk_count = 0

    def _get_iteration(self, n: int) -> IterationInfo:
        if n not in self.iterations:
            self.iterations[n] = IterationInfo(number=n)
        return self.iterations[n]

    def _sub_color(self, agent_name: str) -> int:
        if agent_name not in self.sub_agents:
            self.sub_agents[agent_name] = self._sub_color_counter
            self._sub_color_counter += 1
        return self.sub_agents[agent_name]

    def ingest(self, chunk: dict):
        self._chunks_seen += 1
        if self.collapsed and self._chunks_seen > self._last_chunk_count:
            self.collapsed = False
        self.name = chunk.get("agent", "") or self.name
        self.iteration = chunk.get("iter", self.iteration)
        self.max_iter = chunk.get("max_iter", self.max_iter)
        self.tokens_used = chunk.get("tokens_used", self.tokens_used)
        self.tokens_max = chunk.get("tokens_max", self.tokens_max)
        self.persona = chunk.get("persona", self.persona) or self.persona
        if chunk.get("skills"):
            self.skills = chunk["skills"]

        is_sub = chunk.get("is_sub", False) or bool(chunk.get("_sub_agent_id", ""))
        sub_id = chunk.get("_sub_agent_id", "")
        sub_name = sub_id or (chunk.get("agent", "") if is_sub else "")

        t = chunk.get("type", "")
        it = self._get_iteration(self.iteration)
        it.tokens_used = self.tokens_used
        it.tokens_max = self.tokens_max

        # Ensure iteration graph node
        iter_nid = f"iter_{self.iteration}"
        if self.iteration > 0 and iter_nid not in self.graph.nodes:
            self.graph.add_node(iter_nid, f"iter {self.iteration}", "iteration")
            prev = f"iter_{self.iteration - 1}"
            if prev in self.graph.nodes:
                self.graph.add_edge(prev, iter_nid)

        if t == "content":
            text = chunk.get("chunk", "")
            self.content_lines[-1] += text
            while "\n" in self.content_lines[-1]:
                head, rest = self.content_lines[-1].split("\n", 1)
                self.content_lines[-1] = head
                self.content_lines.append(rest)
            it.content.append(text)

        elif t == "reasoning":
            thought = chunk.get("chunk", "")  # FULL, no truncation
            self.last_thought = thought
            self.thoughts.append(thought)
            it.thoughts.append(thought)
            # Graph node
            nid = f"think_{len(self.thoughts)}"
            self.graph.add_node(nid, _short(thought.replace("\n", " "), 35),
                                "thought", is_sub=is_sub,
                                sub_color_idx=self._sub_color(sub_name) if is_sub else 0)
            if iter_nid in self.graph.nodes:
                self.graph.add_edge(iter_nid, nid)

        elif t == "tool_start":
            name = chunk.get("name", "?")
            args_raw = chunk.get("args", "")
            self.last_tool = name
            self.last_tool_status = "..."
            args_parsed = None
            try:
                args_parsed = _json.loads(args_raw) if isinstance(args_raw, str) and args_raw.strip() else args_raw
            except Exception:
                pass
            ev = ToolEvent(
                name=name, args_raw=args_raw, args_parsed=args_parsed,
                started=time.time(), iteration=self.iteration,
                is_sub=is_sub, sub_agent=sub_name,
            )
            self._last_tool_event = ev
            self.tool_history.append(ev)
            it.tools.append(ev)
            # Graph node
            nid = f"tool_{len(self.tool_history)}"
            self.graph.add_node(nid, name, "tool", is_sub=is_sub,
                                sub_color_idx=self._sub_color(sub_name) if is_sub else 0)
            if iter_nid in self.graph.nodes:
                self.graph.add_edge(iter_nid, nid)

        elif t == "tool_result":
            self.last_tool = chunk.get("name", self.last_tool)
            result_raw = chunk.get("result", "")
            success = True
            result_parsed = None
            try:
                result_parsed = _json.loads(result_raw) if isinstance(result_raw, str) and result_raw.strip() else result_raw
                if isinstance(result_parsed, dict):
                    success = result_parsed.get("success", True)
            except Exception:
                result_parsed = result_raw
            self.last_tool_status = SYM["ok"] if success else SYM["fail"]
            if self._last_tool_event:
                self._last_tool_event.result_raw = str(result_raw)
                self._last_tool_event.result_parsed = result_parsed
                self._last_tool_event.success = success
                self._last_tool_event.elapsed = time.time() - self._last_tool_event.started
                if any(x in self.last_tool for x in ("vfs_write", "vfs_edit", "vfs_create", "vfs_save", "vfs_patch")):
                    p = (self._last_tool_event.args_parsed or {}).get("path") or (self._last_tool_event.args_parsed or {}).get("vfs_path")
                    if p: self.files_touched.add(str(p))
            # Update graph
            nid = f"tool_{len(self.tool_history)}"
            self.graph.update_status(nid, "done" if success else "failed")
            self.graph.update_status(nid, "done" if success else "failed")


        elif t == "done":
            self.phase = "done" if chunk.get("success", True) else "failed"
        elif t == "error":
            self.phase = "error"
        elif t == "final_answer":
            text = chunk.get("answer", "")
            if text:
                self.content_lines.append("")
                for line in text.split("\n"):
                    self.content_lines.append(line)

        # Sub-agent graph node
        if is_sub and sub_name:
            sub_nid = f"sub_{sub_name}"
            if sub_nid not in self.graph.nodes:
                ci = self._sub_color(sub_name)
                self.graph.add_node(sub_nid, sub_name, "sub_agent",
                                    is_sub=True, sub_color_idx=ci)
                if iter_nid in self.graph.nodes:
                    self.graph.add_edge(iter_nid, sub_nid)

    # ── Render: compact (grid) ──────────────────────────────

    def render_compact(self, w: int, h: int) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        out: list[tuple[str, str]] = []
        if self.collapsed:
            phase_col = STATUS_COLOR.get(self.phase, C["dim"])
            out.append(("fg:#6b7280", f" ▸ {_short(self.name, w // 3)}"))
            out.append((f"fg:{phase_col}", f" {self.phase} "))
            out.append(("fg:#6b7280", f"it:{self.iteration}/{self.max_iter} "))
            out.append(("fg:#6b7280", f"[collapsed]\n"))
            return out
        bar = _bar(self.iteration, self.max_iter, min(12, w // 6))
        out.append(("fg:#67e8f9 bold", f" {SYM['agent']} {_short(self.name, w // 3)}"))
        out.append(("fg:#6b7280", f" {bar} {self.iteration}/{self.max_iter}"))
        if self.tokens_max > 0:
            pct = min(100, int(100 * self.tokens_used / self.tokens_max))
            tc = C["dim"] if pct < 50 else (C["amber"] if pct < 80 else C["red"])
            out.append((f"fg:{tc}", f" {pct}%"))
        # Sub-agent indicator
        if self.sub_agents:
            out.append(("fg:#f472b6", f" ✦{len(self.sub_agents)}"))
            elapsed = time.time() - self.started_at
            out.append(("fg:#6b7280", f" {_fmt_elapsed(elapsed)}"))
        out.append(("", "\n"))

        content_h = max(1, h - 4)
        visible = self.content_lines[-content_h:]
        for line in visible:
            out.append(("fg:#e5e7eb", f" {line[:w - 2]}\n"))
        for _ in range(content_h - len(visible)):
            out.append(("", "\n"))

        if self.last_thought:
            out.append(("fg:#6b7280", f" {SYM['think']} {_short(self.last_thought.replace(chr(10), ' '), w - 6)}\n"))
        else:
            out.append(("", "\n"))

        parts: list[tuple[str, str]] = []
        if self.last_tool:
            tc = C["green"] if self.last_tool_status == SYM["ok"] else C["cyan"]
            parts.append((f"fg:{tc}", f" {SYM['tool']}{self.last_tool} {self.last_tool_status}"))
        if self.phase in ("done", "failed", "error"):
            col = STATUS_COLOR.get(self.phase, C["dim"])
            parts.append((f"fg:{col} bold", f" {SYM['done'] if self.phase == 'done' else SYM['fail']} {self.phase}"))
        out.extend(parts or [("", "")])
        out.append(("", "\n"))
        return out

    # ── Render: focus ───────────────────────────────────────

    def render_focus(self, w: int, h: int) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []

        bar = _bar(self.iteration, self.max_iter, 20)
        out.append(("fg:#67e8f9 bold", f"  {SYM['agent']} {self.name}"))
        out.append(("fg:#6b7280", f"  {bar} {self.iteration}/{self.max_iter}"))
        if self.tokens_max > 0:
            pct = min(100, int(100 * self.tokens_used / self.tokens_max))
            tc = C["dim"] if pct < 50 else (C["amber"] if pct < 80 else C["red"])
            out.append((f"fg:{tc}", f"  tok:{pct}%"))
        if self.sub_agents:
            out.append(("fg:#f472b6", f"  ✦{len(self.sub_agents)} sub-agents"))
            elapsed = time.time() - self.started_at
            out.append(("fg:#6b7280", f"  ⏱{_fmt_elapsed(elapsed)}"))
        out.append(("", "\n"))

        meta = []
        if self.persona and self.persona != "default":
            meta.append(f"⬡ {self.persona}")
        if self.skills:
            meta.append(f"⚙ {', '.join(self.skills[:5])}")
        if meta:
            out.append(("fg:#6b7280", f"  {'  '.join(meta)}\n"))

        out.append(("fg:#374151", f"  {'─' * (w - 4)}\n"))

        reserved = 9 + (1 if meta else 0)
        content_h = max(1, h - reserved)
        total = len(self.content_lines)
        start = max(0, min(self.scroll_offset, total - content_h))
        end = start + content_h
        visible = self.content_lines[start:end]

        for line in visible:
            out.append(("fg:#e5e7eb", f"  {line[:w - 4]}\n"))
        for _ in range(content_h - len(visible)):
            out.append(("", "\n"))

        if total > content_h:
            out.append(("fg:#6b7280", f"  ↕ {start + 1}-{min(end, total)}/{total}\n"))
        else:
            out.append(("", "\n"))

        out.append(("fg:#374151", f"  {'─' * (w - 4)}\n"))

        if self.last_thought:
            out.append(("fg:#6b7280", f"  {SYM['think']} {_short(self.last_thought.replace(chr(10), ' '), w - 8)}\n"))

        if self.last_tool:
            tc = C["green"] if self.last_tool_status == SYM["ok"] else C["cyan"]
            out.append((f"fg:{tc}", f"  {SYM['tool']} {self.last_tool} {self.last_tool_status}"))
            out.append(("fg:#6b7280", f"  ({len(self.tool_history)} total)\n"))

        # Shortcuts
        hints = []
        if self.graph.nodes:
            hints.append("[g]raph")
        if self.tool_history:
            hints.append("[t]ools")
        if self.iterations:
            hints.append("[i]terations")
        if self.thoughts:
            hints.append("[h]thoughts")
        if hints:
            out.append(("fg:#6b7280", f"  {' '.join(hints)}\n"))

        if self.phase in ("done", "failed", "error"):
            col = STATUS_COLOR.get(self.phase, C["dim"])
            out.append((f"fg:{col} bold", f"  {SYM['done'] if self.phase == 'done' else SYM['fail']} {self.phase}\n"))
        return out

    # ── Render: detail views ────────────────────────────────

    def render_detail(self, w: int, h: int, dtype: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        out.append(("fg:#67e8f9 bold", f"  {SYM['agent']} {self.name}"))
        out.append(("fg:#6b7280", f"  → {dtype}\n"))
        out.append(("fg:#374151", f"  {'─' * (w - 4)}\n"))

        dispatch = {
            "graph": lambda: self.graph.render(w, h - 3),
            "tools": lambda: self._render_tools_full(w, h - 3),
            "iterations": lambda: self._render_iterations(w, h - 3),
            "thoughts": lambda: self._render_thoughts_full(w, h - 3),
        }
        out.extend(dispatch.get(dtype, lambda: [("fg:#6b7280", "  unknown view\n")])())
        return out

    def _render_tools_full(self, w: int, h: int) -> list[tuple[str, str]]:
        """Tool list with FULL detail on selection."""
        out: list[tuple[str, str]] = []
        if not self.tool_history:
            out.append(("fg:#6b7280", "  no tools called\n"))
            return out

        # Calculate how much space the selected tool needs
        sel_ev = None
        if 0 <= self.selected_item < len(self.tool_history):
            sel_ev = self.tool_history[self.selected_item]

        # Show list with selected expanded
        start = max(0, self.selected_item - (h // 4))
        lines_used = 0

        for idx in range(start, len(self.tool_history)):
            if lines_used >= h - 2:
                break
            ev = self.tool_history[idx]
            is_sel = idx == self.selected_item
            prefix = SYM["select"] if is_sel else " "
            icon = SYM["ok"] if ev.success else SYM["fail"]
            col = C["green"] if ev.success else C["red"]
            elapsed = f" {_fmt_elapsed(ev.elapsed)}" if ev.elapsed > 0 else ""
            sub_tag = f" ✦{_short(ev.sub_agent, 8)}" if ev.is_sub else ""

            sel_style = "fg:#67e8f9 bold" if is_sel else "fg:#6b7280"
            out.append((sel_style, f"  {prefix} "))
            out.append((f"fg:{col}", f"{icon} "))
            out.append(("fg:#e5e7eb bold" if is_sel else "fg:#e5e7eb", f"{ev.name:<16}"))
            out.append(("fg:#6b7280", f" it:{ev.iteration}{elapsed}"))
            if sub_tag:
                out.append(("fg:#f472b6", sub_tag))
            out.append(("", "\n"))
            lines_used += 1

            # FULL DETAIL when selected
            if is_sel:
                detail_h = h - lines_used - 4
                detail_lines = self._render_tool_expanded(ev, w, detail_h)
                out.extend(detail_lines)
                lines_used += sum(1 for _, t in detail_lines if "\n" in t)

        out.append(("", "\n"))
        out.append(("fg:#6b7280", f"  ↑↓ navigate  Esc back  {len(self.tool_history)} tools\n"))
        return out

    def _render_tool_expanded(self, ev: ToolEvent, w: int, max_h: int) -> list[tuple[str, str]]:
        """Full tool detail with syntax highlighting."""
        out: list[tuple[str, str]] = []
        indent = "      "

        # Args
        if ev.args_raw:
            out.append(("fg:#67e8f9", f"{indent}─── args ───\n"))
            args_hl = syntax_highlight(ev.args_raw)
            lines = 0
            for style, text in args_hl:
                for line in text.split("\n"):
                    if line or not text.endswith("\n"):
                        out.append((style, f"{indent}{line}"))
                    if "\n" in text:
                        out.append(("", "\n"))
                        lines += 1
                        if lines > max_h // 3:
                            out.append(("fg:#6b7280", f"{indent}... ({len(ev.args_raw)} chars total)\n"))
                            break

        # Result
        if ev.result_raw:
            out.append(("fg:#67e8f9", f"{indent}─── result ───\n"))

            # Intelligent display based on tool name (like ZenRendererV2)
            display_text = ev.result_raw
            result_parsed = ev.result_parsed

            # For file-read tools: show content with syntax highlighting
            if isinstance(result_parsed, dict):
                content = result_parsed.get("content", "")
                file_type = result_parsed.get("file_type", "")
                if content and len(content) > 10:
                    display_text = content
                    if file_type in ("json", "md", "markdown", "yaml", "toml", "cfg", "ini"):
                        pass  # will be highlighted by syntax_highlight

            result_hl = syntax_highlight(display_text)
            lines = 0
            for style, text in result_hl:
                for sub in text.split("\n"):
                    if sub or not text.endswith("\n"):
                        out.append((style, f"{indent}{sub[:w - 8]}"))
                    if "\n" in text:
                        out.append(("", "\n"))
                        lines += 1
                        if lines > max_h * 2 // 3:
                            out.append(("fg:#6b7280", f"{indent}... ({len(display_text)} chars total)\n"))
                            break

        return out

    def _render_iterations(self, w: int, h: int) -> list[tuple[str, str]]:
        """Iteration history — selecting shows full thoughts + tools."""
        out: list[tuple[str, str]] = []
        iters = sorted(self.iterations.values(), key=lambda x: x.number)
        if not iters:
            out.append(("fg:#6b7280", "  no iterations yet\n"))
            return out

        start = max(0, self.selected_item - (h // 4))
        lines_used = 0

        for idx in range(start, len(iters)):
            if lines_used >= h - 2:
                break
            it = iters[idx]
            is_sel = idx == self.selected_item
            prefix = SYM["select"] if is_sel else " "
            sel_style = "fg:#67e8f9 bold" if is_sel else "fg:#6b7280"

            # Iteration header
            pct = f" {min(100, int(100 * it.tokens_used / it.tokens_max))}%" if it.tokens_max > 0 else ""
            out.append((sel_style, f"  {prefix} {SYM['iter']} iter {it.number}"))
            out.append(("fg:#6b7280", f"  {len(it.tools)} tools  {len(it.thoughts)} thoughts{pct}\n"))
            lines_used += 1

            # EXPANDED when selected
            if is_sel:
                remaining = h - lines_used - 3
                detail = self._render_iteration_expanded(it, w, remaining)
                out.extend(detail)
                lines_used += sum(1 for _, t in detail if "\n" in t)

        out.append(("", "\n"))
        out.append(("fg:#6b7280", f"  ↑↓ navigate  Esc back  {len(iters)} iterations\n"))
        return out

    def _render_iteration_expanded(self, it: IterationInfo, w: int, max_h: int) -> list[tuple[str, str]]:
        """Full iteration detail: all thoughts + all tools."""
        out: list[tuple[str, str]] = []
        indent = "      "
        lines = 0

        # Thoughts (FULL — no truncation)
        for i, thought in enumerate(it.thoughts):
            if lines >= max_h // 2:
                break
            out.append(("fg:#6b7280", f"{indent}{SYM['think']} "))
            # Wrap thought to width
            text = thought.replace("\n", " ")
            while text and lines < max_h // 2:
                chunk = text[:w - 10]
                text = text[w - 10:]
                out.append(("fg:#e5e7eb", f"{chunk}\n"))
                lines += 1
                if text:
                    out.append(("", f"{indent}  "))

        # Tools
        for ev in it.tools:
            if lines >= max_h - 1:
                break
            icon = SYM["ok"] if ev.success else SYM["fail"]
            col = C["green"] if ev.success else C["red"]
            elapsed = f" {ev.elapsed:.2f}s" if ev.elapsed > 0 else ""
            sub = f" ✦{ev.sub_agent}" if ev.is_sub else ""

            out.append((f"fg:{col}", f"{indent}{SYM['tool']} {icon} {ev.name}{elapsed}{sub}\n"))
            lines += 1

            # Show key arg
            if ev.args_parsed and isinstance(ev.args_parsed, dict):
                for k in ("path", "query", "command", "url", "task"):
                    if k in ev.args_parsed:
                        out.append(("fg:#6b7280", f"{indent}  {k}: {_short(str(ev.args_parsed[k]), w - 14)}\n"))
                        lines += 1
                        break

        return out

    def _render_thoughts_full(self, w: int, h: int) -> list[tuple[str, str]]:
        """Thought history — FULL text, no truncation."""
        out: list[tuple[str, str]] = []
        if not self.thoughts:
            out.append(("fg:#6b7280", "  no thoughts\n"))
            return out

        start = max(0, self.selected_item - (h // 4))
        lines_used = 0

        for idx in range(start, len(self.thoughts)):
            if lines_used >= h - 2:
                break
            thought = self.thoughts[idx]
            is_sel = idx == self.selected_item
            prefix = SYM["select"] if is_sel else " "
            sel_style = "fg:#67e8f9 bold" if is_sel else "fg:#6b7280"

            out.append((sel_style, f"  {prefix} {SYM['think']} "))

            if is_sel:
                # FULL thought — wrapped to width
                text = thought
                first = True
                for line in text.split("\n"):
                    while line:
                        chunk = line[:w - 10]
                        line = line[w - 10:]
                        out.append(("fg:#e5e7eb", f"{chunk}\n"))
                        lines_used += 1
                        if line or not first:
                            out.append(("", "        "))
                    first = False
                    if lines_used >= h - 4:
                        out.append(("fg:#6b7280", f"        ... ({len(thought)} chars)\n"))
                        break
            else:
                out.append(("fg:#e5e7eb", f"{_short(thought.replace(chr(10), ' '), w - 10)}\n"))
                lines_used += 1

        out.append(("", "\n"))
        out.append(("fg:#6b7280", f"  ↑↓ navigate  Esc back  {len(self.thoughts)} thoughts\n"))
        return out

    def _detail_item_count(self, dtype: str) -> int:
        counts = {
            "tools": len(self.tool_history),
            "graph": len(self.graph.nodes),
            "iterations": len(self.iterations),
            "thoughts": len(self.thoughts),
        }
        return counts.get(dtype, 0)


# ═══════════════════════════════════════════════════════════════
# ZenPlus Singleton
# ═══════════════════════════════════════════════════════════════

class ZenPlus:
    """
    Singleton fullscreen TUI. Multi-agent + jobs + 3D graph.
    Grid → Focus → Detail (graph/tools/iterations/thoughts).
    Esc at Grid = exit back to Zen (deactivates zen_plus_mode).
    """

    _instance: Optional["ZenPlus"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "ZenPlus":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    def __init__(self):
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._panes: Dict[str, AgentPane] = {}
        self._jobs: Dict[str, JobInfo] = {}
        self._focus: str = ""
        self._app: Optional[Application] = None
        self._running = False
        self._consumer_task: Optional[asyncio.Task] = None
        self._on_exit: Optional[Callable] = None
        self._stream_done = False
        self._exit_to_bg = False
        self._confirm_exit = False
        self._view = ViewMode.GRID
        self._grid_index = 0
        self._detail_type = "graph"
        # Global cross-agent graph
        self._global_graph = GlobalGraph()
        self._global_rebuild_counter = 0
        self._dismissed: set[str] = set()

    @property
    def active(self) -> bool:
        return self._running

    def feed_chunk(self, chunk: dict):
        try:
            self._queue.put_nowait(chunk)
        except Exception:
            pass

    def signal_stream_done(self):
        self._stream_done = True

    def inject_job(self, task_id: str, agent_name: str, query: str,
                   status: str = "running", run_id: str = "", kind: str = "job"):
        self._jobs[task_id] = JobInfo(
            task_id=task_id, agent_name=agent_name, query=query,
            status=status, started_at=time.time(), run_id=run_id, kind=kind,
        )
        if self._app:
            self._app.invalidate()

    def update_job(self, task_id: str, status: str):
        if task_id in self._jobs:
            self._jobs[task_id].status = status
            if self._app:
                self._app.invalidate()

    def remove_job(self, task_id: str):
        self._jobs.pop(task_id, None)

    async def start(self, on_exit: Optional[Callable] = None):
        if self._running:
            return
        self._running = True
        self._stream_done = False
        self._exit_to_bg = False
        self._view = ViewMode.GRID
        self._on_exit = on_exit

        self._app = Application(
            layout=Layout(self._build_layout()),
            key_bindings=self._build_keybindings(),
            full_screen=True,
            mouse_support=False,
        )
        self._consumer_task = asyncio.create_task(self._consume())

        try:
            await self._app.run_async()
        finally:
            self._running = False
            if self._consumer_task and not self._consumer_task.done():
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass
            self._app = None
            if self._on_exit:
                cb = self._on_exit
                self._on_exit = None
                if asyncio.iscoroutinefunction(cb):
                    await cb()
                else:
                    cb()

    async def stop(self):
        if self._app:
            self._app.exit()

    async def reopen(self, on_exit: Optional[Callable] = None):
        """Reopen the TUI after q/Esc exit. Preserves all pane state."""
        if self._running:
            return
        self._stream_done = False
        self._exit_to_bg = False
        await self.start(on_exit=on_exit)

    def clear_panes(self):
        self._panes.clear()
        self._jobs.clear()
        self._focus = ""
        self._stream_done = False
        self._exit_to_bg = False
        self._confirm_exit = False
        self._view = ViewMode.GRID
        self._grid_index = 0
        self._global_graph = GlobalGraph()
        self._global_rebuild_counter = 0
        self._dismissed.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _get_pane(self, name: str) -> AgentPane:
        if name not in self._panes:
            self._panes[name] = AgentPane(name)
            if not self._focus:
                self._focus = name
        return self._panes[name]

    def _ordered_names(self) -> list[str]:
        return [n for n in self._panes.keys() if n not in self._dismissed]

    def _focused_pane(self) -> Optional[AgentPane]:
        return self._panes.get(self._focus)

    async def _consume(self):
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.25)
                agent = chunk.get("agent", "") or "default"
                pane = self._get_pane(agent)
                if agent in self._dismissed and chunk.get("type") not in ("done", "error"):
                    self._dismissed.discard(agent)
                pane.ingest(chunk)
                self._global_rebuild_counter += 1
                # Rebuild global graph every 4 chunks or on tool events
                if (self._global_rebuild_counter % 4 == 0 or
                        chunk.get("type") in ("tool_result", "done")):
                    self._global_graph.rebuild(self._panes)
                if self._app:
                    self._app.invalidate()
            except asyncio.TimeoutError:
                if self._app:
                    self._app.invalidate()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _reset_confirm_exit(self):
        await asyncio.sleep(2.0)
        self._confirm_exit = False
        if self._app:
            self._app.invalidate()

    # ── Layout ──────────────────────────────────────────────

    def _build_layout(self) -> HSplit:
        return HSplit([
            Window(FormattedTextControl(self._render_title), height=1, style="bg:#1a1a2e"),
            Window(FormattedTextControl(self._render_main), style="bg:#0f0f1a"),
            Window(FormattedTextControl(self._render_jobs), height=self._jobs_height, style="bg:#111122"),
            Window(FormattedTextControl(self._render_status), height=1, style="bg:#1a1a2e"),
        ])

    @property
    def _jobs_height(self) -> int:
        active = [j for j in self._jobs.values() if j.status in ("running", "pending", "waiting")]
        return min(max(1, len(active) + 1), 6)

    def _render_title(self) -> list[tuple[str, str]]:
        mode = self._view.value.upper()
        parts = [
            ("fg:#67e8f9 bold", " ZEN+ "),
            ("fg:#374151", "│ "),
            ("fg:#a78bfa", f"{mode} "),
            ("fg:#374151", "│ "),
        ]
        hints = {
            ViewMode.GRID:  "Tab/↑↓←→=select  Enter=focus  G=global  c=dismiss  C=dismiss done  Esc=exit",
            ViewMode.FOCUS: "↑↓=scroll  g=graph  t=tools  i=iterations  h=thoughts  Esc=grid",
            ViewMode.DETAIL: (f"↑↓=select  ←→=edges  Enter=detail/jump  Esc=back  [{self._detail_type}]"
                              if self._detail_type in ("graph", "global") else
                              f"↑↓=navigate  Esc=back  [{self._detail_type}]"),
        }
        parts.append(("fg:#6b7280", hints.get(self._view, "")))
        parts.append(("fg:#374151", " │ "))
        parts.append(("fg:#4ade80" if self._stream_done else "fg:#fbbf24",
                       "● done" if self._stream_done else "◎ streaming"))
        return parts

    def _render_main(self) -> FormattedText:
        size = self._app.output.get_size() if self._app else None
        w = size.columns if size else 100
        h = (size.rows if size else 24) - 2 - max(1, self._jobs_height)

        all_idle = self._stream_done and all(p.phase in ("done", "failed", "error") for p in self._panes.values())

        if all_idle and self._view == ViewMode.GRID:
            return FormattedText([("bg:#000000 fg:#6b7280",
                                   f"\n\n\n{' ' * (w // 2 - 10)}RUN COMPLETED\n{' ' * (w // 2 - 14)}Press Esc to see summary")])

        if self._view == ViewMode.GRID:
            return FormattedText(self._render_grid(w, h))
        elif self._view == ViewMode.FOCUS:
            pane = self._focused_pane()
            if not pane:
                return FormattedText([("fg:#6b7280", "  waiting...\n")])
            return FormattedText(pane.render_focus(w, h))
        elif self._view == ViewMode.DETAIL:
            if self._detail_type == "global":
                return FormattedText(self._render_global_detail(w, h))
            pane = self._focused_pane()
            if not pane:
                return FormattedText([("fg:#6b7280", "  no pane\n")])
            return FormattedText(pane.render_detail(w, h, self._detail_type))
        return FormattedText([("", "\n")])

    def _render_global_detail(self, w: int, h: int) -> list[tuple[str, str]]:
        """Render the global cross-agent graph."""
        out: list[tuple[str, str]] = []
        out.append(("fg:#a78bfa bold", "  ◉ GLOBAL GRAPH"))
        n_agents = len(self._panes)
        n_cross = sum(1 for e in self._global_graph.edges if e.rel != REL_PARENT)
        out.append(("fg:#6b7280", f"  {n_agents} agents  {n_cross} cross-links\n"))
        out.append(("fg:#374151", f"  {'─' * (w - 4)}\n"))
        out.extend(self._global_graph.render(w, h - 3))
        return out

    def _render_grid(self, w: int, h: int) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        names = self._ordered_names()
        if not names:
            out.append(("fg:#6b7280", "  waiting for agent streams...\n"))
            return out

        n = len(names)
        cols = 1 if n == 1 else (2 if n <= 4 else min(3, n))
        rows = math.ceil(n / cols)
        pane_w = max(20, (w - cols + 1) // cols)
        n_expanded = sum(1 for name in names if not self._panes[name].collapsed)
        n_collapsed = n - n_expanded
        collapsed_h = 1
        if n_expanded > 0:
            expanded_h = max(6, (h - n_collapsed * collapsed_h - rows + 1) // max(1, math.ceil(n_expanded / cols)))
        else:
            expanded_h = max(6, (h - rows + 1) // rows)
        pane_h = max(6, (h - rows + 1) // rows)

        for row in range(min(rows, math.ceil(n / cols))):
            row_panes: list[list[list[tuple[str, str]]]] = []
            for col in range(cols):
                idx = row * cols + col
                if idx >= n:
                    row_panes.append([[("", " " * pane_w)] for _ in range(pane_h)])
                    continue
                name = names[idx]
                pane = self._panes[name]
                is_sel = idx == self._grid_index
                cur_pane_h = 1 if pane.collapsed else pane_h
                ft = pane.render_compact(pane_w - 2, cur_pane_h - 1)

                lines: list[list[tuple[str, str]]] = [[]]
                for style, text in ft:
                    if "\n" in text:
                        parts = text.split("\n")
                        lines[-1].append((style, parts[0]))
                        for p in parts[1:]:
                            lines.append([(style, p)])
                    else:
                        lines[-1].append((style, text))
                while len(lines) < cur_pane_h:
                    lines.append([])
                lines = lines[:pane_h]

                border_col = C["cyan"] if is_sel else C["deep"]
                bordered = []
                for lp in lines:
                    bordered.append([(f"fg:{border_col}", SYM["vline"])] + lp)
                row_panes.append(bordered)

            max_pane_h = max(1 if self._panes[names[row * cols + c]].collapsed else pane_h
                             for c in range(cols) if row * cols + c < n) if n > 0 else pane_h
            for li in range(max_pane_h):
                for ci, pl in enumerate(row_panes):
                    if li < len(pl):
                        out.extend(pl[li])
                        actual = sum(len(t) for _, t in pl[li])
                        if actual < pane_w:
                            out.append(("", " " * (pane_w - actual)))
                out.append(("", "\n"))
            if row < rows - 1:
                out.append(("fg:#374151", f"{'─' * w}\n"))
        return out

    def _render_jobs(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if not self._jobs:
            out.append(("fg:#374151", f" {'─' * 20}\n"))
            return out
        out.append(("fg:#a78bfa", f" {SYM['job']} Jobs({len(self._jobs)}) "))
        for jid, job in self._jobs.items():
            col = STATUS_COLOR.get(job.status, C["dim"])
            kind_sym = {"job": SYM["job"], "bg": SYM["bg"],
                        "delegate": SYM["task"]}.get(job.kind, SYM["job"])
            elapsed = time.time() - job.started_at
            elapsed_s = f"{_fmt_elapsed(elapsed)}"
            out.append(("fg:#374151", "│ "))
            out.append((f"fg:{col}", f"{kind_sym}{_short(job.agent_name, 10)} "))
            out.append(("fg:#6b7280", f"{_short(job.query, 18)} "))
            out.append((f"fg:{col}", f"{job.status[:4]} {elapsed_s} "))
        out.append(("", "\n"))
        return out

    def _render_status(self) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        if self._confirm_exit:
            parts.append(("fg:#ef4444 bold", " ⚠ Nochmal Esc/q zum Beenden "))
            parts.append(("fg:#374151", "│"))
        names = self._ordered_names()
        for i, name in enumerate(names):
            pane = self._panes[name]
            phase_col = STATUS_COLOR.get(pane.phase, C["dim"])
            if name == self._focus and self._view != ViewMode.GRID:
                parts.append(("fg:#67e8f9 bold", f" ▸{_short(name, 12)} "))
            elif i == self._grid_index and self._view == ViewMode.GRID:
                parts.append(("fg:#67e8f9", f" [{_short(name, 12)}] "))
            else:
                parts.append((f"fg:{phase_col}", f"  {_short(name, 12)} "))
            elapsed = time.time() - pane.started_at
            parts.append(("fg:#6b7280", f"{_fmt_elapsed(elapsed)} "))
        running = sum(1 for j in self._jobs.values() if j.status == "running")
        if running:
            parts.append(("fg:#374151", "│"))
            parts.append(("fg:#a78bfa", f" {SYM['job']}{running} "))
        if self._dismissed:
            parts.append(("fg:#374151", "│"))
            parts.append(("fg:#6b7280", f" ×{len(self._dismissed)} hidden "))
        if not parts:
            parts.append(("fg:#6b7280", " no agents"))
        return parts

    # ── Key bindings ────────────────────────────────────────

    def _build_keybindings(self) -> KeyBindings:
        kb = KeyBindings()
        zp = self

        @kb.add("escape")
        def _back(event):
            if zp._view == ViewMode.DETAIL:
                zp._confirm_exit = False
                pane = zp._focused_pane()
                if zp._detail_type == "global":
                    if zp._global_graph._detail_node:
                        zp._global_graph._detail_node = None
                        return
                    zp._view = ViewMode.GRID
                    return
                if zp._detail_type == "graph" and pane and pane.graph._detail_node:
                    pane.graph._detail_node = None
                    return
                zp._view = ViewMode.FOCUS
                if pane:
                    pane.selected_item = 0
            elif zp._view == ViewMode.FOCUS:
                zp._confirm_exit = False
                zp._view = ViewMode.GRID
            else:
                # GRID level: double-Esc to exit
                if zp._confirm_exit:
                    zp._confirm_exit = False
                    zp._exit_to_bg = True
                    event.app.exit()
                else:
                    zp._confirm_exit = True
                    asyncio.get_event_loop().create_task(zp._reset_confirm_exit())

        @kb.add("q")
        def _quit(event):
            if zp._confirm_exit:
                zp._confirm_exit = False
                event.app.exit()
            else:
                zp._confirm_exit = True
                asyncio.get_event_loop().create_task(zp._reset_confirm_exit())

        @kb.add("tab")
        def _next(event):
            names = zp._ordered_names()
            if not names:
                return
            if zp._view == ViewMode.GRID:
                zp._grid_index = (zp._grid_index + 1) % len(names)
            else:
                idx = names.index(zp._focus) if zp._focus in names else -1
                zp._focus = names[(idx + 1) % len(names)]
                if zp._view == ViewMode.DETAIL:
                    pane = zp._focused_pane()
                    if pane:
                        pane.selected_item = 0

        @kb.add("s-tab")
        def _prev(event):
            names = zp._ordered_names()
            if not names:
                return
            if zp._view == ViewMode.GRID:
                zp._grid_index = (zp._grid_index - 1) % len(names)
            else:
                idx = names.index(zp._focus) if zp._focus in names else 0
                zp._focus = names[(idx - 1) % len(names)]

        @kb.add("enter")
        def _enter(event):
            names = zp._ordered_names()
            if not names:
                return
            if zp._view == ViewMode.GRID:
                if 0 <= zp._grid_index < len(names):
                    zp._focus = names[zp._grid_index]
                    zp._view = ViewMode.FOCUS
                    pane = zp._focused_pane()
                    if pane:
                        pane.scroll_offset = max(0, len(pane.content_lines) - 10)
            elif zp._view == ViewMode.DETAIL and zp._detail_type == "graph":
                pane = zp._focused_pane()
                if pane:
                    pane.graph.toggle_detail()
            elif zp._view == ViewMode.DETAIL and zp._detail_type == "global":
                gg = zp._global_graph
                if gg._detail_node and gg._detail_node in gg.nodes:
                    gnd = gg.nodes[gg._detail_node]
                    # Jump to agent's individual graph
                    if gnd.agent in zp._panes and gnd.original_nid:
                        zp._focus = gnd.agent
                        pane = zp._focused_pane()
                        if pane:
                            pane.graph.select_by_id(gnd.original_nid)
                            pane.selected_item = 0
                            zp._detail_type = "graph"
                        return
                gg.toggle_detail()

        @kb.add("up")
        def _up(event):
            if zp._view == ViewMode.GRID:
                names = zp._ordered_names()
                n = len(names)
                cols = 1 if n <= 1 else (2 if n <= 4 else 3)
                zp._grid_index = max(0, zp._grid_index - cols)
            elif zp._view == ViewMode.FOCUS:
                pane = zp._focused_pane()
                if pane:
                    pane.scroll_offset = max(0, pane.scroll_offset - 3)
            elif zp._view == ViewMode.DETAIL:
                if zp._detail_type == "global":
                    keys = list(zp._global_graph.nodes.keys())
                    if keys and zp._global_graph._selected:
                        idx = keys.index(zp._global_graph._selected) if zp._global_graph._selected in keys else 0
                        zp._global_graph.select(max(0, idx - 1))
                else:
                    pane = zp._focused_pane()
                    if pane:
                        pane.selected_item = max(0, pane.selected_item - 1)
                        if zp._detail_type == "graph":
                            pane.graph.select(pane.selected_item)

        @kb.add("down")
        def _down(event):
            if zp._view == ViewMode.GRID:
                names = zp._ordered_names()
                n = len(names)
                cols = 1 if n <= 1 else (2 if n <= 4 else 3)
                zp._grid_index = min(len(names) - 1, zp._grid_index + cols)
            elif zp._view == ViewMode.FOCUS:
                pane = zp._focused_pane()
                if pane:
                    pane.scroll_offset = min(
                        max(0, len(pane.content_lines) - 1),
                        pane.scroll_offset + 3)
            elif zp._view == ViewMode.DETAIL:
                if zp._detail_type == "global":
                    keys = list(zp._global_graph.nodes.keys())
                    if keys and zp._global_graph._selected:
                        idx = keys.index(zp._global_graph._selected) if zp._global_graph._selected in keys else 0
                        zp._global_graph.select(min(len(keys) - 1, idx + 1))
                else:
                    pane = zp._focused_pane()
                    if pane:
                        mx = pane._detail_item_count(zp._detail_type) - 1
                        pane.selected_item = min(max(0, mx), pane.selected_item + 1)
                        if zp._detail_type == "graph":
                            pane.graph.select(pane.selected_item)

        @kb.add("left")
        def _left(event):
            if zp._view == ViewMode.GRID:
                zp._grid_index = max(0, zp._grid_index - 1)
            elif zp._view == ViewMode.DETAIL and zp._detail_type == "graph":
                pane = zp._focused_pane()
                if pane:
                    pane.graph.navigate_edge(-1)
            elif zp._view == ViewMode.DETAIL and zp._detail_type == "global":
                zp._global_graph.navigate_edge(-1)

        @kb.add("right")
        def _right(event):
            names = zp._ordered_names()
            if zp._view == ViewMode.GRID and names:
                zp._grid_index = min(len(names) - 1, zp._grid_index + 1)
            elif zp._view == ViewMode.DETAIL and zp._detail_type == "graph":
                pane = zp._focused_pane()
                if pane:
                    pane.graph.navigate_edge(+1)
            elif zp._view == ViewMode.DETAIL and zp._detail_type == "global":
                zp._global_graph.navigate_edge(+1)

        # Detail shortcuts
        @kb.add("g")
        def _graph(event):
            if zp._view == ViewMode.GRID:
                # Global graph from grid view
                zp._global_graph.rebuild(zp._panes)
                if zp._global_graph.nodes:
                    zp._detail_type = "global"
                    zp._view = ViewMode.DETAIL
            elif zp._view == ViewMode.FOCUS:
                pane = zp._focused_pane()
                if pane and pane.graph.nodes:
                    zp._detail_type = "graph"
                    if pane.graph._current and pane.graph._current in pane.graph.nodes:
                        pane.graph.select_by_id(pane.graph._current)
                    else:
                        pane.graph.select(len(pane.graph.nodes) - 1)
                    pane.selected_item = 0
                    zp._view = ViewMode.DETAIL

        @kb.add("t")
        def _tools(event):
            if zp._view == ViewMode.FOCUS:
                pane = zp._focused_pane()
                if pane and pane.tool_history:
                    zp._detail_type = "tools"
                    pane.selected_item = len(pane.tool_history) - 1
                    zp._view = ViewMode.DETAIL

        @kb.add("i")
        def _iters(event):
            if zp._view == ViewMode.FOCUS:
                pane = zp._focused_pane()
                if pane and pane.iterations:
                    zp._detail_type = "iterations"
                    pane.selected_item = len(pane.iterations) - 1
                    zp._view = ViewMode.DETAIL

        @kb.add("h")
        def _thoughts(event):
            if zp._view == ViewMode.FOCUS:
                pane = zp._focused_pane()
                if pane and pane.thoughts:
                    zp._detail_type = "thoughts"
                    pane.selected_item = len(pane.thoughts) - 1
                    zp._view = ViewMode.DETAIL

        @kb.add("c")
        def _dismiss(event):
            if zp._view == ViewMode.GRID:
                names = zp._ordered_names()
                if 0 <= zp._grid_index < len(names):
                    name = names[zp._grid_index]
                    zp._dismissed.add(name)
                    # Fix grid index
                    new_names = zp._ordered_names()
                    if new_names:
                        zp._grid_index = min(zp._grid_index, len(new_names) - 1)
                    else:
                        zp._grid_index = 0

        @kb.add("C")
        def _dismiss_all_done(event):
            if zp._view == ViewMode.GRID:
                for name, pane in zp._panes.items():
                    if pane.phase in ("done", "failed", "error"):
                        zp._dismissed.add(name)
                new_names = zp._ordered_names()
                if new_names:
                    zp._grid_index = min(zp._grid_index, len(new_names) - 1)
                else:
                    zp._grid_index = 0

        return kb


