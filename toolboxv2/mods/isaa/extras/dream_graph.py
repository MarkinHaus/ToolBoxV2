"""
DreamGraph3D — Live 3D Force-Graph im Terminal für Dreamer/Agent/Memory.
=========================================================================

Singleton Observer der sich an Dreamer-Pipeline, SubAgentManager,
MemoryKnowledgeActor und ExecutionEngine hängt. Rendert einen
force-directed 3D-Graphen mit Z-Sorting und Depth-Fog direkt in
prompt_toolkit HTML.

Integration:
    from toolboxv2.mods.isaa.extras.dream_graph import DreamGraph3D
    graph = DreamGraph3D.instance()
    graph.attach(engine)          # hooks AgentLiveState
    graph.attach_dreamer(dreamer) # hooks dream phases
    renderer.set_graph(graph)     # ZenRendererV2 pane

Author: FlowAgent V3 — Markin / TU Berlin
"""

import math
import time
import threading
import asyncio
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

from prompt_toolkit import print_formatted_text, HTML

from toolboxv2.mods.isaa.base.Agent.dreamer import DreamConfig

# ── Re-use ZenRendererV2 palette & symbols ──────────────────────────
C = {
    "dim": "#6b7280", "cyan": "#67e8f9", "green": "#4ade80",
    "red": "#f87171", "amber": "#fbbf24", "white": "#e5e7eb",
    "bright": "#ffffff", "blue": "#60a5fa", "magenta": "#c084fc",
    "deep": "#3b3b3b",
}

# ── Node categories ─────────────────────────────────────────────────

class NK(Enum):
    """Node Kind — maps to visual symbol + color + Y-bias."""
    DREAMER  = ("◎", C["amber"],   0.8)   # top layer
    AGENT    = ("◯", C["cyan"],    0.5)
    SUB      = ("◈", C["blue"],    0.3)
    SKILL    = ("◇", C["green"],   0.0)
    MEMORY   = ("○", C["magenta"], -0.4)
    CLUSTER  = ("▣", C["dim"],     -0.2)
    PHASE    = ("▸", C["amber"],    0.6)
    ERROR    = ("✗", C["red"],     -0.6)

    def __init__(self, sym, color, y_bias):
        self.sym = sym
        self.color = color
        self.y_bias = y_bias


class ES(Enum):
    """Edge Semantic."""
    SPAWNED   = ("spawned",    C["cyan"],  "╌")
    EVOLVED   = ("evolved",    C["green"], "─")
    REMEMBERS = ("remembers",  C["magenta"], "┄")
    PHASE_SEQ = ("phase_seq",  C["dim"],   "─")
    USES      = ("uses",       C["blue"],  "╌")
    ERROR_OF  = ("error_of",   C["red"],   "╌")
    RELATION  = ("relation",   C["magenta"], "═")
    PUBLISHED = ("published",  C["green"],  "─")

    def __init__(self, label, color, ch):
        self.label_ = label
        self.color = color
        self.ch = ch


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class N3:
    """Node with 3D position + metadata."""
    id: str
    kind: NK
    label: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    active: bool = False
    success: Optional[bool] = None  # None=pending, True=ok, False=fail
    born: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)

    @property
    def age(self) -> float:
        return time.time() - self.born


@dataclass
class E3:
    """Directed edge."""
    src: str
    dst: str
    kind: ES
    weight: float = 1.0


# ── 3D Math (inline, no numpy) ─────────────────────────────────────

def _rot_y(x, y, z, a):
    """Rotate around Y axis by angle a (radians)."""
    ca, sa = math.cos(a), math.sin(a)
    return x * ca + z * sa, y, -x * sa + z * ca

def _rot_x(x, y, z, a):
    """Rotate around X axis by angle a (radians)."""
    ca, sa = math.cos(a), math.sin(a)
    return x, y * ca - z * sa, y * sa + z * ca

def _project(x, y, z, az, el, zoom, cols, rows):
    """3D → 2D orthographic projection with camera rotation."""
    rx, ry, rz = _rot_y(x, y, z, az)
    rx, ry, rz = _rot_x(rx, ry, rz, el)
    # aspect correction: terminal chars are ~2:1 (h:w)
    sc = int((rx * zoom + 1) * cols / 2)
    sr = int((-ry * zoom + 1) * rows / 2)
    return sc, sr, rz


# ── Force Simulation ────────────────────────────────────────────────

def _force_step(nodes: dict[str, N3], edges: list[E3], dt: float = 0.15):
    """One iteration of Fruchterman-Reingold in 3D, with Y-bias per kind."""
    nlist = list(nodes.values())
    n = len(nlist)
    if n < 2:
        return

    # Repulsion (all pairs, O(n²) — fine for <200 nodes)
    k_rep = 0.8
    for i in range(n):
        a = nlist[i]
        for j in range(i + 1, n):
            b = nlist[j]
            dx = a.x - b.x
            dy = a.y - b.y
            dz = a.z - b.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01
            f = k_rep / (dist * dist)
            fx, fy, fz = dx * f, dy * f, dz * f
            a.vx += fx;  a.vy += fy;  a.vz += fz
            b.vx -= fx;  b.vy -= fy;  b.vz -= fz

    # Attraction (edges only)
    k_att = 0.12
    for e in edges:
        a, b = nodes.get(e.src), nodes.get(e.dst)
        if not a or not b:
            continue
        dx = b.x - a.x
        dy = b.y - a.y
        dz = b.z - a.z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01
        f = k_att * dist * e.weight
        fx, fy, fz = dx * f, dy * f, dz * f
        a.vx += fx;  a.vy += fy;  a.vz += fz
        b.vx -= fx;  b.vy -= fy;  b.vz -= fz

    # Apply velocities + Y-bias gravity + damping
    damping = 0.7
    y_pull = 0.05
    for nd in nlist:
        # Y-bias: pull toward kind's preferred layer
        nd.vy += (nd.kind.y_bias - nd.y) * y_pull
        # Active nodes push to front (z→+1)
        if nd.active:
            nd.vz += (0.6 - nd.z) * 0.08
        else:
            nd.vz += (-0.3 - nd.z) * 0.02

        nd.x += nd.vx * dt
        nd.y += nd.vy * dt
        nd.z += nd.vz * dt
        nd.vx *= damping
        nd.vy *= damping
        nd.vz *= damping
        # Clamp to [-1, 1]
        nd.x = max(-1, min(1, nd.x))
        nd.y = max(-1, min(1, nd.y))
        nd.z = max(-1, min(1, nd.z))


# ── Singleton 3D Graph ──────────────────────────────────────────────

class DreamGraph3D:
    """
    Singleton live 3D graph observer.

    Hooks into:
      - ExecutionEngine (AgentLiveState phase transitions)
      - Dreamer (_phase_* callbacks)
      - SubAgentManager (spawn events)
      - MemoryKnowledgeActor (add_data_point, add_relation)
    """
    _instance: Optional['DreamGraph3D'] = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> 'DreamGraph3D':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Test helper — reset singleton."""
        cls._instance = None

    def __init__(self):
        self.nodes: dict[str, N3] = {}
        self.edges: list[E3] = []
        self._edge_set: set[tuple[str, str, str]] = set()  # dedup
        # Camera
        self.azimuth = 0.3       # radians
        self.elevation = -0.25
        self.zoom = 0.85
        # Viewport (auto-detected, overridable)
        self.cols = 100
        self.rows = 30
        # State
        self._frame = 0
        self._attached_engines: set[int] = set()
        self._paused = False

    # ── Graph mutation API ──────────────────────────────────────────

    def add_node(self, id_: str, kind: NK, label: str, active: bool = False,
                 success: Optional[bool] = None, data: dict = None) -> N3:
        if id_ in self.nodes:
            nd = self.nodes[id_]
            nd.active = active
            if success is not None:
                nd.success = success
            if label:
                nd.label = label
            return nd
        # Spawn at random-ish position near kind's Y-bias
        import random
        nd = N3(
            id=id_, kind=kind, label=label[:20], active=active, success=success,
            x=random.uniform(-0.5, 0.5),
            y=kind.y_bias + random.uniform(-0.15, 0.15),
            z=0.5 if active else random.uniform(-0.3, 0.2),
            data=data or {},
        )
        self.nodes[id_] = nd
        return nd

    def add_edge(self, src: str, dst: str, kind: ES, weight: float = 1.0):
        key = (src, dst, kind.name)
        if key in self._edge_set:
            return
        self._edge_set.add(key)
        self.edges.append(E3(src=src, dst=dst, kind=kind, weight=weight))

    def set_active(self, id_: str, active: bool = True):
        if id_ in self.nodes:
            self.nodes[id_].active = active

    def set_result(self, id_: str, success: bool):
        if id_ in self.nodes:
            self.nodes[id_].success = success
            self.nodes[id_].active = False

    # ── Attach hooks ────────────────────────────────────────────────

    def attach(self, engine: Any):
        """Hook into ExecutionEngine's AgentLiveState for phase/tool events."""
        eid = id(engine)
        if eid in self._attached_engines:
            return
        self._attached_engines.add(eid)

        live = engine.live
        agent_id = f"agent:{live.agent_name}"
        self.add_node(agent_id, NK.AGENT, live.agent_name, active=True)

        # Monkey-patch live.enter to emit graph events
        _orig_enter = live.enter
        graph = self

        def _patched_enter(phase, msg=""):
            _orig_enter(phase, msg)
            phase_id = f"phase:{live.agent_name}:{phase.value}"
            graph.add_node(phase_id, NK.PHASE, f"{phase.value}", active=True)
            graph.add_edge(agent_id, phase_id, ES.PHASE_SEQ)
            # Deactivate previous phases
            for nd in graph.nodes.values():
                if nd.kind == NK.PHASE and nd.id != phase_id and nd.id.startswith(f"phase:{live.agent_name}:"):
                    nd.active = False
            if phase.value == "done":
                graph.set_result(agent_id, True)

        live.enter = _patched_enter

        # Hook sub-agent spawning if SubAgentManager available
        sam = getattr(engine, '_sub_agent_manager', None)
        if sam and hasattr(sam, 'spawn'):
            _orig_spawn = sam.spawn

            async def _patched_spawn(*a, **kw):
                result = await _orig_spawn(*a, **kw)
                sub_id = f"sub:{result}" if isinstance(result, str) else f"sub:{id(result)}"
                task = kw.get('task', a[0] if a else '')
                graph.add_node(sub_id, NK.SUB, str(task)[:18], active=True)
                graph.add_edge(agent_id, sub_id, ES.SPAWNED)
                return result

            sam.spawn = _patched_spawn

    def attach_dreamer(self, dreamer: Any):
        """Hook into Dreamer's phase pipeline for live dream tracking."""
        dream_id = f"dreamer:{dreamer.agent.amd.name}"
        self.add_node(dream_id, NK.DREAMER, "Dreamer", active=True)
        agent_id = f"agent:{dreamer.agent.amd.name}"
        if agent_id in self.nodes:
            self.add_edge(agent_id, dream_id, ES.SPAWNED)

        # Wrap each _phase_* method
        phase_names = [
            "_phase_harvest", "_phase_cluster", "_phase_analyze",
            "_phase_reconcile", "_phase_publish", "_phase_memory_sync",
        ]
        graph = self
        for pname in phase_names:
            orig = getattr(dreamer, pname, None)
            if not orig:
                continue
            short = pname.replace("_phase_", "")

            async def _wrap(state, _orig=orig, _short=short):
                pid = f"dphase:{_short}"
                graph.add_node(pid, NK.PHASE, _short, active=True)
                graph.add_edge(dream_id, pid, ES.PHASE_SEQ)
                try:
                    result = await _orig(state)
                    graph.set_result(pid, True)
                    # Track outputs
                    report = state.get("report")
                    if report:
                        for sid in getattr(report, 'skills_evolved', []):
                            nid = f"skill:{sid}"
                            graph.add_node(nid, NK.SKILL, sid[:16], success=True)
                            graph.add_edge(pid, nid, ES.EVOLVED)
                        for sid in getattr(report, 'skills_created', []):
                            nid = f"skill:{sid}"
                            graph.add_node(nid, NK.SKILL, sid[:16], active=True)
                            graph.add_edge(pid, nid, ES.EVOLVED)
                        for err in getattr(report, 'errors', []):
                            if err not in [n.label for n in graph.nodes.values()]:
                                eid = f"err:{len(graph.nodes)}"
                                graph.add_node(eid, NK.ERROR, err[:16], success=False)
                                graph.add_edge(pid, eid, ES.ERROR_OF)
                    # Clusters
                    clusters = state.get("clusters", {})
                    for cid, records in clusters.items():
                        if f"cluster:{cid}" not in graph.nodes:
                            graph.add_node(f"cluster:{cid}", NK.CLUSTER, f"{cid}({len(records)})")
                            graph.add_edge(pid, f"cluster:{cid}", ES.USES)
                    return result
                except Exception as e:
                    graph.set_result(pid, False)
                    eid = f"err:{pid}"
                    graph.add_node(eid, NK.ERROR, str(e)[:16], success=False)
                    graph.add_edge(pid, eid, ES.ERROR_OF)
                    raise

            setattr(dreamer, pname, _wrap)

    def attach_memory(self, memory_actor: Any):
        """Hook MemoryKnowledgeActor for entity/relation tracking."""
        graph = self
        mem_root = f"mem:{id(memory_actor)}"
        self.add_node(mem_root, NK.MEMORY, "Memory", active=True)

        if hasattr(memory_actor, 'add_data_point'):
            _orig_add = memory_actor.add_data_point

            async def _patched_add(*a, **kw):
                result = await _orig_add(*a, **kw)
                text = kw.get('text', a[0] if a else '')
                nid = f"mementry:{len(graph.nodes)}"
                graph.add_node(nid, NK.MEMORY, str(text)[:14], success=True)
                graph.add_edge(mem_root, nid, ES.REMEMBERS)
                return result

            memory_actor.add_data_point = _patched_add

        if hasattr(memory_actor, 'add_relation'):
            _orig_rel = memory_actor.add_relation

            async def _patched_rel(*a, **kw):
                result = await _orig_rel(*a, **kw)
                src = kw.get('source_id', a[0] if len(a) > 0 else '?')
                dst = kw.get('target_id', a[1] if len(a) > 1 else '?')
                # Try to find existing entity nodes or create
                src_id = f"entity:{src}"
                dst_id = f"entity:{dst}"
                graph.add_node(src_id, NK.MEMORY, str(src)[:14])
                graph.add_node(dst_id, NK.MEMORY, str(dst)[:14])
                graph.add_edge(src_id, dst_id, ES.RELATION)
                return result

            memory_actor.add_relation = _patched_rel

    # ── Simulation tick ─────────────────────────────────────────────

    def tick(self):
        """Advance physics one step."""
        if not self._paused:
            _force_step(self.nodes, self.edges)
        self._frame += 1

    # ── Camera controls ─────────────────────────────────────────────

    def rotate(self, daz: float = 0.0, del_: float = 0.0):
        self.azimuth += daz
        self.elevation = max(-1.2, min(1.2, self.elevation + del_))

    def zoom_in(self):
        self.zoom = min(2.0, self.zoom + 0.1)

    def zoom_out(self):
        self.zoom = max(0.3, self.zoom - 0.1)

    def toggle_pause(self):
        self._paused = not self._paused

    # ── Rendering ───────────────────────────────────────────────────

    def render(self, cols: int = 0, rows: int = 0) -> str:
        """
        Render graph to prompt_toolkit HTML string.
        Returns full frame as HTML, compatible with ZenRendererV2._print().
        """
        import html as _html

        cols = cols or self.cols
        rows = rows or self.rows
        self.tick()

        if not self.nodes:
            return f"<style fg='{C['dim']}'>  ◎ DreamGraph: no nodes</style>"

        # Character buffer + Z-buffer
        buf = [[' '] * cols for _ in range(rows)]
        cbuf = [[C["deep"]] * cols for _ in range(rows)]  # color buffer
        zbuf = [[float('-inf')] * cols for _ in range(rows)]

        # Project all nodes
        projected: list[tuple[int, int, float, N3]] = []
        for nd in self.nodes.values():
            sc, sr, sz = _project(nd.x, nd.y, nd.z, self.azimuth, self.elevation, self.zoom, cols, rows)
            if 0 <= sc < cols and 0 <= sr < rows:
                projected.append((sc, sr, sz, nd))

        # Draw edges first (background layer)
        for e in self.edges:
            a, b = self.nodes.get(e.src), self.nodes.get(e.dst)
            if not a or not b:
                continue
            ac, ar, az_ = _project(a.x, a.y, a.z, self.azimuth, self.elevation, self.zoom, cols, rows)
            bc, br, bz_ = _project(b.x, b.y, b.z, self.azimuth, self.elevation, self.zoom, cols, rows)
            # Bresenham-lite
            steps = max(abs(bc - ac), abs(br - ar), 1)
            ez = (az_ + bz_) / 2
            for s in range(1, steps):
                t = s / steps
                c = int(ac + (bc - ac) * t)
                r = int(ar + (br - ar) * t)
                if 0 <= c < cols and 0 <= r < rows and zbuf[r][c] < ez - 0.5:
                    buf[r][c] = e.kind.ch[0]
                    cbuf[r][c] = C["deep"] if ez < 0 else C["dim"]
                    zbuf[r][c] = ez - 0.5  # edges behind nodes

        # Draw nodes (Z-sorted, far to near)
        projected.sort(key=lambda p: p[2])
        for sc, sr, sz, nd in projected:
            if zbuf[sr][sc] > sz:
                continue  # occluded
            zbuf[sr][sc] = sz

            # Depth fog: color by Z
            if sz > 0.3:
                col = C["bright"] if nd.active else nd.kind.color
            elif sz > -0.2:
                col = nd.kind.color
            elif sz > -0.6:
                col = C["dim"]
            else:
                col = C["deep"]

            # Override for success/failure
            if nd.success is True:
                col = C["green"]
            elif nd.success is False:
                col = C["red"]

            # Pulse animation for active nodes
            if nd.active and self._frame % 6 < 3:
                col = C["bright"]

            buf[sr][sc] = nd.kind.sym
            cbuf[sr][sc] = col

            # Label (only for near nodes, z > 0)
            if sz > 0.1 and len(nd.label) > 0:
                lbl = nd.label[:10]
                start = sc + 1
                for k, ch in enumerate(lbl):
                    c2 = start + k
                    if 0 <= c2 < cols and zbuf[sr][c2] <= sz:
                        buf[sr][c2] = ch
                        cbuf[sr][c2] = C["dim"]
                        zbuf[sr][c2] = sz

        # Compose HTML — run-length encode colors for efficiency
        lines = []
        for r in range(rows):
            parts = []
            cur_col = None
            seg = []
            for c_ in range(cols):
                if cbuf[r][c_] != cur_col:
                    if seg:
                        parts.append(f"<style fg='{cur_col}'>{_html.escape(''.join(seg))}</style>")
                    seg = [buf[r][c_]]
                    cur_col = cbuf[r][c_]
                else:
                    seg.append(buf[r][c_])
            if seg:
                parts.append(f"<style fg='{cur_col}'>{_html.escape(''.join(seg))}</style>")
            lines.append("".join(parts))

        # Stats bar
        n_active = sum(1 for nd in self.nodes.values() if nd.active)
        n_ok = sum(1 for nd in self.nodes.values() if nd.success is True)
        n_fail = sum(1 for nd in self.nodes.values() if nd.success is False)
        n_edges = len(self.edges)

        stats = (
            f"<style fg='{C['cyan']}'>◎ {len(self.nodes)}nodes</style>"
            f" <style fg='{C['dim']}'>{n_edges}edges</style>"
            f" <style fg='{C['green']}'>✓{n_ok}</style>"
            f" <style fg='{C['red']}'>✗{n_fail}</style>"
            f" <style fg='{C['bright']}'>●{n_active}active</style>"
            f" <style fg='{C['dim']}'>{'⏸' if self._paused else '▸'}f{self._frame}"
            f" az={self.azimuth:.1f} el={self.elevation:.1f}</style>"
        )
        lines.append(stats)

        return "\n".join(lines)

    def print_frame(self, cols: int = 0, rows: int = 0):
        """Print one frame to terminal via prompt_toolkit."""
        html_str = self.render(cols, rows)
        try:
            print_formatted_text(HTML(html_str))
        except Exception:
            import re
            print(re.sub(r"<[^>]+>", "", html_str))

    # ── Legend ───────────────────────────────────────────────────────

    @staticmethod
    def print_legend():
        parts = []
        for nk in NK:
            parts.append(f"<style fg='{nk.color}'>{nk.sym} {nk.name.lower()}</style>")
        legend = "  ".join(parts)
        edge_parts = []
        for es in ES:
            edge_parts.append(f"<style fg='{es.color}'>{es.ch}{es.label_}</style>")
        edges_legend = "  ".join(edge_parts)
        controls = (
            f"<style fg='{C['dim']}'>"
            "q/e:rotate  w/s:tilt  +/-:zoom  Space:pause  Tab:focus  g:toggle"
            "</style>"
        )
        try:
            print_formatted_text(HTML(f"  {legend}\n  {edges_legend}\n  {controls}"))
        except Exception:
            pass


# ── ZenRendererV2 Integration Mixin ─────────────────────────────────

class GraphPaneMixin:
    """
    Drop-in mixin for ZenRendererV2 to add 3D graph pane.

    Usage:
        class MyRenderer(GraphPaneMixin, ZenRendererV2):
            pass
        renderer = MyRenderer(engine)
        renderer.set_graph(DreamGraph3D.instance())
    """
    _graph: Optional[DreamGraph3D] = None
    _graph_visible: bool = False

    def set_graph(self, graph: DreamGraph3D):
        self._graph = graph

    def toggle_graph(self):
        self._graph_visible = not self._graph_visible
        if self._graph_visible and self._graph:
            self._graph.print_legend()

    def render_graph_pane(self, cols: int = 100, rows: int = 24):
        """Render 3D graph pane if visible. Call from your render loop."""
        if not self._graph_visible or not self._graph:
            return
        self._graph.print_frame(cols, rows)

    def handle_graph_key(self, key: str):
        """Handle keyboard input for graph camera. Returns True if consumed."""
        if not self._graph:
            return False
        km = {
            'q': lambda: self._graph.rotate(daz=-0.2),
            'e': lambda: self._graph.rotate(daz=0.2),
            'w': lambda: self._graph.rotate(del_=0.15),
            's': lambda: self._graph.rotate(del_=-0.15),
            '+': self._graph.zoom_in,
            '-': self._graph.zoom_out,
            ' ': self._graph.toggle_pause,
            'g': self.toggle_graph,
        }
        if key in km:
            km[key]()
            return True
        return False


# ── Convenience: one-call hookup ────────────────────────────────────

def hookup_dream_graph(engine=None, dreamer=None, memory_actor=None) -> DreamGraph3D:
    """
    One-liner to wire everything up:

        graph = hookup_dream_graph(engine, dreamer, memory_actor)

    Returns the singleton DreamGraph3D instance.
    """
    g = DreamGraph3D.instance()
    if engine:
        g.attach(engine)
    if dreamer:
        g.attach_dreamer(dreamer)
    if memory_actor:
        g.attach_memory(memory_actor)
    return g




# ── Live Render Loop ────────────────────────────────────────────────

async def _render_loop(graph: DreamGraph3D, interval: float = 0.25):
    """Background task: rendert den Graphen ~4 FPS bis kein active Node mehr da."""
    cols = int(os.get_terminal_size().columns) if hasattr(os, 'get_terminal_size') else 110
    rows = min(28, int(os.get_terminal_size().lines) - 4) if hasattr(os, 'get_terminal_size') else 24

    # Warte bis es Nodes gibt
    while not graph.nodes:
        await asyncio.sleep(0.1)

    graph.print_legend()

    while any(n.active for n in graph.nodes.values()):
        graph.print_frame(cols, rows)
        await asyncio.sleep(interval)

    # Finaler Frame
    graph.print_frame(cols, rows)


# ── Hauptfunktion ───────────────────────────────────────────────────

async def dream_with_viz(
    isaa_tools,
    agent_name: str = "default",
    config: Optional[DreamConfig] = None,
    show_graph: bool = True,
):
    """
    Startet a_dream() mit live 3D-Graph Visualisierung.

    Args:
        isaa_tools: Die ISAA Tools-Instanz (self in Tools)
        agent_name: Name des Agents
        config: DreamConfig oder None für defaults
        show_graph: 3D Graph anzeigen (False = nur dream ohne viz)

    Returns:
        DreamReport

    Usage in Tools:
        report = await dream_with_viz(self, "myagent", DreamConfig(max_budget=3000))
    """
    config = config or DreamConfig()
    agent = await isaa_tools.get_agent(agent_name)

    if not show_graph:
        return await agent.a_dream(config)

    # ── Graph singleton holen + hooks setzen ──

    graph = DreamGraph3D.instance()

    # Root-Node für den Agent
    graph.add_node(f"agent:{agent_name}", NK.AGENT, agent_name, active=True)

    # Dreamer hook — erzeugt den Dreamer falls nötig
    if not hasattr(agent, '_dreamer'):
        from toolboxv2.mods.isaa.base.Agent.dreamer import Dreamer
        agent._dreamer = Dreamer(agent)
    graph.attach_dreamer(agent._dreamer)

    # Engine hook — falls schon eine engine existiert
    engine = getattr(agent, '_current_engine', None)
    if engine:
        graph.attach(engine)

    # Memory hook — über session_manager
    try:
        memory = agent.session_manager._get_memory()
        if memory and hasattr(memory, 'add_data_point'):
            graph.attach_memory(memory)
    except Exception:
        pass

    # ── Parallel: Dream + Render Loop ──

    render_task = asyncio.create_task(_render_loop(graph))

    try:
        report = await agent._dreamer.dream(config)
    finally:
        # Alle Nodes deaktivieren → render_loop stoppt
        for nd in graph.nodes.values():
            nd.active = False
        await asyncio.sleep(0.3)  # letzter Frame
        render_task.cancel()
        try:
            await render_task
        except asyncio.CancelledError:
            pass

    # ── Zusammenfassung ──
    _print_report(report, graph)
    return report


def _print_report(report, graph: DreamGraph3D):
    """Kompakte Zusammenfassung nach dem Dream."""
    from prompt_toolkit import print_formatted_text, HTML
    C = {"cyan": "#67e8f9", "green": "#4ade80", "red": "#f87171",
         "dim": "#6b7280", "amber": "#fbbf24", "bright": "#ffffff"}

    n_ok = sum(1 for n in graph.nodes.values() if n.success is True)
    n_fail = sum(1 for n in graph.nodes.values() if n.success is False)

    lines = [
        f"\n<style fg='{C['cyan']}'>◎ Dream Complete: {report.dream_id}</style>",
        f"<style fg='{C['dim']}'>  Logs gescannt:  {report.logs_scanned}</style>",
        f"<style fg='{C['dim']}'>  Cluster:        {report.clusters_found}</style>",
        f"<style fg='{C['green']}'>  Skills evolved: {', '.join(report.skills_evolved) or '—'}</style>",
        f"<style fg='{C['green']}'>  Skills created: {', '.join(report.skills_created) or '—'}</style>",
        f"<style fg='{C['amber']}'>  Skills split:   {', '.join(report.skills_split) or '—'}</style>",
        f"<style fg='{C['cyan']}'>  Published:      {', '.join(report.skills_published) or '—'}</style>",
        f"<style fg='{C['dim']}'>  Memory entries:  {report.memory_entries_added}</style>",
        f"<style fg='{C['dim']}'>  Budget used:     {report.budget_used}</style>",
        f"<style fg='{C['bright']}'>  Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges</style>",
        f"<style fg='{C['green']}'>  ✓ {n_ok} ok</style>"
        f"  <style fg='{C['red']}'>✗ {n_fail} errors</style>",
    ]
    if report.errors:
        lines.append(f"<style fg='{C['red']}'>  Errors:</style>")
        for e in report.errors[:5]:
            lines.append(f"<style fg='{C['red']}'>    ✗ {e[:60]}</style>")

    try:
        for l in lines:
            print_formatted_text(HTML(l))
    except Exception:
        import re
        for l in lines:
            print(re.sub(r'<[^>]+>', '', l))
